"""FastAPI service for the HID face liveness + authentication kiosk.

Implements the shared frontend/backend contract:
    GET  /health
    POST /enroll      -> also returns a TOTP provisioning QR
    POST /verify      -> ALLOW | STEP_UP | DENY | RECAPTURE
    POST /challenge   -> active challenge check, falls back to TOTP
    POST /otp         -> TOTP check

Dev-only helpers (DEV_MODE=1, the default):
    GET  /dev/users
    GET  /dev/force?decision=ALLOW|STEP_UP|DENY|RECAPTURE|off
    GET  /dev/totp/{name}
    GET  /dev/session/{session_id}

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import logging
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import decision as dec
import face as face_mod
import liveness as liveness_mod
import totp as totp_mod

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

# Step 3 replaces this with a real schedule lookup via policy.py.
SCHEDULE_PENDING = "Schedule enforcement not yet enabled"


# ------------------------------------------------------------------- app state

class State:
    """Everything loaded once at startup lives here."""

    def __init__(self):
        self.face = face_mod.FaceEngine()
        self.liveness = liveness_mod.LivenessEngine()
        self.sessions: Dict[str, dict] = {}
        self.forced: Optional[str] = None  # dev override

    def new_session(self, name: Optional[str], liveness: float, match: float) -> str:
        self.prune_sessions()
        sid = uuid.uuid4().hex[:6]
        self.sessions[sid] = {
            "name": name,
            "liveness": liveness,
            "match": match,
            "created": time.time(),
            "attempts": 0,
            "challenge": None,
            "totp_required": False,
        }
        return sid

    def prune_sessions(self):
        cutoff = time.time() - config.SESSION_TTL_S
        for sid in [s for s, v in self.sessions.items() if v["created"] < cutoff]:
            self.sessions.pop(sid, None)


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    state.face.load()
    state.liveness.load()
    log.info("ready in %.2fs (MODE=%s, face=%s, DEV_MODE=%s)",
             time.time() - t0, config.MODE, state.face.backend_name, config.DEV_MODE)
    yield
    log.info("shutting down")


app = FastAPI(title="HID Face Liveness Kiosk", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------- request models

class EnrollRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    frames: List[str] = Field(default_factory=list)


class VerifyRequest(BaseModel):
    frames: List[str] = Field(default_factory=list)


class ChallengeRequest(BaseModel):
    session_id: str
    frames: List[str] = Field(default_factory=list)


class OtpRequest(BaseModel):
    session_id: str
    code: str


# -------------------------------------------------------------------- helpers

def _response(verdict: str, name: Optional[str], liveness: float, match: float,
              reasons: List[str], t0: float, session_id: str,
              challenge: Optional[dict] = None, totp_required: bool = False,
              schedule_reason: str = SCHEDULE_PENDING,
              scores: Optional[dict] = None,
              extra: Optional[dict] = None) -> dict:
    elapsed = int((time.time() - t0) * 1000)
    if elapsed > config.TARGET_MS:
        log.warning("over budget: %d ms (target %d ms)", elapsed, config.TARGET_MS)
    body = {
        "decision": verdict,
        "name": name,
        "liveness": round(liveness, 2),
        "match": round(match, 2),
        "mode": "online",            # step 4 sets this from the central link state
        "schedule_reason": schedule_reason,
        "totp_required": totp_required,
        "reasons": reasons,
        "scores": scores or {},      # per-signal breakdown for the demo panel
        "time_ms": elapsed,
        "session_id": session_id,
        "challenge": challenge,
    }
    if extra:
        body.update(extra)
    return body


def _scores(live, match: float) -> dict:
    """Flatten the per-signal liveness parts + match into one small dict."""
    p = live.parts
    return {"passive": p.get("passive"), "motion": p.get("motion"),
            "blink": p.get("blink"), "blink_detected": p.get("blink_detected"),
            "device": p.get("device"), "yaw_range": p.get("yaw_range"),
            "match": round(match, 2),
            "latency_ms": {k: p[k] for k in ("passive_ms", "blinkmotion_ms", "device_ms")
                           if k in p}}


def _apply_force(verdict: str) -> str:
    """Dev override so the frontend can exercise every branch on demand."""
    return state.forced or verdict


# ------------------------------------------------------------------- endpoints

@app.get("/")
def root():
    """Friendly status page -- this is the API, the kiosk UI is the frontend."""
    return {
        "service": "HID Face Liveness Kiosk API",
        "ok": True,
        "mode": config.MODE,
        "face_backend": state.face.backend_name,
        "note": "This is the API only. Open the frontend (Frontend/index.html) for the kiosk UI.",
        "endpoints": ["GET /health", "POST /enroll", "POST /verify",
                      "POST /challenge", "POST /otp", "GET /docs"],
    }


@app.get("/health")
def health():
    return {"ok": True, "mode": config.MODE, "face_backend": state.face.backend_name}


@app.post("/enroll")
def enroll(req: EnrollRequest):
    name = req.name.strip()
    images = face_mod.decode_frames(req.frames)
    quality = face_mod.assess_quality(images, config.MIN_FRAMES_ENROLL)
    if not quality.ok:
        return {"ok": False, "name": name, "faces_used": 0, "reason": quality.reason}

    secret, qr = totp_mod.enroll_secret(name)
    faces_used, error = state.face.enroll(name, images, secret)
    if error:
        return {"ok": False, "name": name, "faces_used": 0, "reason": error}

    log.info("enrolled %s: %d face(s), backend=%s",
             name, faces_used, state.face.backend_name)
    body = {"ok": True, "name": name, "faces_used": faces_used, "totp_qr": qr}
    if config.DEV_MODE:
        body["totp_secret"] = secret
    return body


@app.post("/verify")
def verify(req: VerifyRequest):
    t0 = time.time()
    rng = random.Random()

    t_decode = time.time()
    images = face_mod.decode_frames(req.frames)
    quality = face_mod.assess_quality(images, config.MIN_FRAMES_VERIFY)
    t_quality = time.time()
    if not quality.ok:
        sid = state.new_session(None, 0.0, 0.0)
        return _response(_apply_force(dec.RECAPTURE), None, 0.0, 0.0,
                         [quality.reason], t0, sid)

    name, match, burst = state.face.identify(images)
    t_identify = time.time()
    log.info("verify stages: decode+quality=%.0fms identify=%.0fms",
             (t_quality - t_decode) * 1000, (t_identify - t_quality) * 1000)

    # No usable face is a capture problem, not a spoof -- ask for another try.
    # The passive model needs a bbox, so it only runs once a face is found.
    if burst.query is None:
        sid = state.new_session(None, 0.0, 0.0)
        return _response(_apply_force(dec.RECAPTURE), None, 0.0, 0.0,
                         [burst.note], t0, sid)

    live = state.liveness.score(images, burst.detections, rng)

    verdict, reasons = dec.decide(live.score, match, name,
                                  state.face.match_allow, state.face.match_deny,
                                  live.device_seen, live.spoof_detected)
    verdict = _apply_force(verdict)

    sid = state.new_session(name, live.score, match)
    challenge = None
    if verdict == dec.STEP_UP:
        challenge = rng.choice(config.CHALLENGES)
        state.sessions[sid]["challenge"] = challenge

    return _response(verdict, name, live.score, match, reasons + live.reasons,
                     t0, sid, challenge, scores=_scores(live, match))


@app.post("/challenge")
def challenge(req: ChallengeRequest):
    t0 = time.time()
    rng = random.Random()

    state.prune_sessions()
    session = state.sessions.get(req.session_id)
    if session is None:
        return _response(dec.DENY, None, 0.0, 0.0,
                         ["Session expired, please start again"], t0, req.session_id)

    expected = session.get("challenge") or rng.choice(config.CHALLENGES)
    enrolled = session["name"]

    images = face_mod.decode_frames(req.frames)
    quality = face_mod.assess_quality(images, config.MIN_FRAMES_CHALLENGE)
    if not quality.ok:
        return _response(_apply_force(dec.RECAPTURE), enrolled, session["liveness"],
                         session["match"], [quality.reason], t0, req.session_id,
                         challenge=expected)

    # Re-check identity and anti-spoof on the challenge frames (not just the action).
    name, match, burst = state.face.identify(images)
    if burst.query is None:
        return _response(_apply_force(dec.RECAPTURE), enrolled, 0.0, 0.0,
                         [burst.note], t0, req.session_id, challenge=expected)
    passive_avg, _, _ = state.liveness.passive.score_detections(burst.detections)
    spoof = passive_avg is not None and passive_avg < config.ANTISPOOF_SPOOF_VETO

    # Did they perform the requested action?
    passed, chal_score, chal_reasons = state.liveness.check_challenge(expected["action"], images)
    same_person = name == enrolled and match >= state.face.match_allow

    scores = {"passive": round(passive_avg, 3) if passive_avg is not None else None,
              "match": round(match, 2), "challenge": chal_score}

    # A spoof or a different face during the challenge is a hard DENY.
    if spoof:
        return _response(_apply_force(dec.DENY), enrolled, session["liveness"], match,
                         chal_reasons + [f"Print/screen texture detected ({passive_avg:.2f})"],
                         t0, req.session_id, scores=scores)
    if not same_person:
        return _response(_apply_force(dec.DENY), enrolled, session["liveness"], match,
                         chal_reasons + ["Face no longer matches the enrolled user"],
                         t0, req.session_id, scores=scores)

    if passed:
        live_score = max(session["liveness"], config.LIVENESS_ALLOW + 0.05)
        session["liveness"] = live_score
        log.info("challenge '%s' passed for %s", expected["action"], enrolled)
        return _response(_apply_force(dec.ALLOW), enrolled, live_score, match,
                         chal_reasons + [f"Identity confirmed as {enrolled} ({match:.2f})"],
                         t0, req.session_id, scores=scores)

    # Action failed -> fall back to TOTP.
    session["totp_required"] = True
    log.info("challenge '%s' failed for %s, TOTP required", expected["action"], enrolled)
    return _response(_apply_force(dec.STEP_UP), enrolled, session["liveness"], match,
                     chal_reasons + ["Challenge not completed, enter the code from your authenticator app"],
                     t0, req.session_id, totp_required=True, scores=scores)


@app.post("/otp")
def otp(req: OtpRequest):
    """TOTP check. Replaces the old SMS-style OTP."""
    state.prune_sessions()
    session = state.sessions.get(req.session_id)
    if session is None:
        return {"decision": dec.DENY, "name": None, "reason": "Session expired"}
    if not session.get("totp_required"):
        return {"decision": dec.DENY, "name": session["name"],
                "reason": "No step-up pending for this session"}

    name = session["name"]
    secret = state.face.gallery.totp_secret(name) if name else None
    if not secret:
        return {"decision": dec.DENY, "name": name, "reason": "No authenticator enrolled"}

    ok, step, error = totp_mod.verify(secret, req.code,
                                      state.face.gallery.totp_last_step(name))
    if not ok:
        log.info("TOTP rejected for %s: %s", name, error)
        return {"decision": dec.DENY, "name": name, "reason": error}

    state.face.gallery.consume_totp_step(name, step)
    session["totp_required"] = False
    log.info("TOTP accepted for %s", name)
    return {"decision": dec.ALLOW, "name": name, "reason": "Authenticator code accepted"}


# ------------------------------------------------------------- dev-only routes

if config.DEV_MODE:

    @app.get("/dev/users")
    def dev_users():
        return {"backend": state.face.backend_name,
                "users": state.face.gallery.names(),
                "usable": state.face.gallery.usable_for(state.face.backend_name),
                "match_allow": state.face.match_allow,
                "match_deny": state.face.match_deny}

    @app.get("/dev/force")
    def dev_force(decision: str = "off"):
        """Pin every subsequent decision, so the kiosk UI can be exercised."""
        wanted = decision.strip().upper()
        if wanted in ("OFF", ""):
            state.forced = None
        elif wanted in dec.ALL:
            state.forced = wanted
        else:
            return {"ok": False, "error": f"use one of {', '.join(dec.ALL)} or off"}
        return {"ok": True, "forced": state.forced}

    @app.get("/dev/totp/{name}")
    def dev_totp(name: str):
        """Current valid code, so the demo works without an authenticator app."""
        secret = state.face.gallery.totp_secret(name)
        if not secret:
            return {"error": "unknown user or no secret"}
        return {"name": name, "code": totp_mod.current_code(secret)}

    @app.get("/dev/session/{session_id}")
    def dev_session(session_id: str):
        return state.sessions.get(session_id) or {"error": "unknown session"}
