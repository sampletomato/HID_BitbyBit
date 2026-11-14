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
        "time_ms": elapsed,
        "session_id": session_id,
        "challenge": challenge,
    }
    if extra:
        body.update(extra)
    return body


def _apply_force(verdict: str) -> str:
    """Dev override so the frontend can exercise every branch on demand."""
    return state.forced or verdict


# ------------------------------------------------------------------- endpoints

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

    images = face_mod.decode_frames(req.frames)
    quality = face_mod.assess_quality(images, config.MIN_FRAMES_VERIFY)
    if not quality.ok:
        sid = state.new_session(None, 0.0, 0.0)
        return _response(_apply_force(dec.RECAPTURE), None, 0.0, 0.0,
                         [quality.reason], t0, sid)

    name, match, burst = state.face.identify(images)

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
                     t0, sid, challenge)


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
    session["attempts"] += 1

    images = face_mod.decode_frames(req.frames)
    quality = face_mod.assess_quality(images, config.MIN_FRAMES_CHALLENGE)
    if not quality.ok:
        return _response(_apply_force(dec.RECAPTURE), session["name"], session["liveness"],
                         session["match"], [quality.reason], t0, req.session_id,
                         challenge=expected)

    passed, chal_score, reasons = state.liveness.check_challenge(
        expected["action"], images, rng)

    if passed:
        # Passing the challenge lifts liveness confidence above the ALLOW bar.
        boosted = max(session["liveness"], config.LIVENESS_ALLOW + 0.05)
        session["liveness"] = boosted
        verdict, dec_reasons = dec.decide(boosted, session["match"], session["name"],
                                          state.face.match_allow, state.face.match_deny)
        return _response(_apply_force(verdict), session["name"], boosted,
                         session["match"], reasons + dec_reasons, t0, req.session_id)

    if session["attempts"] < config.MAX_CHALLENGE_ATTEMPTS:
        retry = rng.choice(config.CHALLENGES)
        session["challenge"] = retry
        return _response(_apply_force(dec.STEP_UP), session["name"], session["liveness"],
                         session["match"], reasons + ["Please try one more challenge"],
                         t0, req.session_id, challenge=retry)

    session["totp_required"] = True
    log.info("challenge exhausted for %s, TOTP required", session["name"])
    return _response(_apply_force(dec.STEP_UP), session["name"], session["liveness"],
                     session["match"],
                     reasons + ["Challenge failed, enter the code from your authenticator app"],
                     t0, req.session_id, totp_required=True)


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
