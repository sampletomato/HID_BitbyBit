"""FastAPI service for the HID face liveness + authentication kiosk.

Implements the shared frontend/backend contract:
    GET  /health
    POST /enroll      -> also returns a TOTP provisioning QR
    POST /verify      -> ALLOW | STEP_UP | DENY | RECAPTURE
    POST /challenge   -> active challenge check, falls back to TOTP
    POST /otp         -> TOTP check

/verify, /challenge and /otp all accept an optional "sim_time" (ISO datetime)
so the demo can show the campus schedule rules without waiting for the clock.

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
import messages as msg
import policy
import totp as totp_mod

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

# Shown while no user is identified yet (a retake, an unknown face): the zone
# this kiosk guards, without claiming anything about a schedule we cannot look
# up. Once a name is known, policy.py supplies the real line.
ZONE_LINE = f"{policy.zone_label(config.KIOSK_ZONE)} {policy.DOT} access schedule applies"


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


def _warmup():
    """Run the full verify pipeline a few times at startup so the first real
    request is not cold. Uses the cached sample faces when present; otherwise
    best-effort and skipped silently (e.g. a fresh deploy)."""
    import glob
    import random as _random
    import cv2
    sample = sorted(glob.glob(str(config.STORE_DIR / "sample_frames" / "*.jpg")))
    if not sample:
        log.info("warmup skipped (no sample frames)")
        return
    img = cv2.imread(sample[0])
    if img is None:
        return
    frames = [img] * 10
    t0 = time.time()
    for _ in range(3):
        try:
            _, _, burst = state.face.identify(frames)
            if burst.detections:
                state.liveness.score(frames, burst.detections, _random.Random())
        except Exception as exc:
            log.warning("warmup iteration failed: %s", exc)
            break
    log.info("warmup done in %.2fs", time.time() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    state.face.load()
    state.liveness.load()
    # Pull the schedule now so the first real /verify does not pay for the
    # Supabase round trip, and so the log says which source is in play.
    rows = policy.store.load(force=True)
    log.info("schedule: %d row(s) from %s, zone=%s, grace=%d min",
             len(rows), policy.store.source, config.KIOSK_ZONE,
             config.SCHEDULE_GRACE_MIN)
    _warmup()
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


class EnrollConfirmRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    code: str


class VerifyRequest(BaseModel):
    frames: List[str] = Field(default_factory=list)
    # Demo only: pretend it is this wall-clock time when applying the schedule.
    sim_time: Optional[str] = None


class ChallengeRequest(BaseModel):
    session_id: str
    frames: List[str] = Field(default_factory=list)
    sim_time: Optional[str] = None


class OtpRequest(BaseModel):
    session_id: str
    code: str
    sim_time: Optional[str] = None


# -------------------------------------------------------------------- helpers

def _response(verdict: str, name: Optional[str], liveness: float, match: float,
              reasons: List[str], t0: float, session_id: str,
              challenge: Optional[dict] = None, totp_required: bool = False,
              schedule_reason: Optional[str] = None,
              scores: Optional[dict] = None,
              reason_code: Optional[str] = None,
              extra: Optional[dict] = None) -> dict:
    elapsed = int((time.time() - t0) * 1000)
    if elapsed > config.TARGET_MS:
        log.warning("over budget: %d ms (target %d ms)", elapsed, config.TARGET_MS)

    # One clear sentence for the kiosk. A pending TOTP normally wins the copy, so
    # the screen asks for the code rather than repeating the challenge -- except
    # for the schedule grace band, which has its own wording and also wants a code.
    if (verdict == dec.STEP_UP and totp_required
            and reason_code != msg.SCHEDULE_GRACE):
        reason_code = msg.STEP_UP_TOTP
    code = msg.code_for(verdict, reason_code)
    title, message = msg.render(code, name=name,
                                challenge=(challenge or {}).get("text"))

    body = {
        "decision": verdict,
        "name": name,
        "title": title,
        "message": message,
        "reason_code": code,
        "liveness": round(liveness, 2),
        "match": round(match, 2),
        "mode": "online",            # step 4 sets this from the central link state
        "schedule_reason": schedule_reason or ZONE_LINE,
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


# Schedule outcomes that nothing can rescue: the person may be live and
# recognised, but they are not authorised to be here now.
SCHEDULE_HARD_DENY = (policy.OUT_OF_WINDOW, policy.WRONG_ZONE,
                      policy.EXPIRED, policy.NO_SCHEDULE)


def _apply_schedule(verdict: str, name: Optional[str], sim_time: Optional[str]):
    """Fold the campus schedule into a biometric verdict.

    Returns (verdict, reason_code, schedule_reason, totp_required, reasons).
    `reason_code` is None when the schedule had nothing to add, leaving the
    biometric code in place. A biometric DENY/RECAPTURE never reaches here --
    the schedule may restrict access, never grant it.
    """
    sched = policy.evaluate(name, policy.parse_sim_time(sim_time))
    reasons = [f"Schedule: {sched.reason}"]

    if sched.status in SCHEDULE_HARD_DENY:
        return (dec.DENY, msg.SCHEDULE_DENY, sched.reason, False,
                reasons + ["Not authorised to enter at this time"])

    if verdict == dec.ALLOW and sched.grace:
        return (dec.STEP_UP, msg.SCHEDULE_GRACE, sched.reason, True,
                reasons + [f"Within the {config.SCHEDULE_GRACE_MIN}-minute grace "
                           "band, authenticator code required"])

    # Inside the window (or a biometric STEP_UP that still has its own
    # challenge to pass -- the schedule is re-checked once it does).
    return verdict, None, sched.reason, False, reasons


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


@app.post("/enroll/confirm")
def enroll_confirm(req: EnrollConfirmRequest):
    """Confirm a freshly enrolled user's authenticator by checking a TOTP code."""
    name = req.name.strip()
    secret = state.face.gallery.totp_secret(name)
    if not secret:
        return {"ok": False, "name": name, "reason": "Unknown user"}

    ok, step, error = totp_mod.verify(secret, req.code,
                                      state.face.gallery.totp_last_step(name))
    if not ok:
        log.info("enroll confirm rejected for %s: %s", name, error)
        return {"ok": False, "name": name, "reason": error}

    state.face.gallery.consume_totp_step(name, step)
    state.face.gallery.mark_confirmed(name)
    log.info("authenticator confirmed for %s", name)
    return {"ok": True, "name": name}


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
                         [quality.reason], t0, sid,
                         reason_code=msg.recapture_code(quality.reason))

    name, match, burst = state.face.identify(images)
    t_identify = time.time()
    log.info("verify stages: decode+quality=%.0fms identify=%.0fms",
             (t_quality - t_decode) * 1000, (t_identify - t_quality) * 1000)

    # No usable face is a capture problem, not a spoof -- ask for another try.
    # The passive model needs a bbox, so it only runs once a face is found.
    if burst.query is None:
        sid = state.new_session(None, 0.0, 0.0)
        return _response(_apply_force(dec.RECAPTURE), None, 0.0, 0.0,
                         [burst.note], t0, sid,
                         reason_code=msg.recapture_code(burst.note))

    live = state.liveness.score(images, burst.detections, rng)

    # Low light is a capture problem, not an attack: the passive model reads an
    # underexposed real face as flat print texture. Ask for a retake instead of
    # denying a real person. (The hard spoof veto still applies in normal light.)
    if live.too_dark:
        sid = state.new_session(name, live.score, match)
        return _response(_apply_force(dec.RECAPTURE), name, live.score, match,
                         ["Too dark to check liveness, please face the light"],
                         t0, sid, scores=_scores(live, match),
                         reason_code=msg.RECAPTURE_DARK)

    verdict, reasons, code = dec.decide(live.score, match, name,
                                  state.face.match_allow, state.face.match_deny,
                                  live.device_seen, live.spoof_detected)

    # A genuine biometric DENY (spoof, device in frame, unknown face, or match
    # below the deny threshold) is a hard reject. /dev/force may demo the other
    # branches, but it must never upgrade a real DENY into STEP_UP/ALLOW: an
    # unknown or spoofed person never gets a step-up and never carries a name.
    if verdict == dec.DENY:
        sid = state.new_session(None, live.score, match)
        return _response(dec.DENY, None, live.score, match, reasons + live.reasons,
                         t0, sid, scores=_scores(live, match), reason_code=code)

    verdict = _apply_force(verdict)

    # Being live and recognised is not authorisation: the campus schedule has
    # the last word. It can only downgrade (ALLOW -> STEP_UP -> DENY).
    verdict, sched_code, sched_reason, totp_required, sched_reasons = (
        _apply_schedule(verdict, name, req.sim_time))
    code = sched_code or code
    reasons = reasons + sched_reasons

    sid = state.new_session(name, live.score, match)
    state.sessions[sid]["sim_time"] = req.sim_time
    challenge = None
    if verdict == dec.STEP_UP and totp_required:
        # Grace band: a code only, never a face challenge.
        state.sessions[sid]["totp_required"] = True
    elif verdict == dec.STEP_UP:
        challenge = rng.choice(config.CHALLENGES)
        state.sessions[sid]["challenge"] = challenge

    return _response(verdict, name, live.score, match, reasons + live.reasons,
                     t0, sid, challenge, totp_required=totp_required,
                     schedule_reason=sched_reason, scores=_scores(live, match),
                     reason_code=code)


@app.post("/challenge")
def challenge(req: ChallengeRequest):
    t0 = time.time()
    rng = random.Random()

    state.prune_sessions()
    session = state.sessions.get(req.session_id)
    if session is None:
        return _response(dec.DENY, None, 0.0, 0.0,
                         ["Session expired, please start again"], t0, req.session_id,
                         reason_code=msg.DENY_VERIFICATION_FAILED)

    expected = session.get("challenge") or rng.choice(config.CHALLENGES)
    enrolled = session["name"]
    # A sim_time on this request wins; otherwise reuse the one /verify saw, so
    # the whole demo flow stays at the same simulated moment.
    sim_time = req.sim_time or session.get("sim_time")
    session["sim_time"] = sim_time

    images = face_mod.decode_frames(req.frames)
    quality = face_mod.assess_quality(images, config.MIN_FRAMES_CHALLENGE)
    if not quality.ok:
        return _response(_apply_force(dec.RECAPTURE), enrolled, session["liveness"],
                         session["match"], [quality.reason], t0, req.session_id,
                         challenge=expected,
                         reason_code=msg.recapture_code(quality.reason))

    # Re-check identity and anti-spoof on the challenge frames (not just the action).
    name, match, burst = state.face.identify(images)
    if burst.query is None:
        return _response(_apply_force(dec.RECAPTURE), enrolled, 0.0, 0.0,
                         [burst.note], t0, req.session_id, challenge=expected,
                         reason_code=msg.recapture_code(burst.note))
    passive_avg, _, _ = state.liveness.passive.score_detections(burst.detections)
    brightness = liveness_mod.face_brightness(burst.detections)
    too_dark = brightness is not None and brightness < config.MIN_FACE_BRIGHTNESS
    # Same rule as /verify: in low light the passive model is not trusted, so a
    # dark face gets a retake rather than a spoof DENY.
    spoof = (passive_avg is not None
             and passive_avg < config.ANTISPOOF_HARD_DENY and not too_dark)

    scores = {"passive": round(passive_avg, 3) if passive_avg is not None else None,
              "face_brightness": round(brightness, 1) if brightness is not None else None,
              "match": round(match, 2)}

    if too_dark:
        return _response(_apply_force(dec.RECAPTURE), enrolled, session["liveness"],
                         match, ["Too dark to check liveness, please face the light"],
                         t0, req.session_id, challenge=expected, scores=scores,
                         reason_code=msg.RECAPTURE_DARK)

    # Hard veto FIRST: a spoof during the challenge is an immediate DENY, before
    # we even look at the action. This is not overridable by /dev/force, and it
    # never offers TOTP -- a spoofed presentation is rejected outright.
    if spoof:
        session["totp_required"] = False
        return _response(dec.DENY, None, 0.0, match,
                         ["Spoof detected: print/screen texture"],
                         t0, req.session_id, scores=scores,
                         reason_code=msg.DENY_SPOOF)

    # Did they perform the requested action?
    passed, chal_score, chal_reasons = state.liveness.check_challenge(expected["action"], images)
    same_person = name == enrolled and match >= state.face.match_allow
    scores["challenge"] = chal_score
    # A face that no longer matches the enrolled user is a hard DENY: no name, no
    # TOTP, not overridable by /dev/force. Only a failed ACTION by a matching,
    # live face (handled below) may fall back to TOTP.
    if not same_person:
        session["totp_required"] = False
        return _response(dec.DENY, None, 0.0, match,
                         chal_reasons + ["Face no longer matches the enrolled user"],
                         t0, req.session_id, scores=scores,
                         reason_code=msg.DENY_UNKNOWN_FACE)

    if passed:
        live_score = max(session["liveness"], config.LIVENESS_ALLOW + 0.05)
        session["liveness"] = live_score
        log.info("challenge '%s' passed for %s", expected["action"], enrolled)
        # Passing the challenge settles the biometric half only -- the schedule
        # must still authorise the entry.
        verdict, sched_code, sched_reason, totp_required, sched_reasons = (
            _apply_schedule(dec.ALLOW, enrolled, sim_time))
        if totp_required:
            session["totp_required"] = True
        elif verdict == dec.DENY:
            session["totp_required"] = False
        return _response(_apply_force(verdict) if verdict != dec.DENY else verdict,
                         enrolled, live_score, match,
                         chal_reasons + [f"Identity confirmed as {enrolled} ({match:.2f})"]
                         + sched_reasons,
                         t0, req.session_id, totp_required=totp_required,
                         schedule_reason=sched_reason, scores=scores,
                         reason_code=sched_code or msg.ALLOW)

    # Action failed -> fall back to TOTP, unless the schedule rules it out
    # anyway (no point burning a code on an entry that cannot be authorised).
    sched = policy.evaluate(enrolled, policy.parse_sim_time(sim_time))
    if sched.status in SCHEDULE_HARD_DENY:
        session["totp_required"] = False
        log.info("challenge '%s' failed for %s and schedule denies (%s)",
                 expected["action"], enrolled, sched.status)
        return _response(dec.DENY, enrolled, session["liveness"], match,
                         chal_reasons + [f"Schedule: {sched.reason}"],
                         t0, req.session_id, schedule_reason=sched.reason,
                         scores=scores, reason_code=msg.SCHEDULE_DENY)

    session["totp_required"] = True
    log.info("challenge '%s' failed for %s, TOTP required", expected["action"], enrolled)
    return _response(_apply_force(dec.STEP_UP), enrolled, session["liveness"], match,
                     chal_reasons + ["Challenge not completed, enter the code from your authenticator app"],
                     t0, req.session_id, totp_required=True,
                     schedule_reason=sched.reason, scores=scores,
                     reason_code=msg.STEP_UP_TOTP)


def _otp_response(verdict: str, name: Optional[str], reason: str,
                  reason_code: str, schedule_reason: Optional[str] = None) -> dict:
    """/otp keeps its short shape but carries the same title/message/reason_code."""
    code = msg.code_for(verdict, reason_code)
    title, message = msg.render(code, name=name)
    return {"decision": verdict, "name": name, "title": title, "message": message,
            "reason_code": code, "reason": reason,
            "schedule_reason": schedule_reason or ZONE_LINE}


@app.post("/otp")
def otp(req: OtpRequest):
    """TOTP check. Replaces the old SMS-style OTP."""
    state.prune_sessions()
    session = state.sessions.get(req.session_id)
    if session is None:
        return _otp_response(dec.DENY, None, "Session expired",
                             msg.DENY_VERIFICATION_FAILED)
    if not session.get("totp_required"):
        return _otp_response(dec.DENY, session["name"],
                             "No step-up pending for this session",
                             msg.DENY_VERIFICATION_FAILED)

    name = session["name"]
    secret = state.face.gallery.totp_secret(name) if name else None
    if not secret:
        return _otp_response(dec.DENY, name, "No authenticator enrolled",
                             msg.DENY_VERIFICATION_FAILED)

    ok, step, error = totp_mod.verify(secret, req.code,
                                      state.face.gallery.totp_last_step(name))
    if not ok:
        log.info("TOTP rejected for %s: %s", name, error)
        return _otp_response(dec.DENY, name, error, msg.DENY_VERIFICATION_FAILED)

    state.face.gallery.consume_totp_step(name, step)
    session["totp_required"] = False

    # Last gate: a correct code still does not override the schedule. The grace
    # band is fine here -- a code is exactly what it asked for.
    sim_time = req.sim_time or session.get("sim_time")
    session["sim_time"] = sim_time
    sched = policy.evaluate(name, policy.parse_sim_time(sim_time))
    if sched.status in SCHEDULE_HARD_DENY:
        log.info("TOTP accepted for %s but schedule denies (%s)", name, sched.status)
        return _otp_response(dec.DENY, name, sched.reason, msg.SCHEDULE_DENY,
                             sched.reason)

    log.info("TOTP accepted for %s", name)
    return _otp_response(dec.ALLOW, name, "Authenticator code accepted", msg.ALLOW,
                         sched.reason)


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
