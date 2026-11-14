# Backend — HID face liveness kiosk

FastAPI service implementing the shared API contract.

**Real so far:** frame decode, quality gate, **InsightFace buffalo_l face
recognition** (YuNet+SFace fallback), **passive anti-spoof** (MiniFASNetV2-SE),
the decision policy, and **TOTP** step-up. Blink and device-in-frame liveness
sub-signals are still stubbed (`STUB-STEP5`).

Two run modes via `MODE` (one codebase):
- `MODE=edge` (default) — kiosk laptop, runs the face models.
- `MODE=central` — EC2, owns the schedule + audit log, no face models. *(wired in step 3)*

## Run (Linux / EC2)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --only-binary=:all: -r requirements.txt   # avoids compiling numpy
pip install insightface==0.7.3                         # needs gcc; see fallback below
uvicorn app:app --host 0.0.0.0 --port 8000
```

First launch downloads the buffalo_l model pack (~280 MB) into `models/`.

## Run (Windows, local dev)

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --only-binary=:all: -r requirements.txt
pip install insightface==0.7.3
uvicorn app:app --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

## Face backend

`FACE_BACKEND=auto` (default) tries InsightFace buffalo_l first, then falls back
to OpenCV YuNet + SFace. If InsightFace is painful to install on EC2, use the
lighter fallback (opencv + onnxruntime only, ~37 MB of models):

```bash
python tools/fetch_models.py
export FACE_BACKEND=opencv
```

Match thresholds differ per backend (the two produce non-comparable cosine
scores) — both sets live in `config.MATCH_THRESHOLDS`. Re-enroll everyone if you
switch backends; the server logs a warning for enrollments made under the other.

## Passive anti-spoof

MiniFASNetV2-SE (INT8) from
[facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx)
(Apache-2.0). **Required** — `tools/fetch_models.py` downloads it (~0.6 MB).
Crop + preprocess are copied verbatim from that repo (`liveness.py`): RGB,
bbox expanded 1.5×, letterbox to 128×128, `/255`, CHW. Output logits
`[real, spoof]` → real-prob = sigmoid(real − spoof).

It runs on the 3 sharpest **already-detected** frames (reusing the InsightFace
bbox, no second detector pass), averaging the real-probability — ~20 ms added.
A real-prob below `config.ANTISPOOF_SPOOF_VETO` (0.30) hard-vetoes to **DENY**
on its own, so an obvious print/screen attack is rejected even while blink and
device are still stubbed. Reason strings: *"Real skin texture (0.97)"* /
*"Print/screen texture detected (0.08)"*.

```bash
python tools/test_antispoof.py   # scores sample faces + measures latency
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"ok": true, "mode": ..., "face_backend": ...}` |
| POST | `/enroll` | register a name from 5 frames; returns a TOTP QR |
| POST | `/verify` | decide from a ~1 s, 10-frame burst |
| POST | `/challenge` | check an active challenge from a ~2 s, 20-frame burst |
| POST | `/otp` | TOTP step-up check |

Frames are 640×480 JPEG (quality 0.8) base64 data-URL strings. Bare base64
without the `data:image/jpeg;base64,` prefix is also accepted.

### Contract (current)

`/verify` and `/challenge` return:

```json
{
  "decision": "ALLOW" | "STEP_UP" | "DENY" | "RECAPTURE",
  "name": "Divya" | null,
  "liveness": 0.93, "match": 0.71,
  "mode": "online" | "offline",
  "schedule_reason": "string",
  "totp_required": false,
  "reasons": ["..."], "time_ms": 1240,
  "session_id": "a1b2c3",
  "challenge": null | {"action": "BLINK_TWICE", "text": "Please blink twice"}
}
```

`/enroll` returns `{ok, name, faces_used, totp_qr}` — `totp_qr` is a PNG data URL
for Google Authenticator. `totp_secret` is also returned while `DEV_MODE=1`.
`/otp` takes `{session_id, code}` and checks a TOTP code (`pyotp`), with replay
protection and a ±1 step skew window.

> **Renamed:** the old `VERIFY` decision is now `STEP_UP`. `schedule_reason` is a
> placeholder until schedule enforcement lands in step 3; `mode` is always
> `online` until the offline fail-safe lands in step 4.

## Dev helpers

Available while `DEV_MODE=1` (the default). Set `DEV_MODE=0` for the final demo.

```bash
curl localhost:8000/dev/users                     # enrolled names + thresholds
curl "localhost:8000/dev/force?decision=STEP_UP"  # pin every decision
curl "localhost:8000/dev/force?decision=off"      # back to normal
curl localhost:8000/dev/totp/Divya                # current valid TOTP code
curl localhost:8000/dev/session/a1b2c3            # inspect a session
```

`/dev/force` drives the kiosk UI through ALLOW, STEP_UP, DENY and RECAPTURE
without a real spoof in front of the camera. `/dev/totp/<name>` lets the demo
complete the step-up without an authenticator app.

## Tests

```bash
python tools/test_face_match.py   # genuine vs impostor separation + timing (no server)
python tools/smoke_test.py        # full API walk (needs a running server)
python tools/smoke_test.py http://1.2.3.4:8000
```

`test_face_match.py` is the source of the FAR/FRR numbers for the eval report.
Both use the face photos bundled inside the `insightface` package — no webcam.

## Notes

- All thresholds live in `config.py`. Nothing else hard-codes one.
- Models load once in the FastAPI lifespan handler, never per request.
- Enrollments (embeddings + TOTP secret) persist to `backend/store/gallery.json`
  (gitignored). This file stays on the **edge** device — embeddings and secrets
  are never sent to the central server.
- `ORT_THREADS` caps the ONNX thread pool (default `min(4, ncpu)`). The two face
  sessions share settings deliberately: uncapped, they fought for threads and
  cost ~2.3 s/frame instead of ~110 ms.
