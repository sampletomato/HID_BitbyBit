# Backend — HID face liveness kiosk

FastAPI service implementing the shared API contract.

**Real so far:** frame decode, quality gate, **InsightFace buffalo_l face
recognition** (YuNet+SFace fallback), a full **liveness stack** (passive
anti-spoof + MediaPipe blink/head-motion + YOLO device-in-frame), the decision
policy, **TOTP** step-up, and the **campus access schedule** (a live,
recognised person is still only authorised inside their window).

Two run modes via `MODE` (one codebase):
- `MODE=edge` (default) — kiosk laptop, runs the face models.
- `MODE=central` — EC2, owns the audit log, no face models. *(audit in step 4)*

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
bbox, no second detector pass), averaging the real-probability — ~25 ms added.
Reason strings: *"Real skin texture (0.97)"* /
*"Print/screen texture detected (0.08)"*.

**The veto is deliberately loose** (recalibrated after it rejected real faces
in ordinary indoor light at 0.30). It fires only on a *confident* spoof:

- the burst average real-prob is below `ANTISPOOF_HARD_DENY` (**0.15**), **or**
- any single frame is below `ANTISPOOF_FRAME_PANIC` (**0.05**) — so one clearly
  spoofed frame is not averaged away by two good ones.

Anything between that and `LIVENESS_ALLOW` is **not** denied: it goes to the
random active challenge, which is what actually catches a replay. A graded
liveness score no longer rejects anyone on its own.

```bash
python tools/test_antispoof.py   # scores sample faces + measures latency
```

## Liveness fusion (passive + blink/motion + device)

Final liveness is a weighted mix of three graded signals, with two hard DENY
flags on top. Weights and thresholds are all in `config.py`.

| signal | model | contributes |
|---|---|---|
| passive | MiniFASNetV2-SE | `W_PASSIVE` (0.60) + hard veto (see above) |
| head motion | MediaPipe Face Landmarker | `W_MOTION` (0.25) — yaw/pitch range across the 10 frames |
| blink | MediaPipe Face Landmarker | `W_BLINK` (0.15) — EAR dip (open→closed) |
| device | YOLOv8n | **hard DENY** if a `cell phone`/`laptop`/`tv` box overlaps the face box |

- **Motion**: a flat photo has ~0 out-of-plane rotation, so yaw/pitch range ≈ 0.
  If the landmarker yields < 2 head poses, falls back to frame-difference motion.
- **Blink**: eye-aspect-ratio per MediaPipe's 478-point mesh; needs an actual
  open→closed swing (`EAR_BLINK_DELTA`), so a static photo never "blinks".
- **Device**: `DEVICE_FACE_OVERLAP` — a phone/screen *showing* a face surrounds
  the face box (ratio ≈ 1.0), which is exactly the replay-attack geometry.
- Each signal adds a human-readable reason. Per-signal latency is logged on
  every `/verify` and surfaced in the `parts` of the test tools.

The reported liveness is the **weighted average** of the three signals. It used
to be `min(passive, fused)`, which let one soft passive reading cap the whole
score and pushed real faces below the old deny bar — that gate is gone.

**Low light comes first.** If the face crop's mean grey level is below
`MIN_FACE_BRIGHTNESS` (45) the passive model is not trusted at all: the answer
is `RECAPTURE` *"Too dark — please face a light."*, checked **before** any spoof
veto. The frame-wide bar (`MIN_BRIGHTNESS`, 20) stays relaxed — a dim frame can
still be recognised; it is the face crop that needs real light for the texture
model to mean anything.

```bash
python tools/test_liveness.py    # all signals + per-signal latency (no webcam)
python tools/webcam_test.py      # live overlay: real_prob, box, ms  (needs a camera)
```

Blink/motion needs `mediapipe`; device needs `ultralytics` (+ CPU `torch`). If
either is absent the server still runs — that signal degrades (motion falls back
to frame-difference, device check is skipped) and logs a warning. Weights are
fetched by `tools/fetch_models.py` (`face_landmarker.task`, `yolov8n.pt`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"ok": true, "mode": ..., "face_backend": ...}` |
| POST | `/enroll` | register a name from 5 frames; returns a TOTP QR |
| POST | `/verify` | decide from a ~1 s, 10-frame burst |
| POST | `/challenge` | check an active challenge from a ~2 s, 20-frame burst |
| POST | `/otp` | TOTP step-up check |

`/verify`, `/challenge` and `/otp` also accept an optional `"sim_time"` (ISO
datetime, e.g. `"2026-11-16T18:20"`): the schedule is then applied at that
wall-clock moment instead of now. Demo only — it changes nothing biometric.

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

> **Renamed:** the old `VERIFY` decision is now `STEP_UP`. `mode` is always
> `online` until the offline fail-safe lands in step 4.

## Access schedule

Biometrics answer *"is this really Divya, live, at the kiosk?"*; the schedule
answers *"is Divya allowed into this zone right now?"*. `policy.py` folds the
second into the first, and can only ever restrict:

| biometric | schedule | final |
|---|---|---|
| `ALLOW` | inside the window | `ALLOW` |
| `ALLOW` | within `SCHEDULE_GRACE_MIN` (30) of it | `STEP_UP` — authenticator code only, no face challenge (`SCHEDULE_GRACE`) |
| any | outside / wrong zone / expired / no row | `DENY` (`SCHEDULE_DENY`) |
| `STEP_UP` | allows | challenge as before, then the schedule is re-checked on the pass |
| `DENY` | anything | `DENY` — a schedule never upgrades a denial |

Every response carries `schedule_reason` in plain words, e.g.
`"Lab B · Mon–Fri 09:00–18:00"`.

**Source.** The Supabase table `schedules(user_name, zone, days, start_time,
end_time, valid_until)`, read over its REST API using `SUPABASE_URL` /
`SUPABASE_SERVICE_KEY` from `backend/.env` (see `.env.example`; the real file is
git-ignored and the keys are never logged). If Supabase is unset or unreachable
it falls back to `backend/schedule.json`, same shape, so the demo works offline.
Rows are cached for 60 s.

Which door this kiosk guards is `KIOSK_ZONE` in `config.py` (`lab_b`); a row
only counts when its `zone` matches.

```bash
python tools/smoke_schedule.py    # the rules, no camera needed
```

## Dev helpers

Available while `DEV_MODE=1` (the default). Set `DEV_MODE=0` for the final demo.

```bash
curl localhost:8000/dev/users                     # enrolled names + thresholds
curl "localhost:8000/dev/force?decision=STEP_UP"  # pin every decision
curl "localhost:8000/dev/force?decision=off"      # back to normal
curl localhost:8000/dev/totp/Divya                # current valid TOTP code
curl localhost:8000/dev/session/a1b2c3            # inspect a session
curl "localhost:8000/dev/calibrate?label=real"    # start recording calibration
```

`/dev/force` drives the kiosk UI through ALLOW, STEP_UP, DENY and RECAPTURE
without a real spoof in front of the camera. `/dev/totp/<name>` lets the demo
complete the step-up without an authenticator app.

## Latency

Every `/verify` response carries a `stage_ms` breakdown (also merged into
`scores.latency_ms` for the demo panel, and logged):

```
stage_ms = {quality_ms, identify_ms, liveness_ms}
liveness internals = {passive_ms, blinkmotion_ms, device_ms, calibmetrics_ms}
```

Measured on the dev box (16 logical cores), 10-frame burst, warm:

| stage | ms |
|---|---|
| decode + quality gate | ~35 |
| identify (detect + embed, 4 frames) | ~390 |
| liveness (all signals, 2 of them concurrent) | ~105 |
| **total** | **~535** |

Inside liveness: passive ~25 ms, blink/motion ~100 ms, device ~40 ms,
calibration metrics ~2 ms. Blink/motion and device run concurrently in a
thread pool while passive runs on the request thread, so the liveness total is
roughly the slowest of the three rather than their sum.

That is well inside the 2 s budget (`TARGET_MS`), and a single `analyze` call
is ~85 ms. **A multi-second `/verify` is contention or a cold start, not the
algorithm** — the same code measured 3.9 s while the machine was loaded and
0.5 s idle. Frame-level parallelism does not help (ONNX Runtime already uses
`ORT_THREADS` intra-op threads, so extra threads just contend: 4 frames took
373 ms sequential vs 233 ms threaded, with a worse tail). If a real kiosk run
is slow, read `stage_ms` from the response — it says which stage blew up.

## Threshold calibration

The anti-spoof thresholds are room-dependent, so measure instead of guessing
(guessing is what caused the false rejections). With the server running:

```bash
curl "localhost:8000/dev/calibrate?label=real"    # then ~10 real attempts
curl "localhost:8000/dev/calibrate?label=spoof"   # then ~10 spoof attempts
curl "localhost:8000/dev/calibrate?label=off"
python tools/suggest_thresholds.py
```

Each `/verify` appends one row to `eval/calibration.csv` (git-ignored: it is
measurement data): per-frame anti-spoof scores, moiré, face brightness, face
blur, motion, blink, match, decision and the stage timings.
`suggest_thresholds.py` prints the real and spoof distributions per signal and
the cut-off with the fewest mistakes — and says plainly when the two ranges
overlap, i.e. when that signal *cannot* separate them and tightening it would
only reject real people again. Recording never changes a decision; the numbers
go into `config.py` by hand.

**Moiré** (`face_moire` in `liveness.py`) is the share of the face crop's FFT
energy in the high-frequency band — a camera pointed at an LCD picks up the
panel's pixel grid. It is **recorded only** and wired into nothing: no
threshold is defensible before the calibration run says where real and spoof
actually separate in the room.

## Tests

```bash
python tools/smoke_liveness.py    # recalibrated spoof/real decision rules
python tools/smoke_schedule.py    # campus schedule rules
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
