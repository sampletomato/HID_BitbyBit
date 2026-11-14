"""All tunable knobs for the liveness + authentication pipeline.

Nothing else in the backend should hard-code a threshold. Tune here, restart.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "store"
MODEL_DIR = BASE_DIR / "models"
STORE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
GALLERY_PATH = STORE_DIR / "gallery.json"

# edge = kiosk with face models; central = EC2 with schedule + audit (step 3).
MODE = os.getenv("MODE", "edge").strip().lower()

# Dev mode exposes /dev/* helper routes and returns the TOTP secret in plain
# text so the demo can show it. Set DEV_MODE=0 on EC2 for the final demo.
DEV_MODE = os.getenv("DEV_MODE", "1") == "1"

# ----------------------------------------------------------------- face models
# "auto" tries InsightFace buffalo_l, then falls back to OpenCV YuNet + SFace.
# Force one with FACE_BACKEND=insightface | opencv.
FACE_BACKEND = os.getenv("FACE_BACKEND", "auto").strip().lower()

# ONNX Runtime threading. Each session spawns its own pool, and two pools with
# default settings fight each other badly enough to cost ~2 s per frame, so
# both sessions are created with an explicit cap and spin-waiting disabled.
ORT_THREADS = int(os.getenv("ORT_THREADS", "0")) or min(4, os.cpu_count() or 2)

DET_SIZE = 320          # detector input square; 320 is ~2x faster than 640 on CPU
DET_SCORE_MIN = 0.60    # discard weak detections
EMBED_FRAMES = 4        # embed only the N sharpest frames, not the whole burst
MIN_FACE_PX = 70        # a face smaller than this is too far away to trust

YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

# ---------------------------------------------------------------- decisioning
# Liveness is 0..1. Above ALLOW is confident, below DENY is a hard reject, the
# band between falls through to an active challenge.
LIVENESS_ALLOW = 0.75
LIVENESS_DENY = 0.40

# Match thresholds are cosine similarity and are NOT comparable across
# backends: buffalo_l returns 512-d embeddings, SFace 128-d with a tighter
# spread. FaceEngine picks the pair matching the loaded backend.
MATCH_THRESHOLDS = {
    "insightface": {"allow": 0.45, "deny": 0.25},
    "opencv": {"allow": 0.36, "deny": 0.20},
    "stub": {"allow": 0.45, "deny": 0.25},
}

# ------------------------------------------------------------- fusion weights
# Used by liveness.py once the real models land (step 5). Must sum to 1.0.
W_PASSIVE = 0.50   # MiniFASNet passive anti-spoof
W_MOTION = 0.25    # frame-to-frame micro-motion / texture consistency
W_BLINK = 0.15     # MediaPipe eye-aspect-ratio blink evidence
W_DEVICE = 0.10    # YOLO phone / laptop / tv penalty

# ---------------------------------------------------------------- quality gate
MIN_FRAMES_ENROLL = 3
MIN_FRAMES_VERIFY = 5
MIN_FRAMES_CHALLENGE = 10
MIN_BLUR_VAR = 45.0        # variance of Laplacian, lower = too blurry
MIN_BRIGHTNESS = 45.0      # mean grey level, lower = too dark
MAX_BRIGHTNESS = 225.0     # higher = blown out / backlit
MIN_GOOD_FRAME_RATIO = 0.5  # at least half the burst must pass

# -------------------------------------------------------------------- sessions
SESSION_TTL_S = 180
MAX_CHALLENGE_ATTEMPTS = 2

# ------------------------------------------------------------------------ TOTP
TOTP_ISSUER = "HID Kiosk"
TOTP_DIGITS = 6
TOTP_PERIOD_S = 30
TOTP_WINDOW = 1   # accept the adjacent steps, to tolerate clock skew

# Active challenges. The kiosk renders `text` verbatim.
CHALLENGES = [
    {"action": "BLINK_TWICE", "text": "Please blink twice"},
    {"action": "TURN_LEFT", "text": "Please turn your head left"},
    {"action": "TURN_RIGHT", "text": "Please turn your head right"},
    {"action": "SMILE", "text": "Please smile"},
]

# Budget for a single /verify call. Logged as a warning when exceeded.
TARGET_MS = 2000
