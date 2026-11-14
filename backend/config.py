"""All tunable knobs for the liveness + authentication pipeline.

Nothing else in the backend should hard-code a threshold. Tune here, restart.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# backend/.env holds the Supabase credentials (SUPABASE_URL,
# SUPABASE_SERVICE_KEY). It is read here, before anything calls os.getenv, and
# is never committed -- see .gitignore. Missing file is fine: policy.py then
# falls back to backend/schedule.json.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:          # python-dotenv not installed: env vars only
    pass
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

# ------------------------------------------------------------ access schedule
# Which door this kiosk guards. A schedule row only counts when its `zone`
# matches this value, so one gallery can serve several kiosks.
KIOSK_ZONE = os.getenv("KIOSK_ZONE", "lab_b").strip()

# Minutes before the start / after the end of a window that still count as
# "nearly on time": inside this band a biometric ALLOW becomes a STEP_UP for an
# authenticator code instead of a flat DENY.
SCHEDULE_GRACE_MIN = int(os.getenv("SCHEDULE_GRACE_MIN", "30"))

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

# --------------------------------------------------- passive anti-spoof model
# MiniFASNetV2-SE (INT8) from github.com/facenox/face-antispoof-onnx (Apache-2.0).
# 128x128 RGB input, output logits [real, spoof]. Crop + preprocess copied
# verbatim from that repo (see liveness.py) -- do not "improve" them, the model
# was trained against that exact pipeline.
ANTISPOOF_MODEL = MODEL_DIR / "best_model_quantized.onnx"   # INT8, primary
ANTISPOOF_MODEL_FP32 = MODEL_DIR / "best_model.onnx"        # fp32, fallback
ANTISPOOF_INPUT = 128
ANTISPOOF_BBOX_EXPANSION = 1.5   # facenox default; the model expects this context
ANTISPOOF_FRAMES = 3             # run on the 3 sharpest detected frames of the burst
ANTISPOOF_REAL_MIN = 0.5         # real-probability above which a frame reads "live"
# Hard gate: real_prob below this is an immediate DENY -- motion or match can
# never rescue a failed anti-spoof. The decision liveness is also gated at
# min(passive, fused), so a low passive caps the whole score. Genuine faces >> this.
ANTISPOOF_HARD_DENY = 0.30

# --------------------------------------------------- blink + motion (MediaPipe)
# Face Landmarker (Tasks API) run on every verify frame. EAR = eye aspect ratio.
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
EAR_BLINK_THRESHOLD = 0.21    # per-eye EAR below this = eye closed
EAR_BLINK_DELTA = 0.06        # open->closed swing needed to count as a real blink
# Combined yaw+pitch range (degrees) across the burst that maps to motion = 1.0.
# A flat photo held still has ~0 out-of-plane rotation, so it scores near 0.
MOTION_YAWPITCH_FULL_DEG = 6.0

# ----------------------------------------------------- device detection (YOLO)
YOLO_MODEL = MODEL_DIR / "yolov8n.pt"
YOLO_CONF = 0.25            # lowered to catch phones/screens more readily
YOLO_IMGSZ = 320             # 320 is ~3x faster than 640 on CPU; phones/screens are large
YOLO_FRAMES = 2              # run on 2 frames of the burst (CPU cost)
YOLO_DEVICE_CLASSES = ("cell phone", "laptop", "tv")
# A replay attack puts the face INSIDE the device: the test is therefore "how
# much of the face box lies inside the device box", not "do the boxes touch".
# A TV or monitor in the background overlaps the frame but not the face, so it
# no longer trips the check.
DEVICE_FACE_INSIDE = 0.60     # tv/laptop: >=60% of the face area inside the box
DEVICE_FACE_OVERLAP = 0.20    # "cell phone": a phone held near the face is enough
# A tv/laptop box covering more than this fraction of the frame is the room, not
# a replay device -- ignored unless the face is genuinely inside it.
DEVICE_MAX_FRAME_FRAC = 0.50

# ------------------------------------------------------------- fusion weights
# Weighted mix of the three graded liveness signals. Must sum to 1.0. The device
# check is NOT in this mix -- a detected phone/screen is a hard DENY on its own.
W_PASSIVE = 0.60   # MiniFASNetV2-SE passive anti-spoof
W_MOTION = 0.25    # head yaw/pitch micro-motion across the burst
W_BLINK = 0.15     # eye-aspect-ratio blink evidence

# ---------------------------------------------------------------- quality gate
MIN_FRAMES_ENROLL = 3
MIN_FRAMES_VERIFY = 3      # relaxed for a dim, low-quality demo webcam (was 5)
MIN_FRAMES_CHALLENGE = 10
MIN_BLUR_VAR = 10.0        # variance of Laplacian, lower = too blurry (relaxed, was 45)
MIN_BRIGHTNESS = 20.0      # mean grey level, lower = too dark (relaxed, was 45)
MAX_BRIGHTNESS = 240.0     # higher = blown out / backlit
# Mean grey level of the FACE crop below which the passive anti-spoof model is
# not trusted: a real face in the dark reads as "flat print texture", so a dark
# face gets RECAPTURE ("Too dark") instead of a DENY. Same scale as MIN_BRIGHTNESS.
MIN_FACE_BRIGHTNESS = MIN_BRIGHTNESS
MIN_GOOD_FRAME_RATIO = 0.3  # need 3 good frames of 10 (relaxed, was 0.5)

# -------------------------------------------------------------------- sessions
SESSION_TTL_S = 180
MAX_CHALLENGE_ATTEMPTS = 2

# ------------------------------------------------- active challenge detection
# Verified from the ~20 challenge frames using the MediaPipe landmarks.
CHALLENGE_BLINKS_REQUIRED = 2    # BLINK_TWICE
CHALLENGE_YAW_DEG = 12.0         # TURN_LEFT / TURN_RIGHT: head yaw magnitude
# Sign convention: with this = -1, the user's left turn reads as negative yaw.
# If a live test shows left/right swapped, flip to +1 (one-line change).
CHALLENGE_YAW_LEFT_SIGN = -1
SMILE_RATIO_DELTA = 0.12         # SMILE: rise in mouth-width / inter-eye-width
SMILE_MIN_RATIO = 1.55           # or an absolute wide-mouth ratio

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
