"""Download the ONNX model weights (all gitignored).

  * passive anti-spoof -- MiniFASNetV2-SE from facenox/face-antispoof-onnx
    (Apache-2.0). Required for real liveness.
  * YuNet + SFace -- the OpenCV fallback face backend, only needed when
    InsightFace is unavailable.

InsightFace fetches buffalo_l itself on first load, into backend/models/.

    python tools/fetch_models.py
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
FACENOX = "https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/models"
MP = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1"
YOLO = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
MODELS = [
    # Passive anti-spoof: INT8 primary, fp32 fallback. The published top-level
    # quantized model is the 98.20 variant; the matching fp32 lives in best/98.20/.
    (f"{FACENOX}/best_model_quantized.onnx", config.ANTISPOOF_MODEL),
    (f"{FACENOX}/best/98.20/best_model.onnx", config.ANTISPOOF_MODEL_FP32),
    # Blink + head-motion (MediaPipe Face Landmarker bundle).
    (f"{MP}/face_landmarker.task", config.FACE_LANDMARKER_MODEL),
    # Device-in-frame detector (YOLOv8n, COCO; has 'cell phone','laptop','tv').
    (f"{YOLO}/yolov8n.pt", config.YOLO_MODEL),
    (f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
     config.YUNET_MODEL),
    (f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
     config.SFACE_MODEL),
]


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have   {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"get    {dest.name} ...", end="", flush=True)
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            tmp.write_bytes(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"\nfailed: {url}\n  {exc}")
    tmp.replace(dest)
    print(f" {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for url, dest in MODELS:
        fetch(url, dest)
    print(f"\nmodels in {config.MODEL_DIR}")
    print("anti-spoof is required; YuNet+SFace only for FACE_BACKEND=opencv")
