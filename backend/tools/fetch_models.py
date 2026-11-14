"""Download the OpenCV YuNet + SFace ONNX models (the fallback face backend).

Only needed when InsightFace is unavailable. InsightFace fetches buffalo_l
itself on first load, into backend/models/.

    python tools/fetch_models.py
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = [
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
    print("force this backend with: FACE_BACKEND=opencv")
