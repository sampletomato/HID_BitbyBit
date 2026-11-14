"""Real face frames for testing, with no webcam and no network.

Uses the sample photos bundled inside the insightface package: a group photo
(several distinct identities) and a separate portrait. Each detected face is
cropped and resized to a 640x480 "webcam frame", then jittered to build a burst
that looks like consecutive frames rather than N copies of one image.
"""

import base64
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRAME_W, FRAME_H = 640, 480


def _images_dir() -> Path:
    import insightface
    return Path(insightface.__file__).resolve().parent / "data" / "images"


def _crop_to_frame(img: np.ndarray, box, margin: float = 0.9) -> np.ndarray:
    """Crop a detected face with generous margin, letterboxed into 640x480."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    x0, x1 = int(max(0, cx - half)), int(min(img.shape[1], cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(img.shape[0], cy + half))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img
    scale = min(FRAME_W / crop.shape[1], FRAME_H / crop.shape[0])
    resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    canvas = np.full((FRAME_H, FRAME_W, 3), 96, dtype=np.uint8)
    oy = (FRAME_H - resized.shape[0]) // 2
    ox = (FRAME_W - resized.shape[1]) // 2
    canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    return canvas


def _jitter(frame: np.ndarray, i: int) -> np.ndarray:
    """Small shift, brightness drift and grain, so a burst is not N identical frames."""
    rng = np.random.default_rng(i)
    dx, dy = int(rng.integers(-4, 5)), int(rng.integers(-3, 4))
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(frame, m, (FRAME_W, FRAME_H), borderMode=cv2.BORDER_REPLICATE)
    out = cv2.convertScaleAbs(out, alpha=float(rng.uniform(0.97, 1.03)),
                              beta=float(rng.integers(-6, 7)))
    grain = rng.normal(0, 2.5, out.shape).astype(np.float32)
    return np.clip(out.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def burst(frame: np.ndarray, n: int, offset: int = 0) -> List[np.ndarray]:
    """A burst of n frames. Use a different offset so enroll and verify bursts
    are not the same pixels -- otherwise the match score is meaninglessly high."""
    return [_jitter(frame, offset + i) for i in range(n)]


def to_data_url(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "store" / "sample_frames"


def identities(limit: int = 4) -> Dict[str, np.ndarray]:
    """One base frame per distinct sample identity, keyed person0..personN.

    Crops are cached to store/sample_frames/ on first run. Later runs read the
    cache, so tests never spin up a second detector -- two ONNX sessions
    competing for threads is exactly the contention the engine avoids.
    """
    cache = _cache_dir()
    cached = sorted(cache.glob("*.jpg")) if cache.exists() else []
    if cached:
        return {p.stem: cv2.imread(str(p)) for p in cached}
    return _build_identities(limit)


def _build_identities(limit: int = 4) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    d = _images_dir()

    group = cv2.imread(str(d / "t1.jpg"))
    if group is not None:
        boxes = _detect_all(group)
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        for i, box in enumerate(boxes[:limit]):
            out[f"person{i}"] = _crop_to_frame(group, box)

    portrait = cv2.imread(str(d / "Tom_Hanks_54745.png"))
    if portrait is not None:
        boxes = _detect_all(portrait)
        if boxes:
            out["portrait"] = _crop_to_frame(portrait, max(boxes, key=lambda b: b[2] * b[3]))

    if not out:
        raise RuntimeError(f"no faces found in sample images under {d}")

    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    for label, frame in out.items():
        cv2.imwrite(str(cache / f"{label}.jpg"), frame)
    return out


def _detect_all(img: np.ndarray):
    """All face boxes in an image, using insightface's detector directly."""
    import config
    from insightface.app import FaceAnalysis
    global _APP
    try:
        app = _APP
    except NameError:
        app = None
    if app is None:
        app = FaceAnalysis(name="buffalo_l", root=str(config.MODEL_DIR),
                           providers=["CPUExecutionProvider"],
                           allowed_modules=["detection"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        globals()["_APP"] = app
    faces = app.get(img)
    return [(int(f.bbox[0]), int(f.bbox[1]),
             int(f.bbox[2] - f.bbox[0]), int(f.bbox[3] - f.bbox[1])) for f in faces]


if __name__ == "__main__":
    ids = _build_identities()
    print(f"{len(ids)} sample identities: {', '.join(ids)}")
    print(f"cached frames in {_cache_dir()}")
