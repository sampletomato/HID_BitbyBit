"""Passive anti-spoof: score real face crops and measure added latency.

Runs the engine directly (no server). Reports the averaged real-probability for
each sample face and the per-call latency on 3 frames (target < 100 ms).

    python tools/test_antispoof.py

NOTE: the definitive acceptance test -- a live face scoring high and a photo of
that face on a phone scoring low -- needs a real webcam and phone, so it is run
from the kiosk, not here. This harness verifies the pipeline, the output range,
and the latency budget on the faces we have.
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
config.GALLERY_PATH = Path(tempfile.mkdtemp(prefix="hid_as_")) / "gallery.json"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import face as face_mod       # noqa: E402
import liveness as liveness_mod  # noqa: E402
import sample_faces           # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def recompress(frame: np.ndarray, quality: int, scale: float) -> np.ndarray:
    """Crude 'photo on a screen' proxy: downscale + low-quality JPEG round-trip.

    This is NOT a faithful presentation attack (no moire, no reflection), so a
    high score here is not a failure -- it just shows texture degradation alone
    is not what the model keys on. The real spoof test is on the kiosk.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", up, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main() -> int:
    face = face_mod.FaceEngine()
    face.load()
    passive = liveness_mod.PassiveAntiSpoof()
    passive.load()
    if not passive.available:
        print("anti-spoof model unavailable -- run: python tools/fetch_models.py")
        return 1
    print(f"backend: {face.backend_name}  |  anti-spoof: MiniFASNetV2-SE\n")

    ids = sample_faces.identities()
    print(f"{'sample':<12}{'real_prob':>10}{'frames':>8}{'ms':>8}   per-frame")
    print("-" * 64)

    latencies = []
    for label in sorted(ids):
        frames = sample_faces.burst(ids[label], 10, offset=200)
        burst = face.embed_burst(frames, max_frames=config.EMBED_FRAMES)
        if burst.query is None:
            print(f"{label:<12}{'no face':>10}")
            continue
        t = time.time()
        avg, per_frame, n = passive.score_detections(burst.detections)
        ms = (time.time() - t) * 1000
        latencies.append(ms)
        print(f"{label:<12}{avg:>10.3f}{n:>8}{ms:>8.0f}   {per_frame}")

    # Degraded variant of the first identity, for a rough differentiation read.
    label = sorted(ids)[0]
    deg = [recompress(f, quality=18, scale=0.25)
           for f in sample_faces.burst(ids[label], 10, offset=200)]
    burst = face.embed_burst(deg, max_frames=config.EMBED_FRAMES)
    if burst.query is not None:
        avg, per_frame, n = passive.score_detections(burst.detections)
        print(f"{label + '~deg':<12}{avg:>10.3f}{n:>8}{'':>8}   {per_frame}  (recompressed proxy)")

    print("-" * 64)
    print(f"passive latency on {config.ANTISPOOF_FRAMES} frames: "
          f"{min(latencies):.0f}-{max(latencies):.0f} ms (target < 100 ms)")
    ok = max(latencies) < 100
    print("\nlatency OK" if ok else "\nlatency OVER BUDGET")
    print("note: live-vs-phone discrimination is validated on the kiosk, not here.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
