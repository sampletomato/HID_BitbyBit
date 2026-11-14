"""Exercise the full liveness stack and report per-signal latency.

Runs the real LivenessEngine (passive anti-spoof + MediaPipe blink/motion +
YOLO device) on sample face bursts, with no server and no webcam.

    python tools/test_liveness.py

NOTE: sample frames are a jittered still image -- no real head rotation or
blink -- so motion/blink score near zero here (a static burst is spoof-like by
design). The point of this harness is to confirm every signal loads, runs, and
fits the latency budget. Live discrimination is tested with tools/webcam_test.py.
"""

import logging
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
config.GALLERY_PATH = Path(tempfile.mkdtemp(prefix="hid_liv_")) / "gallery.json"

import face as face_mod          # noqa: E402
import liveness as liveness_mod  # noqa: E402
import sample_faces              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    face = face_mod.FaceEngine()
    face.load()
    live = liveness_mod.LivenessEngine()
    live.load()

    print(f"\nsignals: passive={live.passive.available} "
          f"blink/motion={live.blinkmotion.available} device={live.device.available}\n")

    ids = sample_faces.identities()
    budget_ok = True
    for label in sorted(ids)[:2]:
        images = sample_faces.burst(ids[label], 10, offset=300)
        burst = face.embed_burst(images, max_frames=config.EMBED_FRAMES)
        if burst.query is None:
            print(f"{label}: no face")
            continue
        t = time.time()
        res = live.score(images, burst.detections, random.Random())
        total_ms = (time.time() - t) * 1000

        print(f"--- {label} ---")
        print(f"  fused liveness : {res.score}")
        print(f"  per-signal ms  : passive={res.parts['passive_ms']} "
              f"blinkmotion={res.parts['blinkmotion_ms']} device={res.parts['device_ms']}")
        print(f"  liveness total : {total_ms:.0f} ms")
        for r in res.reasons:
            print(f"    - {r}")
        if total_ms >= config.TARGET_MS:
            budget_ok = False

    print("\nliveness latency OK" if budget_ok else "\nliveness OVER BUDGET")
    print("note: run tools/webcam_test.py for live face vs phone vs print.")
    return 0 if budget_ok else 1


if __name__ == "__main__":
    sys.exit(main())
