"""Live webcam diagnostic for the passive anti-spoof model.

Opens the webcam, runs our InsightFace detector + facenox anti-spoof on every
3rd frame, and overlays the real-probability, the face box (green if >= the
threshold, red if not) and the per-frame latency.

    python tools/webcam_test.py [--camera 0] [--threshold 0.5] [--every 3]

Keys:  q = quit   s = save a screenshot to eval/screens/

Use it to compare a live face vs a photo of the face on a phone vs a printed
photo -- the live face should stay green/high, the spoofs red/low.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config          # noqa: E402
import face as face_mod            # noqa: E402
import liveness as liveness_mod    # noqa: E402

SCREENS_DIR = config.BASE_DIR.parent / "eval" / "screens"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=config.ANTISPOOF_REAL_MIN)
    p.add_argument("--every", type=int, default=config.ANTISPOOF_FRAMES,
                   help="run detection+anti-spoof on every Nth frame")
    return p.parse_args()


def main():
    args = parse_args()

    print("loading models ...")
    face = face_mod.FaceEngine()
    face.load()
    passive = liveness_mod.PassiveAntiSpoof()
    passive.load()
    if not passive.available:
        sys.exit("anti-spoof model unavailable -- run: python tools/fetch_models.py")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit(f"could not open camera {args.camera}")

    print(f"backend={face.backend_name}  threshold={args.threshold}  "
          f"every={args.every} frames\nkeys: q quit, s screenshot")

    # Last known result, so we can keep drawing between the processed frames.
    last = {"prob": None, "box": None, "ms": 0.0}
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        if frame_i % args.every == 0:
            t = time.time()
            det = face.backend.analyze(frame)
            if det is not None:
                x, y, w, h = det.bbox
                avg, per_frame, n = passive.score_detections([(frame, det.bbox)])
                last = {"prob": avg, "box": (x, y, w, h),
                        "ms": (time.time() - t) * 1000}
            else:
                last = {"prob": None, "box": None, "ms": (time.time() - t) * 1000}

        _draw(frame, last, args.threshold)

        cv2.imshow("HID anti-spoof live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            _save(frame)

    cap.release()
    cv2.destroyAllWindows()


def _draw(frame, last, threshold):
    prob, box, ms = last["prob"], last["box"], last["ms"]
    if box is not None and prob is not None:
        x, y, w, h = box
        is_real = prob >= threshold
        color = (0, 200, 0) if is_real else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{'REAL' if is_real else 'SPOOF'} {prob:.2f}"
        cv2.putText(frame, label, (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(frame, "no face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.putText(frame, f"{ms:.0f} ms", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def _save(frame):
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    name = SCREENS_DIR / f"shot_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(name), frame)
    print(f"saved {name}")


if __name__ == "__main__":
    main()
