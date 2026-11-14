"""Measure genuine vs impostor match scores on the real recognition backend.

Runs the FaceEngine directly (no server), against the bundled sample faces:
enrolls one identity, then verifies that identity and every other one.

    python tools/test_face_match.py

Prints the score separation and checks it against the configured thresholds.
These are the numbers that feed the FAR/FRR section of the eval report.
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Keep the real enrollment gallery untouched.
config.GALLERY_PATH = Path(tempfile.mkdtemp(prefix="hid_test_")) / "gallery.json"

import face as face_mod      # noqa: E402
import sample_faces          # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    engine = face_mod.FaceEngine()
    t0 = time.time()
    engine.load()
    print(f"backend      : {engine.backend_name} (loaded in {time.time() - t0:.1f}s)")
    print(f"thresholds   : allow>={engine.match_allow}  deny<{engine.match_deny}")

    ids = sample_faces.identities()
    labels = sorted(ids)
    enrolled = labels[0]

    faces_used, err = engine.enroll(enrolled, sample_faces.burst(ids[enrolled], 5), "SECRET")
    if err:
        print(f"enroll failed: {err}")
        return 1
    print(f"enrolled     : {enrolled} ({faces_used} faces)\n")

    print(f"{'probe':<12}{'match':>8}{'name':>12}{'ms':>7}   verdict")
    print("-" * 52)

    genuine, impostors, times = None, [], []
    for label in labels:
        frames = sample_faces.burst(ids[label], 10, offset=500)
        t = time.time()
        name, score, burst = engine.identify(frames)
        ms = int((time.time() - t) * 1000)
        times.append(ms)

        if burst.query is None:
            print(f"{label:<12}{'--':>8}{'no face':>12}{ms:>7}   {burst.note}")
            continue

        is_genuine = label == enrolled
        if is_genuine:
            genuine = score
        else:
            impostors.append(score)
        tag = "GENUINE" if is_genuine else "impostor"
        print(f"{label:<12}{score:>8.3f}{str(name):>12}{ms:>7}   {tag}")

    print("-" * 52)
    worst_impostor = max(impostors) if impostors else 0.0
    print(f"genuine score          : {genuine:.3f}" if genuine is not None else "genuine: n/a")
    print(f"worst impostor score   : {worst_impostor:.3f}")
    if genuine is not None:
        print(f"separation             : {genuine - worst_impostor:.3f}")
    print(f"identify time          : {min(times)}-{max(times)} ms "
          f"(budget {config.TARGET_MS} ms)")

    ok = True
    if genuine is None or genuine < engine.match_allow:
        print(f"\nFAIL: genuine score below allow threshold {engine.match_allow}")
        ok = False
    if worst_impostor >= engine.match_deny:
        print(f"\nFAIL: an impostor scored at or above the deny threshold "
              f"{engine.match_deny}")
        ok = False
    if max(times) > config.TARGET_MS:
        print(f"\nFAIL: identify exceeded the {config.TARGET_MS} ms budget")
        ok = False

    print("\nface match OK" if ok else "\nface match FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
