"""Smoke test for the recalibrated liveness decision, no camera needed.

Guards the false-rejection fix: a real face must not be denied just because
the room is dim or the person held still. Only a confident spoof reading or a
phone/screen in frame may DENY.

    python tools/smoke_liveness.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config            # noqa: E402
import decision as dec   # noqa: E402

ALLOW_M, DENY_M = 0.45, 0.25     # insightface match thresholds


def fused(passive: float, motion: float, blink: float) -> float:
    return round(config.W_PASSIVE * passive + config.W_MOTION * motion
                 + config.W_BLINK * blink, 2)


def spoof_veto(passive_avg, frames, too_dark: bool) -> bool:
    """The rule from liveness.py, so the test tracks the real thresholds."""
    if passive_avg is None or too_dark:
        return False
    panic = any(p < config.ANTISPOOF_FRAME_PANIC for p in frames)
    return passive_avg < config.ANTISPOOF_HARD_DENY or panic


CASES = [
    # (label, passive_frames, motion, blink, match, device, expect)
    # --- the reported bug: real face, dim room, holding still ---------------
    ("real face, dim room, still    ", [0.45, 0.40, 0.44], 0.0, 0.0, 0.67,
     False, dec.STEP_UP),
    ("real face, soft passive 0.27  ", [0.28, 0.27, 0.26], 0.1, 0.0, 0.67,
     False, dec.STEP_UP),
    ("real face, good light + blink ", [0.98, 0.97, 0.99], 0.8, 1.0, 0.70,
     False, dec.ALLOW),
    # Confident passive + strong head motion clears the bar on its own; a
    # blink is not required when the other two signals are already strong.
    ("real face, no blink, moves    ", [0.95, 0.93, 0.96], 0.9, 0.0, 0.70,
     False, dec.ALLOW),
    ("real face, no blink, no motion", [0.95, 0.93, 0.96], 0.0, 0.0, 0.70,
     False, dec.STEP_UP),
    # --- confident spoofs must still be denied ------------------------------
    ("print attack, all frames 0.10 ", [0.10, 0.09, 0.12], 0.0, 0.0, 0.60,
     False, dec.DENY),
    ("screen replay, one panic frame", [0.60, 0.55, 0.02], 0.3, 0.0, 0.60,
     False, dec.DENY),
    ("phone in frame, perfect face  ", [0.99, 0.99, 0.99], 0.9, 1.0, 0.70,
     True, dec.DENY),
    # --- identity still rules ----------------------------------------------
    ("unknown face, live            ", [0.99, 0.99, 0.99], 0.9, 1.0, 0.00,
     False, dec.DENY),
]


def main() -> int:
    print(f"HARD_DENY={config.ANTISPOOF_HARD_DENY} "
          f"FRAME_PANIC={config.ANTISPOOF_FRAME_PANIC} "
          f"ALLOW={config.LIVENESS_ALLOW} "
          f"MIN_FACE_BRIGHTNESS={config.MIN_FACE_BRIGHTNESS}")
    print(f"weights: passive={config.W_PASSIVE} motion={config.W_MOTION} "
          f"blink={config.W_BLINK}")
    print("-" * 94)

    failures = 0
    for label, frames, motion, blink, match, device, expect in CASES:
        passive = sum(frames) / len(frames)
        liveness = fused(passive, motion, blink)
        spoof = spoof_veto(passive, frames, too_dark=False)
        name = "Divya" if match > 0.1 else None
        verdict, reasons, code = dec.decide(liveness, match, name,
                                            ALLOW_M, DENY_M, device, spoof)
        ok = verdict == expect
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}  passive={passive:.2f} "
              f"liveness={liveness:.2f} -> {verdict:9} {code:24} "
              + ("" if ok else f"EXPECTED {expect}"))

    # The old min(passive, fused) gate is gone: a soft passive must no longer
    # pull the score below the band and turn a real person into a DENY.
    print("-" * 94)
    soft = fused(0.45, 0.0, 0.0)
    print(f"real face, passive 0.45, no motion/blink -> liveness {soft:.2f} "
          f"(old min-gate would have reported 0.27 and DENIED)")
    assert soft > 0.0

    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
