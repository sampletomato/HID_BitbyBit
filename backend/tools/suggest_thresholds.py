"""Read eval/calibration.csv and print thresholds that separate real from spoof.

Collect the data first (see backend/calibration.py):

    curl "http://localhost:8000/dev/calibrate?label=real"     # ~10 real attempts
    curl "http://localhost:8000/dev/calibrate?label=spoof"    # ~10 spoof attempts
    curl "http://localhost:8000/dev/calibrate?label=off"
    python tools/suggest_thresholds.py

For each signal it prints the real and spoof distributions, whether they
overlap, and the cut-off that makes the fewest mistakes -- with the error count
at that cut-off, so a signal that cannot separate the two is obvious rather
than quietly turned into a threshold.

Nothing is written: the numbers go into config.py by hand, deliberately.
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

# (column, direction, config knob). direction "low_is_spoof" means a spoof
# scores LOWER than a real face, so the rule is "deny below the threshold".
SIGNALS = [
    ("passive_avg", "low_is_spoof", "ANTISPOOF_HARD_DENY"),
    ("passive_min", "low_is_spoof", "ANTISPOOF_FRAME_PANIC"),
    ("moire", "high_is_spoof", "(not wired into the decision)"),
    ("face_blur", "high_is_spoof", "(diagnostic only)"),
    ("face_brightness", "none", "MIN_FACE_BRIGHTNESS"),
    ("motion", "low_is_spoof", "W_MOTION input"),
    ("blink", "low_is_spoof", "W_BLINK input"),
    ("liveness", "low_is_spoof", "LIVENESS_ALLOW"),
]


def load(path) -> Tuple[Dict[str, List[dict]], int]:
    rows = {"real": [], "spoof": []}
    skipped = 0
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip().lower()
            if label in rows:
                rows[label].append(row)
            else:
                skipped += 1
    return rows, skipped


def values(rows: List[dict], column: str) -> List[float]:
    out = []
    for row in rows:
        raw = (row.get(column) or "").strip()
        if raw in ("", "None"):
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def describe(vals: List[float]) -> str:
    if not vals:
        return "no data"
    vals = sorted(vals)
    mid = vals[len(vals) // 2]
    return (f"n={len(vals):2d} min={vals[0]:.4g} p50={mid:.4g} "
            f"max={vals[-1]:.4g}")


def best_cut(real: List[float], spoof: List[float],
             direction: str) -> Optional[Tuple[float, int, int]]:
    """Threshold with the fewest mistakes.

    Returns (threshold, false_rejects, false_accepts) where a false reject is a
    real face the rule would deny -- the failure we are trying to stop.
    """
    if not real or not spoof:
        return None
    candidates = sorted(set(real) | set(spoof))
    # Midpoints between observed values, so the cut sits in the gap.
    cuts = [(a + b) / 2 for a, b in zip(candidates, candidates[1:])]
    cuts += [min(candidates) - 1e-6, max(candidates) + 1e-6]
    best = None
    for cut in cuts:
        if direction == "low_is_spoof":        # deny below the cut
            false_reject = sum(1 for v in real if v < cut)
            false_accept = sum(1 for v in spoof if v >= cut)
        else:                                   # deny above the cut
            false_reject = sum(1 for v in real if v > cut)
            false_accept = sum(1 for v in spoof if v <= cut)
        # Weight a false reject the same as a false accept, but break ties
        # towards fewer false rejects -- that is the bug being fixed.
        key = (false_reject + false_accept, false_reject)
        if best is None or key < best[0]:
            best = (key, cut, false_reject, false_accept)
    return (best[1], best[2], best[3]) if best else None


def main() -> int:
    path = config.CALIBRATION_CSV
    if not os.path.exists(path):
        print(f"No calibration data at {path}\n\n"
              "Record some first:\n"
              '  curl "http://localhost:8000/dev/calibrate?label=real"   '
              "# then ~10 real attempts\n"
              '  curl "http://localhost:8000/dev/calibrate?label=spoof"  '
              "# then ~10 spoof attempts\n"
              '  curl "http://localhost:8000/dev/calibrate?label=off"')
        return 1

    rows, skipped = load(path)
    n_real, n_spoof = len(rows["real"]), len(rows["spoof"])
    print(f"{path}\n{n_real} real, {n_spoof} spoof"
          + (f", {skipped} unlabelled (ignored)" if skipped else ""))
    if n_real < 5 or n_spoof < 5:
        print("\nWARNING: fewer than 5 samples on one side. The suggestions "
              "below are not yet trustworthy -- record more.")
    print("=" * 78)

    for column, direction, knob in SIGNALS:
        real, spoof = values(rows["real"], column), values(rows["spoof"], column)
        print(f"\n{column}   [{knob}]")
        print(f"  real : {describe(real)}")
        print(f"  spoof: {describe(spoof)}")
        if direction == "none":
            print("  (no spoof/real direction -- recorded for context only)")
            continue
        cut = best_cut(real, spoof, direction)
        if cut is None:
            print("  not enough data on both sides to suggest a threshold")
            continue
        threshold, false_reject, false_accept = cut
        rule = "<" if direction == "low_is_spoof" else ">"
        total = false_reject + false_accept
        print(f"  suggested: deny when {column} {rule} {threshold:.4g}")
        print(f"             -> {false_reject} real face(s) wrongly denied, "
              f"{false_accept} spoof(s) wrongly allowed "
              f"({total}/{len(real) + len(spoof)} wrong)")
        # Do the two groups occupy the same range? If so, no cut-off on this
        # signal can work, however good the best one looks.
        if direction == "low_is_spoof":
            separated = min(real) > max(spoof)
        else:
            separated = max(real) < min(spoof)
        if total == 0 and separated:
            print("             clean separation in this sample")
        elif separated:
            print("             separated, but thin -- keep the challenge as backstop")
        elif min(real) <= max(spoof) and min(spoof) <= max(real):
            print("             RANGES OVERLAP: this signal cannot separate real "
                  "from spoof here.")
            print("             Do not tighten the veto on it -- let the active "
                  "challenge do the work.")

    print("\n" + "=" * 78)
    print("Current config:")
    print(f"  ANTISPOOF_HARD_DENY   = {config.ANTISPOOF_HARD_DENY}")
    print(f"  ANTISPOOF_FRAME_PANIC = {config.ANTISPOOF_FRAME_PANIC}")
    print(f"  MIN_FACE_BRIGHTNESS   = {config.MIN_FACE_BRIGHTNESS}")
    print(f"  LIVENESS_ALLOW        = {config.LIVENESS_ALLOW}")
    print("\nA spoof that lands between the deny bar and LIVENESS_ALLOW is not "
          "let in:\nit gets the random active challenge, which is what catches "
          "a replay.\nPrefer the looser deny bar and rely on the challenge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
