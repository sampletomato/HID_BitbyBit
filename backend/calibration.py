"""Threshold calibration: record what the signals actually read in this room.

The anti-spoof thresholds were guessed, and guessed too strict -- a real face
in ordinary indoor light was being denied. Rather than guess again, turn on
recording, present ~10 real faces and ~10 spoofs (phone, print, laptop screen),
and let tools/suggest_thresholds.py read the numbers off the result:

    curl "http://localhost:8000/dev/calibrate?label=real"     # then 10 real attempts
    curl "http://localhost:8000/dev/calibrate?label=spoof"    # then 10 spoof attempts
    curl "http://localhost:8000/dev/calibrate?label=off"
    python tools/suggest_thresholds.py

One row per /verify is appended to eval/calibration.csv while a label is set.
Recording never changes a decision -- it only observes.
"""

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config

log = logging.getLogger("calibration")

LABELS = ("real", "spoof")

# One row per /verify. Column order is also the order suggest_thresholds.py
# expects, so keep new columns at the end.
FIELDS = [
    "timestamp", "label", "name", "decision", "reason_code",
    "passive_avg", "passive_min", "passive_frames",
    "moire", "face_brightness", "face_blur",
    "motion", "blink", "blink_detected", "liveness", "match",
    "device", "too_dark", "frame_panic",
    "time_ms", "quality_ms", "identify_ms", "liveness_ms",
    "passive_ms", "blinkmotion_ms", "device_ms",
]


class Recorder:
    """Appends one labelled row per verify while a label is active."""

    def __init__(self):
        self.label: Optional[str] = None
        self.count = 0
        self._lock = threading.Lock()

    def set_label(self, label: str) -> Optional[str]:
        """Returns the new label, or None when recording was switched off."""
        wanted = (label or "").strip().lower()
        with self._lock:
            if wanted in ("off", "", "none", "stop"):
                self.label = None
            elif wanted in LABELS:
                self.label = wanted
            else:
                raise ValueError(f"label must be one of {', '.join(LABELS)} or off")
            self.count = 0
            return self.label

    @property
    def active(self) -> bool:
        return self.label is not None

    def rows_on_disk(self) -> int:
        path = Path(config.CALIBRATION_CSV)
        if not path.exists():
            return 0
        with path.open(encoding="utf-8", newline="") as fh:
            return max(0, sum(1 for _ in fh) - 1)

    def record(self, *, decision: str, reason_code: str, name: Optional[str],
               liveness: float, match: float, parts: dict,
               time_ms: int, stages: Optional[dict] = None) -> None:
        """Append one row. Never raises into the request path."""
        with self._lock:
            label = self.label
        if label is None:
            return
        try:
            frames: List[float] = list(parts.get("passive_frames") or [])
            stages = stages or {}
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "label": label,
                "name": name or "",
                "decision": decision,
                "reason_code": reason_code,
                "passive_avg": parts.get("passive"),
                "passive_min": round(min(frames), 4) if frames else "",
                # Semicolons, so the field survives a CSV reader unquoted.
                "passive_frames": ";".join(f"{p:.4f}" for p in frames),
                "moire": parts.get("moire"),
                "face_brightness": parts.get("face_brightness"),
                "face_blur": parts.get("face_blur"),
                "motion": parts.get("motion"),
                "blink": parts.get("blink"),
                "blink_detected": parts.get("blink_detected"),
                "liveness": liveness,
                "match": match,
                "device": parts.get("device"),
                "too_dark": parts.get("too_dark"),
                "frame_panic": parts.get("frame_panic"),
                "time_ms": time_ms,
                "quality_ms": stages.get("quality_ms"),
                "identify_ms": stages.get("identify_ms"),
                "liveness_ms": stages.get("liveness_ms"),
                "passive_ms": parts.get("passive_ms"),
                "blinkmotion_ms": parts.get("blinkmotion_ms"),
                "device_ms": parts.get("device_ms"),
            }
            path = Path(config.CALIBRATION_CSV)
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists() or path.stat().st_size == 0
            with path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            with self._lock:
                self.count += 1
            log.info("calibration: recorded %s sample #%d (passive=%s moire=%s "
                     "brightness=%s)", label, self.count, row["passive_avg"],
                     row["moire"], row["face_brightness"])
        except Exception as exc:
            # A calibration failure must never break a verify.
            log.warning("calibration row not written: %s", exc)


recorder = Recorder()
