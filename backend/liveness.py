"""Liveness: passive anti-spoof, motion, blink, and device-in-frame checks.

Step 1: all four signals are stubs marked STUB-STEP3. They return plausible
sub-scores and fuse them with the real weights from config, so the fusion code
and the reason strings the kiosk displays are already the final ones.
"""

import logging
import random
from typing import List, Tuple

import numpy as np

import config

log = logging.getLogger("liveness")


class LivenessResult:
    def __init__(self, score: float, reasons: List[str], parts: dict, device_seen: bool):
        self.score = score
        self.reasons = reasons
        self.parts = parts
        self.device_seen = device_seen


class LivenessEngine:
    """All anti-spoof models. Loaded once at startup, never per request."""

    def __init__(self):
        self.backend = "stub"

    def load(self):
        """STUB-STEP3: load MiniFASNet, MediaPipe FaceMesh and YOLOv8n here."""
        log.info("LivenessEngine loading (backend=%s)", self.backend)
        self.backend = "stub"
        log.info("LivenessEngine ready")

    # -- individual signals ----------------------------------------------------

    def _passive(self, images: List[np.ndarray], rng: random.Random) -> float:
        """STUB-STEP3: MiniFASNet real/spoof probability, averaged over frames."""
        return rng.uniform(0.78, 0.97)

    def _motion(self, images: List[np.ndarray], rng: random.Random) -> float:
        """Frame-to-frame change. A printed photo held still barely moves.

        Real signal already: mean absolute difference between consecutive
        greyscale frames, squashed into 0..1. STUB-STEP3 will add texture and
        moire analysis on top.
        """
        if len(images) < 2:
            return 0.5
        diffs = []
        prev = images[0].mean(axis=2)
        for img in images[1:]:
            cur = img.mean(axis=2)
            if cur.shape != prev.shape:
                prev = cur
                continue
            diffs.append(float(np.abs(cur - prev).mean()))
            prev = cur
        if not diffs:
            return 0.5
        # ~1.5 grey levels of mean change across a burst is a comfortably live face.
        return float(min(1.0, np.mean(diffs) / 1.5))

    def _blink(self, images: List[np.ndarray], rng: random.Random) -> Tuple[float, bool]:
        """STUB-STEP3: MediaPipe eye-aspect-ratio dip detection."""
        detected = rng.random() < 0.8
        return (rng.uniform(0.75, 0.95) if detected else rng.uniform(0.2, 0.45)), detected

    def _device(self, images: List[np.ndarray], rng: random.Random) -> Tuple[float, bool]:
        """STUB-STEP3: YOLOv8n looking for phone / laptop / tv bezels."""
        seen = rng.random() < 0.08
        return (0.0 if seen else 1.0), seen

    # -- fusion ----------------------------------------------------------------

    def score(self, images: List[np.ndarray], rng: random.Random) -> LivenessResult:
        passive = self._passive(images, rng)
        motion = self._motion(images, rng)
        blink, blink_seen = self._blink(images, rng)
        device, device_seen = self._device(images, rng)

        fused = (
            config.W_PASSIVE * passive
            + config.W_MOTION * motion
            + config.W_BLINK * blink
            + config.W_DEVICE * device
        )
        fused = round(float(np.clip(fused, 0.0, 1.0)), 2)

        reasons = []
        reasons.append("Passive anti-spoof passed" if passive >= 0.6
                       else "Passive anti-spoof flagged a spoof")
        reasons.append("Natural micro-motion detected" if motion >= 0.4
                       else "Frames look static, possible printed photo")
        reasons.append("Blink detected" if blink_seen else "No blink detected")
        reasons.append("Screen or phone detected in frame" if device_seen
                       else "No phone or screen in frame")

        parts = {"passive": round(passive, 2), "motion": round(motion, 2),
                 "blink": round(blink, 2), "device": round(device, 2)}
        return LivenessResult(fused, reasons, parts, device_seen)

    # -- active challenge ------------------------------------------------------

    def check_challenge(self, action: str, images: List[np.ndarray],
                        rng: random.Random) -> Tuple[bool, float, List[str]]:
        """Did the user perform the requested action?

        STUB-STEP3: FaceMesh landmark tracking per action (eye-aspect-ratio for
        BLINK_TWICE, yaw for TURN_*, mouth corners for SMILE).
        """
        passed = rng.random() < 0.75
        score = round(rng.uniform(0.76, 0.95) if passed else rng.uniform(0.25, 0.55), 2)
        label = action.replace("_", " ").lower()
        reasons = [f"Challenge '{label}' {'completed' if passed else 'not completed'}"]
        return passed, score, reasons
