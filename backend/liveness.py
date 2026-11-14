"""Liveness: passive anti-spoof (real), motion (real), blink + device (stubs).

The passive anti-spoof signal is a real model: MiniFASNetV2-SE (INT8) from
github.com/facenox/face-antispoof-onnx (Apache-2.0). It runs on face crops
taken from our existing InsightFace bounding boxes -- the detector is never run
a second time. Blink and device-in-frame are still STUB-STEP5.
"""

import logging
import random
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

log = logging.getLogger("liveness")

Detection = Tuple[np.ndarray, Tuple[int, int, int, int]]  # (frame_bgr, bbox_xywh)


# ---------------------------------------------------------------------------
# Crop + preprocess copied VERBATIM from facenox/face-antispoof-onnx
# (src/inference/preprocess.py, Apache-2.0). The model was trained against this
# exact pipeline -- letterbox to 128, reflect-101 pad, /255, CHW, RGB, bbox
# expanded 1.5x. Do not "clean up" or substitute a plain resize.
# ---------------------------------------------------------------------------

def _fx_crop(img: np.ndarray, bbox: tuple, bbox_expansion_factor: float) -> np.ndarray:
    """facenox crop(): square face crop from (x1,y1,x2,y2) bbox, reflect-padded."""
    original_height, original_width = img.shape[:2]
    x, y, w, h = bbox
    w = w - x
    h = h - y
    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox dimensions")

    max_dim = max(w, h)
    center_x = x + w / 2
    center_y = y + h / 2

    x = int(center_x - max_dim * bbox_expansion_factor / 2)
    y = int(center_y - max_dim * bbox_expansion_factor / 2)
    crop_size = int(max_dim * bbox_expansion_factor)

    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(original_width, x + crop_size)
    crop_y2 = min(original_height, y + crop_size)

    top_pad = int(max(0, -y))
    left_pad = int(max(0, -x))
    bottom_pad = int(max(0, (y + crop_size) - original_height))
    right_pad = int(max(0, (x + crop_size) - original_width))

    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    else:
        img = np.zeros((0, 0, 3), dtype=img.dtype)

    result = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                                cv2.BORDER_REFLECT_101)
    if result.shape[0] != crop_size or result.shape[1] != crop_size:
        raise ValueError(f"Crop size mismatch: expected {crop_size}x{crop_size}, "
                         f"got {result.shape[0]}x{result.shape[1]}")
    return result


def _fx_preprocess(img: np.ndarray, model_img_size: int) -> np.ndarray:
    """facenox preprocess(): letterbox resize, normalise to [0,1], CHW."""
    new_size = model_img_size
    old_size = img.shape[:2]
    ratio = float(new_size) / max(old_size)
    scaled_shape = tuple(int(x * ratio) for x in old_size)

    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (scaled_shape[1], scaled_shape[0]), interpolation=interpolation)

    delta_w = new_size - scaled_shape[1]
    delta_h = new_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    return img.transpose(2, 0, 1).astype(np.float32) / 255.0

# ---------------------------------------------------------------------------


class PassiveAntiSpoof:
    """MiniFASNetV2-SE real/spoof model, loaded once at startup."""

    def __init__(self):
        self.session = None
        self.input_name = None
        self.available = False

    def load(self):
        import ort_utils
        if not config.ANTISPOOF_MODEL.exists():
            log.error("anti-spoof model missing at %s -- run: python tools/fetch_models.py. "
                      "Passive liveness will be neutral until then.", config.ANTISPOOF_MODEL)
            return
        self.session = ort_utils.make_session(config.ANTISPOOF_MODEL)
        self.input_name = self.session.get_inputs()[0].name
        self.available = True
        log.info("passive anti-spoof ready (MiniFASNetV2-SE, %dx%d)",
                 config.ANTISPOOF_INPUT, config.ANTISPOOF_INPUT)

    def _real_probs(self, crops_chw: np.ndarray) -> np.ndarray:
        """Batch of (N,3,H,W) -> real probability per crop via softmax over logits."""
        logits = self.session.run([], {self.input_name: crops_chw})[0]  # (N, 2)
        real, spoof = logits[:, 0], logits[:, 1]
        # softmax of 2 logits = sigmoid of their difference
        return 1.0 / (1.0 + np.exp(-(real - spoof)))

    def score_detections(self, detections: List[Detection]
                         ) -> Tuple[Optional[float], List[float], int]:
        """Average real-probability over up to ANTISPOOF_FRAMES detected frames.

        Returns (avg_real_prob, per_frame_probs, frames_used). avg is None when
        the model is unavailable or no crop could be built.
        """
        if not self.available or not detections:
            return None, [], 0

        batch = []
        for frame_bgr, (x, y, w, h) in detections[:config.ANTISPOOF_FRAMES]:
            try:
                face = _fx_crop(frame_bgr, (x, y, x + w, y + h),
                                config.ANTISPOOF_BBOX_EXPANSION)
            except ValueError:
                continue
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            batch.append(_fx_preprocess(rgb, config.ANTISPOOF_INPUT))

        if not batch:
            return None, [], 0
        probs = self._real_probs(np.stack(batch))
        per_frame = [round(float(p), 3) for p in probs]
        return float(np.mean(probs)), per_frame, len(per_frame)


class LivenessResult:
    def __init__(self, score: float, reasons: List[str], parts: dict,
                 device_seen: bool, spoof_detected: bool = False):
        self.score = score
        self.reasons = reasons
        self.parts = parts
        self.device_seen = device_seen
        self.spoof_detected = spoof_detected


class LivenessEngine:
    """All anti-spoof models. Loaded once at startup, never per request."""

    def __init__(self):
        self.passive = PassiveAntiSpoof()

    def load(self):
        self.passive.load()
        log.info("LivenessEngine ready (passive=%s)",
                 "real" if self.passive.available else "unavailable")

    # -- individual signals ----------------------------------------------------

    def _motion(self, images: List[np.ndarray]) -> float:
        """Frame-to-frame change. A printed photo held still barely moves."""
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

    def _blink(self, rng: random.Random) -> Tuple[float, bool]:
        """STUB-STEP5: MediaPipe eye-aspect-ratio dip detection."""
        detected = rng.random() < 0.8
        return (rng.uniform(0.75, 0.95) if detected else rng.uniform(0.2, 0.45)), detected

    def _device(self, rng: random.Random) -> Tuple[float, bool]:
        """STUB-STEP5: YOLOv8n looking for phone / laptop / tv bezels."""
        seen = rng.random() < 0.08
        return (0.0 if seen else 1.0), seen

    # -- fusion ----------------------------------------------------------------

    def score(self, images: List[np.ndarray], detections: List[Detection],
              rng: random.Random) -> LivenessResult:
        t0 = time.time()
        passive_avg, passive_frames, n_passive = self.passive.score_detections(detections)
        passive_ms = (time.time() - t0) * 1000

        # When the model is unavailable, don't fail open: fall back to a neutral
        # 0.5 and flag it loudly in the reasons.
        passive = passive_avg if passive_avg is not None else 0.5
        motion = self._motion(images)
        blink, blink_seen = self._blink(rng)
        device, device_seen = self._device(rng)

        fused = (config.W_PASSIVE * passive + config.W_MOTION * motion
                 + config.W_BLINK * blink + config.W_DEVICE * device)
        fused = round(float(np.clip(fused, 0.0, 1.0)), 2)

        # A confident spoof from the passive model vetoes to DENY on its own.
        spoof_detected = passive_avg is not None and passive_avg < config.ANTISPOOF_SPOOF_VETO

        reasons = []
        if passive_avg is None:
            reasons.append("Passive model unavailable, liveness reduced")
        elif passive >= config.ANTISPOOF_REAL_MIN:
            reasons.append(f"Real skin texture ({passive:.2f})")
        else:
            reasons.append(f"Print/screen texture detected ({passive:.2f})")
        reasons.append("Natural micro-motion detected" if motion >= 0.4
                       else "Frames look static, possible printed photo")
        reasons.append("Blink detected" if blink_seen else "No blink detected")
        reasons.append("Screen or phone detected in frame" if device_seen
                       else "No phone or screen in frame")

        log.info("liveness fused=%.2f passive=%.3f(n=%d,%.0fms) motion=%.2f",
                 fused, passive, n_passive, passive_ms, motion)

        parts = {"passive": round(passive, 3), "passive_frames": passive_frames,
                 "passive_ms": round(passive_ms, 1), "motion": round(motion, 2),
                 "blink": round(blink, 2), "device": round(device, 2)}
        return LivenessResult(fused, reasons, parts, device_seen, spoof_detected)

    # -- active challenge ------------------------------------------------------

    def check_challenge(self, action: str, images: List[np.ndarray],
                        rng: random.Random) -> Tuple[bool, float, List[str]]:
        """Did the user perform the requested action?

        STUB-STEP5: FaceMesh landmark tracking per action (eye-aspect-ratio for
        BLINK_TWICE, yaw for TURN_*, mouth corners for SMILE).
        """
        passed = rng.random() < 0.75
        score = round(rng.uniform(0.76, 0.95) if passed else rng.uniform(0.25, 0.55), 2)
        label = action.replace("_", " ").lower()
        reasons = [f"Challenge '{label}' {'completed' if passed else 'not completed'}"]
        return passed, score, reasons
