"""Liveness: passive anti-spoof (real), motion (real), blink + device (stubs).

The passive anti-spoof signal is a real model: MiniFASNetV2-SE (INT8) from
github.com/facenox/face-antispoof-onnx (Apache-2.0). It runs on face crops
taken from our existing InsightFace bounding boxes -- the detector is never run
a second time. Blink and device-in-frame are still STUB-STEP5.
"""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
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


def face_brightness(detections: List[Detection]) -> Optional[float]:
    """Mean grey level inside the face box, averaged over the detected frames.

    The passive anti-spoof model reads an underexposed real face as flat,
    print-like texture, so a dark face must not be vetoed as a spoof -- the
    caller turns a low value into RECAPTURE ("Too dark") instead.
    """
    values = []
    for frame_bgr, (x, y, w, h) in detections[:config.ANTISPOOF_FRAMES]:
        fh, fw = frame_bgr.shape[:2]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(fw, int(x + w)), min(fh, int(y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame_bgr[y1:y2, x1:x2]
        values.append(float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()))
    return float(np.mean(values)) if values else None


class PassiveAntiSpoof:
    """MiniFASNetV2-SE real/spoof model, loaded once at startup."""

    def __init__(self):
        self.session = None
        self.input_name = None
        self.available = False

    def load(self):
        import ort_utils
        # Prefer the INT8 model; fall back to the fp32 one if it is missing.
        path = next((p for p in (config.ANTISPOOF_MODEL, config.ANTISPOOF_MODEL_FP32)
                     if p.exists()), None)
        if path is None:
            log.error("anti-spoof model missing in %s -- run: python tools/fetch_models.py. "
                      "Passive liveness will be neutral until then.", config.MODEL_DIR)
            return
        self.session = ort_utils.make_session(path)
        self.input_name = self.session.get_inputs()[0].name
        self.available = True
        log.info("passive anti-spoof ready (MiniFASNetV2-SE, %dx%d, %s)",
                 config.ANTISPOOF_INPUT, config.ANTISPOOF_INPUT, path.name)

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


# ----------------------------------------------------- blink + head motion ---
# 6-point eye-aspect-ratio landmarks on MediaPipe's 478-point face mesh.
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def _ear(pts: np.ndarray) -> float:
    """Eye aspect ratio from 6 points: (|p2-p6|+|p3-p5|) / (2|p1-p4|)."""
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return float((a + b) / (2.0 * c)) if c > 1e-6 else 0.0


def _yaw_pitch_deg(matrix: np.ndarray) -> Tuple[float, float]:
    """Yaw, pitch (degrees) from a 4x4 facial transformation matrix."""
    r = np.asarray(matrix)[:3, :3]
    sy = float(np.hypot(r[0, 0], r[1, 0]))
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arctan2(-r[2, 0], sy))
    return float(yaw), float(pitch)


class BlinkMotion:
    """MediaPipe Face Landmarker: EAR-based blink + yaw/pitch micro-motion."""

    def __init__(self):
        self.landmarker = None
        self.available = False

    def load(self):
        if not config.FACE_LANDMARKER_MODEL.exists():
            log.error("face_landmarker.task missing at %s -- run tools/fetch_models.py; "
                      "blink/motion will be neutral", config.FACE_LANDMARKER_MODEL)
            return
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(config.FACE_LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        # Warm up: the first detect() builds the TFLite/XNNPACK graph (~0.5 s),
        # which would otherwise land on the first real request.
        import mediapipe as mp
        self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=np.zeros((480, 640, 3), dtype=np.uint8)))
        self.available = True
        log.info("blink/motion ready (MediaPipe Face Landmarker)")

    def analyze(self, images: List[np.ndarray]) -> dict:
        """Returns blink/motion evidence over the burst.

        {blink_detected, blink_score, motion_score, min_ear, yaw_range,
         pitch_range, frames_with_face}. Values are None-free; when the model is
         unavailable the caller falls back to a frame-difference motion proxy.
        """
        import mediapipe as mp
        ears, yaws, pitches = [], [], []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)
            if not res.face_landmarks:
                continue
            h, w = img.shape[:2]
            lms = res.face_landmarks[0]
            coords = np.array([[lms[i].x * w, lms[i].y * h] for i in
                               (*_LEFT_EYE, *_RIGHT_EYE)], dtype=np.float32)
            left_ear = _ear(coords[:6])
            right_ear = _ear(coords[6:])
            ears.append((left_ear + right_ear) / 2.0)
            if res.facial_transformation_matrixes:
                yaw, pitch = _yaw_pitch_deg(res.facial_transformation_matrixes[0])
                yaws.append(yaw)
                pitches.append(pitch)

        if not ears:
            return {"blink_detected": False, "blink_score": 0.0, "motion_score": None,
                    "min_ear": None, "yaw_range": 0.0, "pitch_range": 0.0,
                    "frames_with_face": 0}

        min_ear, max_ear = min(ears), max(ears)
        blink_detected = (min_ear < config.EAR_BLINK_THRESHOLD
                          and (max_ear - min_ear) >= config.EAR_BLINK_DELTA)
        blink_score = 1.0 if blink_detected else float(np.clip(
            (max_ear - min_ear) / config.EAR_BLINK_DELTA, 0.0, 0.8))

        yaw_range = (max(yaws) - min(yaws)) if len(yaws) >= 2 else 0.0
        pitch_range = (max(pitches) - min(pitches)) if len(pitches) >= 2 else 0.0
        motion_score = None
        if len(yaws) >= 2:
            motion_score = float(np.clip((yaw_range + pitch_range)
                                         / config.MOTION_YAWPITCH_FULL_DEG, 0.0, 1.0))
        return {"blink_detected": blink_detected, "blink_score": blink_score,
                "motion_score": motion_score, "min_ear": round(min_ear, 3),
                "yaw_range": round(yaw_range, 2), "pitch_range": round(pitch_range, 2),
                "frames_with_face": len(ears)}

    def landmark_series(self, images: List[np.ndarray]) -> List[dict]:
        """Per-frame {ear, yaw, mouth_ratio} for active-challenge checking.

        mouth_ratio = mouth corner width / inter-eye width, so it is scale and
        distance invariant -- a smile widens it.
        """
        import mediapipe as mp
        series = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                continue
            h, w = img.shape[:2]
            lms = res.face_landmarks[0]

            def px(i):
                return np.array([lms[i].x * w, lms[i].y * h], dtype=np.float32)

            eye = np.array([px(i) for i in (*_LEFT_EYE, *_RIGHT_EYE)])
            ear = (_ear(eye[:6]) + _ear(eye[6:])) / 2.0
            yaw = 0.0
            if res.facial_transformation_matrixes:
                yaw, _ = _yaw_pitch_deg(res.facial_transformation_matrixes[0])
            inter_eye = float(np.linalg.norm(px(33) - px(263))) or 1.0
            mouth = float(np.linalg.norm(px(61) - px(291))) / inter_eye
            series.append({"ear": ear, "yaw": yaw, "mouth_ratio": mouth})
        return series


class DeviceDetector:
    """YOLO (ultralytics): phone / laptop / tv overlapping the face box."""

    def __init__(self):
        self.model = None
        self.available = False

    def load(self):
        if not config.YOLO_MODEL.exists():
            log.error("YOLO weights missing at %s -- run tools/fetch_models.py; "
                      "device check disabled", config.YOLO_MODEL)
            return
        from ultralytics import YOLO
        self.model = YOLO(str(config.YOLO_MODEL))
        # Warm up so the first real request is not penalised by lazy init.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8),
                           imgsz=config.YOLO_IMGSZ, conf=config.YOLO_CONF,
                           verbose=False, device="cpu")
        self.available = True
        log.info("device detector ready (YOLO %s)", config.YOLO_MODEL.name)

    @staticmethod
    def _inside_frac(dev_xyxy, face_xywh) -> float:
        """Fraction of the FACE box that lies inside the device box (0..1)."""
        dx1, dy1, dx2, dy2 = dev_xyxy
        fx, fy, fw, fh = face_xywh
        fx2, fy2 = fx + fw, fy + fh
        ix1, iy1 = max(dx1, fx), max(dy1, fy)
        ix2, iy2 = min(dx2, fx2), min(dy2, fy2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        face_area = max(1.0, fw * fh)
        return (iw * ih) / face_area

    def analyze(self, images: List[np.ndarray],
                face_xywh: Tuple[int, int, int, int]) -> Tuple[bool, Optional[str]]:
        """True only when a device is plausibly REPLAYING the face.

        A replay puts the face inside the device's screen, so the test is how
        much of the face box lies inside the device box:

          * tv / laptop -> at least DEVICE_FACE_INSIDE of the face inside it, and
            a box covering more than DEVICE_MAX_FRAME_FRAC of the frame (i.e. a
            monitor or TV in the room) is ignored unless the face really is
            inside it. A screen merely visible in the background never trips.
          * cell phone  -> a looser DEVICE_FACE_OVERLAP, because a phone held up
            next to the face is already a presentation attack.
        """
        for img in images[:config.YOLO_FRAMES]:
            frame_area = max(1.0, float(img.shape[0] * img.shape[1]))
            res = self.model.predict(img, imgsz=config.YOLO_IMGSZ, conf=config.YOLO_CONF,
                                     verbose=False, device="cpu")[0]
            names = res.names
            for box in res.boxes:
                label = names[int(box.cls)]
                if label not in config.YOLO_DEVICE_CLASSES:
                    continue
                dx1, dy1, dx2, dy2 = box.xyxy[0].tolist()
                inside = self._inside_frac((dx1, dy1, dx2, dy2), face_xywh)
                dev_frac = (max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)) / frame_area

                if label == "cell phone":
                    if inside >= config.DEVICE_FACE_OVERLAP:
                        return True, label
                    continue

                # tv / laptop: background screens are large and do not contain
                # the face -- both conditions must fail for it to be ignored.
                if dev_frac > config.DEVICE_MAX_FRAME_FRAC and inside < config.DEVICE_FACE_INSIDE:
                    log.debug("ignoring background %s (%.0f%% of frame, face %.0f%% inside)",
                              label, dev_frac * 100, inside * 100)
                    continue
                if inside >= config.DEVICE_FACE_INSIDE:
                    return True, label
        return False, None


class LivenessResult:
    def __init__(self, score: float, reasons: List[str], parts: dict,
                 device_seen: bool, spoof_detected: bool = False,
                 too_dark: bool = False):
        self.score = score
        self.reasons = reasons
        self.parts = parts
        self.device_seen = device_seen
        self.spoof_detected = spoof_detected
        # The face was too dark to judge: the caller asks for a retake instead
        # of denying a real person for bad lighting.
        self.too_dark = too_dark


class LivenessEngine:
    """All anti-spoof models. Loaded once at startup, never per request."""

    def __init__(self):
        self.passive = PassiveAntiSpoof()
        self.blinkmotion = BlinkMotion()
        self.device = DeviceDetector()

    def load(self):
        for name, engine in (("passive", self.passive), ("blink/motion", self.blinkmotion),
                             ("device", self.device)):
            try:
                engine.load()
            except Exception as exc:
                log.warning("%s signal failed to load: %s", name, exc)
        log.info("LivenessEngine ready (passive=%s blink/motion=%s device=%s)",
                 self.passive.available, self.blinkmotion.available, self.device.available)

    def _motion_fallback(self, images: List[np.ndarray]) -> float:
        """Frame-difference motion, used only when the landmarker is unavailable."""
        if len(images) < 2:
            return 0.5
        diffs, prev = [], images[0].mean(axis=2)
        for img in images[1:]:
            cur = img.mean(axis=2)
            if cur.shape == prev.shape:
                diffs.append(float(np.abs(cur - prev).mean()))
            prev = cur
        return float(min(1.0, np.mean(diffs) / 1.5)) if diffs else 0.5

    # -- fusion ----------------------------------------------------------------

    def score(self, images: List[np.ndarray], detections: List[Detection],
              rng: random.Random) -> LivenessResult:
        timings = {}
        face_box = detections[0][1] if detections else None
        bm_default = {"blink_detected": False, "blink_score": 0.0, "motion_score": None,
                      "min_ear": None, "yaw_range": 0.0, "pitch_range": 0.0,
                      "frames_with_face": 0}

        # Blink/motion (MediaPipe) and device (YOLO) are the two slow, independent
        # checks -- run them concurrently. Both release the GIL during inference,
        # so this roughly halves the liveness latency.
        def run_blinkmotion():
            t = time.time()
            out = self.blinkmotion.analyze(images) if self.blinkmotion.available else bm_default
            return out, round((time.time() - t) * 1000, 1)

        def run_device():
            t = time.time()
            seen, label = (self.device.analyze(images, face_box)
                           if self.device.available and face_box is not None else (False, None))
            return (seen, label), round((time.time() - t) * 1000, 1)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_bm = pool.submit(run_blinkmotion)
            f_dev = pool.submit(run_device)
            # Passive is cheap; run it on this thread while the others work.
            t = time.time()
            passive_avg, passive_frames, _ = self.passive.score_detections(detections)
            timings["passive_ms"] = round((time.time() - t) * 1000, 1)
            bm, timings["blinkmotion_ms"] = f_bm.result()
            (device_seen, device_label), timings["device_ms"] = f_dev.result()

        # --- graded signals -------------------------------------------------
        passive = passive_avg if passive_avg is not None else 0.5
        blink = bm["blink_score"]
        motion = bm["motion_score"]
        if motion is None:                       # landmarker gave < 2 head poses
            motion = self._motion_fallback(images)

        fused = (config.W_PASSIVE * passive + config.W_MOTION * motion
                 + config.W_BLINK * blink)
        fused = round(float(np.clip(fused, 0.0, 1.0)), 2)

        # Anti-spoof is a GATE, not just a weighted input: the liveness score
        # used for the decision is min(passive, fused), so a low passive caps
        # the whole score and motion can never rescue a spoof.
        gated = round(min(passive, fused), 2)

        # --- hard flags -----------------------------------------------------
        # A detected device, or a confident passive spoof, each force DENY --
        # except in low light, where the passive model cannot tell a real face
        # from a print. Then the veto is suppressed and the caller asks for a
        # retake. The veto stays hard at normal brightness.
        brightness = face_brightness(detections)
        too_dark = brightness is not None and brightness < config.MIN_FACE_BRIGHTNESS
        spoof_detected = (passive_avg is not None
                          and passive_avg < config.ANTISPOOF_HARD_DENY
                          and not too_dark)

        device_ran = self.device.available and face_box is not None

        # --- reasons --------------------------------------------------------
        reasons = []
        if device_seen:
            reasons.append(f"Phone/screen detected around face ({device_label})")
        if too_dark:
            reasons.append(f"Face too dark to judge liveness (brightness {brightness:.0f})")
        if spoof_detected:
            reasons.append("Spoof detected: print/screen texture")
        elif passive_avg is None:
            reasons.append("Passive model unavailable, liveness reduced")
        elif passive >= config.ANTISPOOF_REAL_MIN:
            reasons.append(f"Real skin texture ({passive:.2f})")
        else:
            reasons.append(f"Print/screen texture detected ({passive:.2f})")
        reasons.append(f"Blink detected (EAR {bm['min_ear']})" if bm["blink_detected"]
                       else "No blink detected")
        reasons.append(f"Head micro-motion present ({bm['yaw_range']:.1f}deg yaw)"
                       if motion >= 0.4 else "Frames look static, possible printed photo")

        log.info("liveness gated=%.2f (fused=%.2f) passive=%.3f motion=%.2f blink=%.2f "
                 "device=%s spoof=%s dark=%s (face brightness %s) | %s",
                 gated, fused, passive, motion, blink,
                 device_label or ("clear" if device_ran else "n/a"), spoof_detected,
                 too_dark, "n/a" if brightness is None else f"{brightness:.0f}", timings)

        # device: label when detected, False when it ran and was clear, None if it
        # could not run -- so the response always carries a device result.
        device_report = device_label if device_seen else (False if device_ran else None)
        parts = {"passive": round(passive, 3), "passive_frames": passive_frames,
                 "face_brightness": round(brightness, 1) if brightness is not None else None,
                 "too_dark": too_dark,
                 "fused": fused, "motion": round(motion, 2), "blink": round(blink, 2),
                 "blink_detected": bm["blink_detected"], "device": device_report,
                 "device_seen": device_seen, "device_ran": device_ran,
                 "yaw_range": bm["yaw_range"], "pitch_range": bm["pitch_range"],
                 **timings}
        return LivenessResult(gated, reasons, parts, device_seen, spoof_detected,
                              too_dark)

    # -- active challenge ------------------------------------------------------

    def check_challenge(self, action: str,
                        images: List[np.ndarray]) -> Tuple[bool, float, List[str]]:
        """Did the user perform the requested action? Real landmark detection.

        BLINK_TWICE -> count EAR dips; TURN_LEFT/RIGHT -> head yaw magnitude and
        direction; SMILE -> rise in mouth-width / inter-eye-width.
        """
        if not self.blinkmotion.available:
            return False, 0.0, ["Landmark model unavailable"]

        series = self.blinkmotion.landmark_series(images)
        if len(series) < 3:
            return False, 0.0, ["Could not track your face during the challenge"]

        ears = [s["ear"] for s in series]
        yaws = [s["yaw"] for s in series]
        mouths = [s["mouth_ratio"] for s in series]

        if action == "BLINK_TWICE":
            blinks, closed = 0, False
            for e in ears:
                if e < config.EAR_BLINK_THRESHOLD and not closed:
                    blinks += 1
                    closed = True
                elif e >= config.EAR_BLINK_THRESHOLD:
                    closed = False
            passed = blinks >= config.CHALLENGE_BLINKS_REQUIRED
            score = min(1.0, blinks / config.CHALLENGE_BLINKS_REQUIRED)
            return passed, round(score, 2), [f"Detected {blinks} blink(s)"]

        if action in ("TURN_LEFT", "TURN_RIGHT"):
            want_left = action == "TURN_LEFT"
            sign = config.CHALLENGE_YAW_LEFT_SIGN if want_left else -config.CHALLENGE_YAW_LEFT_SIGN
            reached = min(yaws) if sign < 0 else max(yaws)
            passed = (sign * reached) >= config.CHALLENGE_YAW_DEG
            score = min(1.0, abs(reached) / config.CHALLENGE_YAW_DEG)
            return passed, round(score, 2), [f"Head yaw reached {reached:.0f}deg"]

        if action == "SMILE":
            rise = max(mouths) - min(mouths)
            passed = rise >= config.SMILE_RATIO_DELTA or max(mouths) >= config.SMILE_MIN_RATIO
            score = min(1.0, rise / config.SMILE_RATIO_DELTA)
            return passed, round(score, 2), [f"Mouth widened by {rise:.2f}"]

        return False, 0.0, [f"Unknown challenge '{action}'"]
