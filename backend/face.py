"""Frame decoding, quality gating, face detection/embedding, and the gallery.

Two interchangeable recognition backends sit behind one interface:
  * insightface -- buffalo_l, 512-d embeddings (preferred)
  * opencv      -- YuNet detector + SFace recogniser, 128-d (fallback)

Embeddings are stored L2-normalised, so cosine similarity is a plain dot
product. Thresholds differ per backend -- see config.MATCH_THRESHOLDS.
"""

import base64
import binascii
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

log = logging.getLogger("face")

_DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z+]+;base64,", re.IGNORECASE)


# --------------------------------------------------------------------- decoding

def decode_data_url(frame: str) -> Optional[np.ndarray]:
    """base64 data-URL (or bare base64) JPEG -> BGR ndarray. None if unusable."""
    if not isinstance(frame, str) or not frame:
        return None
    payload = _DATA_URL_RE.sub("", frame.strip())
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def decode_frames(frames: List[str]) -> List[np.ndarray]:
    """Decode a burst, silently dropping frames that fail."""
    out = []
    for f in frames or []:
        img = decode_data_url(f)
        if img is not None and img.size:
            out.append(img)
    return out


# ---------------------------------------------------------------------- quality

class Quality:
    """Per-burst image quality verdict driving the RECAPTURE decision."""

    def __init__(self, ok: bool, reason: Optional[str], blur: float, brightness: float,
                 good_frames: int, total_frames: int):
        self.ok = ok
        self.reason = reason
        self.blur = blur
        self.brightness = brightness
        self.good_frames = good_frames
        self.total_frames = total_frames

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "blur": round(self.blur, 1),
                "brightness": round(self.brightness, 1), "good_frames": self.good_frames,
                "total_frames": self.total_frames}


def _sharpness(img: np.ndarray) -> float:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def _frame_metrics(img: np.ndarray) -> Tuple[float, float]:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return blur, float(grey.mean())


def assess_quality(images: List[np.ndarray], min_frames: int) -> Quality:
    if len(images) < min_frames:
        return Quality(False, f"Need at least {min_frames} usable frames, got {len(images)}",
                       0.0, 0.0, 0, len(images))

    blurs, brights, good = [], [], 0
    for img in images:
        blur, brightness = _frame_metrics(img)
        blurs.append(blur)
        brights.append(brightness)
        if blur >= config.MIN_BLUR_VAR and config.MIN_BRIGHTNESS <= brightness <= config.MAX_BRIGHTNESS:
            good += 1

    med_blur = float(np.median(blurs))
    med_bright = float(np.median(brights))

    log.info("quality: median blur=%.1f (min %.0f)  brightness=%.1f (min %.0f)  "
             "good=%d/%d", med_blur, config.MIN_BLUR_VAR, med_bright,
             config.MIN_BRIGHTNESS, good, len(images))

    if good / len(images) < config.MIN_GOOD_FRAME_RATIO:
        # Brightness first: an over- or under-exposed frame is also flat, so it
        # would otherwise be misreported as blur.
        if med_bright < config.MIN_BRIGHTNESS:
            reason = "Too dark, please face the light"
        elif med_bright > config.MAX_BRIGHTNESS:
            reason = "Too bright, please avoid backlight"
        else:
            reason = "Image too blurry, please hold still"
        return Quality(False, reason, med_blur, med_bright, good, len(images))

    return Quality(True, None, med_blur, med_bright, good, len(images))


def pick_sharpest(images: List[np.ndarray], k: int) -> List[np.ndarray]:
    """The k sharpest frames. Embedding every frame blows the 2 s budget."""
    if len(images) <= k:
        return images
    ranked = sorted(images, key=_sharpness, reverse=True)
    return ranked[:k]


# ------------------------------------------------------------------- embeddings

def l2norm(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


class DetectedFace:
    def __init__(self, bbox: Tuple[int, int, int, int], embedding: np.ndarray,
                 det_score: float):
        self.bbox = bbox          # x, y, w, h
        self.embedding = embedding
        self.det_score = det_score

    @property
    def size_px(self) -> int:
        return min(self.bbox[2], self.bbox[3])


class InsightFaceBackend:
    """buffalo_l detection + recognition on CPU.

    Deliberately does NOT use insightface's FaceAnalysis wrapper. FaceAnalysis
    builds each ONNX session with default options, so the detector and the
    recogniser each spawn a full spin-waiting thread pool and starve one
    another -- measured at ~2.3 s per frame. Owning the sessions and capping
    the pools brings that to ~110 ms.
    """

    name = "insightface"
    dim = 512

    def __init__(self):
        self.det = None
        self.rec = None
        self._align = None

    def load(self):
        import ort_utils
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.model_zoo.retinaface import RetinaFace
        from insightface.utils import face_align, storage

        self._align = face_align

        # Downloads buffalo_l into config.MODEL_DIR on first run.
        model_dir = Path(storage.ensure_available("models", "buffalo_l",
                                                  root=str(config.MODEL_DIR)))
        det_path = model_dir / "det_10g.onnx"
        rec_path = model_dir / "w600k_r50.onnx"
        for p in (det_path, rec_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} missing from the buffalo_l pack")

        self.det = RetinaFace(model_file=str(det_path),
                              session=ort_utils.make_session(det_path))
        self.det.prepare(ctx_id=-1, input_size=(config.DET_SIZE, config.DET_SIZE))

        self.rec = ArcFaceONNX(model_file=str(rec_path),
                               session=ort_utils.make_session(rec_path))
        self.rec.prepare(ctx_id=-1)

    def analyze(self, img: np.ndarray) -> Optional[DetectedFace]:
        bboxes, kpss = self.det.detect(img, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        i = int(np.argmax(areas))
        score = float(bboxes[i][4])
        if score < config.DET_SCORE_MIN or kpss is None:
            return None
        aligned = self._align.norm_crop(img, kpss[i], self.rec.input_size[0])
        emb = np.asarray(self.rec.get_feat(aligned), dtype=np.float32).flatten()
        x1, y1, x2, y2 = [int(v) for v in bboxes[i][:4]]
        return DetectedFace((x1, y1, x2 - x1, y2 - y1), l2norm(emb), score)


class OpenCVBackend:
    """YuNet detector + SFace recogniser. Needs the two ONNX files in models/."""

    name = "opencv"
    dim = 128

    def __init__(self):
        self.detector = None
        self.recognizer = None
        self._input_size = None

    def load(self):
        for path in (config.YUNET_MODEL, config.SFACE_MODEL):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path.name} missing -- run: python tools/fetch_models.py")
        self.detector = cv2.FaceDetectorYN.create(
            str(config.YUNET_MODEL), "", (config.DET_SIZE, config.DET_SIZE),
            config.DET_SCORE_MIN, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(config.SFACE_MODEL), "")

    def analyze(self, img: np.ndarray) -> Optional[DetectedFace]:
        h, w = img.shape[:2]
        if self._input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self.detector.detect(img)
        if faces is None or not len(faces):
            return None
        best = max(faces, key=lambda f: f[2] * f[3])
        if float(best[14]) < config.DET_SCORE_MIN:
            return None
        aligned = self.recognizer.alignCrop(img, best)
        feat = self.recognizer.feature(aligned).flatten()
        return DetectedFace((int(best[0]), int(best[1]), int(best[2]), int(best[3])),
                            l2norm(np.asarray(feat, dtype=np.float32)),
                            float(best[14]))


class BurstResult:
    """What we could extract from one burst of frames.

    `detections` carries (frame_bgr, bbox_xywh) for each frame a face was found
    in, so downstream consumers (passive anti-spoof) can reuse the detection
    instead of running the detector a second time.
    """

    def __init__(self, query: Optional[np.ndarray], faces_found: int,
                 note: Optional[str], embeddings: Optional[List[np.ndarray]] = None,
                 detections: Optional[List[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = None):
        self.query = query            # averaged, normalised query embedding
        self.faces_found = faces_found
        self.note = note
        self.embeddings = embeddings or []
        self.detections = detections or []


# ---------------------------------------------------------------------- gallery

class Gallery:
    """Enrolled identities: embeddings + TOTP secret, persisted as JSON.

    Lives only on the edge device. Embeddings and secrets are never sent to the
    central server.
    """

    def __init__(self, path=config.GALLERY_PATH):
        self.path = path
        self.users: dict = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            self.users = json.loads(self.path.read_text(encoding="utf-8"))
            log.info("gallery loaded: %d user(s)", len(self.users))
        except (OSError, ValueError) as exc:
            log.warning("gallery unreadable (%s), starting empty", exc)
            self.users = {}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.users, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, name: str, embeddings: List[np.ndarray], backend: str,
            faces_used: int, totp_secret: str):
        self.users[name] = {
            "backend": backend,
            "embeddings": [e.tolist() for e in embeddings],
            "faces_used": faces_used,
            "enrolled_at": time.time(),
            "totp_secret": totp_secret,
            "totp_last_step": None,
        }
        self._save()

    def get(self, name: str) -> Optional[dict]:
        return self.users.get(name)

    def totp_secret(self, name: str) -> Optional[str]:
        rec = self.users.get(name) or {}
        return rec.get("totp_secret")

    def totp_last_step(self, name: str) -> Optional[int]:
        rec = self.users.get(name) or {}
        return rec.get("totp_last_step")

    def consume_totp_step(self, name: str, step: int):
        if name in self.users:
            self.users[name]["totp_last_step"] = step
            self._save()

    def mark_confirmed(self, name: str):
        if name in self.users:
            self.users[name]["authenticator_confirmed"] = True
            self._save()

    def names(self) -> List[str]:
        return sorted(self.users)

    def is_empty(self) -> bool:
        return not self.users

    def usable_for(self, backend: str) -> List[str]:
        """Names whose embeddings came from this backend. Others need re-enroll."""
        return sorted(n for n, r in self.users.items() if r.get("backend") == backend)


# ------------------------------------------------------------------ face engine

class FaceEngine:
    """Detection + embeddings + matching. Loaded once at startup."""

    def __init__(self):
        self.backend = None
        self.gallery = Gallery()

    @property
    def backend_name(self) -> str:
        return self.backend.name if self.backend else "none"

    @property
    def match_allow(self) -> float:
        return config.MATCH_THRESHOLDS[self.backend_name]["allow"]

    @property
    def match_deny(self) -> float:
        return config.MATCH_THRESHOLDS[self.backend_name]["deny"]

    def load(self):
        wanted = config.FACE_BACKEND
        order = {"insightface": [InsightFaceBackend],
                 "opencv": [OpenCVBackend]}.get(wanted,
                                                [InsightFaceBackend, OpenCVBackend])
        errors = []
        for cls in order:
            t0 = time.time()
            try:
                backend = cls()
                backend.load()
                self.backend = backend
                log.info("face backend '%s' ready in %.2fs (dim=%d, det=%d)",
                         backend.name, time.time() - t0, backend.dim, config.DET_SIZE)
                break
            except Exception as exc:
                errors.append(f"{cls.name}: {exc}")
                log.warning("face backend '%s' unavailable: %s", cls.name, exc)
        if self.backend is None:
            raise RuntimeError("no face backend could load -- " + "; ".join(errors))

        stale = [n for n in self.gallery.names()
                 if n not in self.gallery.usable_for(self.backend_name)]
        if stale:
            log.warning("%d enrollment(s) came from another backend and need "
                        "re-enrolling: %s", len(stale), ", ".join(stale))

    # -- embedding -------------------------------------------------------------

    def embed_burst(self, images: List[np.ndarray], max_frames: int = None) -> BurstResult:
        """Detect + embed the sharpest frames, averaged into one query vector."""
        chosen = pick_sharpest(images, max_frames or config.EMBED_FRAMES)
        embeddings, detections, too_small = [], [], 0
        for img in chosen:
            face = self.backend.analyze(img)
            if face is None:
                continue
            if face.size_px < config.MIN_FACE_PX:
                too_small += 1
                continue
            embeddings.append(face.embedding)
            detections.append((img, face.bbox))

        if not embeddings:
            note = ("Face too far from the camera, please step closer"
                    if too_small else "No face detected, please look at the camera")
            return BurstResult(None, 0, note)

        query = l2norm(np.mean(np.stack(embeddings), axis=0))
        return BurstResult(query, len(embeddings), None, embeddings, detections)

    # -- enrollment ------------------------------------------------------------

    def enroll(self, name: str, images: List[np.ndarray],
               totp_secret: str) -> Tuple[int, Optional[str]]:
        """Returns (faces_used, error). Embeds every enrollment frame, not just 4."""
        burst = self.embed_burst(images, max_frames=len(images))
        if burst.query is None:
            return 0, burst.note
        self.gallery.add(name, burst.embeddings, self.backend_name,
                         burst.faces_found, totp_secret)
        return burst.faces_found, None

    # -- identification --------------------------------------------------------

    def match(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        """Best enrolled identity for a query embedding, by cosine similarity."""
        best_name, best_score = None, 0.0
        for name in self.gallery.usable_for(self.backend_name):
            stored = self.gallery.users[name]["embeddings"]
            if not stored:
                continue
            sims = np.asarray(stored, dtype=np.float32) @ query
            score = float(np.max(sims))
            if score > best_score:
                best_name, best_score = name, score
        return best_name, round(max(0.0, best_score), 2)

    def identify(self, images: List[np.ndarray]) -> Tuple[Optional[str], float, BurstResult]:
        burst = self.embed_burst(images)
        if burst.query is None:
            return None, 0.0, burst
        name, score = self.match(burst.query)
        return name, score, burst
