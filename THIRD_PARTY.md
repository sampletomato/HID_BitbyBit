# Third-party models and code

This project uses pretrained models and a small amount of preprocessing code
from the open-source projects below. We do **not** redistribute their source or
weights in this repository: model weights are downloaded at setup time by
`backend/tools/fetch_models.py` into `backend/models/` (gitignored), and the
only third-party code in our tree is the two preprocessing functions noted
below, reproduced under their original license.

## Passive anti-spoof — facenox / face-antispoof-onnx

- **Project:** https://github.com/facenox/face-antispoof-onnx
- **License:** Apache License 2.0
- **Model used:** `best_model_quantized.onnx` (MiniFASNetV2-SE, INT8), with
  `best_model.onnx` (fp32) as a fallback. 128×128 RGB input, logits `[real, spoof]`.
- **Code reproduced:** the `crop()` and `preprocess()` functions from
  `src/inference/preprocess.py` are copied verbatim into
  [`backend/liveness.py`](backend/liveness.py) (marked as such), because the
  model must receive the exact preprocessing it was trained against. Reproduced
  under the Apache-2.0 license.
- **Upstream:** facenox itself is based on the MiniFASNet architecture from the
  **Silent-Face-Anti-Spoofing** project by **Minivision AI**
  (https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), also Apache-2.0.

## Face detection + recognition — InsightFace

- **Project:** https://github.com/deepinsight/insightface
- **Model used:** `buffalo_l` pack (RetinaFace `det_10g` + ArcFace `w600k_r50`),
  downloaded by InsightFace on first run into `backend/models/`.
- **License:** InsightFace code is MIT. The pretrained `buffalo_l` models are
  released by InsightFace **for non-commercial / academic research use**. This
  project is a hackathon prototype; any production or commercial use would need
  models cleared for that purpose.

## Fallback face backend — OpenCV Zoo (YuNet + SFace)

- **Project:** https://github.com/opencv/opencv_zoo
- **Models used (only when `FACE_BACKEND=opencv`):**
  `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`.
- **License:** the YuNet and SFace models in OpenCV Zoo are distributed under
  their respective licenses (MIT / Apache-2.0); see the upstream repo.

---

Full license texts live with each upstream project. Apache-2.0 components are
used in compliance with the license, including preservation of attribution.
