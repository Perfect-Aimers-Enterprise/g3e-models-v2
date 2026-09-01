"""
G3E1Detector — wraps a trained Ultralytics YOLO model behind G3E-1's
output contract (`shared.schemas.G3E1Output`, spec section 7).

Lazy-loaded (model only loads on first `.predict()` call), same rationale
as `g3e2/reasoner.py::G3E2Reasoner` — so this class can be constructed and
passed around in tests/orchestration code without requiring `ultralytics`
or a GPU to be present.

Weights can be a local `.pt` file OR a (repo_id, filename) pair hosted on
Hugging Face Hub — the file is downloaded once (cached by huggingface_hub
in its usual cache dir) and handed to Ultralytics as a local path either way.
"""
from __future__ import annotations

from shared.schemas import G3E1Output, DetectedObject, G3E_CLASSES


class G3E1Detector:
    """
    Usage — local weights:

        detector = G3E1Detector(weights="./checkpoints/g3e1/best.pt")

    Usage — weights hosted on HF Hub:

        detector = G3E1Detector(hf_repo_id="Godsave22/g3e1-yolo-v1", hf_filename="best.pt")
    """

    def __init__(
        self,
        weights: str | None = None,
        hf_repo_id: str | None = None,
        hf_filename: str | None = None,
        hf_token: str | None = None,
        confidence_threshold: float = 0.25,
    ):
        if not weights and not (hf_repo_id and hf_filename):
            raise ValueError("Provide either `weights` (local .pt path) or both `hf_repo_id` and `hf_filename`.")
        self.weights = weights
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename
        self.hf_token = hf_token
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _resolve_weights_path(self) -> str:
        if self.weights:
            return self.weights

        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=self.hf_repo_id, filename=self.hf_filename, token=self.hf_token)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self._resolve_weights_path())

    def predict(self, image_path: str) -> G3E1Output:
        self._ensure_loaded()
        results = self._model.predict(image_path, conf=self.confidence_threshold, verbose=False)
        result = results[0]

        objects = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            objects.append(
                DetectedObject(
                    **{
                        "class": G3E_CLASSES.get(class_id, f"unknown_{class_id}"),
                        "class_id": class_id,
                        "confidence": float(box.conf.item()),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
            )

        return G3E1Output(objects=objects)
