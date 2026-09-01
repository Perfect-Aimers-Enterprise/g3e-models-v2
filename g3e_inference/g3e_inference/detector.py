"""
G3E1Detector — wraps a trained Ultralytics YOLO model behind G3E-1's
output contract (schemas.G3E1Output, spec section 7).

Model resolution is local-first, Hugging Face Hub fallback — see
artifacts.py for the exact precedence rules. G3E1Detector() with zero
arguments works out of the box: it resolves the default HF repo
(defaults.py), which is exactly what makes "pip install g3e-inference"
then immediately calling G3E1Detector().predict(...) "just work" — the
first call downloads and caches the weights; every call after that on the
same machine reuses that cache.

Lazy-loaded (the model only loads on the first .predict() call, not in
__init__) so a G3E1Detector instance is cheap to construct and pass
around — e.g. in tests, or while building a G3EPipeline — without
requiring ultralytics or a GPU to even be importable yet.
"""
from __future__ import annotations

from g3e_inference.schemas import G3E1Output, DetectedObject, G3E_CLASSES
from g3e_inference.artifacts import resolve_weights_file
from g3e_inference.defaults import DEFAULT_G3E1_HF_REPO, DEFAULT_G3E1_HF_FILENAME, DEFAULT_HF_TOKEN


class G3E1Detector:
    """
    Usage — zero-config (resolves the default HF repo, local cache after first run):

        detector = G3E1Detector()

    Usage — your own local weights take priority if present:

        detector = G3E1Detector(weights="./checkpoints/g3e1/best.pt")
        # falls back to the default HF repo automatically if that path doesn't exist

    Usage — a different HF repo (e.g. a newer model version):

        detector = G3E1Detector(hf_repo_id="YourOrg/g3e1-yolo-v2", hf_filename="best.pt")
    """

    def __init__(
        self,
        weights: str | None = None,
        hf_repo_id: str = DEFAULT_G3E1_HF_REPO,
        hf_filename: str = DEFAULT_G3E1_HF_FILENAME,
        hf_token: str | None = DEFAULT_HF_TOKEN,
        confidence_threshold: float = 0.25,
    ):
        self.weights = weights
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename
        self.hf_token = hf_token
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        weights_path = resolve_weights_file(self.weights, self.hf_repo_id, self.hf_filename, self.hf_token)
        self._model = YOLO(weights_path)

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
