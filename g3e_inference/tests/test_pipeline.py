import json

import numpy as np
from PIL import Image

from g3e_inference.pipeline import G3EPipeline
from g3e_inference.schemas import G3E1Output, G3E2Output, DetectedObject


class FakeDetector:
    """Deterministic stand-in for G3E1Detector — no ultralytics/GPU needed."""
    def __init__(self, objects=None):
        self.objects = objects or [
            DetectedObject(**{"class": "person", "class_id": 0, "confidence": 0.98, "bbox": [10, 10, 50, 50]})
        ]
        self.calls = []

    def predict(self, image_path):
        self.calls.append(image_path)
        return G3E1Output(objects=self.objects)


class FakeReasoner:
    """Deterministic stand-in for G3E2Reasoner — no torch/transformers/GPU needed."""
    def __init__(self, output=None):
        self.output = output or G3E2Output(state="normal", severity="none", description="d", reason="")
        self.calls = []

    def predict(self, image_path, detections):
        self.calls.append((image_path, detections))
        return self.output


def _make_image(tmp_path):
    p = tmp_path / "frame.jpg"
    Image.fromarray(np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)).save(p)
    return p


def test_pipeline_calls_detector_then_reasoner_in_order(tmp_path):
    detector = FakeDetector()
    reasoner = FakeReasoner()
    pipeline = G3EPipeline(detector=detector, reasoner=reasoner)

    img = _make_image(tmp_path)
    pipeline.run(str(img), tmp_path / "out")

    assert detector.calls == [str(img)]
    assert reasoner.calls[0][0] == str(img)


def test_pipeline_passes_detector_output_to_reasoner_as_dicts(tmp_path):
    detector = FakeDetector(objects=[
        DetectedObject(**{"class": "knife", "class_id": 4, "confidence": 0.9, "bbox": [1, 2, 3, 4]})
    ])
    reasoner = FakeReasoner()
    pipeline = G3EPipeline(detector=detector, reasoner=reasoner)

    img = _make_image(tmp_path)
    pipeline.run(str(img), tmp_path / "out")

    passed_detections = reasoner.calls[0][1]
    assert passed_detections == [{"class": "knife", "class_id": 4, "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]}]


def test_pipeline_writes_annotated_image_and_event_json(tmp_path):
    pipeline = G3EPipeline(detector=FakeDetector(), reasoner=FakeReasoner())
    img = _make_image(tmp_path)
    out_dir = tmp_path / "out"

    event = pipeline.run(str(img), out_dir)

    assert (out_dir / "annotated.jpg").exists()
    assert (out_dir / "event.json").exists()

    saved = json.loads((out_dir / "event.json").read_text())
    assert saved["event_id"] == event.event_id
    assert saved["g3e1"]["objects"][0]["class"] == "person"


def test_pipeline_uses_provided_event_id(tmp_path):
    pipeline = G3EPipeline(detector=FakeDetector(), reasoner=FakeReasoner())
    img = _make_image(tmp_path)
    event = pipeline.run(str(img), tmp_path / "out", event_id="evt_fixed_id")
    assert event.event_id == "evt_fixed_id"


def test_pipeline_event_reflects_reasoner_output():
    pass  # covered by test below with a non-default reasoner output


def test_pipeline_propagates_reasoner_semantic_state(tmp_path):
    reasoner = FakeReasoner(output=G3E2Output(
        state="potential_threat", severity="high", description="d", reason="r", recommended_action="alert_user",
    ))
    pipeline = G3EPipeline(detector=FakeDetector(), reasoner=reasoner)
    img = _make_image(tmp_path)

    event = pipeline.run(str(img), tmp_path / "out")
    assert event.g3e2.state == "potential_threat"
    assert event.g3e2.recommended_action == "alert_user"


def test_pipeline_zero_config_construction_does_not_require_torch_or_ultralytics():
    """G3EPipeline() must be constructible without torch/ultralytics/transformers
    installed — those are only needed once .run() actually executes."""
    pipeline = G3EPipeline()
    assert pipeline.detector is not None
    assert pipeline.reasoner is not None
