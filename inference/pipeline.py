"""
Full production pipeline (spec section 21):

    image -> G3E-1 (detect) -> OpenCV (render annotated frame)
                             -> G3E-2 (reason, given image + detections)
          -> event_builder -> G3EEvent

`G3EPipeline` takes its detector and reasoner as constructor arguments
rather than constructing `G3E1Detector`/`G3E2Reasoner` internally — this
is what makes `run()`'s orchestration logic (call order, event assembly,
annotated-frame path wiring, error handling) fully testable with fake
detector/reasoner objects, without needing ultralytics, torch,
transformers, or a GPU. See tests/test_pipeline.py.

Real usage wires in the real classes:

    from g3e1.detector import G3E1Detector
    from g3e2.reasoner import G3E2Reasoner
    from inference.pipeline import G3EPipeline

    pipeline = G3EPipeline(
        detector=G3E1Detector(hf_repo_id="Godsave22/g3e1-yolo-v1", hf_filename="best.pt"),
        reasoner=G3E2Reasoner(adapter="Godsave22/g3e2-lora-v1"),
    )
    event = pipeline.run("frame.jpg", output_dir="./events/evt_001")
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from shared.schemas import G3E1Output, G3E2Output, G3EEvent
from inference.event_builder import build_event, save_event, generate_event_id
from inference.opencv_pipeline import draw_detections


class Detector(Protocol):
    def predict(self, image_path: str) -> G3E1Output: ...


class Reasoner(Protocol):
    def predict(self, image_path: str, detections: list[dict]) -> G3E2Output: ...


class G3EPipeline:
    def __init__(self, detector: Detector, reasoner: Reasoner):
        self.detector = detector
        self.reasoner = reasoner

    def run(self, image_path: str, output_dir: str | Path, event_id: str | None = None) -> G3EEvent:
        """
        Runs the full pipeline for one image and writes, under
        `output_dir`: the annotated frame (`annotated.jpg`) and the event
        JSON (`event.json`). Returns the `G3EEvent` object too, for
        callers that want to act on it immediately (e.g. push an alert)
        rather than re-reading the file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        event_id = event_id or generate_event_id()

        g3e1_output = self.detector.predict(image_path)

        detections_for_g3e2 = [d.model_dump(by_alias=True) for d in g3e1_output.objects]
        g3e2_output = self.reasoner.predict(image_path, detections_for_g3e2)

        annotated_path = output_dir / "annotated.jpg"
        draw_detections(image_path, g3e1_output.objects, annotated_path)

        event = build_event(
            image_original=str(image_path),
            image_annotated=str(annotated_path),
            g3e1_detections=g3e1_output.objects,
            g3e2_output=g3e2_output,
            event_id=event_id,
        )
        save_event(event, output_dir / "event.json")
        return event
