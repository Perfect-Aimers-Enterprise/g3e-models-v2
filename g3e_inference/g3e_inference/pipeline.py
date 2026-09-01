"""
G3EPipeline — the full production pipeline (spec section 21):

    image -> G3E-1 (detect) -> OpenCV (render annotated frame)
                             -> G3E-2 (reason, given image + detections)
          -> event_builder -> G3EEvent

Zero-config usage — this is the integration surface an app should use:

    from g3e_inference import G3EPipeline

    pipeline = G3EPipeline()
    event = pipeline.run("frame.jpg", output_dir="./events/evt_001")
    print(event.model_dump_json(by_alias=True, indent=2))

G3EPipeline() with no arguments constructs a default G3E1Detector() and
G3E2Reasoner() — both resolve their models local-first, Hugging Face Hub
fallback (see artifacts.py/defaults.py). The FIRST call to .run() anywhere
on a machine downloads and caches both models; every call after that
(including in a new process) reuses that cache and makes no network calls
at all.

detector/reasoner can still be passed explicitly — this is what makes the
orchestration logic itself (call order, event assembly, file writing)
fully testable with fake objects, with zero GPU/model/network dependency.
See tests/test_pipeline.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from g3e_inference.schemas import G3E1Output, G3E2Output, G3EEvent
from g3e_inference.event_builder import build_event, save_event, generate_event_id
from g3e_inference.renderer import draw_detections


class Detector(Protocol):
    def predict(self, image_path: str) -> G3E1Output: ...


class Reasoner(Protocol):
    def predict(self, image_path: str, detections: list[dict]) -> G3E2Output: ...


class G3EPipeline:
    def __init__(self, detector: Detector | None = None, reasoner: Reasoner | None = None):
        if detector is None:
            from g3e_inference.detector import G3E1Detector

            detector = G3E1Detector()
        if reasoner is None:
            from g3e_inference.reasoner import G3E2Reasoner

            reasoner = G3E2Reasoner()

        self.detector = detector
        self.reasoner = reasoner

    def run(self, image_path: str, output_dir: str | Path, event_id: str | None = None) -> G3EEvent:
        """
        INPUT:
          - image_path: path to a JPG/PNG frame on disk. Any resolution —
            both models resize internally as needed.
          - output_dir: directory to write outputs into (created if missing).
          - event_id: optional — auto-generated (evt_<12 hex chars>) if omitted.

        OUTPUT (all three always produced together):
          - Returns a G3EEvent (schemas.py) — the full spec section 22
            shape: event_id, timestamp, image paths, G3E-1 detections,
            G3E-2 semantic judgment.
          - Writes {output_dir}/annotated.jpg — the source frame with
            detection boxes + "CLASS NN%" labels drawn on it.
          - Writes {output_dir}/event.json — the same event, serialized
            with by_alias=True (so class_name correctly appears as
            "class" on disk, matching the spec's documented shape).

        Call .model_dump_json(by_alias=True) yourself on the returned
        event for an in-memory JSON string (e.g. to push over a queue)
        without needing to re-read event.json from disk.
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
