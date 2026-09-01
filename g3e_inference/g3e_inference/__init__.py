"""
g3e-inference
=============
Production inference for G3E-1 (detection) + G3E-2 (semantic reasoning).

Quickstart:

    from g3e_inference import G3EPipeline

    pipeline = G3EPipeline()  # zero-config - resolves models local-first, HF Hub fallback
    event = pipeline.run("frame.jpg", output_dir="./events/evt_001")
    print(event.model_dump_json(by_alias=True, indent=2))

See README.md for the full input/output contract, environment variable
configuration (which HF repos to pull from), and how model resolution
(local vs. Hugging Face Hub) works.
"""
from g3e_inference.schemas import (
    G3E_CLASSES,
    G3E_CLASS_NAME_TO_ID,
    SEMANTIC_STATES,
    SEMANTIC_SEVERITIES,
    DetectedObject,
    G3E1Output,
    G3E2Output,
    EventImage,
    G3EEvent,
)
from g3e_inference.detector import G3E1Detector
from g3e_inference.reasoner import G3E2Reasoner
from g3e_inference.pipeline import G3EPipeline
from g3e_inference.event_builder import build_event, save_event, to_json, generate_event_id

__all__ = [
    "G3E_CLASSES",
    "G3E_CLASS_NAME_TO_ID",
    "SEMANTIC_STATES",
    "SEMANTIC_SEVERITIES",
    "DetectedObject",
    "G3E1Output",
    "G3E2Output",
    "EventImage",
    "G3EEvent",
    "G3E1Detector",
    "G3E2Reasoner",
    "G3EPipeline",
    "build_event",
    "save_event",
    "to_json",
    "generate_event_id",
]

__version__ = "0.1.0"
