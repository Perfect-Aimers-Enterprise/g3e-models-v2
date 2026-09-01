"""
Builds the final `G3EEvent` (spec section 22) from a G3E-1 output, a G3E-2
output, image paths, and a timestamp. This is the one place that assembly
happens — `g3e2/predict.py`, `inference/pipeline.py`, and any future
consumer should all go through `build_event()` rather than constructing
`G3EEvent(...)` by hand, so the `by_alias=True` serialization requirement
(see `save_event`/`to_json`) is never forgotten at a second call site.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from g3e_inference.schemas import G3EEvent, EventImage, G3E1Output, G3E2Output, DetectedObject


def generate_event_id() -> str:
    """`evt_<12 hex chars>` — short enough to be readable in logs, unique
    enough that collisions across a real deployment are not a concern."""
    return f"evt_{uuid.uuid4().hex[:12]}"


def current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_detections(objects: list[dict] | list[DetectedObject]) -> list[DetectedObject]:
    return [o if isinstance(o, DetectedObject) else DetectedObject(**o) for o in objects]


def build_event(
    image_original: str,
    g3e1_detections: list[dict] | list[DetectedObject],
    g3e2_output: dict | G3E2Output,
    image_annotated: str | None = None,
    event_id: str | None = None,
    timestamp: str | None = None,
) -> G3EEvent:
    """
    `g3e1_detections` accepts either `DetectedObject` instances or plain
    dicts with a `"class"` key (matching G3E-1's raw output contract, spec
    section 7) — callers coming straight from a YOLO wrapper or a JSON file
    on disk don't need to construct `DetectedObject` themselves first.
    Same for `g3e2_output` — a plain dict (e.g. straight from
    `g3e2/predict.py`'s `parse_model_output`) or a `G3E2Output` both work.
    """
    g3e2 = g3e2_output if isinstance(g3e2_output, G3E2Output) else G3E2Output(**g3e2_output)

    return G3EEvent(
        event_id=event_id or generate_event_id(),
        timestamp=timestamp or current_timestamp(),
        image=EventImage(original=image_original, annotated=image_annotated),
        g3e1=G3E1Output(objects=_coerce_detections(g3e1_detections)),
        g3e2=g3e2,
    )


def to_json(event: G3EEvent, indent: int | None = 2) -> str:
    """
    ALWAYS use this (or save_event) to serialize an event, never
    `event.model_dump_json()` directly — `DetectedObject.class_name`
    aliases to `"class"` on the wire (see shared/schemas.py), and only
    `by_alias=True` produces that key instead of `"class_name"`, which
    would silently break every downstream consumer expecting the spec's
    documented shape.
    """
    return event.model_dump_json(by_alias=True, indent=indent)


def save_event(event: G3EEvent, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(event))
    return path
