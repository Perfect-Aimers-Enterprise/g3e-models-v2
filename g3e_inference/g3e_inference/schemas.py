"""
Schemas — the full input/output contract for the g3e-inference library.

This is the CANONICAL copy for anything published to PyPI as
`g3e-inference` — it is deliberately self-contained (pydantic only, no
import from a sibling `shared`/`g3e1`/`g3e2` package) because after
`pip install g3e-inference`, none of this repo's other training-side
folders exist on the installing machine. The g3e-models training repo's
own `shared/schemas.py` is a separate, matching copy — keep the two in
sync by hand if you change one; there are only ~60 lines here, so this is
a deliberate simplicity trade-off rather than adding a shared-package
dependency between "the training repo" and "the thing we publish."
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Frozen v1 ontology — MUST match g3e-data-engine's metadata/classes.json
# exactly (same names, same ids). Do not edit this list independently of
# that file; see DATASET_SPEC section 3 in the data engine repo. A future
# class change is a new dataset/model version, not an edit here.
G3E_CLASSES: dict[int, str] = {
    0: "person",
    1: "fire",
    2: "smoke",
    3: "gun",
    4: "knife",
    5: "car",
    6: "dog",
    7: "cat",
}
G3E_CLASS_NAME_TO_ID: dict[str, int] = {v: k for k, v in G3E_CLASSES.items()}

SEMANTIC_STATES = {"normal", "caution", "hazard", "potential_threat"}
SEMANTIC_SEVERITIES = {"none", "low", "medium", "high", "critical"}


# ---------------------------------------------------------------------------
# G3E-1 (detection) output contract — see spec section 7
# ---------------------------------------------------------------------------
class DetectedObject(BaseModel):
    class_name: str = Field(alias="class")
    class_id: int
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] — PIXEL coordinates, never [x,y,w,h]. See spec section 7.

    model_config = {"populate_by_name": True}


class G3E1Output(BaseModel):
    objects: list[DetectedObject] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# G3E-2 (semantic reasoning) output contract — see spec section 15
# ---------------------------------------------------------------------------
class G3E2Output(BaseModel):
    state: str
    severity: str
    description: str
    reason: str
    recommended_action: str = ""
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Final event schema — see spec section 22
# ---------------------------------------------------------------------------
class EventImage(BaseModel):
    original: str
    annotated: str | None = None


class G3EEvent(BaseModel):
    event_id: str
    timestamp: str
    image: EventImage
    g3e1: G3E1Output
    g3e2: G3E2Output
