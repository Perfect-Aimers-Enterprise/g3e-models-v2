import os

from g3e_inference.schemas import DetectedObject, G3E1Output, G3E2Output, G3EEvent, EventImage, G3E_CLASSES


def test_g3e_classes_are_frozen_v1_ontology():
    assert G3E_CLASSES == {
        0: "person", 1: "fire", 2: "smoke", 3: "gun", 4: "knife", 5: "car", 6: "dog", 7: "cat",
    }


def test_detected_object_class_alias_round_trips():
    obj = DetectedObject(**{"class": "gun", "class_id": 3, "confidence": 0.95, "bbox": [0, 0, 1, 1]})
    dumped = obj.model_dump(by_alias=True)
    assert dumped["class"] == "gun"
    assert "class_name" not in dumped


def test_g3e2_output_optional_confidence_defaults_to_none():
    out = G3E2Output(state="normal", severity="none", description="d", reason="")
    assert out.confidence is None


def test_defaults_read_from_environment(monkeypatch):
    monkeypatch.setenv("G3E1_HF_REPO", "TestOrg/test-repo")
    # reimport to pick up the env var at module load time
    import importlib
    import g3e_inference.defaults as defaults_module
    importlib.reload(defaults_module)
    assert defaults_module.DEFAULT_G3E1_HF_REPO == "TestOrg/test-repo"
    importlib.reload(defaults_module)  # reset for other tests in the same process
