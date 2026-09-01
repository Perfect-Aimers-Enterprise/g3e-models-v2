import json

from g3e_inference.event_builder import build_event, to_json, save_event, generate_event_id


def test_generate_event_id_format():
    eid = generate_event_id()
    assert eid.startswith("evt_")
    assert len(eid) == len("evt_") + 12


def test_build_event_accepts_plain_dicts_for_detections():
    detections = [{"class": "person", "class_id": 0, "confidence": 0.98, "bbox": [120, 80, 420, 620]}]
    g3e2_output = {"state": "normal", "severity": "none", "description": "x", "reason": ""}

    event = build_event("frame.jpg", detections, g3e2_output)
    assert event.g3e1.objects[0].class_name == "person"
    assert event.g3e2.state == "normal"


def test_to_json_uses_class_alias_not_class_name():
    detections = [{"class": "knife", "class_id": 4, "confidence": 0.91, "bbox": [1, 2, 3, 4]}]
    g3e2_output = {"state": "potential_threat", "severity": "high", "description": "d", "reason": "r"}
    event = build_event("frame.jpg", detections, g3e2_output)

    parsed = json.loads(to_json(event))
    assert parsed["g3e1"]["objects"][0]["class"] == "knife"
    assert "class_name" not in parsed["g3e1"]["objects"][0]


def test_build_event_uses_provided_event_id_and_timestamp_when_given():
    event = build_event(
        "frame.jpg", [], {"state": "normal", "severity": "none", "description": "x", "reason": ""},
        event_id="evt_custom", timestamp="2020-01-01T00:00:00Z",
    )
    assert event.event_id == "evt_custom"
    assert event.timestamp == "2020-01-01T00:00:00Z"


def test_build_event_generates_id_and_timestamp_when_omitted():
    event = build_event("frame.jpg", [], {"state": "normal", "severity": "none", "description": "x", "reason": ""})
    assert event.event_id.startswith("evt_")
    assert event.timestamp.endswith("Z")


def test_save_event_writes_valid_json_file(tmp_path):
    event = build_event("frame.jpg", [], {"state": "normal", "severity": "none", "description": "x", "reason": ""})
    path = save_event(event, tmp_path / "sub" / "event.json")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["event_id"] == event.event_id
