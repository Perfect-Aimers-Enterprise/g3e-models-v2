import numpy as np
import cv2
from PIL import Image

from g3e_inference.renderer import draw_detections


def test_draw_detections_writes_correct_output_size(tmp_path):
    src = tmp_path / "src.jpg"
    Image.fromarray(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)).save(src)

    detections = [{"class": "person", "class_id": 0, "confidence": 0.98, "bbox": [120, 80, 420, 460]}]
    out = draw_detections(src, detections, tmp_path / "out.jpg")

    result = cv2.imread(str(out))
    assert result.shape == (480, 640, 3)


def test_draw_detections_modifies_pixels(tmp_path):
    src = tmp_path / "src.jpg"
    arr = np.zeros((480, 640, 3), dtype=np.uint8)  # solid black
    Image.fromarray(arr).save(src)

    detections = [{"class": "fire", "class_id": 1, "confidence": 0.8, "bbox": [100, 100, 300, 300]}]
    out_path = tmp_path / "out.jpg"
    draw_detections(src, detections, out_path)

    result = cv2.imread(str(out_path))
    assert result.mean() > 0  # boxes/labels were actually drawn onto the black frame


def test_draw_detections_handles_empty_detection_list(tmp_path):
    src = tmp_path / "src.jpg"
    Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)).save(src)
    out = draw_detections(src, [], tmp_path / "out.jpg")
    assert out.exists()


def test_draw_detections_raises_on_unreadable_image(tmp_path):
    import pytest
    bad = tmp_path / "not_an_image.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        draw_detections(bad, [], tmp_path / "out.jpg")


def test_draw_detections_accepts_detected_object_instances(tmp_path):
    from g3e_inference.schemas import DetectedObject

    src = tmp_path / "src.jpg"
    Image.fromarray(np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)).save(src)
    obj = DetectedObject(**{"class": "car", "class_id": 5, "confidence": 0.7, "bbox": [10, 10, 50, 50]})
    out = draw_detections(src, [obj], tmp_path / "out.jpg")
    assert out.exists()
