"""
OpenCV-based detection rendering — spec section 9. OpenCV draws the
annotated frame; it is NOT the detector (that's G3E-1/YOLO). This module
takes an image + a list of already-computed detections and produces the
"PERSON 98% / KNIFE 91%"-style annotated output.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from g3e_inference.schemas import DetectedObject

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
TEXT_BG_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


def _to_detected_objects(detections: list[dict] | list[DetectedObject]) -> list[DetectedObject]:
    return [d if isinstance(d, DetectedObject) else DetectedObject(**d) for d in detections]


def draw_detections(
    image_path: str | Path,
    detections: list[dict] | list[DetectedObject],
    output_path: str | Path,
) -> Path:
    """
    Draws each detection's box + "CLASS NN%" label onto a copy of the
    source image and writes it to `output_path`. Boxes are expected in
    PIXEL [x1, y1, x2, y2] coordinates (G3E-1's output contract) — the same
    convention used everywhere else in this repo (see
    shared/schemas.py:DetectedObject).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cv2 could not read image: {image_path}")

    for det in _to_detected_objects(detections):
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        label = f"{det.class_name.upper()} {det.confidence * 100:.0f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        label_y1 = max(0, y1 - text_h - baseline - 4)
        label_y2 = label_y1 + text_h + baseline + 4
        cv2.rectangle(image, (x1, label_y1), (x1 + text_w + 4, label_y2), TEXT_BG_COLOR, -1)
        cv2.putText(image, label, (x1 + 2, label_y2 - baseline - 2), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path
