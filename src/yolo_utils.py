"""YOLO model loading and frame annotation utilities."""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - runtime import
    YOLO = None  # type: ignore


def load_yolo_model(weights: str = "yolov8s.pt"):
    """Load and return an Ultralytics YOLO model, downloading weights if needed."""
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not available in the environment.")

    model = YOLO(weights)
    return model


def annotate_frame(frame: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray, confidences: np.ndarray, names: Dict[int, str]) -> np.ndarray:
    """Draw bounding boxes, labels and confidence scores on a frame.

    Args:
        frame: BGR image as numpy array.
        boxes: array of shape (N,4) with xyxy coordinates.
        class_ids: array of integer class ids.
        confidences: array of floats [0..1].
        names: mapping from class id to class name.

    Returns:
        Annotated BGR image.
    """
    annotated = frame.copy()
    for (x1, y1, x2, y2), cls, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(int(cls), str(int(cls)))} {conf:0.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 6), (x1 + w + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated
