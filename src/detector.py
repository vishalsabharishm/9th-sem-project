"""Object detection routines for images and videos using YOLOv8."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from config import ensure_directories, OUTPUTS_DIR, DEFAULT_VIDEO_PATH
except Exception:  # pragma: no cover - supports package or script execution
    from src.config import ensure_directories, OUTPUTS_DIR, DEFAULT_VIDEO_PATH

try:
    from yolo_utils import annotate_frame, load_yolo_model
except Exception:  # pragma: no cover - supports package or script execution
    from src.yolo_utils import annotate_frame, load_yolo_model


def detect_image(model, image_path: Path, output_path: Optional[Path] = None, conf: float = 0.25) -> Path:
    ensure_directories()

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    results = model(img, conf=conf)
    r = results[0]

    if hasattr(r, "boxes") and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
    else:
        boxes = np.empty((0, 4))
        confidences = np.array([])
        class_ids = np.array([])

    annotated = annotate_frame(img, boxes, class_ids, confidences, model.names)

    output_path = output_path or (OUTPUTS_DIR / f"annotated_{image_path.name}")
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved annotated image to {output_path}")
    return output_path


def detect_video(model, video_path: Path, output_path: Optional[Path] = None, conf: float = 0.25) -> Path:
    ensure_directories()

    video_path = Path(video_path) if video_path else DEFAULT_VIDEO_PATH
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    output_path = output_path or (OUTPUTS_DIR / f"annotated_{video_path.name}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        results = model(frame, conf=conf)
        r = results[0]

        if hasattr(r, "boxes") and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes = np.empty((0, 4))
            confidences = np.array([])
            class_ids = np.array([])

        annotated = annotate_frame(frame, boxes, class_ids, confidences, model.names)
        writer.write(annotated)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Saved annotated video to {output_path}")
    return output_path


if __name__ == "__main__":
    # Minimal demo when run directly
    model = load_yolo_model()
    detect_video(model, DEFAULT_VIDEO_PATH)
