"""Frame-by-frame YOLOv8 object detection for surveillance videos.

The module is intentionally limited to detection and annotation. Tracking,
event reasoning, and risk assessment can consume ``DetectionRecord`` objects
later without changing this video-processing interface.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depends on deployment environment
    YOLO = None  # type: ignore[misc, assignment]

try:
    from config import DEFAULT_VIDEO_PATH, MODELS_DIR, OUTPUTS_DIR, ensure_directories
    from video_loader import VideoLoadError, open_video
except ImportError:  # pragma: no cover - supports package execution
    from src.config import (
        DEFAULT_VIDEO_PATH,
        MODELS_DIR,
        OUTPUTS_DIR,
        ensure_directories,
    )
    from src.video_loader import VideoLoadError, open_video


PathLike = Union[str, Path]
BoundingBox = Tuple[int, int, int, int]
DEFAULT_MODEL_PATH = MODELS_DIR / "yolov8s.pt"


class DetectionError(RuntimeError):
    """Raised when a detection model or output artifact cannot be used."""


@dataclass(frozen=True)
class DetectionRecord:
    """One model prediction in a video frame.

    ``frame_number`` is one-based, matching the numbering convention used by
    video tools and making exported records easier to inspect manually.
    """

    frame_number: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class DetectionRunResult:
    """Detection records and files produced during one video-processing run."""

    detections: List[DetectionRecord]
    annotated_video_path: Path
    sample_frame_path: Path


def load_yolo_model(model_path: PathLike = DEFAULT_MODEL_PATH) -> "YOLO":
    """Load local YOLOv8 weights from the project's ``models`` directory."""
    if YOLO is None:
        raise DetectionError(
            "Ultralytics is not installed. Run 'pip install -r requirements.txt'."
        )

    weights_path = Path(model_path).expanduser().resolve()
    if not weights_path.is_file():
        raise DetectionError(
            f"YOLOv8 weights were not found at: {weights_path}. "
            "Place the pretrained .pt file in the models directory."
        )

    try:
        return YOLO(str(weights_path))
    except Exception as error:
        raise DetectionError(
            f"Unable to load YOLOv8 weights from {weights_path}: {error}"
        ) from error


def draw_detection(
    frame: np.ndarray,
    bounding_box: BoundingBox,
    class_name: str,
    confidence: float,
) -> None:
    """Draw one labelled bounding box in-place on an OpenCV BGR frame."""
    x1, y1, x2, y2 = bounding_box
    label = f"{class_name} {confidence:.2f}"
    color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
    (label_width, label_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    label_top = max(y1 - label_height - baseline - 6, 0)
    cv2.rectangle(
        frame,
        (x1, label_top),
        (x1 + label_width + 6, y1),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, max(y1 - baseline - 3, label_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _class_name(model_names: Union[Dict[int, str], List[str]], class_id: int) -> str:
    """Return a class label for either Ultralytics name-map representation."""
    if isinstance(model_names, dict):
        return str(model_names.get(class_id, class_id))
    if 0 <= class_id < len(model_names):
        return str(model_names[class_id])
    return str(class_id)


def detect_frame(
    model: "YOLO",
    frame: np.ndarray,
    frame_number: int,
    confidence_threshold: float = 0.25,
) -> Tuple[np.ndarray, List[DetectionRecord]]:
    """Detect objects in one frame and return its annotation and records."""
    if frame is None or frame.size == 0:
        raise DetectionError("Cannot run detection on an empty frame.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0.")

    try:
        result = model.predict(
            source=frame,
            conf=confidence_threshold,
            verbose=False,
        )[0]
    except Exception as error:
        message = f"YOLOv8 inference failed on frame {frame_number}: {error}"
        raise DetectionError(message) from error

    annotated_frame = frame.copy()
    records: List[DetectionRecord] = []
    if result.boxes is None or len(result.boxes) == 0:
        return annotated_frame, records

    # Convert tensors once per frame to keep the record-generation loop simple.
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    frame_height, frame_width = frame.shape[:2]

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        # Clamping protects drawing and downstream consumers from edge boxes.
        bounding_box = (
            max(0, min(x1, frame_width - 1)),
            max(0, min(y1, frame_height - 1)),
            max(0, min(x2, frame_width - 1)),
            max(0, min(y2, frame_height - 1)),
        )
        class_name = _class_name(model.names, int(class_id))
        record = DetectionRecord(
            frame_number=frame_number,
            class_name=class_name,
            confidence=float(confidence),
            bounding_box=bounding_box,
        )
        records.append(record)
        draw_detection(
            annotated_frame,
            bounding_box,
            class_name,
            record.confidence,
        )

    return annotated_frame, records


def _create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create an MP4 writer and fail early if the codec is unavailable."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise DetectionError(f"Could not create annotated video: {output_path}")
    return writer


def detect_video(
    video_path: PathLike = DEFAULT_VIDEO_PATH,
    model_path: PathLike = DEFAULT_MODEL_PATH,
    output_video_path: Optional[PathLike] = None,
    sample_frame_path: Optional[PathLike] = None,
    confidence_threshold: float = 0.25,
) -> DetectionRunResult:
    """Run YOLOv8 detection frame-by-frame and save annotated artifacts.

    The input resolution is retained for output so annotation coordinates and
    video dimensions remain aligned for later tracking or explanation modules.
    """
    ensure_directories()
    input_path = Path(video_path).expanduser()
    output_path = Path(output_video_path or OUTPUTS_DIR / "annotated_video.mp4")
    default_frame_path = OUTPUTS_DIR / "annotated_sample_frame.jpg"
    frame_path = Path(sample_frame_path or default_frame_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_yolo_model(model_path)
    capture: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None
    detections: List[DetectionRecord] = []

    try:
        capture, metadata = open_video(input_path)
        fps = metadata.fps if metadata.fps > 0 else 20.0
        writer = _create_video_writer(output_path, fps, metadata.width, metadata.height)

        frame_number = 0
        sample_saved = False
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            annotated_frame, frame_detections = detect_frame(
                model,
                frame,
                frame_number,
                confidence_threshold,
            )
            detections.extend(frame_detections)
            writer.write(annotated_frame)

            if not sample_saved:
                if not cv2.imwrite(str(frame_path), annotated_frame):
                    message = f"Could not save annotated frame: {frame_path}"
                    raise DetectionError(message)
                sample_saved = True

        if frame_number == 0:
            raise DetectionError(f"Video contains no readable frames: {input_path}")
        if not sample_saved:
            raise DetectionError("No sample frame could be written.")
    except VideoLoadError as error:
        raise DetectionError(f"Unable to process input video: {error}") from error
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    return DetectionRunResult(detections, output_path, frame_path)
