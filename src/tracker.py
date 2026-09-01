"""Lightweight detection-based tracker for Phase 3 and Phase 4.

This implements a simple greedy IoU-based tracker that assigns persistent IDs
to detections across frames, and maintains centroid history for drawing
trail lines. It also overlays abnormal-event detections on the video output.
"""
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2


LOGGER = logging.getLogger(__name__)


def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(bb_test[0], bb_gt[0])
    y1 = max(bb_test[1], bb_gt[1])
    x2 = min(bb_test[2], bb_gt[2])
    y2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area1 = max(0.0, bb_test[2] - bb_test[0]) * max(0.0, bb_test[3] - bb_test[1])
    area2 = max(0.0, bb_gt[2] - bb_gt[0]) * max(0.0, bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    bbox: np.ndarray  # xyxy
    class_id: int
    conf: float
    last_seen: int
    hits: int = 1
    history: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class TrackingSnapshot:
    """Simple container for the tracking state at a single frame."""

    frame_idx: int
    tracks: List[Track]


def build_tracking_snapshot(frame_idx: int, tracks: List[Track]) -> TrackingSnapshot:
    """Create a serializable snapshot for downstream analysis modules."""
    return TrackingSnapshot(frame_idx=frame_idx, tracks=list(tracks))


def active_tracks(tracks: List[Track], tracker_frame_idx: int) -> List[Track]:
    """Return only the tracks matched to a detection in the current frame.

    ``SimpleTracker.update`` keeps a track for ``max_age`` frames after its
    last detection, and a retained track keeps its final bounding box. That
    frozen box has zero centroid motion, so ``BehaviorAnalyzer`` scores it
    stationary and the rule engine reports an object that is no longer there.
    Anything reasoning about what is present *now* -- rule evaluation, and
    drawing -- must filter on ``last_seen`` first.

    ``tracker_frame_idx`` is ``SimpleTracker.frame_idx`` read after ``update``.
    """
    return [track for track in tracks if track.last_seen == tracker_frame_idx]


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 1, trail_length: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.frame_idx = 0
        self.trail_length = trail_length

    def update(self, boxes: np.ndarray, class_ids: np.ndarray, confidences: np.ndarray) -> List[Track]:
        """Update tracks with current frame detections and return active tracks."""
        self.frame_idx += 1

        if boxes is None or len(boxes) == 0:
            # mark all tracks as unseen
            to_del = []
            for tid, tr in list(self.tracks.items()):
                if self.frame_idx - tr.last_seen > self.max_age:
                    to_del.append(tid)
            for tid in to_del:
                del self.tracks[tid]
            return list(self.tracks.values())

        detections = [np.array(b) for b in boxes]
        matched_tracks = set()
        matched_dets = set()

        # Compute IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=float)
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = iou(self.tracks[tid].bbox, det)

        # Greedy matching: find best IoU pairs above threshold
        if iou_matrix.size > 0:
            for _ in range(min(iou_matrix.shape[0], iou_matrix.shape[1])):
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                best = iou_matrix[i, j]
                if best < self.iou_threshold:
                    break
                tid = track_ids[i]
                # assign
                det = detections[j]
                self.tracks[tid].bbox = det
                self.tracks[tid].class_id = int(class_ids[j])
                self.tracks[tid].conf = float(confidences[j])
                self.tracks[tid].last_seen = self.frame_idx
                self.tracks[tid].hits += 1
                cx = int((det[0] + det[2]) / 2)
                cy = int((det[1] + det[3]) / 2)
                self.tracks[tid].history.append((cx, cy))
                if len(self.tracks[tid].history) > self.trail_length:
                    self.tracks[tid].history.pop(0)
                matched_tracks.add(tid)
                matched_dets.add(j)
                # invalidate row and col
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            cx = int((det[0] + det[2]) / 2)
            cy = int((det[1] + det[3]) / 2)
            tr = Track(id=tid, bbox=det, class_id=int(class_ids[j]), conf=float(confidences[j]), last_seen=self.frame_idx, hits=1, history=[(cx, cy)])
            self.tracks[tid] = tr

        # Remove stale tracks
        to_del = []
        for tid, tr in list(self.tracks.items()):
            if self.frame_idx - tr.last_seen > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        return list(self.tracks.values())


def draw_tracks(frame: np.ndarray, tracks: List[Track], names: Dict[int, str], abnormal_ids: set[int] | None = None) -> np.ndarray:
    out = frame.copy()
    abnormal_ids = abnormal_ids or set()

    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.bbox]
        color = (0, 255, 0)
        if tr.id in abnormal_ids:
            color = (0, 0, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(int(tr.class_id), str(tr.class_id))} {tr.conf:0.2f} ID:{tr.id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - h - 6), (x1 + w + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # draw trail
        if len(tr.history) >= 2:
            for i in range(1, len(tr.history)):
                cv2.line(out, tr.history[i - 1], tr.history[i], color, 2)

    return out


def track_video(model, video_path, output_path=None, conf: float = 0.25, iou_threshold: float = 0.3, max_age: int = 30):
    from pathlib import Path

    try:
        from config import ensure_directories, OUTPUTS_DIR, DEFAULT_VIDEO_PATH
    except Exception:  # pragma: no cover - supports package or script execution
        from src.config import ensure_directories, OUTPUTS_DIR, DEFAULT_VIDEO_PATH

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

    output_path = Path(output_path) if output_path else (OUTPUTS_DIR / f"tracked_{video_path.name}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video writer: {output_path}")

    tracker = SimpleTracker(iou_threshold=iou_threshold, max_age=max_age)
    from abnormal_event_detector import AbnormalEventDetector

    detector = AbnormalEventDetector()
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

        tracks = tracker.update(boxes, class_ids, confidences)
        snapshot = build_tracking_snapshot(frame_idx, tracks)
        events = detector.process_snapshot(snapshot)
        abnormal_ids = {event.object_id for event in events if event.object_id is not None}
        annotated = draw_tracks(frame, tracks, model.names, abnormal_ids)

        if events:
            cv2.putText(annotated, "ABNORMAL EVENT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            for idx, event in enumerate(events):
                y_pos = 60 + idx * 24
                cv2.putText(annotated, event.event_type, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        writer.write(annotated)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Saved tracked video to {output_path}")
    return output_path


# The functions below are the current tracking API. The classes above remain
# only for backwards compatibility with the project's earlier Phase 3/4 code.
PathLike = Union[str, Path]
BoundingBox = Tuple[int, int, int, int]
BYTE_TRACKER = "bytetrack.yaml"
BOT_SORT_TRACKER = "botsort.yaml"


class TrackingError(RuntimeError):
    """Raised when model tracking or tracking-output generation fails."""


@dataclass(frozen=True)
class TrackingRecord:
    """One tracked object associated with a single one-based frame number."""

    frame_number: int
    track_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation with the required fields."""
        return {
            "frame_number": self.frame_number,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "bounding_box": list(self.bounding_box),
        }


@dataclass(frozen=True)
class TrackingRunResult:
    """Artifacts and structured records generated by one tracking run."""

    records: List[TrackingRecord]
    tracked_video_path: Path
    results_json_path: Path
    tracker_name: str


def _get_class_name(model_names: Any, class_id: int) -> str:
    """Resolve a class ID from either Ultralytics label-map representation."""
    if isinstance(model_names, dict):
        return str(model_names.get(class_id, class_id))
    if isinstance(model_names, list) and 0 <= class_id < len(model_names):
        return str(model_names[class_id])
    return str(class_id)


def _draw_tracking_record(frame: np.ndarray, record: TrackingRecord) -> None:
    """Draw one track annotation in-place on a BGR OpenCV frame."""
    x1, y1, x2, y2 = record.bounding_box
    color = (0, 200, 255)
    label = f"ID {record.track_id} | {record.class_name} {record.confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    text_top = max(0, y1 - text_height - baseline - 6)
    cv2.rectangle(
        frame,
        (x1, text_top),
        (x1 + text_width + 6, y1),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, max(text_height, y1 - baseline - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


class UltralyticsObjectTracker:
    """Stateful ByteTrack runner with a BoT-SORT runtime fallback.

    Passing ``persist=True`` to ``model.track`` preserves tracker state between
    calls, which is essential for one object to keep the same ID in subsequent
    frames. ByteTrack is selected first; BoT-SORT is only selected when the
    installed Ultralytics environment cannot initialize ByteTrack.
    """

    def __init__(self, model: Any, confidence_threshold: float = 0.25) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")

        self.model = model
        self.confidence_threshold = confidence_threshold
        self.tracker_name = BYTE_TRACKER

    def track_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> Tuple[np.ndarray, List[TrackingRecord]]:
        """Track objects in one frame and return annotations and records."""
        if frame is None or frame.size == 0:
            raise TrackingError("Cannot track objects in an empty frame.")

        try:
            result = self._run_tracker(frame)
        except Exception as error:
            if self.tracker_name != BYTE_TRACKER:
                message = (
                    f"{self.tracker_name} tracking failed on frame "
                    f"{frame_number}: {error}"
                )
                raise TrackingError(
                    message
                ) from error

            # The fallback is intentionally activated once. It then persists
            # across all remaining frames, preserving BoT-SORT track IDs.
            LOGGER.warning(
                "ByteTrack could not initialize (%s). Falling back to BoT-SORT.",
                error,
            )
            self.tracker_name = BOT_SORT_TRACKER
            try:
                result = self._run_tracker(frame)
            except Exception as fallback_error:
                raise TrackingError(
                    "Neither ByteTrack nor BoT-SORT could track frame "
                    f"{frame_number}: {fallback_error}"
                ) from fallback_error

        annotated_frame = frame.copy()
        records = self._extract_records(result, frame_number, frame.shape)
        for record in records:
            _draw_tracking_record(annotated_frame, record)
        return annotated_frame, records

    def _run_tracker(self, frame: np.ndarray) -> Any:
        """Run the active Ultralytics tracker while retaining tracker state."""
        return self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            persist=True,
            tracker=self.tracker_name,
            verbose=False,
        )[0]

    def _extract_records(
        self,
        result: Any,
        frame_number: int,
        frame_shape: Tuple[int, ...],
    ) -> List[TrackingRecord]:
        """Convert Ultralytics boxes and IDs into stable, serializable records."""
        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.boxes.id is None:
            # Ultralytics can report detections before the tracker confirms an
            # association. Excluding these prevents fabricated track IDs.
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.int().cpu().tolist()
        height, width = frame_shape[:2]
        records: List[TrackingRecord] = []

        for box, confidence, class_id, track_id in zip(
            boxes,
            confidences,
            class_ids,
            track_ids,
        ):
            x1, y1, x2, y2 = box.astype(int)
            bounding_box = (
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)),
                max(0, min(y2, height - 1)),
            )
            records.append(
                TrackingRecord(
                    frame_number=frame_number,
                    track_id=int(track_id),
                    class_name=_get_class_name(self.model.names, int(class_id)),
                    confidence=float(confidence),
                    bounding_box=bounding_box,
                )
            )

        return records


def _create_tracking_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create the MP4 writer used for tracked-video visualization."""
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise TrackingError(f"Could not create tracked video: {output_path}")
    return writer


def _write_tracking_results(records: List[TrackingRecord], output_path: Path) -> None:
    """Write only JSON-native data, keeping results ready for later analysis."""
    try:
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump([record.as_dict() for record in records], file_handle, indent=2)
    except OSError as error:
        message = f"Could not save tracking results: {output_path}"
        raise TrackingError(message) from error


def run_object_tracking(
    video_path: PathLike,
    model_path: PathLike,
    output_video_path: Optional[PathLike] = None,
    results_json_path: Optional[PathLike] = None,
    confidence_threshold: float = 0.25,
) -> TrackingRunResult:
    """Track objects sequentially and write the video and JSON deliverables.

    The model is loaded once and ``UltralyticsObjectTracker`` is retained for
    the full stream. This avoids resetting the tracker's association state.
    """
    try:
        from config import OUTPUTS_DIR, ensure_directories
        from detection import DetectionError, load_yolo_model
        from video_loader import VideoLoadError, open_video
    except ImportError:  # pragma: no cover - supports package execution
        from src.config import OUTPUTS_DIR, ensure_directories
        from src.detection import DetectionError, load_yolo_model
        from src.video_loader import VideoLoadError, open_video

    ensure_directories()
    video_output = Path(output_video_path or OUTPUTS_DIR / "tracked_video.mp4")
    json_output = Path(results_json_path or OUTPUTS_DIR / "tracking_results.json")
    video_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        model = load_yolo_model(model_path)
    except DetectionError as error:
        raise TrackingError(f"Unable to load tracking model: {error}") from error
    tracker = UltralyticsObjectTracker(model, confidence_threshold)
    capture: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None
    records: List[TrackingRecord] = []

    try:
        capture, metadata = open_video(video_path)
        fps = metadata.fps if metadata.fps > 0 else 20.0
        writer = _create_tracking_writer(
            video_output,
            fps,
            metadata.width,
            metadata.height,
        )

        frame_number = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            annotated_frame, frame_records = tracker.track_frame(frame, frame_number)
            records.extend(frame_records)
            writer.write(annotated_frame)

        if frame_number == 0:
            raise TrackingError(f"Video contains no readable frames: {video_path}")
    except VideoLoadError as error:
        raise TrackingError(f"Unable to process input video: {error}") from error
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    _write_tracking_results(records, json_output)
    LOGGER.info(
        "Tracking complete with %s records using %s.",
        len(records),
        tracker.tracker_name,
    )
    return TrackingRunResult(
        records=records,
        tracked_video_path=video_output,
        results_json_path=json_output,
        tracker_name=tracker.tracker_name,
    )
