"""Lightweight detection-based tracker for Phase 3.

This implements a simple greedy IoU-based tracker that assigns persistent IDs
to detections across frames, and maintains centroid history for drawing
trail lines.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import cv2


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


def draw_tracks(frame: np.ndarray, tracks: List[Track], names: Dict[int, str]) -> np.ndarray:
    out = frame.copy()
    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.bbox]
        color = (0, 200, 255)
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
        # TODO: feed snapshot into the future abnormal event detector pipeline.
        annotated = draw_tracks(frame, tracks, model.names)
        writer.write(annotated)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Saved tracked video to {output_path}")
    return output_path
