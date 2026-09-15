"""Behavior analysis for Phase 4.

This module maintains lightweight history for each tracked object so the
abnormal event detector can reason about motion persistence and stationarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracker import TrackingSnapshot


@dataclass
class BehaviorSummary:
    """High-level summary of a tracked object's behavior over time."""

    object_id: int
    frame_count: int = 0
    total_distance: float = 0.0
    stationary_frames: int = 0
    last_movement: float = 0.0
    is_stationary: bool = False
    notes: List[str] = field(default_factory=list)


class BehaviorAnalyzer:
    """Track motion-related history for each object across frames."""

    def __init__(self) -> None:
        self.summaries: Dict[int, BehaviorSummary] = {}
        self._bbox_history: Dict[int, List[np.ndarray]] = {}

    def _centroid(self, bbox: Optional[np.ndarray]) -> Tuple[float, float]:
        if bbox is None:
            return (0.0, 0.0)
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _movement_pixels(self, prev_bbox: Optional[np.ndarray], curr_bbox: Optional[np.ndarray]) -> float:
        if prev_bbox is None or curr_bbox is None:
            return 0.0
        prev_center = self._centroid(prev_bbox)
        curr_center = self._centroid(curr_bbox)
        return float(np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]))

    def update_history(
        self,
        snapshot: TrackingSnapshot,
        movement_threshold_pixels: float = 3.0,
        stationary_frame_threshold: int = 5,
    ) -> List[BehaviorSummary]:
        """Update object-history summaries for one tracking snapshot."""
        summaries: List[BehaviorSummary] = []

        for track in snapshot.tracks:
            previous_bbox = None
            if track.id in self._bbox_history and self._bbox_history[track.id]:
                previous_bbox = self._bbox_history[track.id][-1]

            movement = self._movement_pixels(previous_bbox, track.bbox)
            summary = self.summaries.get(track.id)
            if summary is None:
                summary = BehaviorSummary(object_id=track.id)
                self.summaries[track.id] = summary

            summary.frame_count += 1
            summary.total_distance += movement
            summary.last_movement = movement

            if previous_bbox is None:
                summary.stationary_frames = 1 if movement <= movement_threshold_pixels else 0
            elif movement <= movement_threshold_pixels:
                summary.stationary_frames += 1
            else:
                summary.stationary_frames = 0

            summary.is_stationary = summary.stationary_frames >= stationary_frame_threshold
            if summary.is_stationary:
                summary.notes = ["stationary for too long"]
            else:
                summary.notes = []

            self._bbox_history.setdefault(track.id, []).append(np.asarray(track.bbox, dtype=float))
            if len(self._bbox_history[track.id]) > 50:
                self._bbox_history[track.id].pop(0)

            summaries.append(summary)

        return summaries

    def get_previous_bbox(self, object_id: int) -> Optional[np.ndarray]:
        """Return the second-most-recent bounding box for an object.

        NOTE ON NAMING -- read before using this for a speed.

        This returns ``history[-2]``, not ``history[-1]``. Callers that invoke
        it BEFORE ``update_history`` has recorded the current frame therefore
        get the track's bbox from TWO appearances ago, and a displacement
        measured against ``track.bbox`` spans two appearances rather than one
        transition. The frozen spatial feature is computed exactly that way; it
        is a deliberate, locked behaviour, documented in
        docs/SPATIAL_FEATURE_SEMANTICS.md.

        History records only frames in which the track was matched to a
        detection, so across a tracker dropout that two-appearance gap can span
        many more wall-clock frames.

        ``update_history`` itself does not use this accessor -- it reads
        ``history[-1]`` directly, so the per-frame ``movement`` it records for
        the stationary rule IS a one-transition displacement. The two are
        different measurements and are not interchangeable.

        Behaviour here is unchanged and must stay unchanged: it is what
        produced the frozen thresholds and the locked primary results.
        """
        history = self._bbox_history.get(object_id, [])
        if len(history) < 2:
            return None
        return history[-2]
