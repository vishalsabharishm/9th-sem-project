"""Stateful generic crowding and proximity signals for tracked persons.

These signals describe geometric conditions only. They are not classifiers for
violence, stabbing, or any other specific behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from tracker import TrackingSnapshot


@dataclass(frozen=True)
class InteractionSignal:
    """A generic crowding or sustained-proximity condition from one frame."""

    event_type: str
    description: str
    object_ids: List[int]
    evidence: List[str]


class CrowdInteractionAnalyzer:
    """Evaluate configurable persistence rules without assigning confidence."""

    def __init__(self) -> None:
        self._crowding_frames = 0
        self._crowding_active = False
        self._pair_frames: Dict[Tuple[int, int], int] = {}
        self._pair_active: set[Tuple[int, int]] = set()

    @staticmethod
    def _persons(snapshot: TrackingSnapshot):
        return [track for track in snapshot.tracks if int(track.class_id) == 0]

    @staticmethod
    def _centroid(track) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in track.bbox]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def evaluate_crowding(
        self,
        snapshot: TrackingSnapshot,
        minimum_person_count: int,
        persistence_frames: int,
    ) -> List[InteractionSignal]:
        persons = self._persons(snapshot)
        if len(persons) >= minimum_person_count:
            self._crowding_frames += 1
        else:
            self._crowding_frames = 0
            self._crowding_active = False

        if self._crowding_active or self._crowding_frames < persistence_frames:
            return []
        self._crowding_active = True
        person_ids = sorted(track.id for track in persons)
        return [
            InteractionSignal(
                event_type="Crowding",
                description="Configured simultaneous-person threshold persisted.",
                object_ids=person_ids,
                evidence=[
                    f"person_count={len(persons)}",
                    f"minimum_person_count={minimum_person_count}",
                    f"persistence_frames={self._crowding_frames}",
                ],
            )
        ]

    def evaluate_proximity(
        self,
        snapshot: TrackingSnapshot,
        normalized_distance_threshold: float,
        persistence_frames: int,
    ) -> List[InteractionSignal]:
        persons = self._persons(snapshot)
        active_pairs: set[Tuple[int, int]] = set()
        signals: List[InteractionSignal] = []
        x_values = [float(track.bbox[0]) for track in persons] + [float(track.bbox[2]) for track in persons]
        y_values = [float(track.bbox[1]) for track in persons] + [float(track.bbox[3]) for track in persons]
        # ``TrackingSnapshot`` does not carry image dimensions. Distance is
        # therefore normalized by the diagonal enclosing current person boxes.
        group_diagonal = float(np.hypot(max(x_values) - min(x_values), max(y_values) - min(y_values))) if persons else 0.0

        for first, second in combinations(persons, 2):
            pair = tuple(sorted((first.id, second.id)))
            first_center = self._centroid(first)
            second_center = self._centroid(second)
            distance = float(np.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))
            normalized_distance = distance / group_diagonal if group_diagonal > 0 else float("inf")

            if normalized_distance <= normalized_distance_threshold:
                active_pairs.add(pair)
                self._pair_frames[pair] = self._pair_frames.get(pair, 0) + 1
                if self._pair_frames[pair] >= persistence_frames and pair not in self._pair_active:
                    self._pair_active.add(pair)
                    signals.append(
                        InteractionSignal(
                            event_type="Proximity/Interaction",
                            description="Tracked persons remained within the configured normalized distance.",
                            object_ids=list(pair),
                            evidence=[
                                f"normalized_centroid_distance={normalized_distance:.4f}",
                                f"person_group_diagonal={group_diagonal:.4f}",
                                f"distance_threshold={normalized_distance_threshold}",
                                f"persistence_frames={self._pair_frames[pair]}",
                            ],
                        )
                    )

        for pair in list(self._pair_frames):
            if pair not in active_pairs:
                self._pair_frames.pop(pair, None)
                self._pair_active.discard(pair)
        return signals
