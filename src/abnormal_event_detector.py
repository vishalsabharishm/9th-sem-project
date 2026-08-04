"""Rule-based abnormal event detection for Phase 4.

This module evaluates tracked objects against a small set of simple rules:
1. A person remains stationary for too many consecutive frames.
2. A person enters a predefined restricted region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from behavior_analyzer import BehaviorAnalyzer
from event_rules import EventRuleEngine, build_default_rules, evaluate_restricted_region_rule, evaluate_stationary_rule
from tracker import TrackingSnapshot


@dataclass
class EventDetection:
    """Represents a detected abnormal event candidate."""

    event_type: str
    description: str
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    object_id: Optional[int] = None


class AbnormalEventDetector:
    """Evaluate tracked persons against simple abnormal-event rules."""

    def __init__(self, rule_engine: Optional[EventRuleEngine] = None) -> None:
        self.events: List[EventDetection] = []
        self.behavior_analyzer = BehaviorAnalyzer()
        self.rule_engine = rule_engine or EventRuleEngine(build_default_rules())

    def _find_rule(self, *fragments: str):
        for rule in self.rule_engine.rules:
            name = rule.name.lower()
            if all(fragment.lower() in name for fragment in fragments):
                return rule
        return None

    def _is_person(self, track) -> bool:
        return int(track.class_id) == 0

    def process_snapshot(self, snapshot: TrackingSnapshot) -> List[EventDetection]:
        """Analyze one tracking snapshot and return event candidates."""
        self.events = []

        stationary_rule = self._find_rule("stationary")
        restricted_rule = self._find_rule("restricted")

        movement_threshold = 3.0
        stationary_threshold = 5
        if stationary_rule is not None:
            movement_threshold = stationary_rule.parameters.get("movement_threshold_pixels", movement_threshold)
            stationary_threshold = stationary_rule.parameters.get("stationary_frame_threshold", stationary_threshold)

        summaries = self.behavior_analyzer.update_history(
            snapshot,
            movement_threshold_pixels=movement_threshold,
            stationary_frame_threshold=stationary_threshold,
        )
        summary_lookup = {summary.object_id: summary for summary in summaries}

        for track in snapshot.tracks:
            if not self._is_person(track):
                continue

            summary = summary_lookup.get(track.id)
            previous_bbox = self.behavior_analyzer.get_previous_bbox(track.id)

            if stationary_rule is not None and stationary_rule.enabled and summary and evaluate_stationary_rule(summary, stationary_rule):
                self.events.append(
                    EventDetection(
                        event_type="Stationary Person",
                        description="Person remained stationary for too many consecutive frames.",
                        confidence=0.9,
                        evidence=[f"stationary_frames={summary.stationary_frames}"],
                        object_id=track.id,
                    )
                )

            if restricted_rule is not None and restricted_rule.enabled and evaluate_restricted_region_rule(track, previous_bbox, restricted_rule):
                self.events.append(
                    EventDetection(
                        event_type="Restricted Area Entry",
                        description="Person entered the predefined restricted region.",
                        confidence=0.95,
                        evidence=[f"bbox={list(track.bbox)}"],
                        object_id=track.id,
                    )
                )

        return self.events

    def reset(self) -> None:
        """Clear any accumulated state."""
        self.events.clear()
        self.behavior_analyzer = BehaviorAnalyzer()
