"""Phase 4 scaffolding for abnormal event detection.

This module will eventually receive tracking snapshots and evaluate them against
configured rules to identify abnormal or suspicious behavior in surveillance
video streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tracker import TrackingSnapshot


@dataclass
class EventDetection:
    """Represents a detected abnormal event candidate."""

    event_type: str
    description: str
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


class AbnormalEventDetector:
    """Placeholder detector interface for future abnormal event logic."""

    def __init__(self) -> None:
        self.events: List[EventDetection] = []

    def process_snapshot(self, snapshot: TrackingSnapshot) -> List[EventDetection]:
        """Analyze one tracking snapshot and return event candidates.

        TODO: Implement rule-based or model-based abnormal event detection here.
        """
        self.events = []
        return self.events

    def reset(self) -> None:
        """Clear any accumulated state."""
        self.events.clear()
