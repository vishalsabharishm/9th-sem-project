"""Transparent, configurable rule-based risk assessment for Phase 5."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from abnormal_event_detector import EventDetection
except ImportError:  # pragma: no cover - supports package execution
    from src.abnormal_event_detector import EventDetection


DEFAULT_RISK_MAPPING: Dict[str, str] = {
    "Stationary Person": "Low",
    "Restricted Area Entry": "Medium",
    "Crowding": "Medium",
    "Proximity/Interaction": "Low",
}
DEFAULT_RISK_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "risk_assessment.json"


@dataclass(frozen=True)
class RiskAssessment:
    """A transparent assessment derived from one abnormal-event record."""

    event_type: str
    risk_level: str
    reason: str
    confidence: Optional[float] = None
    frame_number: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    object_id: Optional[int] = None
    object_ids: List[int] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return {
            "event_type": self.event_type,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "confidence": self.confidence,
            "frame_number": self.frame_number,
            "timestamp_seconds": self.timestamp_seconds,
            "object_id": self.object_id,
            "object_ids": self.object_ids,
            "evidence": self.evidence,
        }


class RiskAssessor:
    """Map known Phase 4 event types to configured, explainable risk levels."""

    def __init__(self, risk_mapping: Optional[Dict[str, str]] = None) -> None:
        self.risk_mapping = dict(DEFAULT_RISK_MAPPING if risk_mapping is None else risk_mapping)

    def assess(
        self,
        event: EventDetection,
        frame_number: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
    ) -> RiskAssessment:
        """Assess one event without generating any new confidence value."""
        risk_level = self.risk_mapping.get(event.event_type, "Unknown")
        if risk_level == "Unknown":
            reason = f"No risk mapping is configured for event type '{event.event_type}'."
        else:
            reason = f"Configured mapping assigns {event.event_type} to {risk_level} risk."

        confidence = event.confidence if event.confidence is not None and event.confidence > 0 else None
        return RiskAssessment(
            event_type=event.event_type,
            risk_level=risk_level,
            reason=reason,
            confidence=confidence,
            frame_number=frame_number if frame_number is not None else event.frame_number,
            timestamp_seconds=timestamp_seconds if timestamp_seconds is not None else event.timestamp_seconds,
            object_id=event.object_id,
            object_ids=list(event.object_ids),
            evidence=list(event.evidence),
        )

    def assess_events(
        self,
        events: Iterable[EventDetection],
        frame_number: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
    ) -> List[RiskAssessment]:
        """Assess a collection of event records with shared optional context."""
        return [self.assess(event, frame_number, timestamp_seconds) for event in events]


def write_risk_assessments(
    assessments: Iterable[RiskAssessment],
    output_path: Path = DEFAULT_RISK_OUTPUT_PATH,
) -> Path:
    """Write risk assessments as JSON for reporting or future consumers."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([assessment.as_dict() for assessment in assessments], indent=2),
        encoding="utf-8",
    )
    return path
