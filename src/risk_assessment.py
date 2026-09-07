"""Transparent, configurable rule-based risk assessment for Phase 5."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from abnormal_event_detector import EventDetection
    from temporal_event_adapter import (
        DECLARED_CONSTANT_EVENT_TYPES,
        PROVENANCE_DECLARED,
        PROVENANCE_MEASURED,
        TEMPORAL_EVENT_TYPE,
    )
except ImportError:  # pragma: no cover - supports package execution
    from src.abnormal_event_detector import EventDetection
    from src.temporal_event_adapter import (
        DECLARED_CONSTANT_EVENT_TYPES,
        PROVENANCE_DECLARED,
        PROVENANCE_MEASURED,
        TEMPORAL_EVENT_TYPE,
    )


DEFAULT_RISK_MAPPING: Dict[str, str] = {
    "Stationary Person": "Low",
    "Restricted Area Entry": "Medium",
    "Crowding": "Medium",
    "Proximity/Interaction": "Low",
    TEMPORAL_EVENT_TYPE: "High",
}
DEFAULT_RISK_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "risk_assessment.json"

# ---------------------------------------------------------------------------
# What this layer is, stated so a consumer cannot mistake it for something else.
#
# The four concerns below are deliberately separated. Conflating them is what
# turns an engineering convenience into an unsupported scientific claim, and the
# risk level in particular is a CONFIGURED CONSTANT, not a measurement.
# ---------------------------------------------------------------------------

RISK_LAYER_DECLARATION = {
    "layers": {
        "1_event_detection": {
            "what": "did a rule or the temporal model fire on this frame or clip",
            "source": "AbnormalEventDetector rules and the frozen temporal aggregation",
            "empirically_evaluated": True,
            "note": (
                "the temporal branch has a measured operating point on held-out data; "
                "the spatial rules have NO violence ground truth and were never "
                "evaluated as detectors"
            ),
        },
        "2_evidence": {
            "what": "the geometric or probabilistic quantities behind the firing",
            "source": "rule evidence strings and the R3D window probability",
            "empirically_evaluated": True,
            "note": "quantities are measured; their sufficiency as evidence is not established",
        },
        "3_confidence": {
            "what": "the number attached to an event",
            "source": "either a measured model probability or an engineer-declared constant",
            "empirically_evaluated": "partly",
            "note": (
                "confidence_provenance distinguishes the two. Stationary and Restricted "
                "carry HARDCODED constants (0.9, 0.95); Crowding and Proximity carry no "
                "confidence at all. Only the temporal value is a model probability."
            ),
        },
        "4_risk_interpretation": {
            "what": "the Low / Medium / High label",
            "source": "a fixed dictionary from event type to level",
            "empirically_evaluated": False,
            "note": (
                "this is a configured mapping chosen by an engineer. It is NOT a risk "
                "model: it is not calibrated, not learned, not validated against any "
                "outcome, and no risk ground truth exists in RWF-2000."
            ),
        },
    },
    "is_validated_risk_model": False,
    "risk_ground_truth_available": False,
    "what_may_be_claimed": (
        "the system reports which events fired, the evidence behind them, and a "
        "configured severity label"
    ),
    "what_may_not_be_claimed": (
        "that the risk level is calibrated, probabilistic, operationally validated, "
        "or comparable across event types. A 'High' label is a routing convenience, "
        "not an estimated probability of harm."
    ),
    "to_validate_this_layer": (
        "risk-annotated data with outcome labels would be required. No such data "
        "exists in this project, so the limitation is structural rather than pending."
    ),
}


def describe_risk_layer() -> Dict[str, object]:
    """Return the layer declaration, for reports and UI copy.

    Exposed as a function so that a caller reporting a risk level has a
    single obvious place to obtain the caveat that must travel with it.
    """
    return dict(RISK_LAYER_DECLARATION)



def _confidence_provenance(event_type: str, confidence: Optional[float]) -> Optional[str]:
    """Tag a confidence value's origin so a report can never confuse a
    calibrated model probability with an engineer's fixed rule weight.

    Returns ``None`` when there is no confidence to tag (Crowding /
    Proximity events are reported without a manufactured number).
    """
    if confidence is None:
        return None
    if event_type in DECLARED_CONSTANT_EVENT_TYPES:
        return PROVENANCE_DECLARED
    if event_type == TEMPORAL_EVENT_TYPE:
        return PROVENANCE_MEASURED
    return None


@dataclass(frozen=True)
class RiskAssessment:
    """A transparent assessment derived from one abnormal-event record."""

    event_type: str
    risk_level: str
    reason: str
    confidence: Optional[float] = None
    confidence_provenance: Optional[str] = None
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
            "confidence_provenance": self.confidence_provenance,
            # Travels with every record so a consumer cannot render a risk level
            # without also having the fact that it is a configured constant.
            "risk_level_is_validated": False,
            "risk_level_basis": "configured mapping from event type; not calibrated or learned",
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
            confidence_provenance=_confidence_provenance(event.event_type, confidence),
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
