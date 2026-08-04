"""Rule definitions for basic abnormal event analysis.

The rules are intentionally lightweight so they can be used directly with the
tracking pipeline without adding external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


DEFAULT_STATIONARY_FRAME_THRESHOLD = 5
DEFAULT_MOVEMENT_THRESHOLD_PIXELS = 3.0
DEFAULT_RESTRICTED_REGION = (0, 0, 200, 200)


@dataclass
class EventRule:
    """Simple rule container for abnormal event detection."""

    name: str
    description: str
    enabled: bool = True
    parameters: dict = field(default_factory=dict)


class EventRuleEngine:
    """Manage and evaluate the configured rules."""

    def __init__(self, rules: Optional[List[EventRule]] = None) -> None:
        self.rules = rules or []

    def add_rule(self, rule: EventRule) -> None:
        """Register a new rule."""
        self.rules.append(rule)

    def evaluate(self, snapshot) -> List[EventRule]:
        """Return the enabled rules for a tracking snapshot."""
        return [rule for rule in self.rules if rule.enabled]


def build_default_rules() -> List[EventRule]:
    """Create the default rule set for Phase 4."""
    return [
        EventRule(
            name="stationary_person",
            description="Person remains stationary for too many frames",
            parameters={
                "stationary_frame_threshold": DEFAULT_STATIONARY_FRAME_THRESHOLD,
                "movement_threshold_pixels": DEFAULT_MOVEMENT_THRESHOLD_PIXELS,
            },
        ),
        EventRule(
            name="restricted_area_entry",
            description="Person enters the predefined restricted region",
            parameters={"restricted_region": DEFAULT_RESTRICTED_REGION},
        ),
    ]


def get_rule_parameter(rule: EventRule, key: str, default=None):
    """Return a rule parameter, falling back to the default value."""
    return rule.parameters.get(key, default)


def bbox_center(bbox) -> Tuple[float, float]:
    """Compute the center point of a bounding box."""
    if bbox is None:
        return (0.0, 0.0)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_region(point: Tuple[float, float], region) -> bool:
    """Check whether a point lies inside a rectangular region."""
    if region is None:
        return False
    x1, y1, x2, y2 = region
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def evaluate_stationary_rule(summary, rule: EventRule) -> bool:
    """Evaluate whether a tracked object's behavior meets the stationary rule."""
    threshold = int(get_rule_parameter(rule, "stationary_frame_threshold", DEFAULT_STATIONARY_FRAME_THRESHOLD))
    return bool(summary.is_stationary and summary.stationary_frames >= threshold)


def evaluate_restricted_region_rule(track, previous_bbox, rule: EventRule) -> bool:
    """Evaluate whether the object has entered the restricted region."""
    region = get_rule_parameter(rule, "restricted_region", DEFAULT_RESTRICTED_REGION)
    if region is None:
        return False

    current_inside = point_in_region(bbox_center(track.bbox), region)
    previous_inside = point_in_region(bbox_center(previous_bbox), region) if previous_bbox is not None else False
    return current_inside and not previous_inside
