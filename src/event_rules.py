"""Rule definitions for future abnormal event analysis.

This module will hold reusable rule objects and configuration values that the
abnormal event detector can apply to tracking snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class EventRule:
    """Simple rule container for future abnormal behavior detection."""

    name: str
    description: str
    enabled: bool = True
    parameters: dict = field(default_factory=dict)


class EventRuleEngine:
    """Placeholder engine that will evaluate configured rules."""

    def __init__(self, rules: Optional[List[EventRule]] = None) -> None:
        self.rules = rules or []

    def add_rule(self, rule: EventRule) -> None:
        """Register a new rule."""
        self.rules.append(rule)

    def evaluate(self, snapshot) -> List[EventRule]:
        """Evaluate rules against a tracking snapshot.

        TODO: Implement rule evaluation and return matched rules.
        """
        return [rule for rule in self.rules if rule.enabled]
