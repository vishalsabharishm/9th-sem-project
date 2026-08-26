"""Binary classification metrics for the temporal violence classifier.

Implemented on numpy alone so the pipeline needs no extra dependency and
produces identical numbers locally and on a GPU runtime.

``Fight`` is the positive class (index 1, per
:data:`rwf2000_config.LABEL_TO_INDEX`), so recall answers "what fraction
of violent clips did we catch" -- the quantity that matters for a
surveillance alert.

Degenerate inputs return ``None`` for the affected metric rather than a
misleading number: ROC-AUC and PR-AUC are undefined when only one class
is present, and precision is undefined when nothing was predicted
positive. A ``None`` must be reported as "not computable", never as zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np


DEFAULT_DECISION_THRESHOLD = 0.5


class MetricsError(ValueError):
    """Raised when metrics cannot be computed from the given inputs."""


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts for the binary decision, with ``Fight`` positive."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    @property
    def total(self) -> int:
        """Total number of scored samples."""
        return (
            self.true_negative
            + self.false_positive
            + self.false_negative
            + self.true_positive
        )

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return asdict(self)


@dataclass(frozen=True)
class BinaryMetrics:
    """One evaluation result. ``None`` means not computable, not zero."""

    accuracy: float
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: ConfusionMatrix
    threshold: float
    support: Dict[str, int]

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        document = asdict(self)
        document["confusion_matrix"] = self.confusion_matrix.as_dict()
        return document

    def summary(self) -> str:
        """Return a compact one-line summary for logs."""

        def show(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        return (
            f"acc={self.accuracy:.4f} prec={show(self.precision)} "
            f"rec={show(self.recall)} f1={show(self.f1)} "
            f"roc_auc={show(self.roc_auc)} pr_auc={show(self.pr_auc)}"
        )


def _as_arrays(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int64).ravel()
    score = np.asarray(y_score, dtype=np.float64).ravel()
    if truth.size == 0:
        raise MetricsError("Cannot compute metrics from an empty evaluation set.")
    if truth.shape != score.shape:
        raise MetricsError(
            f"y_true and y_score must have equal length, got "
            f"{truth.shape[0]} and {score.shape[0]}."
        )
    if not np.all((truth == 0) | (truth == 1)):
        raise MetricsError("y_true must contain only 0 and 1.")
    return truth, score


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """Return ROC-AUC via the rank (Mann-Whitney U) identity.

    Ranking avoids threshold sweeps entirely and handles ties exactly by
    averaging their ranks, which is what makes this agree with reference
    implementations on scores that repeat.
    """
    truth, score = _as_arrays(y_true, y_score)
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    sorted_scores = score[order]
    index = 0
    while index < sorted_scores.size:
        stop = index
        while stop + 1 < sorted_scores.size and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        # Average rank (1-based) across the tied block.
        average = (index + stop + 2) / 2.0
        ranks[order[index : stop + 1]] = average
        index = stop + 1

    positive_rank_sum = float(ranks[truth == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """Return average precision: the step-wise area under the PR curve.

    Uses the ``sum((R_n - R_{n-1}) * P_n)`` definition rather than
    trapezoidal interpolation, which is the standard for average
    precision and does not optimistically bridge between operating points.
    """
    truth, score = _as_arrays(y_true, y_score)
    positives = int(truth.sum())
    if positives == 0 or positives == truth.size:
        return None

    order = np.argsort(-score, kind="mergesort")
    ordered_truth = truth[order]
    ordered_score = score[order]

    true_positives = np.cumsum(ordered_truth)
    predicted_positives = np.arange(1, truth.size + 1, dtype=np.float64)
    precision = true_positives / predicted_positives
    recall = true_positives / positives

    # Only the last entry of a tied score block is a real operating point.
    keep = np.ones(truth.size, dtype=bool)
    keep[:-1] = ordered_score[1:] != ordered_score[:-1]

    precision = precision[keep]
    recall = recall[keep]
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous_recall) * precision))


def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> BinaryMetrics:
    """Compute the full metric set from labels and positive-class scores.

    ``y_score`` is the predicted probability of ``Fight``. The threshold
    is explicit because the tuned operating point is a later decision --
    ``rwf2000_config.InitialTrainingConfig.decision_threshold`` is
    deliberately unset until validation data exists.
    """
    truth, score = _as_arrays(y_true, y_score)
    if not 0.0 <= float(threshold) <= 1.0:
        raise MetricsError(f"threshold must be in [0, 1], got {threshold}.")

    predicted = (score >= float(threshold)).astype(np.int64)
    true_positive = int(np.sum((truth == 1) & (predicted == 1)))
    true_negative = int(np.sum((truth == 0) & (predicted == 0)))
    false_positive = int(np.sum((truth == 0) & (predicted == 1)))
    false_negative = int(np.sum((truth == 1) & (predicted == 0)))

    accuracy = float((true_positive + true_negative) / truth.size)
    precision = (
        float(true_positive / (true_positive + false_positive))
        if (true_positive + false_positive) > 0
        else None
    )
    recall = (
        float(true_positive / (true_positive + false_negative))
        if (true_positive + false_negative) > 0
        else None
    )
    if precision is None or recall is None or (precision + recall) == 0.0:
        f1 = None
    else:
        f1 = float(2 * precision * recall / (precision + recall))

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc(truth, score),
        pr_auc=pr_auc(truth, score),
        confusion_matrix=ConfusionMatrix(
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            true_positive=true_positive,
        ),
        threshold=float(threshold),
        support={
            "NonFight": int(np.sum(truth == 0)),
            "Fight": int(np.sum(truth == 1)),
        },
    )
