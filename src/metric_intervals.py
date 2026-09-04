"""Confidence intervals for the temporal evaluation metrics.

WHY A SEPARATE MODULE
---------------------
``temporal_metrics.py`` produced every historical number in this project. It is
deliberately left byte-identical: adding to it would change the file whose
output the recorded results came from, for no benefit. Interval estimation is a
separate concern and lives here.

WHICH METRICS CAN HONESTLY RECEIVE AN INTERVAL
----------------------------------------------
Three different situations, and this module keeps them apart rather than
applying one formula to everything.

**Binomial proportions with a denominator fixed by the evaluation design.**
Accuracy (over all N clips), recall/sensitivity (over the Fight clips) and
specificity (over the NonFight clips) each count successes out of a denominator
that was decided before any prediction was made. A Wilson score interval is
appropriate and is what this module reports.

**Binomial proportions with a random denominator.** Precision (TP / predicted
positives) and NPV (TN / predicted negatives) divide by a count the model
chose. A Wilson interval on them is *conditional on the observed number of
predictions of that class* -- it is a standard and defensible thing to report,
but it is not the same object as the recall interval, and it is labelled
``denominator_is_random`` so a reader is not misled.

**Not proportions at all.** F1 is a harmonic mean of two overlapping
proportions; ROC-AUC and PR-AUC are rank statistics over all pairs. None of
them is a count out of a fixed denominator, and applying a binomial formula to
them would be wrong. They get a percentile bootstrap or nothing -- never a
Wilson interval.

THE INDEPENDENCE PROBLEM
------------------------
Every interval below assumes the evaluation units are independent. In the
394-clip primary split they are not: the clips come from only 174 distinct
source videos (inferred from the ``<source_id>_<n>.avi`` filename convention),
a mean of 2.26 clips per source, with individual sources contributing up to 13
clips, and 25 sources contributing clips to *both* classes.

Clips cut from one video share camera, scene, lighting and subjects, so their
outcomes are correlated and the effective sample size is below 394. A
clip-level interval is therefore **anti-conservative** -- too narrow.

This module does not hide that. It reports the conventional clip-level interval
*and* a cluster bootstrap that resamples whole source videos, so the size of
the effect is visible instead of assumed away. See ``cluster_ids`` for the
caveat that the grouping is inferred from filenames and cannot be confirmed
against dataset metadata that this repository does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 42

METHOD_WILSON = "wilson_score"
METHOD_BOOTSTRAP_STRATIFIED = "percentile_bootstrap_stratified_by_class"
METHOD_BOOTSTRAP_CLUSTER = "percentile_bootstrap_over_source_video_clusters"

# Filenames are '<source_id>_<index>.avi'. Verified to match all 394 primary
# clips and all 240 carve clips.
_CLIP_STEM = re.compile(r"^(?P<source>.+)_(?P<index>\d+)$")


class IntervalError(ValueError):
    """Raised when an interval cannot be computed from the given inputs."""


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal critical value. ``statistics`` only -- no scipy."""
    if not 0.0 < confidence < 1.0:
        raise IntervalError(f"confidence must be in (0, 1), got {confidence}.")
    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


@dataclass(frozen=True)
class ProportionInterval:
    """A proportion with a Wilson score interval and its assumptions attached."""

    name: str
    successes: int
    total: int
    point: float
    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE
    method: str = METHOD_WILSON
    denominator_is_random: bool = False
    note: str = ""

    @property
    def half_width_points(self) -> float:
        """Half the interval width, in percentage points (asymmetric: mean of sides)."""
        return 100.0 * ((self.high - self.low) / 2.0)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "successes": self.successes,
            "total": self.total,
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "confidence": self.confidence,
            "method": self.method,
            "denominator_is_random": self.denominator_is_random,
            "note": self.note,
        }

    def render(self) -> str:
        return (
            f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}] "
            f"({self.successes}/{self.total})"
        )


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile bootstrap interval, with everything needed to reproduce it."""

    name: str
    point: Optional[float]
    low: Optional[float]
    high: Optional[float]
    confidence: float = DEFAULT_CONFIDENCE
    method: str = METHOD_BOOTSTRAP_STRATIFIED
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    seed: int = DEFAULT_BOOTSTRAP_SEED
    usable_resamples: int = 0
    discarded_resamples: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "confidence": self.confidence,
            "method": self.method,
            "resamples": self.resamples,
            "seed": self.seed,
            "usable_resamples": self.usable_resamples,
            "discarded_resamples": self.discarded_resamples,
            "note": self.note,
        }

    def render(self) -> str:
        if self.point is None or self.low is None:
            return "not computable"
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) approximation because it stays inside
    [0, 1], does not collapse to zero width at 0 or 100 percent, and keeps
    close to nominal coverage at the sample sizes and near-extreme proportions
    this project reports.

        centre     = (p + z^2/2n) / (1 + z^2/n)
        half-width = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/(4n^2) )

    The interval is asymmetric about ``p``; that is a property of the method,
    not a bug. Bounds are clamped into [0, 1] against floating-point drift.
    """
    if total < 0:
        raise IntervalError(f"total must be non-negative, got {total}.")
    if total == 0:
        raise IntervalError("Cannot form an interval from zero observations.")
    if not 0 <= successes <= total:
        raise IntervalError(
            f"successes must be within [0, {total}], got {successes}."
        )

    z = z_for(confidence)
    n = float(total)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = (z / denominator) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    low = float(max(0.0, centre - half))
    high = float(min(1.0, centre + half))

    # At the extremes the algebra gives exactly 0 and exactly 1; floating-point
    # cancellation leaves a ~1e-17 residue instead. Snap those two cases so a
    # zero-success interval reads as [0, ...] rather than [2.8e-17, ...]. This
    # is the exact value, not a widening.
    if successes == 0:
        low = 0.0
    if successes == total:
        high = 1.0
    return (low, high)


def proportion_interval(
    name: str,
    successes: int,
    total: int,
    confidence: float = DEFAULT_CONFIDENCE,
    denominator_is_random: bool = False,
    note: str = "",
) -> ProportionInterval:
    """Wrap :func:`wilson_interval` with its point estimate and assumptions."""
    low, high = wilson_interval(successes, total, confidence)
    return ProportionInterval(
        name=name,
        successes=int(successes),
        total=int(total),
        point=successes / total,
        low=low,
        high=high,
        confidence=confidence,
        denominator_is_random=denominator_is_random,
        note=note,
    )


RANDOM_DENOMINATOR_NOTE = (
    "the denominator is the number of clips the model predicted for this class, "
    "which is not fixed by the evaluation design; this interval is conditional "
    "on that observed count"
)


def confusion_intervals(
    true_positive: int,
    false_positive: int,
    true_negative: int,
    false_negative: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, ProportionInterval]:
    """Wilson intervals for every metric that is genuinely a proportion.

    Deliberately omits F1, ROC-AUC and PR-AUC: none of them is a count out of a
    denominator, so no binomial interval applies. Use
    :func:`bootstrap_metric_interval` for those, or report them without an
    interval.
    """
    tp, fp, tn, fn = (int(true_positive), int(false_positive),
                      int(true_negative), int(false_negative))
    total = tp + fp + tn + fn
    if total == 0:
        raise IntervalError("Cannot form intervals from an empty confusion matrix.")

    intervals: Dict[str, ProportionInterval] = {
        "accuracy": proportion_interval(
            "accuracy", tp + tn, total, confidence,
            note="successes are correct decisions over all evaluated clips",
        ),
        "recall_sensitivity": proportion_interval(
            "recall_sensitivity", tp, tp + fn, confidence,
            note="denominator is the number of Fight clips, fixed by the split",
        ),
        "specificity": proportion_interval(
            "specificity", tn, tn + fp, confidence,
            note="denominator is the number of NonFight clips, fixed by the split",
        ),
        "prevalence_fight": proportion_interval(
            "prevalence_fight", tp + fn, total, confidence,
            note="class balance of the evaluation split itself",
        ),
    }
    if tp + fp > 0:
        intervals["precision_ppv"] = proportion_interval(
            "precision_ppv", tp, tp + fp, confidence,
            denominator_is_random=True, note=RANDOM_DENOMINATOR_NOTE,
        )
    if tn + fn > 0:
        intervals["npv"] = proportion_interval(
            "npv", tn, tn + fn, confidence,
            denominator_is_random=True, note=RANDOM_DENOMINATOR_NOTE,
        )
    return intervals


# --------------------------------------------------------------------------
# Source-video clustering
# --------------------------------------------------------------------------


def source_video_id(clip: str) -> str:
    """Infer the source video a clip was cut from, via ``<source>_<index>``.

    INFERRED, NOT READ. RWF-2000 ships no per-clip source metadata in anything
    this repository holds, so the grouping comes from the filename convention.
    It matches all 394 primary and all 240 carve clips, and
    ``rwf2000_config.SplitPolicy.validation_grouped_by_source_video`` records
    that the dataset authors treated source video as a real grouping. That is
    supporting evidence, not proof: two unrelated videos sharing an id prefix
    would be merged, and one video released under two ids would be split.

    The class directory is deliberately NOT part of the key: 25 source ids
    contribute clips to both Val_Fight and Val_NonFight, and those clips come
    from the same underlying video.
    """
    stem = Path(clip).stem
    match = _CLIP_STEM.match(stem)
    return match.group("source") if match else stem


def cluster_ids(clips: Sequence[str]) -> np.ndarray:
    """Map each clip to an integer cluster index by inferred source video."""
    lookup: Dict[str, int] = {}
    out = np.empty(len(clips), dtype=np.int64)
    for position, clip in enumerate(clips):
        source = source_video_id(clip)
        out[position] = lookup.setdefault(source, len(lookup))
    return out


def cluster_summary(clips: Sequence[str]) -> dict:
    """Describe how far the evaluation set is from independent observations."""
    if not len(clips):
        raise IntervalError("Cannot summarise clustering of zero clips.")
    ids = cluster_ids(clips)
    _, counts = np.unique(ids, return_counts=True)
    return {
        "clips": int(len(clips)),
        "clusters": int(counts.size),
        "mean_cluster_size": float(counts.mean()),
        "max_cluster_size": int(counts.max()),
        "singleton_clusters": int((counts == 1).sum()),
        "clips_in_multi_clip_clusters": int(counts[counts > 1].sum()),
        "grouping": "inferred from the '<source_id>_<index>.avi' filename convention",
        "caveat": (
            "not confirmed against dataset metadata; this repository holds no "
            "per-clip source-video record"
        ),
    }


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def bootstrap_metric_interval(
    name: str,
    y_true: Sequence[int],
    y_score: Sequence[float],
    statistic: Callable[[np.ndarray, np.ndarray], Optional[float]],
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    clusters: Optional[Sequence[int]] = None,
    note: str = "",
) -> BootstrapInterval:
    """Percentile bootstrap for a statistic that is not a binomial proportion.

    Two resampling schemes, chosen by whether ``clusters`` is supplied:

    *Stratified by class* (``clusters is None``). Positives are resampled from
    the positives and negatives from the negatives, so every resample keeps the
    evaluation split's class sizes. Appropriate here because those sizes are a
    property of the split, not something that would vary if the experiment were
    repeated.

    *Clustered* (``clusters`` supplied). Whole source videos are resampled with
    replacement and all their clips taken together, which propagates the
    within-video correlation the stratified scheme ignores. Class balance then
    varies between resamples, which is the honest consequence of treating the
    video as the sampling unit.

    A resample whose composition makes the statistic undefined -- one class
    absent, so ROC-AUC has no meaning -- is discarded and counted rather than
    silently scored as zero. Fully reproducible from ``seed`` and ``resamples``.

    The percentile method is used, not BCa: it is simple enough to verify by
    eye, and with the cluster dependence dominating the uncertainty here, bias
    correction would be false precision.
    """
    truth = np.asarray(y_true, dtype=np.int64).ravel()
    score = np.asarray(y_score, dtype=np.float64).ravel()
    if truth.shape != score.shape:
        raise IntervalError("y_true and y_score must have equal length.")
    if truth.size == 0:
        raise IntervalError("Cannot bootstrap an empty evaluation set.")
    if resamples < 1:
        raise IntervalError(f"resamples must be >= 1, got {resamples}.")

    point = statistic(truth, score)
    generator = np.random.default_rng(seed)

    if clusters is None:
        positives = np.flatnonzero(truth == 1)
        negatives = np.flatnonzero(truth == 0)
        method = METHOD_BOOTSTRAP_STRATIFIED
    else:
        cluster_array = np.asarray(clusters, dtype=np.int64).ravel()
        if cluster_array.shape != truth.shape:
            raise IntervalError("clusters must align with y_true.")
        unique = np.unique(cluster_array)
        members = [np.flatnonzero(cluster_array == c) for c in unique]
        method = METHOD_BOOTSTRAP_CLUSTER

    values: List[float] = []
    discarded = 0
    for _ in range(resamples):
        if clusters is None:
            picked = np.concatenate(
                [
                    generator.choice(positives, size=positives.size, replace=True),
                    generator.choice(negatives, size=negatives.size, replace=True),
                ]
            )
        else:
            drawn = generator.integers(0, len(members), size=len(members))
            picked = np.concatenate([members[i] for i in drawn])
        value = statistic(truth[picked], score[picked])
        if value is None or not np.isfinite(value):
            discarded += 1
            continue
        values.append(float(value))

    if not values:
        return BootstrapInterval(
            name=name, point=point, low=None, high=None, confidence=confidence,
            method=method, resamples=resamples, seed=seed, usable_resamples=0,
            discarded_resamples=discarded,
            note="every resample was degenerate; no interval can be formed",
        )

    tail = (1.0 - confidence) / 2.0
    ordered = np.sort(np.asarray(values))
    low = float(np.quantile(ordered, tail, method="linear"))
    high = float(np.quantile(ordered, 1.0 - tail, method="linear"))
    return BootstrapInterval(
        name=name, point=point, low=low, high=high, confidence=confidence,
        method=method, resamples=resamples, seed=seed,
        usable_resamples=len(values), discarded_resamples=discarded, note=note,
    )


NO_INTERVAL_CLAIMED = {
    "f1": (
        "F1 is the harmonic mean of precision and recall, which share the TP "
        "count and have different denominators. It is not a binomial "
        "proportion and no closed-form interval applies; only a bootstrap "
        "interval is offered, and it is labelled as such."
    ),
    "roc_auc": (
        "ROC-AUC is a rank statistic over all positive-negative pairs, not a "
        "count out of a denominator. It is not a binomial proportion, so no "
        "closed-form interval applies. A percentile bootstrap is offered; "
        "DeLong's variance estimator was not implemented in this step."
    ),
    "pr_auc": (
        "PR-AUC (average precision) is a rank statistic summarising the "
        "precision-recall curve, not a count out of a denominator. It is "
        "not a binomial proportion, so no closed-form interval applies. A "
        "percentile bootstrap is offered, as for ROC-AUC."
    ),
}
