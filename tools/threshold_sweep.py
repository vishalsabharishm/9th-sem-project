"""Choose the Fight decision threshold from already-computed predictions.

This never loads a model and never touches a video. It reads the
per-clip scores the evaluation run already wrote
(``evaluation/predictions.csv``) and re-thresholds them, so the sweep is
exact, instant, and costs no GPU.

WHY NOT ACCURACY
----------------
On a balanced set, accuracy is maximised near the threshold that trades a
missed fight for a false alarm one-for-one. That is the wrong exchange
rate for surveillance: a missed assault is the failure the system exists
to prevent, while a false alarm costs an operator a few seconds. The
selection here is therefore driven by **recall subject to a false-alarm
ceiling**, with accuracy and F1 reported but not optimised.

    python tools/threshold_sweep.py \\
        --predictions evaluation/predictions.csv \\
        --output-dir evaluation/threshold_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

POSITIVE_LABEL = "Fight"


class SweepError(RuntimeError):
    """Raised when the sweep cannot be performed."""


@dataclass(frozen=True)
class OperatingPoint:
    """Every quantity that matters at one threshold."""

    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    accuracy: float
    precision: Optional[float]
    recall: float
    f1: Optional[float]
    specificity: float
    # Recall and specificity averaged -- the balanced view that does not
    # let the majority class hide missed fights.
    balanced_accuracy: float
    # Youden's J: recall + specificity - 1. Threshold-free ROC optimum.
    youden_j: float

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return asdict(self)


def read_predictions(path: Path) -> Tuple[List[int], List[float]]:
    """Return (truths, fight_probabilities) from an evaluation CSV.

    Expects the columns ``evaluate_temporal_baseline`` writes:
    ``index,true_label,fight_probability,predicted_label``. The stored
    ``predicted_label`` is deliberately ignored -- it encodes the old
    threshold, and re-deriving it is the whole point.
    """
    path = Path(path)
    if not path.is_file():
        raise SweepError(f"predictions file not found: {path}")

    truths: List[int] = []
    scores: List[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"true_label", "fight_probability"} - set(reader.fieldnames or [])
        if missing:
            raise SweepError(f"{path} is missing column(s): {sorted(missing)}")
        for row in reader:
            truths.append(1 if row["true_label"] == POSITIVE_LABEL else 0)
            scores.append(float(row["fight_probability"]))

    if not truths:
        raise SweepError(f"{path} contained no rows")
    if not any(truths) or all(truths):
        raise SweepError("predictions contain only one class; a sweep is meaningless")
    return truths, scores


def evaluate_at(
    truths: Sequence[int], scores: Sequence[float], threshold: float
) -> OperatingPoint:
    """Score one threshold. ``score >= threshold`` predicts Fight."""
    true_positive = false_positive = true_negative = false_negative = 0
    for truth, score in zip(truths, scores):
        predicted = 1 if score >= threshold else 0
        if truth == 1 and predicted == 1:
            true_positive += 1
        elif truth == 0 and predicted == 1:
            false_positive += 1
        elif truth == 0 and predicted == 0:
            true_negative += 1
        else:
            false_negative += 1

    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    total = positives + negatives

    recall = true_positive / positives if positives else 0.0
    specificity = true_negative / negatives if negatives else 0.0
    predicted_positive = true_positive + false_positive
    precision = true_positive / predicted_positive if predicted_positive else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and (precision + recall) > 0
        else None
    )

    return OperatingPoint(
        threshold=round(threshold, 6),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        accuracy=(true_positive + true_negative) / total if total else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        balanced_accuracy=(recall + specificity) / 2,
        youden_j=recall + specificity - 1,
    )


def sweep(
    truths: Sequence[int],
    scores: Sequence[float],
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.05,
) -> List[OperatingPoint]:
    """Evaluate a grid of thresholds from ``start`` to ``stop`` inclusive."""
    if step <= 0:
        raise SweepError("step must be positive")
    points = []
    steps = int(round((stop - start) / step))
    for index in range(steps + 1):
        points.append(evaluate_at(truths, scores, start + index * step))
    return points


def recommend(
    points: Sequence[OperatingPoint],
    min_precision: float = 0.75,
    max_false_positive_rate: float = 0.15,
    min_accuracy: Optional[float] = None,
) -> Tuple[OperatingPoint, str]:
    """Pick the operating point: highest recall inside the alarm budget.

    Guards keep "maximise recall" from collapsing to "alarm on
    everything": predictions must stay at least ``min_precision`` precise,
    the false-positive rate must stay under ``max_false_positive_rate``,
    and -- when ``min_accuracy`` is given -- overall accuracy must not
    fall below it. That last guard is the useful one in practice: setting
    it to the baseline's accuracy expresses "buy as much recall as
    possible, but do not pay for it with overall correctness".

    Among the survivors the highest recall wins; ties break on fewer
    false positives, then on higher F1.
    """
    eligible = [
        point
        for point in points
        if point.precision is not None
        and point.precision >= min_precision
        and (1 - point.specificity) <= max_false_positive_rate
        and (min_accuracy is None or point.accuracy >= min_accuracy)
    ]
    if not eligible:
        # Never silently invent a point: fall back to Youden's J and say so.
        best = max(points, key=lambda p: (p.youden_j, -p.false_positive))
        return best, (
            f"No threshold met precision >= {min_precision:.2f} with a false-positive "
            f"rate <= {max_false_positive_rate:.2f}; fell back to the Youden's-J optimum."
        )

    best = max(
        eligible,
        key=lambda p: (p.recall, -p.false_positive, p.f1 or 0.0),
    )
    floor = (
        "" if min_accuracy is None
        else f" and accuracy >= {min_accuracy:.4f}"
    )
    return best, (
        f"Highest recall among thresholds holding precision >= {min_precision:.2f}, "
        f"false-positive rate <= {max_false_positive_rate:.2f}{floor}."
    )


def render_table(points: Sequence[OperatingPoint]) -> str:
    """Render the sweep as a fixed-width table."""

    def show(value: Optional[float]) -> str:
        return "   n/a" if value is None else f"{value:6.4f}"

    lines = [
        f"{'thr':>5}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}"
        f"{'spec':>8}{'FP':>5}{'FN':>5}",
        "-" * 53,
    ]
    for point in points:
        lines.append(
            f"{point.threshold:5.2f}{point.accuracy:8.4f}{show(point.precision):>8}"
            f"{point.recall:8.4f}{show(point.f1):>8}{point.specificity:8.4f}"
            f"{point.false_positive:5d}{point.false_negative:5d}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min-precision", type=float, default=0.75)
    parser.add_argument("--max-fpr", type=float, default=0.15)
    parser.add_argument("--baseline", type=float, default=0.5)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Accuracy floor for the selection. Pass 'baseline' semantics by "
        "setting this to the baseline threshold's accuracy.",
    )
    parser.add_argument(
        "--min-accuracy-from-baseline",
        action="store_true",
        help="Use the baseline threshold's own accuracy as the floor.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the sweep and write the artifacts."""
    args = build_arg_parser().parse_args(argv)
    try:
        truths, scores = read_predictions(Path(args.predictions))
    except SweepError as error:
        print(f"Sweep failed: {error}", file=sys.stderr)
        return 2

    points = sweep(truths, scores, args.start, args.stop, args.step)
    baseline = evaluate_at(truths, scores, args.baseline)
    floor = args.min_accuracy
    if args.min_accuracy_from_baseline:
        floor = baseline.accuracy
    best, rationale = recommend(points, args.min_precision, args.max_fpr, floor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"clips: {len(truths)}  positives: {sum(truths)}  "
          f"negatives: {len(truths) - sum(truths)}")
    print()
    print(render_table(points))
    print()
    print(f"baseline threshold {baseline.threshold:.2f}: "
          f"recall={baseline.recall:.4f} FN={baseline.false_negative} "
          f"FP={baseline.false_positive} acc={baseline.accuracy:.4f}")
    print(f"selected threshold {best.threshold:.2f}: "
          f"recall={best.recall:.4f} FN={best.false_negative} "
          f"FP={best.false_positive} acc={best.accuracy:.4f}")
    print(f"rationale: {rationale}")

    record = {
        "predictions": str(args.predictions),
        "clips": len(truths),
        "positives": sum(truths),
        "negatives": len(truths) - sum(truths),
        "grid": {"start": args.start, "stop": args.stop, "step": args.step},
        "selection_rule": {
            "min_precision": args.min_precision,
            "max_false_positive_rate": args.max_fpr,
            "min_accuracy": floor,
            "objective": "maximise recall within the alarm budget",
            "rationale": rationale,
        },
        "baseline": baseline.as_dict(),
        "selected": best.as_dict(),
        "sweep": [point.as_dict() for point in points],
    }
    (output_dir / "threshold_sweep.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )

    fields = list(points[0].as_dict())
    with (output_dir / "threshold_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow(point.as_dict())

    (output_dir / "threshold_sweep.txt").write_text(
        render_table(points) + "\n", encoding="utf-8"
    )
    print(f"\nartifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
