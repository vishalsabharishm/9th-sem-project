"""Evaluate a trained R3D-18 violence checkpoint and write the artifacts.

This module reports; it never trains and never writes a checkpoint. It
composes the pieces that already exist rather than re-deriving them:

    temporal_model.R3D18TemporalModel        weights + forward pass
    rwf2000_dataset.build_*_dataset          the official, leak-free splits
    temporal_metrics.binary_metrics          accuracy / P / R / F1 / AUC

EVALUATION SET
--------------
The default is the **primary** evaluation split: the 394 clips left after
the six confirmed train/val duplicates are removed from the official 400.
Those six are excluded from training *and* evaluation, so a number
produced here is not inflated by a clip the model already saw. The full
official 400 is available with ``--split official`` for the one-off final
report, and is labelled as such in the output.

Nothing here re-decides the split. It is read from ``rwf2000_splits``,
which is the same source the training pipeline uses.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from rwf2000_config import CANONICAL_LABELS, LABEL_TO_INDEX, SamplingConfig
    from rwf2000_dataset import (
        DatasetConfig,
        build_official_validation_dataset,
        build_primary_evaluation_dataset,
    )
    from temporal_metrics import BinaryMetrics, binary_metrics
    from temporal_model import (
        R3D18TemporalModel,
        TemporalModelConfig,
        TemporalModelError,
        resolve_device,
    )
    from temporal_preprocessing import PreprocessingConfig
except ImportError:  # pragma: no cover - supports package execution
    from src.rwf2000_config import CANONICAL_LABELS, LABEL_TO_INDEX, SamplingConfig
    from src.rwf2000_dataset import (
        DatasetConfig,
        build_official_validation_dataset,
        build_primary_evaluation_dataset,
    )
    from src.temporal_metrics import BinaryMetrics, binary_metrics
    from src.temporal_model import (
        R3D18TemporalModel,
        TemporalModelConfig,
        TemporalModelError,
        resolve_device,
    )
    from src.temporal_preprocessing import PreprocessingConfig


POSITIVE_CLASS_INDEX = LABEL_TO_INDEX["Fight"]
SPLIT_BUILDERS = {
    "primary": build_primary_evaluation_dataset,
    "official": build_official_validation_dataset,
}


class EvaluationError(RuntimeError):
    """Raised when the evaluation is misconfigured."""


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@torch.no_grad()
def score_dataset(
    module: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[float], float]:
    """Return truths, Fight-probabilities, and seconds spent in forward.

    The timer covers only the forward pass, so the reported per-clip cost
    is the model's, not the video decoder's.
    """
    module.eval()
    truths: List[int] = []
    scores: List[float] = []
    forward_seconds = 0.0

    for clips, labels in loader:
        clips = clips.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = module(clips)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - started

        probabilities = torch.softmax(logits.float(), dim=1)[:, POSITIVE_CLASS_INDEX]
        scores.extend(probabilities.detach().cpu().tolist())
        truths.extend(labels.detach().cpu().tolist())

    return truths, scores, forward_seconds


def classification_report_text(metrics: BinaryMetrics) -> str:
    """Render a per-class report from the confusion matrix.

    Written out rather than imported from scikit-learn so the evaluation
    has no dependency the training pipeline does not already have.
    """
    matrix = metrics.confusion_matrix
    rows = []
    # NonFight is the negative class, Fight the positive one.
    for label, true_positive, false_positive, false_negative in (
        (
            "NonFight",
            matrix.true_negative,
            matrix.false_negative,
            matrix.false_positive,
        ),
        (
            "Fight",
            matrix.true_positive,
            matrix.false_positive,
            matrix.false_negative,
        ),
    ):
        support = true_positive + false_negative
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else None
        )
        recall = true_positive / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall)
            else None
        )
        rows.append((label, precision, recall, f1, support))

    def show(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:8.4f}"

    lines = [
        f"{'class':>10}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}",
        "-" * 52,
    ]
    for label, precision, recall, f1, support in rows:
        lines.append(
            f"{label:>10}{show(precision):>12}{show(recall):>10}"
            f"{show(f1):>10}{support:>10}"
        )
    lines.append("-" * 52)
    lines.append(f"{'accuracy':>10}{metrics.accuracy:>32.4f}{matrix.total:>10}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _roc_points(
    truths: Sequence[int], scores: Sequence[float]
) -> Tuple[List[float], List[float]]:
    """Return (fpr, tpr) points by sweeping every distinct score."""
    truth = np.asarray(truths, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return [], []

    order = np.argsort(-score, kind="mergesort")
    ranked = truth[order]
    true_positive = np.cumsum(ranked == 1)
    false_positive = np.cumsum(ranked == 0)
    tpr = np.concatenate([[0.0], true_positive / positives, [1.0]])
    fpr = np.concatenate([[0.0], false_positive / negatives, [1.0]])
    return fpr.tolist(), tpr.tolist()


def write_plots(
    output_dir: Path,
    metrics: BinaryMetrics,
    truths: Sequence[int],
    scores: Sequence[float],
    history_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Write the confusion matrix, ROC curve, and training curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # noqa: BLE001 - plots are optional
        return {"error": f"matplotlib unavailable: {error}"}

    written: Dict[str, str] = {}
    matrix = metrics.confusion_matrix

    # -- confusion matrix --------------------------------------------------
    grid = np.array(
        [
            [matrix.true_negative, matrix.false_positive],
            [matrix.false_negative, matrix.true_positive],
        ]
    )
    figure, axes = plt.subplots(figsize=(4.6, 4.2))
    axes.imshow(grid, cmap="Blues")
    axes.set_xticks([0, 1], labels=["NonFight", "Fight"])
    axes.set_yticks([0, 1], labels=["NonFight", "Fight"])
    axes.set_xlabel("predicted")
    axes.set_ylabel("actual")
    axes.set_title(f"Confusion matrix (n={matrix.total})")
    threshold = grid.max() / 2 if grid.max() else 0
    for row in range(2):
        for column in range(2):
            axes.text(
                column,
                row,
                str(grid[row, column]),
                ha="center",
                va="center",
                color="white" if grid[row, column] > threshold else "black",
            )
    figure.tight_layout()
    path = output_dir / "confusion_matrix.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written["confusion_matrix"] = str(path)

    # -- ROC curve ---------------------------------------------------------
    fpr, tpr = _roc_points(truths, scores)
    if fpr:
        figure, axes = plt.subplots(figsize=(4.6, 4.2))
        label = "n/a" if metrics.roc_auc is None else f"{metrics.roc_auc:.4f}"
        axes.plot(fpr, tpr, label=f"AUC = {label}")
        axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
        axes.set_xlabel("false positive rate")
        axes.set_ylabel("true positive rate")
        axes.set_title("ROC curve")
        axes.legend(loc="lower right")
        figure.tight_layout()
        path = output_dir / "roc_curve.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written["roc_curve"] = str(path)

    # -- training curves ---------------------------------------------------
    if history_path and Path(history_path).is_file():
        try:
            document = json.loads(Path(history_path).read_text(encoding="utf-8"))
            epochs = document.get("epochs", [])
            if epochs:
                figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
                index = [record["epoch"] for record in epochs]
                axes[0].plot(index, [r["train_loss"] for r in epochs], label="train")
                axes[0].plot(index, [r["eval_loss"] for r in epochs], label="eval")
                axes[0].set_xlabel("epoch")
                axes[0].set_ylabel("loss")
                axes[0].set_title("Loss")
                axes[0].legend()
                axes[1].plot(
                    index,
                    [r["metrics"].get("accuracy") for r in epochs],
                    label="accuracy",
                )
                auc = [r["metrics"].get("roc_auc") for r in epochs]
                if any(value is not None for value in auc):
                    axes[1].plot(index, auc, label="roc_auc")
                axes[1].set_xlabel("epoch")
                axes[1].set_title("Evaluation metrics")
                axes[1].legend()
                figure.tight_layout()
                path = output_dir / "training_curves.png"
                figure.savefig(path, dpi=150)
                plt.close(figure)
                written["training_curves"] = str(path)
        except Exception as error:  # noqa: BLE001 - a plot never fails the run
            written["training_curves_error"] = str(error)

    return written


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint: Path,
    dataset_root: Path,
    output_dir: Path,
    split: str = "primary",
    device: str = "cuda",
    batch_size: int = 8,
    num_workers: int = 2,
    threshold: float = 0.5,
    seed: int = 42,
    history_path: Optional[Path] = None,
    log=print,
) -> dict:
    """Evaluate one checkpoint and write every artifact. Returns the record."""
    if split not in SPLIT_BUILDERS:
        raise EvaluationError(
            f"Unknown split {split!r}; expected one of {sorted(SPLIT_BUILDERS)}."
        )

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise EvaluationError(f"Checkpoint not found: {checkpoint}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_device(device)
    # num_classes MUST be 2 here: the config default is Kinetics-400's 400,
    # and a 400-way head would refuse our 2-class checkpoint outright.
    wrapper = R3D18TemporalModel(
        TemporalModelConfig(
            num_classes=len(CANONICAL_LABELS),
            device=str(resolved),
            pretrained_backbone=False,
            class_labels=CANONICAL_LABELS,
        )
    )
    wrapper.load_checkpoint(checkpoint)
    if not wrapper.is_task_specific:
        raise EvaluationError(
            "Loaded weights are not task-specific; refusing to report metrics."
        )
    log(f"checkpoint   : {checkpoint}")
    log(f"provenance   : {wrapper.provenance}")
    log(f"device       : {resolved}")

    dataset = SPLIT_BUILDERS[split](
        Path(dataset_root),
        DatasetConfig(
            sampling=SamplingConfig(),
            preprocessing=PreprocessingConfig(),
            training=False,
            seed=seed,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    log(f"split        : {split} ({len(dataset)} clips)")

    started = time.perf_counter()
    truths, scores, forward_seconds = score_dataset(wrapper.model, loader, resolved)
    wall_seconds = time.perf_counter() - started

    metrics = binary_metrics(truths, scores, threshold=threshold)
    clips = max(len(truths), 1)
    timing = {
        "clips": len(truths),
        "forward_seconds_total": forward_seconds,
        "forward_ms_per_clip": 1000.0 * forward_seconds / clips,
        "wall_seconds_total": wall_seconds,
        "wall_ms_per_clip": 1000.0 * wall_seconds / clips,
        "batch_size": batch_size,
    }

    record = {
        "checkpoint": str(checkpoint),
        "provenance": wrapper.provenance,
        "dataset_root": str(dataset_root),
        "split": split,
        "split_clips": len(dataset),
        "decision_threshold": threshold,
        "metrics": metrics.as_dict(),
        "timing": timing,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(resolved),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
    }

    # -- artifacts ---------------------------------------------------------
    (output_dir / "metrics.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    flat = {
        "split": split,
        "clips": len(dataset),
        "threshold": threshold,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "true_negative": metrics.confusion_matrix.true_negative,
        "false_positive": metrics.confusion_matrix.false_positive,
        "false_negative": metrics.confusion_matrix.false_negative,
        "true_positive": metrics.confusion_matrix.true_positive,
        "forward_ms_per_clip": timing["forward_ms_per_clip"],
    }
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)

    report = classification_report_text(metrics)
    (output_dir / "classification_report.txt").write_text(report + "\n", encoding="utf-8")

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "true_label", "fight_probability", "predicted_label"])
        for index, (truth, score) in enumerate(zip(truths, scores)):
            predicted = 1 if score >= threshold else 0
            writer.writerow(
                [
                    index,
                    CANONICAL_LABELS[truth] if truth < len(CANONICAL_LABELS) else truth,
                    f"{score:.6f}",
                    CANONICAL_LABELS[predicted]
                    if predicted < len(CANONICAL_LABELS)
                    else predicted,
                ]
            )

    record["plots"] = write_plots(output_dir, metrics, truths, scores, history_path)

    (output_dir / "metrics.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    log("")
    log(report)
    log("")
    log(metrics.summary())
    log(f"forward pass : {timing['forward_ms_per_clip']:.2f} ms/clip")
    log(f"artifacts    : {output_dir}")
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True, help="Extracted RWF-2000 root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        default="primary",
        choices=sorted(SPLIT_BUILDERS),
        help="primary = 394 leak-free clips (default); official = all 400.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history", default=None, help="training_history.json")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    try:
        evaluate_checkpoint(
            checkpoint=Path(args.checkpoint),
            dataset_root=Path(args.root),
            output_dir=Path(args.output_dir),
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            seed=args.seed,
            history_path=Path(args.history) if args.history else None,
        )
    except (EvaluationError, TemporalModelError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
