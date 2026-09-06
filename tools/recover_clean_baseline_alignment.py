#!/usr/bin/env python3
"""
tools/recover_clean_baseline_alignment.py

Establish, empirically, which clip each row of the recovered clean-baseline
``predictions.csv`` refers to.

THE PROBLEM
-----------
The recovered file carries ``index,true_label,fight_probability,predicted_label``
-- no clip identifier. The sliding-window scores carry full clip paths. A paired
comparison between the two needs to know which row is which clip, and guessing
from row order is not evidence.

THE TEST
--------
``src/evaluate_temporal_baseline.py`` writes exactly that header, iterates the
primary evaluation dataset with ``shuffle=False``, and that dataset is built
from ``rwf2000_splits`` sorted relative paths. So the hypothesis is:

    recovered row i  <-->  sorted_primary_clips[i]

That hypothesis is testable now that the checkpoint has been recovered. This
script re-runs whole-clip inference locally, in known clip order, with the same
checkpoint, sampling and preprocessing, and compares the resulting probability
sequence against the recovered one position by position.

Agreement across 394 clips at six decimals would be astronomically unlikely
under a wrong permutation, so a close match is strong evidence for the
hypothesis rather than an assumption of it. A mismatch refutes it, and the
script says so instead of forcing an alignment.

WHAT THIS IS NOT
----------------
It does not retrain, tune, or alter any evaluation artifact. It reads the
committed dataset and the recovered checkpoint and writes one new JSON under
``outputs/``. The recovered ``predictions.csv`` is opened read-only.

Usage:
    python tools/recover_clean_baseline_alignment.py \\
        --clean-predictions "C:/Users/.../Downloads/predictions.csv" \\
        --checkpoint models/temporal_violence/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import file_sha256  # noqa: E402
from rwf2000_config import SamplingConfig  # noqa: E402
from rwf2000_dataset import DatasetConfig, build_primary_evaluation_dataset  # noqa: E402
from temporal_preprocessing import PreprocessingConfig  # noqa: E402
from temporal_runtime import build_violence_model  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "data" / "rwf2000" / "RWF-2000"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "temporal_violence" / "best.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "temporal_risk" / "paired_clean_vs_sliding_window"
DEFAULT_OUTPUT = OUTPUT_DIR / "clean_baseline_alignment.json"

CLEAN_PREDICTIONS_SHA256 = "ee6c26994a29042f630d262754381a28b3d0cc27cc03b42e9a5c818da1a58ce3"
# The recovered file stores six decimals, and this machine runs on CPU while the
# original ran on a T4. Both can shift the last places. This bound is a
# reporting threshold for "these are the same prediction", not a tuned value --
# see the observed maximum in the emitted report.
MATCH_TOLERANCE = 1e-3


def load_clean_predictions(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    import torch

    digest = file_sha256(args.clean_predictions)
    if digest != CLEAN_PREDICTIONS_SHA256:
        print(
            f"ERROR: {args.clean_predictions} has sha256 {digest}, expected the "
            f"clean-baseline {CLEAN_PREDICTIONS_SHA256}. Refusing to proceed with "
            "a file that is not the recovered artifact.",
            file=sys.stderr,
        )
        return 1

    recovered = load_clean_predictions(args.clean_predictions)
    dataset = build_primary_evaluation_dataset(
        args.root,
        DatasetConfig(
            sampling=SamplingConfig(),
            preprocessing=PreprocessingConfig(),
            training=False,
            seed=42,
        ),
    )
    clips = list(dataset.relative_paths)
    if len(clips) != len(recovered):
        print(
            f"ERROR: dataset has {len(clips)} clips but the recovered file has "
            f"{len(recovered)} rows; they cannot be aligned.",
            file=sys.stderr,
        )
        return 1

    model, verdict = build_violence_model(args.checkpoint, args.device)
    print(f"checkpoint : {args.checkpoint}")
    print(f"provenance : {model.provenance}")
    print(f"clips      : {len(clips)} (sorted primary evaluation order)")
    print("re-running whole-clip inference locally; this takes ~15 minutes on CPU")

    rows: List[dict] = []
    started = time.perf_counter()
    for index, clip in enumerate(clips):
        tensor, label = dataset[index]
        with torch.no_grad():
            prediction = model.predict(tensor.unsqueeze(0))
        local = float(prediction.probabilities[0, 1])
        record = recovered[index]
        rows.append(
            {
                "index": index,
                "clip": clip,
                "dataset_label": "Fight" if label == 1 else "NonFight",
                "recovered_label": record["true_label"],
                "recovered_probability": float(record["fight_probability"]),
                "local_probability": local,
                "abs_difference": abs(local - float(record["fight_probability"])),
            }
        )
        if (index + 1) % 25 == 0:
            elapsed = time.perf_counter() - started
            print(f"  {index + 1}/{len(clips)}  ({elapsed:.0f}s elapsed)")

    differences = [row["abs_difference"] for row in rows]
    label_mismatches = [
        row for row in rows if row["dataset_label"] != row["recovered_label"]
    ]
    over_tolerance = [row for row in rows if row["abs_difference"] > MATCH_TOLERANCE]

    report = {
        "tool": "recover_clean_baseline_alignment",
        "hypothesis": "recovered row i corresponds to sorted primary clip i",
        "basis": (
            "evaluate_temporal_baseline.py writes this exact header, iterates the "
            "primary evaluation dataset with shuffle=False, and that dataset is "
            "built from rwf2000_splits sorted relative paths"
        ),
        "clean_predictions": str(args.clean_predictions),
        "clean_predictions_sha256": digest,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "model_provenance": model.provenance,
        "device": args.device,
        "clips": len(clips),
        "match_tolerance": MATCH_TOLERANCE,
        "label_mismatches": len(label_mismatches),
        "rows_over_tolerance": len(over_tolerance),
        "max_abs_difference": max(differences),
        "mean_abs_difference": sum(differences) / len(differences),
        "exact_matches_at_six_decimals": sum(
            1 for row in rows
            if round(row["local_probability"], 6) == round(row["recovered_probability"], 6)
        ),
        "alignment_supported": not label_mismatches and not over_tolerance,
        "worst_rows": sorted(rows, key=lambda r: -r["abs_difference"])[:10],
        "wall_seconds": time.perf_counter() - started,
        "per_clip": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"label mismatches            : {report['label_mismatches']}")
    print(f"rows over tolerance         : {report['rows_over_tolerance']}")
    print(f"exact at six decimals       : {report['exact_matches_at_six_decimals']}/{len(clips)}")
    print(f"max abs difference          : {report['max_abs_difference']:.3e}")
    print(f"mean abs difference         : {report['mean_abs_difference']:.3e}")
    print(f"ALIGNMENT SUPPORTED         : {report['alignment_supported']}")
    print(f"\nWrote {args.output}")
    return 0 if report["alignment_supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
