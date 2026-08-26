"""Load a saved checkpoint in a FRESH process and classify real clips.

This answers one question a training-time metric cannot: does the
checkpoint that was written to disk actually load and predict on its own,
in a process that never saw the training run?

It imports nothing from ``temporal_training``. Clips come from
``build_primary_evaluation_dataset``, so every clip shown here is one the
model was never trained on, and the decode/sample/preprocess path is the
same one the trainer used rather than a re-implementation of it.

    python tools/classify_sample_clips.py \\
        --checkpoint models/temporal_violence/best.pt \\
        --root /kaggle/temp/rwf2000/RWF-2000 --count 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # noqa: E402

from rwf2000_config import CANONICAL_LABELS  # noqa: E402
from rwf2000_dataset import (  # noqa: E402
    DatasetConfig,
    build_primary_evaluation_dataset,
)
from temporal_model import (  # noqa: E402
    R3D18TemporalModel,
    TemporalModelConfig,
    TemporalModelError,
)


def balanced_indices(dataset, count: int, seed: int) -> List[int]:
    """Return roughly class-balanced item indices from the dataset."""
    fight_index = CANONICAL_LABELS.index("Fight")
    fight = [i for i, label in enumerate(dataset.labels) if label == fight_index]
    other = [i for i, label in enumerate(dataset.labels) if label != fight_index]

    rng = random.Random(seed)
    rng.shuffle(fight)
    rng.shuffle(other)
    half = max(count // 2, 1)
    chosen = fight[:half] + other[: max(count - half, 0)]
    rng.shuffle(chosen)
    return chosen[:count]


def main(argv: Optional[List[str]] = None) -> int:
    """Classify a few real clips and report whether each prediction is right."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    try:
        # num_classes MUST be 2: the config default is Kinetics-400's 400.
        wrapper = R3D18TemporalModel(
            TemporalModelConfig(
                num_classes=len(CANONICAL_LABELS),
                device=args.device,
                pretrained_backbone=False,
                class_labels=CANONICAL_LABELS,
            )
        )
        wrapper.load_checkpoint(Path(args.checkpoint))
    except TemporalModelError as error:
        print(f"FAILED to load checkpoint: {error}", file=sys.stderr)
        return 2

    if not wrapper.is_task_specific:
        print("FAILED: loaded weights are not task-specific.", file=sys.stderr)
        return 2

    print(f"checkpoint : {args.checkpoint}")
    print(f"provenance : {wrapper.provenance}")
    print(f"device     : {args.device}")

    dataset = build_primary_evaluation_dataset(
        Path(args.root), DatasetConfig(training=False, seed=args.seed)
    )
    print(f"pool       : {len(dataset)} leak-free held-out clips")

    records = []
    correct = 0
    for item in balanced_indices(dataset, args.count, args.seed):
        clip, truth_index = dataset[item]
        prediction = wrapper.predict(clip.unsqueeze(0))

        truth = CANONICAL_LABELS[truth_index]
        predicted = (
            prediction.predicted_label
            or CANONICAL_LABELS[prediction.predicted_index]
        )
        confidence = float(
            prediction.probabilities.flatten()[prediction.predicted_index]
        )
        hit = predicted == truth
        correct += int(hit)
        records.append(
            {
                "clip": dataset.relative_paths[item],
                "true_label": truth,
                "predicted_label": predicted,
                "confidence": confidence,
                "correct": hit,
            }
        )
        name = Path(dataset.relative_paths[item]).name
        print(
            f"  {'OK  ' if hit else 'MISS'}  true={truth:<8} "
            f"pred={predicted:<8} conf={confidence:.4f}  {name[:44]}"
        )

    print(
        f"\n{correct}/{len(records)} correct on this sample "
        f"-- a spot check that the checkpoint loads and predicts, not a metric."
    )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "provenance": wrapper.provenance,
                    "sampled_clips": len(records),
                    "correct": correct,
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")

    # A load failure is an error; a wrong prediction on a spot check is not.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
