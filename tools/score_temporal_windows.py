#!/usr/bin/env python3
"""
tools/score_temporal_windows.py

Sliding-window violence scoring: reconstructs, in this repository, the step
that produced ``temporal_risk/{carve,primary}_window_scores.csv``.

WHY THIS EXISTS
---------------
``temporal_risk/window_scoring_manifest.json`` names its producer
``score_temporal_windows``. No such file existed in this repository -- it lived
only in Kaggle notebook ``notebookbce85b4c2b`` -- so the two CSVs the entire
demo replays could not be regenerated, extended to a new clip, or checked.
This script closes that gap by composing the project's own modules:

    rwf2000_dataset.decode_frames   read the clip's frames
    temporal_clip_buffer            cut them into 16-frame / stride-8 windows
    temporal_preprocessing          frames -> normalized (1,C,T,H,W) tensor
    temporal_model                  R3D-18 forward pass
    carved_validation / rwf2000_splits   which clips belong to which split

Nothing here is a re-implementation of notebook logic: every stage is the
module the rest of the project already uses.

WINDOW GEOMETRY IS VERIFIABLE WITHOUT A CHECKPOINT
--------------------------------------------------
``--verify-geometry`` runs the windowing alone and checks that the frame
bounds it produces match those recorded in a committed CSV, with no model and
no weights. That closes the audit's open question about whether the recorded
``first_frame``/``last_frame`` columns actually correspond to this project's
``TemporalClipBuffer(clip_length=16, stride=8)``.

SCORING REQUIRES THE CHECKPOINT
-------------------------------
Producing probabilities requires ``best.pt`` (132.74 MB), which is not on this
machine. Without ``--checkpoint`` the script refuses to emit a score rather
than substituting an untrained model's output -- a randomly initialised or
Kinetics-400 head produces numbers, and those numbers would be meaningless.

Usage:
    # No checkpoint needed: check the windowing matches the committed CSV
    python tools/score_temporal_windows.py --verify-geometry \\
        --against temporal_risk/primary_window_scores.csv

    # Full scoring, once a checkpoint is available
    python tools/score_temporal_windows.py --split primary \\
        --checkpoint models/temporal_violence/best.pt \\
        --output temporal_risk/primary_window_scores.regenerated.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from carved_validation import (  # noqa: E402
    CarvedValidationError,
    clip_label,
    load_carved_split,
)
from rwf2000_dataset import decode_frames  # noqa: E402
from rwf2000_splits import clip_path  # noqa: E402
from temporal_clip_buffer import TemporalClipBuffer  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "data" / "rwf2000" / "RWF-2000"
DEFAULT_CLIP_LENGTH = 16
DEFAULT_STRIDE = 8
POSITIVE_CLASS_INDEX = 1  # rwf2000_config.LABEL_TO_INDEX["Fight"]

CSV_FIELDS = (
    "split",
    "clip",
    "window_index",
    "first_frame",
    "last_frame",
    "true_label",
    "fight_probability",
)


class WindowScoringError(RuntimeError):
    """Raised when scoring cannot be performed honestly."""


def window_bounds(
    total_frames: int,
    clip_length: int = DEFAULT_CLIP_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> Tuple[Tuple[int, int], ...]:
    """Return the ``(first_frame, last_frame)`` of every complete window.

    Driven by the project's own :class:`TemporalClipBuffer` rather than by
    arithmetic, so this cannot drift away from what the buffer would actually
    hand a model at inference time. ``last_frame`` is inclusive, matching the
    recorded CSV columns.

    A clip shorter than ``clip_length`` yields no windows at all -- the buffer
    never reports ready, and a partial window is not silently padded.
    """
    buffer = TemporalClipBuffer(clip_length=clip_length, stride=stride)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    bounds: List[Tuple[int, int]] = []
    for index in range(int(total_frames)):
        buffer.add_frame(frame, frame_number=index)
        if buffer.is_ready():
            numbers = buffer.get_clip().frame_numbers
            bounds.append((numbers[0], numbers[-1]))
    return tuple(bounds)


def read_recorded_bounds(csv_path: Path) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """Return ``{clip: ((first,last), ...)}`` from a committed scores CSV."""
    per_clip: Dict[str, List[Tuple[int, int, int]]] = {}
    with open(csv_path, newline="", encoding="utf-8", errors="surrogateescape") as handle:
        for row in csv.DictReader(handle):
            per_clip.setdefault(row["clip"], []).append(
                (int(row["window_index"]), int(row["first_frame"]), int(row["last_frame"]))
            )
    return {
        clip: tuple((first, last) for _, first, last in sorted(rows))
        for clip, rows in per_clip.items()
    }


def verify_geometry(
    csv_path: Path,
    source_frames: int,
    clip_length: int = DEFAULT_CLIP_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> dict:
    """Check the committed CSV's window bounds against this project's buffer.

    Needs no model, no weights and no video files -- it compares recorded
    integers against integers the buffer generates. Returns a report; the
    caller decides whether a mismatch is fatal.
    """
    expected = window_bounds(source_frames, clip_length, stride)
    recorded = read_recorded_bounds(Path(csv_path))
    mismatched = sorted(clip for clip, bounds in recorded.items() if bounds != expected)
    distinct = {bounds for bounds in recorded.values()}
    return {
        "csv": str(csv_path),
        "clips_checked": len(recorded),
        "source_frames_assumed": source_frames,
        "clip_length": clip_length,
        "stride": stride,
        "expected_windows_per_clip": len(expected),
        "expected_bounds": [list(bound) for bound in expected],
        "distinct_bound_sequences_in_csv": len(distinct),
        "clips_matching": len(recorded) - len(mismatched),
        "clips_mismatched": len(mismatched),
        "first_mismatches": mismatched[:5],
        "match": not mismatched,
    }


def _load_model(checkpoint: Path, device: str):
    """Load the fine-tuned 2-class R3D-18, refusing anything non-task-specific."""
    from temporal_model import R3D18TemporalModel, TemporalModelConfig  # local import: torch

    model = R3D18TemporalModel(
        TemporalModelConfig(
            num_classes=2,
            device=device,
            pretrained_backbone=False,
            checkpoint_path=Path(checkpoint),
            class_labels=("NonFight", "Fight"),
        )
    )
    if not model.is_task_specific:
        raise WindowScoringError(
            f"Model provenance is {model.provenance!r}, which is not a "
            "task-specific checkpoint. Refusing to emit violence probabilities "
            "from weights that were never trained for this task."
        )
    return model


def select_clips(root: Path, split: str) -> List[str]:
    """Return the clips of one named split, using the project's own splits."""
    carved = load_carved_split(root)
    if split == "primary":
        return list(carved.primary_evaluation)
    if split == "carve":
        return list(carved.carve_validation)
    if split == "train_remainder":
        return list(carved.train_remainder)
    raise WindowScoringError(
        f"Unknown split {split!r}; expected primary, carve or train_remainder."
    )


def score_clips(
    root: Path,
    clips: Sequence[str],
    split_name: str,
    checkpoint: Path,
    device: str = "cpu",
    clip_length: int = DEFAULT_CLIP_LENGTH,
    stride: int = DEFAULT_STRIDE,
    log=print,
) -> Tuple[List[dict], dict]:
    """Score every complete window of every clip. Returns (rows, manifest)."""
    import torch  # local import so --verify-geometry needs no torch
    from temporal_preprocessing import ClipPreprocessor, PreprocessingConfig

    model = _load_model(checkpoint, device)
    preprocessor = ClipPreprocessor(PreprocessingConfig())

    rows: List[dict] = []
    windows_per_clip: set = set()
    empty: List[str] = []
    forward_seconds = 0.0
    started = time.perf_counter()

    for position, clip in enumerate(clips, start=1):
        path = clip_path(root, clip)
        import cv2

        capture = cv2.VideoCapture(path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        bounds = window_bounds(total_frames, clip_length, stride)
        if not bounds:
            empty.append(clip)
            continue
        windows_per_clip.add(len(bounds))

        indices = sorted({index for first, last in bounds for index in range(first, last + 1)})
        frames = decode_frames(path, indices)
        by_index = {index: frames[offset] for offset, index in enumerate(indices)}

        for window_index, (first, last) in enumerate(bounds):
            stack = np.stack([by_index[i] for i in range(first, last + 1)], axis=0)
            tensor = preprocessor.preprocess(stack)
            forward_started = time.perf_counter()
            with torch.no_grad():
                probabilities = model.predict(tensor).probabilities
            forward_seconds += time.perf_counter() - forward_started
            rows.append(
                {
                    "split": split_name,
                    "clip": clip,
                    "window_index": window_index,
                    "first_frame": first,
                    "last_frame": last,
                    "true_label": clip_label(clip),
                    "fight_probability": f"{float(probabilities[0, POSITIVE_CLASS_INDEX]):.6f}",
                }
            )
        if position % 25 == 0:
            log(f"  scored {position}/{len(clips)} clips")

    manifest = {
        "tool": "score_temporal_windows",
        "mode": "inference_only",
        "performs_training": False,
        "performs_threshold_selection": False,
        "performs_aggregation_selection": False,
        "modifies_checkpoint": False,
        "checkpoint": str(checkpoint),
        "model_provenance": model.provenance,
        "dataset_root": str(root),
        "clip_length": clip_length,
        "stride": stride,
        "split": split_name,
        "clips": len(clips),
        "windows": len(rows),
        "windows_per_clip_observed": sorted(windows_per_clip),
        "clips_with_no_complete_window": empty,
        "class_labels": ["NonFight", "Fight"],
        "positive_class": "Fight",
        "wall_seconds": time.perf_counter() - started,
        "forward_seconds": forward_seconds,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
        },
    }
    return rows, manifest


def write_csv(rows: Iterable[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dataset root.")
    parser.add_argument("--split", choices=("primary", "carve", "train_remainder"),
                        default="primary")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Fine-tuned 2-class R3D-18 checkpoint. Required to score.")
    parser.add_argument("--output", type=Path, default=None, help="Destination CSV.")
    parser.add_argument("--manifest", type=Path, default=None, help="Destination manifest JSON.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip-length", type=int, default=DEFAULT_CLIP_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N clips.")
    parser.add_argument("--verify-geometry", action="store_true",
                        help="Check window bounds against a committed CSV and exit. "
                             "Needs no checkpoint, no torch and no video files.")
    parser.add_argument("--against", type=Path,
                        default=REPO_ROOT / "temporal_risk" / "primary_window_scores.csv",
                        help="CSV to verify geometry against.")
    parser.add_argument("--source-frames", type=int, default=150,
                        help="Frames per source clip (RWF-2000 is 150, verified over all 2000).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.verify_geometry:
        report = verify_geometry(args.against, args.source_frames, args.clip_length, args.stride)
        print(json.dumps(report, indent=2))
        if not report["match"]:
            print("\nGEOMETRY MISMATCH: the committed CSV's window bounds are not what "
                  "TemporalClipBuffer produces with these settings.", file=sys.stderr)
            return 1
        print(f"\nOK: all {report['clips_checked']} clips in {args.against.name} use exactly "
              f"the {report['expected_windows_per_clip']} window bounds that "
              f"TemporalClipBuffer(clip_length={args.clip_length}, stride={args.stride}) "
              f"produces for a {args.source_frames}-frame clip.")
        return 0

    if args.checkpoint is None:
        print(
            "ERROR: --checkpoint is required to produce scores.\n\n"
            "The fine-tuned R3D-18 checkpoint (best.pt) is not present in this\n"
            "repository. This script will not substitute a randomly initialised or\n"
            "Kinetics-400 model: those produce numbers, but the numbers would not be\n"
            "violence probabilities. See docs/EXPERIMENT_REPRODUCIBILITY.md.\n\n"
            "To check the windowing without a checkpoint:\n"
            "    python tools/score_temporal_windows.py --verify-geometry",
            file=sys.stderr,
        )
        return 2

    try:
        clips = select_clips(args.root, args.split)
    except CarvedValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.limit is not None:
        clips = clips[: args.limit]

    print(f"Scoring {len(clips)} clips from split {args.split!r} with {args.checkpoint}")
    rows, manifest = score_clips(
        args.root, clips, args.split, args.checkpoint,
        device=args.device, clip_length=args.clip_length, stride=args.stride,
    )
    output = args.output or (REPO_ROOT / "temporal_risk" / f"{args.split}_window_scores.regenerated.csv")
    write_csv(rows, output)
    manifest_path = args.manifest or output.with_suffix(".manifest.json")
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({len(rows)} rows)")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
