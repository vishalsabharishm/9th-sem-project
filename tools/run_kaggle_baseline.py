"""Drive the whole RWF-2000 baseline phase inside one Kaggle run.

Sequence, each stage gating the next:

    1. restore the extracted dataset from the snapshot tar
    2. run the project test suite (the torch-dependent tests included --
       they cannot run on the CPU-only dev box, so this is where they run)
    3. CPU smoke test of the training wiring on a handful of clips
    4. real fine-tuning on the GPU
    5. evaluation of the best checkpoint on the 394 leak-free clips
    6. a fresh-process checkpoint load + spot classification

A stage that fails stops the run, so a long GPU stage never starts on top
of a broken one. Every artifact lands under ``--work``/outputs so a Kaggle
commit persists it.

    python tools/run_kaggle_baseline.py --stage all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

BANNER = "=" * 74


class StageError(RuntimeError):
    """Raised when a stage fails and the run must stop."""


def announce(title: str) -> None:
    """Print a section header."""
    print(f"\n{BANNER}\n{title}\n{BANNER}", flush=True)


def run_command(
    args: Sequence[str],
    cwd: Path,
    title: str,
    allow_failure: bool = False,
) -> int:
    """Run a subprocess with inherited stdio and return its exit code.

    Output is deliberately NOT captured: the training stage runs for
    hours, and a captured pipe would hide every epoch line until it
    finished. Inheriting stdio streams progress into the notebook live.
    """
    announce(title)
    print("$ " + " ".join(str(item) for item in args), flush=True)
    sys.stdout.flush()
    started = time.perf_counter()
    completed = subprocess.run(
        [str(item) for item in args],
        cwd=str(cwd),
        stdout=None,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    print(f"[exit {completed.returncode} after {elapsed:.1f}s]", flush=True)
    if completed.returncode != 0 and not allow_failure:
        raise StageError(f"{title} failed with exit code {completed.returncode}")
    return completed.returncode


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_restore(args) -> Path:
    """Copy the code somewhere writable and untar the dataset snapshot."""
    announce("STAGE 1 - restore code and dataset")

    project = Path(args.work) / "project"
    if project.exists():
        shutil.rmtree(project)
    shutil.copytree(args.code, project)
    (project / "models").mkdir(parents=True, exist_ok=True)
    (project / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"code       : {args.code} -> {project}")

    destination = Path(args.data)
    root = Path(args.root) if args.root else None

    if root and root.is_dir():
        print(f"dataset    : reusing existing tree at {root}")
    else:
        # Prefer a prepared snapshot -- untarring is minutes, re-extracting
        # the 13-part 7z archive is twenty. Fall back to the archive only
        # when no snapshot has been produced yet.
        candidates = [args.tar] if args.tar else sorted(
            glob.glob("/kaggle/input/**/rwf2000_extracted.tar", recursive=True)
        )
        destination.mkdir(parents=True, exist_ok=True)
        if candidates:
            print(f"snapshot   : {candidates[0]}")
            started = time.perf_counter()
            code = subprocess.run(
                ["tar", "-xf", candidates[0], "-C", str(destination)]
            ).returncode
            if code != 0:
                raise StageError(f"tar extraction failed with exit code {code}")
            print(f"untarred in {time.perf_counter() - started:.1f}s")
            root = destination / "RWF-2000"
        else:
            print("snapshot   : none found -- extracting from the 7z archive")
            target = destination / "rwf2000"
            run_command(
                [
                    sys.executable,
                    "src/rwf2000_kaggle.py",
                    "--target",
                    str(target),
                    "--metadata-dir",
                    str(Path(args.work) / "rwf2000_metadata"),
                    "--validate",
                ],
                cwd=project,
                title="STAGE 1a - extracting RWF-2000 from the 7z archive",
            )
            root = target / "RWF-2000"

    clips = glob.glob(str(root / "**" / "*.avi"), recursive=True)
    print(f"dataset root : {root}")
    print(f"clips        : {len(clips)}")
    if len(clips) != args.expect_clips:
        raise StageError(
            f"expected {args.expect_clips} clips, found {len(clips)} -- "
            "the snapshot is incomplete, refusing to train on it."
        )

    import torch

    print(f"torch        : {torch.__version__}")
    print(f"cuda         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu          : {torch.cuda.get_device_name(0)}")
    return root


def stage_snapshot(args, root: Path) -> None:
    """Archive the extracted tree so later runs untar instead of re-extracting.

    Written into the persisted working directory as a single tar, because
    a Kaggle notebook output is far happier with one 12 GB file than with
    2000 small ones, and because /kaggle/working must hold the archive
    without also holding the tree (hence extracting to /kaggle/temp).
    """
    target = Path(args.work) / "rwf2000_extracted.tar"
    if target.exists():
        print(f"snapshot already present: {target}")
        return
    announce("STAGE 1b - writing the reusable dataset snapshot")
    started = time.perf_counter()
    code = subprocess.run(
        ["tar", "-cf", str(target), "-C", str(Path(root).parent), Path(root).name]
    ).returncode
    if code != 0:
        raise StageError(f"tar failed with exit code {code}")
    print(
        f"wrote {target} ({target.stat().st_size:,} bytes) "
        f"in {time.perf_counter() - started:.1f}s"
    )


def stage_tests(args, project: Path) -> None:
    """Run the whole test suite, including the torch-dependent modules."""
    run_command(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
        cwd=project,
        title="STAGE 2 - project test suite",
    )


def stage_smoke(args, project: Path, root: Path) -> None:
    """Prove the training wiring end to end before spending GPU time."""
    run_command(
        [
            sys.executable,
            "src/temporal_training.py",
            "--validate-pipeline",
            "--root",
            str(root),
            "--checkpoint-dir",
            str(project / "outputs" / "smoke" / "checkpoints"),
            "--history-dir",
            str(project / "outputs" / "smoke"),
        ],
        cwd=project,
        title="STAGE 3 - CPU smoke test (untrained backbone, few clips)",
    )


def stage_train(args, project: Path, root: Path) -> Path:
    """Run the real fine-tuning."""
    checkpoints = project / "outputs" / "training" / "checkpoints"
    history = project / "outputs" / "training"
    command = [
        sys.executable,
        "src/temporal_training.py",
        "--train",
        "--root",
        str(root),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--checkpoint-dir",
        str(checkpoints),
        "--history-dir",
        str(history),
    ]
    if args.amp:
        command.append("--amp")
    run_command(command, cwd=project, title="STAGE 4 - fine-tuning")

    best = checkpoints / "best.pt"
    if not best.is_file():
        raise StageError(f"training produced no best checkpoint at {best}")
    print(f"best checkpoint: {best} ({best.stat().st_size:,} bytes)")
    return best


def stage_evaluate(args, project: Path, root: Path, checkpoint: Path) -> None:
    """Evaluate the best checkpoint and write metrics and plots."""
    results = project / "outputs" / "evaluation"
    history = project / "outputs" / "training" / "training_history.json"
    run_command(
        [
            sys.executable,
            "src/evaluate_temporal_baseline.py",
            "--checkpoint",
            str(checkpoint),
            "--root",
            str(root),
            "--output-dir",
            str(results),
            "--split",
            "primary",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--history",
            str(history),
        ],
        cwd=project,
        title="STAGE 5 - evaluation on the 394 leak-free clips",
    )


def stage_spotcheck(args, project: Path, root: Path, checkpoint: Path) -> None:
    """Load the checkpoint in a fresh process and classify real clips."""
    run_command(
        [
            sys.executable,
            "tools/classify_sample_clips.py",
            "--checkpoint",
            str(checkpoint),
            "--root",
            str(root),
            "--count",
            str(args.spotcheck_clips),
            "--device",
            args.device,
            "--json-out",
            str(project / "outputs" / "evaluation" / "spot_check.json"),
        ],
        cwd=project,
        title="STAGE 6 - fresh-process checkpoint load and classification",
    )


def collect_outputs(args, project: Path) -> None:
    """Copy the small artifacts to a stable, persisted location."""
    announce("COLLECTING ARTIFACTS")
    destination = Path(args.work) / "baseline_results"
    destination.mkdir(parents=True, exist_ok=True)

    for source in (project / "outputs").rglob("*"):
        if not source.is_file():
            continue
        if source.suffix == ".pt" and source.name != "best.pt":
            continue  # keep only the best weights among the large files
        relative = source.relative_to(project / "outputs")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for path in sorted(destination.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(destination)}  ({path.stat().st_size:,} bytes)")
    print(f"\nartifacts under {destination}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "verify", "train"),
        help="verify = restore + tests + smoke only (no GPU time spent on "
        "training); train = the full sequence.",
    )
    parser.add_argument("--code", default="/kaggle/input/rwf2000-project-code")
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--data", default="/kaggle/temp/data")
    parser.add_argument("--tar", default=None)
    parser.add_argument("--root", default=None, help="Skip untar, use this tree.")
    parser.add_argument("--expect-clips", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--spotcheck-clips", type=int, default=8)
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Also write rwf2000_extracted.tar into the working directory "
        "so later runs can untar instead of re-extracting.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the requested stages, stopping at the first failure."""
    args = build_arg_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        root = stage_restore(args)
        project = Path(args.work) / "project"
        if args.snapshot:
            stage_snapshot(args, root)
        stage_tests(args, project)
        stage_smoke(args, project, root)

        if args.stage == "verify":
            announce("VERIFY STAGES COMPLETE - training not started")
            return 0

        checkpoint = stage_train(args, project, root)
        stage_evaluate(args, project, root, checkpoint)
        stage_spotcheck(args, project, root, checkpoint)
        collect_outputs(args, project)
    except StageError as error:
        print(f"\nSTOPPED: {error}", file=sys.stderr, flush=True)
        return 1

    announce(f"BASELINE PHASE COMPLETE in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
