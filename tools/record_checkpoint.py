#!/usr/bin/env python3
"""
tools/record_checkpoint.py

Verify a temporal checkpoint and record its identity in
``models/temporal_violence/CHECKPOINT.md``.

Records SHA-256, size, modification time, retrieval source, training
provenance, validation protocol, monitored metric, best epoch and the training
configuration -- so a checkpoint obtained later can be proven to be the same
file that produced a given set of numbers.

Verification comes first. A file that fails
``checkpoint_identity.verify_trained_checkpoint`` is not recorded as valid;
``--record-anyway`` documents it *as rejected*, with its reasons, rather than
silently accepting it.

The weights themselves are never committed: ``.gitignore`` excludes ``*.pt``,
and this document records identity so a checkpoint can be verified, not
redistributed.

Usage:
    python tools/record_checkpoint.py --inspect models/temporal_violence/best.pt
    python tools/record_checkpoint.py --checkpoint models/temporal_violence/best.pt \\
        --retrieved-from "kaggle notebook7bb9a86555 v3, Output tab"
    python tools/record_checkpoint.py --refresh     # rewrite CHECKPOINT.md from what is present
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import (  # noqa: E402
    CheckpointIntegrityError,
    checkpoint_record,
    render_checkpoint_md,
    verify_trained_checkpoint,
)

CHECKPOINT_DIR = REPO_ROOT / "models" / "temporal_violence"
CHECKPOINT_MD = CHECKPOINT_DIR / "CHECKPOINT.md"
RECORDS_JSON = CHECKPOINT_DIR / "checkpoint_records.json"

# The authentic checkpoint this project depends on but does not have. Recorded
# so CHECKPOINT.md states precisely what is missing and where it came from,
# rather than leaving a silent gap. Every value here is transcribed from
# committed artifacts, not estimated.
KNOWN_ABSENT = [
    {
        "filename": "best.pt",
        "status": "ABSENT from this working tree",
        "expected_size_bytes": 139172577,
        "expected_size_display": "132.74 MB",
        "sha256": "NOT RECORDED -- no hash was taken before the file became unavailable",
        "source": "Kaggle notebook7bb9a86555, version 3, session 344760335",
        "source_path": "clean_experiment/training/checkpoints/best.pt",
        "protocol": "carved_validation (validation carved from train, fraction 0.15)",
        "monitor": "roc_auc on the train-carved validation subset = 0.935342732134176",
        "best_epoch": 12,
        "completed_epochs": 17,
        "frozen_threshold": 0.16,
        "produces": "accuracy 0.8325 / ROC-AUC 0.9251 on the 394-clip primary split",
        "evidence": "temporal_risk/clean_baseline_training_summary.json, "
                    "temporal_risk/clean_baseline_final_evaluation_metrics.json",
        "consequence": "training, window scoring and re-derivation of the clean "
                       "baseline are all blocked until this file is retrieved",
        "retrieval": "download from the Kaggle notebook's Output tab, place at "
                     "models/temporal_violence/best.pt, then run "
                     "`python tools/record_checkpoint.py --checkpoint "
                     "models/temporal_violence/best.pt --retrieved-from '<source>'`",
        "identity_caveat": "because no hash was recorded before the file became "
                           "unavailable, a future download can be checked for "
                           "internal consistency but CANNOT be proven to be the "
                           "exact file that produced the reported metrics",
    }
]


def discover(directory: Path) -> List[Path]:
    """Return checkpoint files present in the directory, sorted."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.pt") if p.is_file())


def load_records() -> List[dict]:
    if RECORDS_JSON.is_file():
        return json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    return []


def save_records(records: List[dict]) -> None:
    RECORDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RECORDS_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_markdown(records: List[dict]) -> Path:
    CHECKPOINT_MD.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_MD.write_text(
        render_checkpoint_md(records, missing=KNOWN_ABSENT), encoding="utf-8"
    )
    return CHECKPOINT_MD


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint to verify and record.")
    parser.add_argument("--inspect", type=Path, default=None,
                        help="Verify and print a verdict without recording anything.")
    parser.add_argument("--retrieved-from", default=None,
                        help="Where this file came from, recorded verbatim.")
    parser.add_argument("--record-anyway", action="store_true",
                        help="Record a checkpoint that failed verification, marked as rejected.")
    parser.add_argument("--refresh", action="store_true",
                        help="Rewrite CHECKPOINT.md from the stored records and the "
                             "known-absent list, without inspecting any file.")
    args = parser.parse_args(argv)

    if args.inspect:
        try:
            verdict = verify_trained_checkpoint(args.inspect)
        except CheckpointIntegrityError as error:
            print(f"REJECTED: {error}", file=sys.stderr)
            return 1
        print(json.dumps(verdict.as_dict(), indent=2, default=str))
        print(f"\nVERDICT: {'ACCEPTED' if verdict.accepted else 'REJECTED'}")
        return 0 if verdict.accepted else 1

    if args.refresh:
        path = write_markdown(load_records())
        print(f"Wrote {path} from {len(load_records())} stored record(s) "
              f"and {len(KNOWN_ABSENT)} known-absent entry.")
        return 0

    if args.checkpoint is None:
        # Default: record whatever is present, and always document what is absent.
        present = discover(CHECKPOINT_DIR)
        if not present:
            path = write_markdown([])
            print(f"No checkpoint files under {CHECKPOINT_DIR}.")
            print(f"Wrote {path} documenting {len(KNOWN_ABSENT)} known-absent checkpoint.")
            return 0
        print(f"Found {len(present)} checkpoint file(s). Verifying each:")
        for candidate in present:
            verdict = verify_trained_checkpoint(candidate)
            print(f"  {candidate.name}: {'ACCEPTED' if verdict.accepted else 'REJECTED'}")
            for reason in verdict.rejections:
                print(f"      - {reason}")
        print("\nPass --checkpoint <path> to record one.")
        return 0

    verdict = verify_trained_checkpoint(args.checkpoint)
    if not verdict.accepted and not args.record_anyway:
        print(f"REJECTED -- {args.checkpoint} is not an acceptable trained checkpoint:",
              file=sys.stderr)
        for reason in verdict.rejections:
            print(f"  - {reason}", file=sys.stderr)
        print("\nNot recorded. Pass --record-anyway to document it as rejected.",
              file=sys.stderr)
        return 1

    record = checkpoint_record(args.checkpoint, retrieved_from=args.retrieved_from)
    records = [r for r in load_records()
               if r["identity"]["sha256"] != record["identity"]["sha256"]]
    records.append(record)
    save_records(records)
    path = write_markdown(records)

    print(f"{'RECORDED' if verdict.accepted else 'RECORDED AS REJECTED'}: "
          f"{record['identity']['filename']}")
    print(f"  sha256 : {record['identity']['sha256']}")
    print(f"  bytes  : {record['identity']['bytes']:,}")
    print(f"Wrote {path}")
    print(f"Wrote {RECORDS_JSON}")
    return 0 if verdict.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
