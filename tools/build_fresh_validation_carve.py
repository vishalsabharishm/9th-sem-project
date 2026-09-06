#!/usr/bin/env python3
"""
tools/build_fresh_validation_carve.py

Construct a FRESH, never-used validation carve from the untouched remainder of
the RWF-2000 training split, for fitting and evaluating a spatial/temporal
fusion rule.

WHY A FRESH CARVE IS REQUIRED
-----------------------------
The existing 240-clip carve has been spent three times: selecting the R3D
checkpoint's best epoch (monitor roc_auc, metrics-at-save support 121/119 --
that split), a 215-candidate aggregation sweep that produced ``max >= 0.14``,
and the v2 matched-FPR aggregation protocol. Any threshold fitted on it now
would inherit three layers of selection. The 394-clip primary split is reserved
for confirmatory evaluation and must not be touched. So a fusion rule that needs
any fitted quantity has nowhere legitimate to learn it -- unless a genuinely
unused split is built first. That is what this does.

PROVENANCE EXCLUSION -- DELIBERATE OVER-EXCLUSION
------------------------------------------------
Five of the 240 existing carve clips are recorded in the Kaggle-produced CSV as
``_urlgot_NNN__<hash>.avi``; locally those files carry mojibake names. There are
27 non-ASCII filenames in Train_Fight and only 5 unmatched carve entries, so
elimination is AMBIGUOUS and the mapping is UNKNOWN. It is not reconstructed
here. Instead every non-ASCII Train_Fight clip AND its complete source group is
removed from the candidate pool. That over-excludes roughly five clean clips for
every contaminated one, which is the correct trade: a fresh split whose
freshness is merely probable is not fresh.

WHAT THIS DOES NOT DO
---------------------
No model of any kind is loaded. No temporal score, no spatial feature and no
prediction is read. Selection uses only filenames, class directories and the
inferred source-video grouping, so the split cannot be tuned toward a result.
The manifest is hashed on write and frozen before any scoring runs.

GROUPING IS MANDATORY, NOT COSMETIC
-----------------------------------
Clips cut from one source video are near-duplicates, and in this pool 107
sources contribute clips under BOTH labels. A clip-level split would put
near-identical footage on both sides of the fit/eval boundary and report an
optimistic number. Whole sources move together, and fit and eval are
source-disjoint by construction.

Usage:
    python tools/build_fresh_validation_carve.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from checkpoint_identity import file_sha256  # noqa: E402
from metric_intervals import source_video_id  # noqa: E402

DATASET_ROOT = REPO_ROOT / "data" / "rwf2000" / "RWF-2000"
CARVE_CSV = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
DEFAULT_OUTPUT = REPO_ROOT / "temporal_risk" / "fresh_validation_carve.json"

PROTOCOL_VERSION = "fresh-carve-v1-2026-09"
SEED = 20260906

# Sized from the existing carve's observed behaviour, before any fresh-carve
# scoring: the frozen temporal rule misses ~30% of Fight clips, so an eval split
# of ~120 Fight clips yields ~36 recoverable misses. Exact McNemar needs roughly
# 6-10 net discordant clips for significance, which that supports if fusion
# recovers even a third of them. The fit split only has to place two univariate
# thresholds, for which ~100 clips is adequate.
TARGET_FIT_CLIPS = 100
TARGET_EVAL_CLIPS = 240


class CarveError(RuntimeError):
    """Raised when a clean, source-disjoint fresh carve cannot be built."""


def list_split(root: Path, split: str, dirs: Tuple[Tuple[str, str], ...]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for directory, label in dirs:
        base = root / split / directory
        if not base.is_dir():
            raise CarveError(f"{base} not found; the dataset is not laid out as expected.")
        for name in sorted(os.listdir(base)):
            if name.lower().endswith(".avi"):
                out.append((f"{split}/{directory}/{name}", label))
    return out


def build_pool(root: Path, carve_clips: set) -> dict:
    """Everything in train that is neither in the old carve nor provenance-suspect."""
    train = list_split(root, "train", (("Train_Fight", "Fight"), ("Train_NonFight", "NonFight")))
    val = list_split(root, "val", (("Val_Fight", "Fight"), ("Val_NonFight", "NonFight")))

    remainder = [(clip, label) for clip, label in train if clip not in carve_clips]

    non_ascii = [
        clip for clip, _ in train
        if clip.startswith("train/Train_Fight/")
        and not all(ord(ch) < 128 for ch in os.path.basename(clip))
    ]
    excluded_sources = {source_video_id(clip) for clip in non_ascii}
    excluded_clips = [c for c, _ in remainder if source_video_id(c) in excluded_sources]

    pool = [(c, l) for c, l in remainder if source_video_id(c) not in excluded_sources]

    carve_sources = {source_video_id(c) for c in carve_clips}
    val_sources = {source_video_id(c) for c, _ in val}
    pool_sources = {source_video_id(c) for c, _ in pool}

    if pool_sources & carve_sources:
        raise CarveError(
            f"{len(pool_sources & carve_sources)} pool sources overlap the existing carve."
        )
    if pool_sources & val_sources:
        raise CarveError(
            f"{len(pool_sources & val_sources)} pool sources overlap the held-out val split."
        )

    return {
        "pool": pool,
        "exclusion": {
            "reason": (
                "5 existing-carve clips cannot be mapped from the Kaggle CSV to local "
                "filenames; 27 non-ASCII Train_Fight names exist and elimination is "
                "ambiguous, so the mapping is UNKNOWN and was not reconstructed. Every "
                "non-ASCII Train_Fight clip and its complete source group is removed."
            ),
            "non_ascii_clips": len(non_ascii),
            "excluded_source_groups": len(excluded_sources),
            "clips_removed_from_remainder": len(excluded_clips),
        },
        "counts": {
            "train_clips": len(train),
            "existing_carve_clips": len(carve_clips),
            "raw_remainder": len(remainder),
            "pool_clips": len(pool),
            "pool_sources": len(pool_sources),
            "pool_labels": dict(collections.Counter(l for _, l in pool)),
        },
        "disjointness": {
            "pool_sources_overlapping_existing_carve": 0,
            "pool_sources_overlapping_val_primary": 0,
        },
    }


def assign_splits(pool: List[Tuple[str, str]], seed: int = SEED) -> Dict[str, List[Tuple[str, str]]]:
    """Assign whole source videos to fit and eval, balancing labels.

    Sources are shuffled once under a fixed seed, then each is placed in the
    split whose class balance it improves most, until that split reaches its
    target. Balance is computed from LABELS ONLY -- no score, feature or
    prediction is consulted, so nothing here can steer the split toward a
    result.
    """
    members: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for clip, label in pool:
        members[source_video_id(clip)].append((clip, label))

    order = sorted(members)  # deterministic base order before shuffling
    generator = np.random.default_rng(seed)
    generator.shuffle(order)

    targets = {"fit": TARGET_FIT_CLIPS, "eval": TARGET_EVAL_CLIPS}
    chosen: Dict[str, List[str]] = {"fit": [], "eval": []}
    counts = {"fit": {"Fight": 0, "NonFight": 0}, "eval": {"Fight": 0, "NonFight": 0}}

    def size(split: str) -> int:
        return counts[split]["Fight"] + counts[split]["NonFight"]

    def imbalance_after(split: str, group: List[Tuple[str, str]]) -> int:
        fight = counts[split]["Fight"] + sum(1 for _, l in group if l == "Fight")
        other = counts[split]["NonFight"] + sum(1 for _, l in group if l == "NonFight")
        return abs(fight - other)

    for source in order:
        group = members[source]
        # eval is filled first: it is the split the conclusion rests on.
        options = [s for s in ("eval", "fit") if size(s) + len(group) <= targets[s]]
        if not options:
            continue
        best = min(options, key=lambda s: (imbalance_after(s, group), targets[s] - size(s), s))
        chosen[best].append(source)
        for _, label in group:
            counts[best][label] += 1

    return {
        split: [pair for source in sources for pair in members[source]]
        for split, sources in chosen.items()
    }, chosen


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    import csv as _csv
    with open(CARVE_CSV, newline="", encoding="utf-8") as handle:
        carve_clips = {row["clip"] for row in _csv.DictReader(handle)}

    info = build_pool(args.root, carve_clips)
    splits, sources = assign_splits(info["pool"])

    fit_sources = set(sources["fit"])
    eval_sources = set(sources["eval"])
    if fit_sources & eval_sources:
        raise CarveError("fit and eval share a source video; the split is not grouped.")

    all_clips = [c for pairs in splits.values() for c, _ in pairs]
    if len(all_clips) != len(set(all_clips)):
        raise CarveError("a clip appears twice across the fresh carve.")

    missing = [c for c in all_clips if not (args.root / c).is_file()]
    if missing:
        raise CarveError(f"{len(missing)} selected clips are not on disk, e.g. {missing[:3]}")

    record = {
        "tool": "build_fresh_validation_carve",
        "protocol_version": PROTOCOL_VERSION,
        "artifact_kind": "fresh validation carve manifest",
        "purpose": (
            "the only split on which spatial/temporal fusion parameters may be "
            "fitted; never used for checkpoint selection or aggregation selection"
        ),
        "fitting_occurred": False,
        "primary_data_accessed": False,
        "model_scores_consulted": False,
        "selection_basis": "filenames, class directories and inferred source grouping only",
        "seed": SEED,
        "dataset_root": str(args.root),
        "existing_carve_csv_sha256": file_sha256(CARVE_CSV),
        "pool": info,
        "splits": {
            split: {
                "clips": len(pairs),
                "sources": len(sources[split]),
                "labels": dict(collections.Counter(l for _, l in pairs)),
                "members": [
                    {"clip": c, "label": l, "source": source_video_id(c)}
                    for c, l in sorted(pairs)
                ],
            }
            for split, pairs in splits.items()
        },
        "disjointness": {
            "fit_eval_shared_sources": 0,
            "duplicate_clips": 0,
            "all_clips_present_on_disk": True,
        },
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool_sha256": file_sha256(Path(__file__).resolve()),
            "python": sys.version.split()[0],
        },
    }

    payload = json.dumps(record, indent=2, sort_keys=True)
    record["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("FRESH VALIDATION CARVE")
    print("=" * 70)
    e = info["exclusion"]
    print(f"  provenance exclusion : {e['non_ascii_clips']} non-ASCII clips -> "
          f"{e['excluded_source_groups']} source groups, "
          f"{e['clips_removed_from_remainder']} clips removed")
    c = info["counts"]
    print(f"  candidate pool       : {c['pool_clips']} clips, {c['pool_sources']} sources, "
          f"{c['pool_labels']}")
    for split in ("fit", "eval"):
        s = record["splits"][split]
        print(f"  {split:<20} : {s['clips']} clips, {s['sources']} sources, {s['labels']}")
    print(f"  fit/eval source overlap: 0")
    print(f"  manifest sha256      : {record['manifest_sha256']}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
