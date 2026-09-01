"""The carved-validation (leakage-safe) evaluation protocol for RWF-2000.

WHY THIS MODULE EXISTS
----------------------
Two temporal protocols exist in this project's history.

``experiment1_primary_monitor`` -- the ORIGINAL/BASELINE protocol, still
    implemented by ``temporal_training.build_dataloaders``. It trains on all
    1600 train clips and early-stops on the 394-clip *primary* set, which is
    also the set headline metrics are reported on. ``rwf2000_config``'s own
    comment calls the threshold this produced "optimistically biased". Kept
    for historical comparison; must not be used for new reported numbers.

``carved_validation`` -- the CORRECT protocol, and the one that produced the
    reported clean baseline (accuracy 0.8325 / ROC-AUC 0.9251 / threshold
    0.16). A validation subset is carved out of *train*; early stopping and
    threshold selection happen there; the primary set is touched once, for
    final reporting. Until now this protocol existed only inside Kaggle
    notebook ``notebook7bb9a86555`` v3 and could not be run from this
    repository at all. This module brings it back.

RECOVERED, NOT RE-DERIVED
-------------------------
The exact membership of the 240-clip carve split is **read from**
``temporal_risk/carve_window_scores.csv``, which records every clip that was
scored on the carve split. It is NOT recomputed from a seed.

That distinction is deliberate and important. The manifest records
``carve_fraction=0.15`` and ``carve_seed=42``, and
``rwf2000_config.SplitPolicy`` declares ``validation_grouped_by_source_video``
-- but the notebook that consumed those parameters is not in this repository,
so the exact partition algorithm is unknown (see
``docs/EXPERIMENT_REPRODUCIBILITY.md``, "Documented uncertainty"). Re-deriving
a split from a seed we cannot verify would produce a *different* 240 clips and
silently invalidate every number the carve split supports. Reading the
recorded membership reproduces the actual experiment exactly.

The result is reproducible in the sense that matters: the same split, every
run, on any machine, verifiable against a committed artifact.

CLIP-NAME RESOLUTION
--------------------
The CSV records clip names as they existed in the Kaggle extraction. Six of
the 240 do not match this machine's filenames, for two distinct and fully
deterministic reasons:

1. Five were extracted on Kaggle under ``rwf2000_kaggle.safe_relative_path``
   names (``<ascii-fragment>__<sha1[:12]>.avi``) because their original CJK
   names exceed the Linux 255-byte filename limit. This machine's extraction
   did not need to rename them (``data/rwf2000_metadata/renamed_clips.json``
   records ``recovered_under_safe_names: 0``), so they kept their original
   names here.
2. One differs only by character encoding: its UTF-8 bytes read back as
   cp1252 give the CSV spelling.

Both mappings are computed, not guessed, and both are asserted by
``tests/test_carved_validation.py``. Resolution never falls back to fuzzy
matching -- an unresolved clip is reported, never approximated.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rwf2000_kaggle import safe_relative_path
    from rwf2000_splits import build_evaluation_splits, split_clip_paths
except ImportError:  # pragma: no cover - supports package execution
    from src.rwf2000_kaggle import safe_relative_path
    from src.rwf2000_splits import build_evaluation_splits, split_clip_paths


PROTOCOL_CARVED = "carved_validation"
PROTOCOL_EXPERIMENT1 = "experiment1_primary_monitor"
SUPPORTED_PROTOCOLS = (PROTOCOL_CARVED, PROTOCOL_EXPERIMENT1)

TRAIN_SPLIT_DIRECTORY = "train"
DEFAULT_CARVE_CSV = (
    Path(__file__).resolve().parent.parent / "temporal_risk" / "carve_window_scores.csv"
)

# The archive prefix under which safe names were generated on Kaggle. Derived
# by checking which prefix reproduces the recorded names, not assumed; see
# tests/test_carved_validation.py::SafeNameResolutionTests.
ARCHIVE_PREFIX = "RWF-2000/"

RESOLUTION_EXACT = "exact_filename"
RESOLUTION_SAFE_NAME = "kaggle_safe_name_sha1"
RESOLUTION_ENCODING = "utf8_cp1252_roundtrip"


class CarvedValidationError(RuntimeError):
    """Raised when the recorded carve split cannot be resolved on this disk."""


@dataclass(frozen=True)
class CarvedSplit:
    """The three disjoint clip sets of the carved-validation protocol.

    All paths are dataset-root-relative, forward-slash separated, and refer to
    files that exist on this machine.
    """

    root: str
    carve_validation: Tuple[str, ...] = ()
    train_remainder: Tuple[str, ...] = ()
    primary_evaluation: Tuple[str, ...] = ()
    leakage_excluded: Tuple[str, ...] = ()
    labels: Dict[str, str] = field(default_factory=dict)
    resolution_methods: Dict[str, str] = field(default_factory=dict)
    carve_csv: str = ""

    @property
    def counts(self) -> Dict[str, int]:
        """Size of each set, for reporting."""
        return {
            "train_remainder": len(self.train_remainder),
            "carve_validation": len(self.carve_validation),
            "primary_evaluation": len(self.primary_evaluation),
            "leakage_excluded": len(self.leakage_excluded),
        }

    def label_counts(self, clips: Tuple[str, ...]) -> Dict[str, int]:
        """Fight/NonFight counts for one set, from the directory label."""
        counts: Dict[str, int] = {}
        for clip in clips:
            label = clip_label(clip)
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict:
        """Return a JSON-native representation for the run record."""
        return {
            "protocol": PROTOCOL_CARVED,
            "root": self.root,
            "carve_csv": self.carve_csv,
            "membership_source": "recovered from carve_window_scores.csv, not re-derived from a seed",
            "counts": self.counts,
            "label_counts": {
                "train_remainder": self.label_counts(self.train_remainder),
                "carve_validation": self.label_counts(self.carve_validation),
                "primary_evaluation": self.label_counts(self.primary_evaluation),
            },
            "resolution_method_counts": _tally(self.resolution_methods.values()),
            "leakage_excluded": list(self.leakage_excluded),
            "carve_validation": list(self.carve_validation),
        }


def _tally(values) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def clip_label(relative_path: str) -> str:
    """Return ``Fight``/``NonFight`` from the class directory in the path.

    RWF-2000 prefixes class directories with the split (``Train_Fight``), so
    the label is the part after the underscore.
    """
    directory = Path(relative_path).parent.name
    _, _, label = directory.partition("_")
    return label or directory


def recorded_carve_clips(carve_csv: Path = DEFAULT_CARVE_CSV) -> Dict[str, str]:
    """Return ``{clip_name_as_recorded: true_label}`` from the carve CSV.

    Reads the committed Kaggle output only. Nothing on disk is touched.
    """
    path = Path(carve_csv)
    if not path.is_file():
        raise CarvedValidationError(
            f"{path} not found. The carved-validation split membership is "
            "recovered from this file; without it the protocol cannot be run."
        )
    labels: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8", errors="surrogateescape") as handle:
        for row in csv.DictReader(handle):
            clip, label = row["clip"], row["true_label"]
            previous = labels.setdefault(clip, label)
            if previous != label:
                raise CarvedValidationError(
                    f"Inconsistent true_label for {clip!r} in {path}."
                )
    return labels


def _disk_index(root: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build the two reverse indexes used to resolve recorded clip names."""
    on_disk = split_clip_paths(root, TRAIN_SPLIT_DIRECTORY)
    by_safe_name: Dict[str, str] = {}
    by_encoding: Dict[str, str] = {}
    for clip in on_disk:
        safe = safe_relative_path(ARCHIVE_PREFIX + clip).rsplit("/", 1)[-1]
        by_safe_name.setdefault(safe, clip)
        try:
            by_encoding.setdefault(clip.encode("utf-8").decode("cp1252"), clip)
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return by_safe_name, by_encoding


def resolve_recorded_clip(
    recorded: str,
    on_disk: set,
    by_safe_name: Dict[str, str],
    by_encoding: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    """Map one recorded clip name to a local path, deterministically.

    Returns ``(local_path, method)``, or ``(None, None)`` when the clip cannot
    be resolved by any of the three exact mechanisms. There is deliberately no
    fuzzy fallback: an unmatched clip must surface as an error, not as a
    plausible-looking substitute.
    """
    if recorded in on_disk:
        return recorded, RESOLUTION_EXACT
    basename = recorded.rsplit("/", 1)[-1]
    if basename in by_safe_name:
        return by_safe_name[basename], RESOLUTION_SAFE_NAME
    if recorded in by_encoding:
        return by_encoding[recorded], RESOLUTION_ENCODING
    return None, None


def load_carved_split(
    root: Path,
    carve_csv: Path = DEFAULT_CARVE_CSV,
    strict: bool = True,
) -> CarvedSplit:
    """Build the carved-validation split for the dataset at ``root``.

    Partitions the official 1600-clip train split into the recorded 240-clip
    carve validation set and the 1360-clip training remainder, and pairs it
    with the leak-free 394-clip primary evaluation set from
    ``rwf2000_splits``. Nothing is moved, copied, deleted or re-labelled --
    the partition is applied by path at load time.

    With ``strict`` set (the default), a recorded carve clip that cannot be
    resolved on this disk is an error. Continuing with a silently smaller
    carve set would change the protocol without saying so.
    """
    root_path = Path(root)
    recorded = recorded_carve_clips(carve_csv)
    on_disk = set(split_clip_paths(root_path, TRAIN_SPLIT_DIRECTORY))
    by_safe_name, by_encoding = _disk_index(root_path)

    carve: List[str] = []
    methods: Dict[str, str] = {}
    labels: Dict[str, str] = {}
    unresolved: List[str] = []
    mismatched: List[str] = []

    for name in sorted(recorded):
        local, method = resolve_recorded_clip(name, on_disk, by_safe_name, by_encoding)
        if local is None:
            unresolved.append(name)
            continue
        carve.append(local)
        methods[local] = method
        labels[local] = recorded[name]
        if clip_label(local) != recorded[name]:
            mismatched.append(f"{local}: directory says {clip_label(local)!r}, CSV says {recorded[name]!r}")

    if unresolved and strict:
        raise CarvedValidationError(
            f"{len(unresolved)} recorded carve clip(s) could not be resolved on "
            f"disk under {root_path}. The carve split would be incomplete and "
            f"any metric from it incomparable to the recorded run. "
            f"First unresolved: {unresolved[0]!r}"
        )
    if mismatched:
        raise CarvedValidationError(
            "Recorded carve labels disagree with the on-disk class directory: "
            + "; ".join(mismatched[:3])
        )

    carve_set = set(carve)
    remainder = tuple(clip for clip in sorted(on_disk) if clip not in carve_set)
    evaluation = build_evaluation_splits(root_path)

    return CarvedSplit(
        root=str(root_path),
        carve_validation=tuple(sorted(carve)),
        train_remainder=remainder,
        primary_evaluation=evaluation.primary_evaluation,
        leakage_excluded=evaluation.leakage_excluded,
        labels=labels,
        resolution_methods=methods,
        carve_csv=str(carve_csv),
    )


def verify_disjoint(split: CarvedSplit) -> List[str]:
    """Return every leakage problem found in a built split; empty means clean.

    This is the check that makes the protocol's central claim auditable rather
    than asserted: no clip may appear in more than one of the three sets, and
    no confirmed-leaked validation clip may appear in any of them.
    """
    problems: List[str] = []
    sets = {
        "train_remainder": set(split.train_remainder),
        "carve_validation": set(split.carve_validation),
        "primary_evaluation": set(split.primary_evaluation),
    }
    names = sorted(sets)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = sets[first] & sets[second]
            if shared:
                problems.append(
                    f"{len(shared)} clip(s) appear in both {first} and {second}: "
                    f"{sorted(shared)[:3]}"
                )
    leaked = set(split.leakage_excluded)
    for name, members in sets.items():
        overlap = members & leaked
        if overlap:
            problems.append(
                f"{len(overlap)} confirmed-leaked clip(s) present in {name}: "
                f"{sorted(overlap)[:3]}"
            )
    total = sum(len(members) for members in sets.values())
    if len(set().union(*sets.values())) != total:
        problems.append("the three sets are not pairwise disjoint by count")
    return problems
