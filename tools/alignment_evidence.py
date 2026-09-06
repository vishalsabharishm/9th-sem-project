#!/usr/bin/env python3
"""
tools/alignment_evidence.py

Regenerate the statistical evidence that the clean-baseline clip alignment is
correct, deterministically and without re-running inference.

WHY THIS EXISTS
---------------
``tools/recover_clean_baseline_alignment.py`` establishes the mapping

    recovered row i  <-->  sorted_primary_clips[i]

by re-running the checkpoint locally and comparing probability sequences. It
records the agreement (0 label mismatches, max abs difference 3.16e-04) but it
does not quantify how surprising that agreement would be under a *wrong*
mapping, and it does not check whether the residual near-ties could move the
paired 2x2 tables.

Those two arguments are what turn "the sequences agree" into "the mapping is
established", so they must be reproducible rather than computed by hand once.
This script regenerates both from the committed alignment report:

  1. PERMUTATION TEST. Re-assign clips at random *within* the label blocks and
     recompute the mean absolute difference. Permuting within blocks is the
     hard null: the label sequence is already fixed by the recovered file, so
     this asks whether the mapping is better than every other mapping that is
     equally consistent with the labels. Seeded, so the p-value reproduces.

  2. TIE ADMISSIBILITY AND TABLE INVARIANCE. Some clips have local
     probabilities too close to separate by value alone. This forms tie
     clusters, discards cross-label swaps (inadmissible -- the recovered file
     states each row's label and none mismatched), and checks whether every
     remaining admissible swap leaves the paired 2x2 tables unchanged.

     The invariance argument, machine-checked below: a swap of clips i and j
     moves clean decision d_i onto clip j and d_j onto clip i, while the
     sliding-window outcome stays with the clip. Admissibility gives
     label_i == label_j, so a clean decision's correctness does not change when
     it moves between them. Hence if d_i == d_j the two contributed cells are
     identical before and after, and every cell count -- so b and c, so every
     McNemar p-value -- is invariant.

Reads the alignment report only. Runs in about a second. Writes one JSON beside
it. No committed evaluation artifact is modified, no checkpoint is loaded, and
no dataset is required.

Usage:
    python tools/alignment_evidence.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = REPO_ROOT / "outputs" / "temporal_risk" / "paired_clean_vs_sliding_window"
ALIGNMENT_JSON = OUTPUT_DIR / "clean_baseline_alignment.json"
PAIRED_JSON = OUTPUT_DIR / "paired_comparison.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "alignment_evidence.json"

PERMUTATIONS = 20000
PERMUTATION_SEED = 42
# Two clips whose locally recomputed probabilities differ by less than this
# cannot be told apart by value matching alone, so a swap between them would
# still reproduce the recovered sequence. Chosen two orders of magnitude below
# the six decimals the recovered file stores, so it is a property of the
# artifact's precision rather than a tuned quantity.
TIE_TOLERANCE = 1e-6


def permutation_test(
    local: np.ndarray,
    recovered: np.ndarray,
    labels: np.ndarray,
    permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> dict:
    """How much better is the hypothesised mapping than a label-consistent one?"""
    observed = float(np.mean(np.abs(local - recovered)))
    generator = np.random.default_rng(seed)
    blocks = [np.flatnonzero(labels == value) for value in np.unique(labels)]

    values = np.empty(permutations, dtype=float)
    order = np.arange(local.size)
    for draw in range(permutations):
        permuted = order.copy()
        for block in blocks:
            permuted[block] = generator.permutation(block)
        values[draw] = float(np.mean(np.abs(local[permuted] - recovered)))

    at_least_as_good = int(np.sum(values <= observed))
    return {
        "observed_mean_abs_difference": observed,
        "permutations": permutations,
        "seed": seed,
        "scheme": (
            "clips re-assigned at random within each label block, which is the "
            "hard null: the recovered file already fixes the label sequence, so "
            "only label-consistent mappings compete"
        ),
        "permuted_min": float(values.min()),
        "permuted_max": float(values.max()),
        "permuted_mean": float(values.mean()),
        "permutations_at_least_as_good_as_observed": at_least_as_good,
        "p_value": (1 + at_least_as_good) / (1 + permutations),
        "p_value_note": (
            "add-one estimator; with zero permutations at least as good this is "
            "bounded by the number of permutations, not by the data"
        ),
        "times_better_than_a_typical_wrong_mapping": float(values.mean() / observed),
    }


def tie_analysis(
    rows: List[dict],
    clean_decisions: Dict[str, str],
    tolerance: float = TIE_TOLERANCE,
) -> dict:
    """Near-ties, which swaps are admissible, and whether the tables survive them."""
    ordered = sorted(rows, key=lambda r: r["local_probability"])

    adjacent = [
        (ordered[i], ordered[i + 1])
        for i in range(len(ordered) - 1)
        if abs(ordered[i]["local_probability"] - ordered[i + 1]["local_probability"]) <= tolerance
    ]
    cross_label = [(x, y) for x, y in adjacent if x["recovered_label"] != y["recovered_label"]]
    same_label = [(x, y) for x, y in adjacent if x["recovered_label"] == y["recovered_label"]]
    same_label_same_decision = [
        (x, y) for x, y in same_label
        if clean_decisions[x["clip"]] == clean_decisions[y["clip"]]
    ]

    # Adjacent pairs understate the freedom: a run of mutually-close clips can
    # be permuted arbitrarily, not just swapped pairwise. Build the transitive
    # closure and check every admissible swap inside each cluster.
    clusters: List[List[dict]] = []
    current = [ordered[0]]
    for previous, row in zip(ordered, ordered[1:]):
        if abs(row["local_probability"] - previous["local_probability"]) <= tolerance:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]
    clusters.append(current)
    tie_clusters = [c for c in clusters if len(c) > 1]

    offending: List[dict] = []
    admissible_swaps = 0
    for cluster in tie_clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                first, second = cluster[i], cluster[j]
                if first["recovered_label"] != second["recovered_label"]:
                    continue  # inadmissible: labels are fixed by the recovered file
                admissible_swaps += 1
                if clean_decisions[first["clip"]] != clean_decisions[second["clip"]]:
                    offending.append(
                        {"first": first["clip"], "second": second["clip"]}
                    )

    return {
        "tolerance": tolerance,
        "adjacent_near_tied_pairs": len(adjacent),
        "adjacent_cross_label_inadmissible": len(cross_label),
        "adjacent_same_label_admissible": len(same_label),
        "adjacent_same_label_and_same_clean_decision": len(same_label_same_decision),
        "tie_clusters": len(tie_clusters),
        "largest_tie_cluster": max((len(c) for c in tie_clusters), default=0),
        "clips_in_tie_clusters": sum(len(c) for c in tie_clusters),
        "admissible_swaps_within_clusters": admissible_swaps,
        "admissible_swaps_that_would_change_a_cell": len(offending),
        "offending_swaps": offending,
        "tables_invariant_to_residual_ambiguity": not offending,
        "invariance_argument": (
            "a swap of clips i and j moves clean decision d_i onto clip j and d_j "
            "onto clip i while the sliding-window outcome stays with the clip. "
            "Admissibility forces label_i == label_j, so a clean decision's "
            "correctness is unchanged by the move; therefore if d_i == d_j the "
            "contributed cells are identical before and after, and b, c and every "
            "McNemar p-value are invariant"
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT_JSON)
    parser.add_argument("--paired", type=Path, default=PAIRED_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=PERMUTATION_SEED)
    args = parser.parse_args(argv)

    for path in (args.alignment, args.paired):
        if not path.is_file():
            print(
                f"ERROR: {path} not found. Run "
                "tools/recover_clean_baseline_alignment.py then "
                "tools/paired_clean_vs_sliding.py first.",
                file=sys.stderr,
            )
            return 1

    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    paired = json.loads(args.paired.read_text(encoding="utf-8"))
    rows = alignment["per_clip"]
    clean_decisions = {r["clip"]: r["clean_predicted"] for r in paired["per_clip"]}

    missing = {r["clip"] for r in rows} - set(clean_decisions)
    if missing:
        print(
            f"ERROR: {len(missing)} aligned clips have no clean decision in the "
            "paired report; the two artifacts do not describe the same run.",
            file=sys.stderr,
        )
        return 1

    permutation = permutation_test(
        np.array([r["local_probability"] for r in rows]),
        np.array([r["recovered_probability"] for r in rows]),
        np.array([r["recovered_label"] for r in rows]),
        permutations=args.permutations,
        seed=args.seed,
    )
    ties = tie_analysis(rows, clean_decisions)

    report = {
        "tool": "alignment_evidence",
        "purpose": (
            "quantify how strongly the recovered clip alignment is supported, and "
            "prove the paired tables cannot be changed by the ambiguity that remains"
        ),
        "clips": len(rows),
        "clean_predictions_sha256": alignment["clean_predictions_sha256"],
        "checkpoint_sha256": alignment["checkpoint_sha256"],
        "agreement_from_alignment_report": {
            "label_mismatches": alignment["label_mismatches"],
            "rows_over_tolerance": alignment["rows_over_tolerance"],
            "exact_matches_at_six_decimals": alignment["exact_matches_at_six_decimals"],
            "max_abs_difference": alignment["max_abs_difference"],
            "mean_abs_difference": alignment["mean_abs_difference"],
        },
        "permutation_test": permutation,
        "tie_analysis": ties,
        "conclusion": (
            "the hypothesised mapping beats every label-consistent alternative "
            "tried, and the residual near-ties provably cannot move a single cell "
            "of the paired tables"
            if permutation["permutations_at_least_as_good_as_observed"] == 0
            and ties["tables_invariant_to_residual_ambiguity"]
            else "ALIGNMENT EVIDENCE IS NOT CONCLUSIVE -- see the fields above"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("ALIGNMENT EVIDENCE")
    print("=" * 72)
    print(f"  observed mean |diff|      : {permutation['observed_mean_abs_difference']:.6e}")
    print(f"  permuted min / mean / max : {permutation['permuted_min']:.4e} / "
          f"{permutation['permuted_mean']:.4e} / {permutation['permuted_max']:.4e}")
    print(f"  permutations >= as good   : "
          f"{permutation['permutations_at_least_as_good_as_observed']} / {args.permutations}")
    print(f"  permutation p-value       : {permutation['p_value']:.3e}")
    print(f"  times better than typical : {permutation['times_better_than_a_typical_wrong_mapping']:.0f}x")
    print()
    print(f"  adjacent near-tied pairs  : {ties['adjacent_near_tied_pairs']}")
    print(f"    cross-label (inadmissible): {ties['adjacent_cross_label_inadmissible']}")
    print(f"    same-label (admissible)   : {ties['adjacent_same_label_admissible']}")
    print(f"    of those, same clean call : {ties['adjacent_same_label_and_same_clean_decision']}")
    print(f"  admissible swaps in clusters: {ties['admissible_swaps_within_clusters']}")
    print(f"  swaps that would move a cell: {ties['admissible_swaps_that_would_change_a_cell']}")
    print(f"  TABLES INVARIANT            : {ties['tables_invariant_to_residual_ambiguity']}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
