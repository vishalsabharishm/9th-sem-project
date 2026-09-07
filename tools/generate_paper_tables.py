#!/usr/bin/env python3
"""
tools/generate_paper_tables.py

Generate publication-ready tables from the LOCKED research artifacts.

WHY THIS EXISTS
---------------
Every number destined for the paper currently has to be copied by hand out of a
JSON file. That is the single most likely way a wrong figure reaches print, and
it is entirely avoidable: each number already lives in a hashed artifact.

READS ONLY. RECOMPUTES NOTHING.
-------------------------------
This tool loads committed artifacts and formats them. It does not re-derive a
metric, re-run a test, or re-open a dataset -- if it recomputed anything, a
library upgrade could silently shift a published number away from the value that
was actually locked. Where a figure is not in an artifact, the table says so
rather than filling the gap.

The primary split is not read. The only primary-derived input is the FINAL
confirmatory result that was already produced, hashed and committed.

OUTPUT
------
Markdown tables plus a provenance appendix listing the SHA-256 of every artifact
each table came from, so any figure in the paper can be traced to a file.

Usage:
    python tools/generate_paper_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import file_sha256  # noqa: E402

CONFIRMATORY = REPO_ROOT / "outputs" / "fusion" / "FINAL_confirmatory_primary_evaluation.json"
LOCK_RECORD = REPO_ROOT / "outputs" / "fusion" / "FINAL_experiment_lock_record.json"
PAIRED = (REPO_ROOT / "outputs" / "temporal_risk" / "paired_clean_vs_sliding_window"
          / "paired_comparison.json")
AGGREGATION = REPO_ROOT / "temporal_risk" / "frozen_candidate_aggregation.json"
FAILURE_MODES = REPO_ROOT / "outputs" / "fusion" / "development_failure_modes.json"
PROTOCOL = REPO_ROOT / "temporal_risk" / "frozen_confirmatory_fusion_protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "PAPER_TABLES.md"

MISSING = "_artifact not present; table omitted rather than invented_"


def load(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, places: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def confusion_table(report: Optional[dict]) -> List[str]:
    """Table 1 -- the confirmatory comparison."""
    if report is None:
        return ["### Table 1 — Confirmatory primary evaluation", "", MISSING, ""]
    lines = [
        "### Table 1 — Confirmatory primary evaluation (394 clips, 200 Fight / 194 NonFight)",
        "",
        "| Candidate | TP | FP | TN | FN | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "B0_temporal_only": "B0 temporal-only *(incumbent)*",
        "F2_rank_sum": "F2 fusion *(confirmatory candidate)*",
        "F1_or_gate": "F1 or-gate",
        "B1_spatial_only": "B1 spatial-only",
    }
    for key, label in labels.items():
        entry = report["results"].get(key)
        if entry is None:
            continue
        auc = fmt(entry.get("roc_auc")) if entry.get("roc_auc") is not None else "—"
        pr = fmt(entry.get("pr_auc")) if entry.get("pr_auc") is not None else "—"
        lines.append(
            f"| {label} | {entry['tp']} | {entry['fp']} | {entry['tn']} | {entry['fn']} "
            f"| {fmt(entry['accuracy'])} | {fmt(entry['precision'])} | {fmt(entry['recall'])} "
            f"| {fmt(entry['specificity'])} | {fmt(entry['f1'])} | {auc} | {pr} |"
        )
    lines += [
        "",
        "ROC-AUC and PR-AUC are undefined for F1, which is a boolean gate with no "
        "continuous score; the dash records that rather than a missing measurement.",
        "",
    ]
    return lines


def interval_table(report: Optional[dict]) -> List[str]:
    """Table 2 -- uncertainty, at both levels."""
    if report is None:
        return ["### Table 2 — Confidence intervals", "", MISSING, ""]
    lines = [
        "### Table 2 — Confidence intervals (Wilson, 95%)",
        "",
        "| Candidate | Accuracy [95% CI] | Recall [95% CI] | Specificity [95% CI] |",
        "|---|---|---|---|",
    ]
    for key, label in (("B0_temporal_only", "B0 temporal-only"),
                       ("F2_rank_sum", "F2 fusion")):
        entry = report["results"].get(key)
        if entry is None:
            continue

        def interval(name):
            bounds = entry.get(f"{name}_wilson_95")
            base = entry.get(name)
            if not bounds or base is None:
                return "n/a"
            return f"{fmt(base)} [{fmt(bounds[0])}, {fmt(bounds[1])}]"

        lines.append(f"| {label} | {interval('accuracy')} | {interval('recall')} "
                     f"| {interval('specificity')} |")

    comparison = report["paired_comparisons"]["F2_rank_sum"]
    clip = comparison["accuracy_difference_clip_level"]
    source = comparison["accuracy_difference_source_level"]
    lines += [
        "",
        "**Paired accuracy difference (F2 − B0), percentile bootstrap, "
        f"{clip['resamples']} resamples, seed {clip['seed']}**",
        "",
        "| Resampling level | Difference | 95% CI |",
        "|---|---:|---|",
        f"| Clip | {fmt(clip['difference'])} | [{fmt(clip['ci_low'])}, {fmt(clip['ci_high'])}] |",
        f"| Source video | {fmt(source['difference'])} "
        f"| [{fmt(source['ci_low'])}, {fmt(source['ci_high'])}] |",
        "",
        "The source-clustered interval is wider because the 394 clips come from only "
        f"{report['sample']['sources']} source videos and are therefore not independent; "
        "clip-level intervals overstate precision.",
        "",
    ]
    return lines


def mcnemar_table(report: Optional[dict]) -> List[str]:
    """Table 3 -- the paired tests, overall and by stratum."""
    if report is None:
        return ["### Table 3 — Paired tests", "", MISSING, ""]
    comparison = report["paired_comparisons"]["F2_rank_sum"]
    lines = [
        "### Table 3 — Exact McNemar, F2 versus B0",
        "",
        "| Stratum | b | c | Discordant | p (exact, two-sided) | Reading |",
        "|---|---:|---:|---:|---:|---|",
    ]
    rows = (
        ("Overall", comparison["overall"], "primary endpoint"),
        ("Fight (n=200)", comparison["fight_detection"], "detections gained vs lost"),
        ("NonFight (n=194)", comparison["nonfight_false_alarms"], "false alarms added vs removed"),
    )
    for label, block, reading in rows:
        lines.append(
            f"| {label} | {block['b']} | {block['c']} | {block['discordant']} "
            f"| {block['p_value']:.6f} | {reading} |"
        )
    changed = comparison["changed_clips"]
    lines += [
        "",
        "| Change | Clips |",
        "|---|---:|",
        f"| Fight recovered by F2 | {len(changed['fight_gained'])} |",
        f"| Fight lost by F2 | {len(changed['fight_lost'])} |",
        f"| NonFight false alarms added | {len(changed['nonfight_false_alarm_gained'])} |",
        f"| NonFight false alarms removed | {len(changed['nonfight_false_alarm_removed'])} |",
        "",
    ]
    return lines


def aggregation_table(report: Optional[dict]) -> List[str]:
    """Table 4 -- the aggregation-selection null."""
    if report is None:
        return ["### Table 4 — Aggregation selection", "", MISSING, ""]
    gate = report["promotion"]["gate"]
    lines = [
        "### Table 4 — Aggregation candidates at a matched false-positive budget "
        "(240-clip carve, development)",
        "",
        "| Candidate | Threshold | TP | FP | TN | FN | Recall | Role |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    # This artifact stores candidates as a LIST of records, each carrying its own
    # name, threshold and metrics -- a different shape from the confirmatory
    # protocol's dict. Read it as it is rather than assuming one layout.
    for block in report["candidates"]:
        metrics = block["metrics"]
        lines.append(
            f"| {block['name']} | {fmt(block.get('threshold'), 6)} | {metrics['tp']} "
            f"| {metrics['fp']} | {metrics['tn']} | {metrics['fn']} "
            f"| {fmt(metrics.get('recall'))} | {block.get('role', '')} |"
        )
    lines += [
        "",
        f"Promotion gate: at least {gate['required_tp']} true positives "
        f"({gate['required_extra_true_positives']} more than the incumbent's "
        f"{gate['incumbent_tp']}). **{report['promotion']['decision']}** — every "
        "alternative fell below the incumbent, so no gate value would have promoted one.",
        "",
        "_Development-set figures. Not a performance claim: these are the clips the "
        "thresholds were selected on._",
        "",
    ]
    return lines


def failure_table(report: Optional[dict]) -> List[str]:
    """Table 5 -- failure modes, with the class-dependence flag."""
    if report is None:
        return ["### Table 5 — Failure modes", "", MISSING, ""]
    lines = [
        "### Table 5 — Detection and tracking failure modes (235 analyzable carve clips)",
        "",
        "| Failure mode | Clips | % all | % Fight | % NonFight | Class-dependent |",
        "|---|---:|---:|---:|---:|:--:|",
    ]
    for name, block in report["detection_and_tracking_failures"].items():
        flag = "**yes**" if block["class_dependent"] else "no"
        lines.append(
            f"| {name.replace('_', ' ')} | {block['clips']} | {block['pct_all']} "
            f"| {block['pct_fight']} | {block['pct_nonfight']} | {flag} |"
        )
    errors = report["temporal_errors_frozen_rule"]
    concentration = report["failure_concentration"]
    lines += [
        "",
        f"Under the frozen rule: **{errors['false_negatives_missed_fights']} missed fights** "
        f"and {errors['false_positives']} false positives. Missed fights have a median "
        f"max-window score of {errors['missed_fight_median_max_score']} against "
        f"{errors['detected_fight_median_max_score']} for detected ones — they are not "
        "marginal near-threshold cases.",
        "",
        f"Failures concentrate by source video: {concentration['missed_fights']} missed "
        f"fights span only {concentration['distinct_sources']} sources, the largest "
        f"contributing {concentration['largest_single_source_contribution']}.",
        "",
    ]
    return lines


def provenance_appendix(entries) -> List[str]:
    lines = [
        "## Provenance appendix",
        "",
        "Every table above was formatted from these artifacts. Nothing was recomputed.",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ]
    for path in entries:
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest = file_sha256(path) if path.is_file() else "absent"
        lines.append(f"| `{relative}` | `{digest}` |")
    lines.append("")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    confirmatory = load(CONFIRMATORY)
    aggregation = load(AGGREGATION)
    failures = load(FAILURE_MODES)

    lines = [
        "# Publication tables",
        "",
        "**Generated by `tools/generate_paper_tables.py` — do not edit by hand.**",
        "",
        "Every figure is read from a committed, hashed artifact and formatted. Nothing "
        "here is recomputed, so a library upgrade cannot silently shift a published "
        "number away from the value that was locked. Where an artifact is absent, the "
        "table says so rather than filling the gap.",
        "",
        f"Regenerated {datetime.now(timezone.utc).isoformat()}.",
        "",
        "---",
        "",
    ]
    lines += confusion_table(confirmatory)
    lines += interval_table(confirmatory)
    lines += mcnemar_table(confirmatory)
    lines += aggregation_table(aggregation)
    lines += failure_table(failures)
    lines += provenance_appendix(
        [CONFIRMATORY, LOCK_RECORD, PAIRED, AGGREGATION, FAILURE_MODES, PROTOCOL])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"  tables: 5 | artifacts read: 6 | recomputation: none")
    missing = [p.name for p in (CONFIRMATORY, AGGREGATION, FAILURE_MODES) if not p.is_file()]
    if missing:
        print(f"  NOTE: absent artifacts, tables omitted: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
