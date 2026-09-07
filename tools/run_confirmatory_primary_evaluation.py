#!/usr/bin/env python3
"""
tools/run_confirmatory_primary_evaluation.py

THE ONE-SHOT CONFIRMATORY EVALUATION on the protected primary 394-clip split.

This runs exactly once. Every parameter it uses was frozen and committed before
primary was opened, in temporal_risk/frozen_confirmatory_fusion_protocol.json
(protocol_sha256 ff2d78c5...). The tool verifies that SHA before doing anything
and refuses if it does not match, so a protocol edited after the fact cannot be
silently used.

WHAT IS FIXED BEFORE PRIMARY IS READ
------------------------------------
    B0  max(window_probabilities) >= 0.14
    B1  spatial >= 0.023439943309201124
    F1  max(...) >= 0.14  OR  spatial >= 0.44120893993145877
    F2  0.5*pct(temporal) + 0.5*pct(spatial) >= 0.49133532299990373

The confirmatory comparison is B0 versus F2. B1 and F1 are reported
descriptively and are NOT permitted to trigger post-hoc selection.

NORMALISATION CANNOT TOUCH PRIMARY
----------------------------------
The percentile reference distributions are read from the frozen artifact -- 235
temporal values and 221 spatial values measured on the development carve.
Primary scores are ranked against those stored values. Nothing from primary is
appended to a reference, and no reference is computed from primary. The tool
asserts the reference sizes against the recorded development counts before use.

TEMPORAL EVIDENCE IS READ, NEVER RECOMPUTED
-------------------------------------------
Primary temporal window scores come from the protected artifact
temporal_risk/primary_window_scores.csv, opened read-only, with its SHA-256
recorded. No model is loaded for the temporal branch, nothing is retrained, and
the artifact is never rewritten.

SPATIAL EVIDENCE
----------------
Computed with the committed runtime configuration -- YOLOv8s at conf 0.25,
SimpleTracker defaults, active_tracks filtering -- and the frozen feature
    s = mean(per-track centroid displacement) / mean(person-group diagonal)
which is a RATIO OF MEANS. Missing-value handling is the frozen abstention rule:
an undefined spatial score never fires.

SCOPE OF THE RESULT
-------------------
The spatial feature is a whole-clip aggregate, so this is a CLIP-LEVEL OFFLINE
CONFIRMATORY CLASSIFICATION. It is not a causal mid-clip alarm, not real-time,
and not early warning. Those words must not be attached to this result.

Usage:
    python tools/run_confirmatory_primary_evaluation.py score   --i-am-authorized
    python tools/run_confirmatory_primary_evaluation.py analyse --i-am-authorized
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import file_sha256  # noqa: E402
from metric_intervals import source_video_id, wilson_interval  # noqa: E402

PROTOCOL = REPO_ROOT / "temporal_risk" / "frozen_confirmatory_fusion_protocol.json"
EXPECTED_PROTOCOL_SHA = "ff2d78c5ae064e16b1144b74193b40eb66d71f63069c8594faf13238fb6384cd"
PRIMARY_WINDOWS = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
DATASET_ROOT = REPO_ROOT / "data" / "rwf2000" / "RWF-2000"
YOLO_WEIGHTS = REPO_ROOT / "models" / "yolov8s.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "fusion"
SPATIAL_OUT = OUTPUT_DIR / "FINAL_primary_spatial_scores.json"
RESULT_OUT = OUTPUT_DIR / "FINAL_confirmatory_primary_evaluation.json"

RUNTIME_CONFIDENCE = 0.25
EXPECTED_FIGHT, EXPECTED_NONFIGHT = 200, 194


class ConfirmatoryError(RuntimeError):
    """Raised when the locked protocol cannot be applied exactly as frozen."""


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    recorded = protocol.get("protocol_sha256")
    if recorded != EXPECTED_PROTOCOL_SHA:
        raise ConfirmatoryError(
            f"protocol_sha256 is {recorded}, expected {EXPECTED_PROTOCOL_SHA}. "
            "The frozen protocol has changed; refusing to run the confirmatory "
            "evaluation against an unlocked protocol."
        )
    # Recompute the hash the same way the freeze did, so an edit anywhere in the
    # artifact is caught rather than just a mismatched stored field.
    copy = dict(protocol)
    copy.pop("protocol_sha256")
    recomputed = hashlib.sha256(
        json.dumps(copy, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if recomputed != EXPECTED_PROTOCOL_SHA:
        raise ConfirmatoryError(
            f"the protocol body hashes to {recomputed}, not {EXPECTED_PROTOCOL_SHA}. "
            "It has been edited since it was frozen."
        )
    return protocol


def load_primary_temporal() -> Dict[str, dict]:
    """Read the protected primary artifact. Read-only, never rewritten."""
    windows: Dict[str, List[float]] = {}
    labels: Dict[str, str] = {}
    with open(PRIMARY_WINDOWS, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "primary":
                raise ConfirmatoryError(f"unexpected split {row['split']!r} in the primary artifact")
            windows.setdefault(row["clip"], []).append(float(row["fight_probability"]))
            labels[row["clip"]] = row["true_label"]
    fight = sum(1 for v in labels.values() if v == "Fight")
    other = len(labels) - fight
    if (fight, other) != (EXPECTED_FIGHT, EXPECTED_NONFIGHT):
        raise ConfirmatoryError(
            f"primary composition is {fight} Fight / {other} NonFight, expected "
            f"{EXPECTED_FIGHT} / {EXPECTED_NONFIGHT}. Refusing to proceed on an "
            "unexpected sample."
        )
    return {clip: {"windows": values, "label": labels[clip]} for clip, values in windows.items()}


def centroid(track) -> tuple:
    x1, y1, x2, y2 = [float(v) for v in track.bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def spatial_score_for(relative: str, model, root: Path) -> dict:
    """The frozen feature: mean(speed) / mean(group diagonal). Ratio of means."""
    import cv2
    from tracker import SimpleTracker, active_tracks, build_tracking_snapshot
    from behavior_analyzer import BehaviorAnalyzer
    from event_rules import (
        DEFAULT_MOVEMENT_THRESHOLD_PIXELS, DEFAULT_STATIONARY_FRAME_THRESHOLD,
    )

    capture = cv2.VideoCapture(str(root / relative))
    if not capture.isOpened():
        return {"clip": relative, "error": "cannot open"}

    tracker, behaviour = SimpleTracker(), BehaviorAnalyzer()
    speeds: List[float] = []
    diagonals: List[float] = []
    counts: List[int] = []
    frames = zero_person = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        detections = model(frame, conf=RUNTIME_CONFIDENCE, verbose=False)[0]
        if detections.boxes is not None and len(detections.boxes):
            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy().astype(int)
            confidences = detections.boxes.conf.cpu().numpy()
        else:
            boxes, classes, confidences = np.empty((0, 4)), np.array([], dtype=int), np.array([])

        tracks = active_tracks(tracker.update(boxes, classes, confidences), tracker.frame_idx)
        snapshot = build_tracking_snapshot(frames - 1, tracks)
        persons = [t for t in tracks if int(t.class_id) == 0]
        counts.append(len(persons))
        if not persons:
            zero_person += 1

        for track in persons:
            previous = behaviour.get_previous_bbox(track.id)
            if previous is not None:
                speeds.append(behaviour._movement_pixels(previous, track.bbox))

        if persons:
            xs = [float(t.bbox[0]) for t in persons] + [float(t.bbox[2]) for t in persons]
            ys = [float(t.bbox[1]) for t in persons] + [float(t.bbox[3]) for t in persons]
            diagonals.append(float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))))

        behaviour.update_history(
            snapshot, DEFAULT_MOVEMENT_THRESHOLD_PIXELS, DEFAULT_STATIONARY_FRAME_THRESHOLD
        )

    capture.release()
    speed_mean = float(np.mean(speeds)) if speeds else None
    diagonal_mean = float(np.mean(diagonals)) if diagonals else None
    spatial = (speed_mean / diagonal_mean) if (speed_mean is not None and diagonal_mean) else None
    return {
        "clip": relative, "frames": frames, "zero_person_frames": zero_person,
        "person_count_mean": float(np.mean(counts)) if counts else None,
        "speed_mean_pixels": speed_mean, "group_diagonal_mean": diagonal_mean,
        "spatial_score": spatial,
    }


def command_score(args) -> int:
    protocol = load_protocol()
    temporal = load_primary_temporal()
    from detection import load_yolo_model

    clips = sorted(temporal)

    # Incremental checkpointing. A previous run of this step was killed by process
    # teardown at 10/394 clips and lost everything, because results were only
    # written at the end. Per-clip scoring is independent -- a fresh tracker and
    # analyzer are constructed inside spatial_score_for -- so resuming from a
    # partial file yields exactly the same per-clip values as one uninterrupted
    # pass. This changes no threshold, feature, normalisation or statistic.
    partial_path = OUTPUT_DIR / "FINAL_primary_spatial_scores.partial.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done: Dict[str, dict] = {}
    if partial_path.is_file():
        cached = json.loads(partial_path.read_text(encoding="utf-8"))
        if cached.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
            raise ConfirmatoryError(
                "the partial scores were produced under a different protocol; refusing "
                "to resume. Delete the partial file to start cleanly."
            )
        done = {r["clip"]: r for r in cached.get("clips", [])}
        print(f"resuming: {len(done)} clips already scored")

    print(f"scoring {len(clips)} PRIMARY clips (spatial branch only; temporal comes "
          f"from the protected artifact)")
    model = load_yolo_model()
    rows, started = [], time.perf_counter()
    scored_this_run = 0
    for index, clip in enumerate(clips):
        if clip in done:
            rows.append(done[clip])
            continue
        rows.append(spatial_score_for(clip, model, args.root))
        scored_this_run += 1
        if scored_this_run % 10 == 0:
            elapsed = time.perf_counter() - started
            remaining = len(clips) - index - 1
            partial_path.write_text(json.dumps(
                {"protocol_sha256": EXPECTED_PROTOCOL_SHA, "clips": rows}, indent=1),
                encoding="utf-8")
            print(f"  {index+1}/{len(clips)}  {elapsed:.0f}s "
                  f"({elapsed/scored_this_run:.1f}s/clip, eta "
                  f"{remaining*elapsed/scored_this_run/60:.0f} min)", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPATIAL_OUT.write_text(json.dumps({
        "tool": "run_confirmatory_primary_evaluation.score",
        "artifact_kind": "FINAL confirmatory primary spatial scores",
        "protocol_sha256": protocol["protocol_sha256"],
        "primary_windows_sha256": file_sha256(PRIMARY_WINDOWS),
        "yolo_sha256": file_sha256(YOLO_WEIGHTS),
        "yolo_confidence": RUNTIME_CONFIDENCE,
        "feature": "mean(per-track centroid displacement) / mean(person-group diagonal); RATIO OF MEANS",
        "camera_motion_compensated": False,
        "wall_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "clips": rows,
    }, indent=2), encoding="utf-8")
    if partial_path.is_file():
        partial_path.unlink()
    print(f"wrote {SPATIAL_OUT}")
    return 0


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------


def confusion(rows, decisions) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        fired, positive = decisions[row["clip"]], row["truth"] == 1
        if positive and fired:
            tp += 1
        elif positive:
            fn += 1
        elif fired:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics(matrix) -> dict:
    tp, fp, tn, fn = matrix["tp"], matrix["fp"], matrix["tn"], matrix["fn"]
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    out = {**matrix,
           "accuracy": (tp + tn) / total, "precision": precision, "recall": recall,
           "specificity": tn / (tn + fp) if tn + fp else None,
           "fpr": fp / (fp + tn) if fp + tn else None, "f1": f1}
    low, high = wilson_interval(tp + tn, total)
    out["accuracy_wilson_95"] = [low, high]
    if recall is not None:
        low, high = wilson_interval(tp, tp + fn)
        out["recall_wilson_95"] = [low, high]
    if matrix["tn"] + matrix["fp"]:
        low, high = wilson_interval(tn, tn + fp)
        out["specificity_wilson_95"] = [low, high]
    if tp + fp:
        low, high = wilson_interval(tp, tp + fp)
        out["precision_wilson_95"] = [low, high]
    return out


def roc_auc(rows, key) -> Optional[float]:
    pos = [r[key] for r in rows if r["truth"] == 1 and r[key] is not None]
    neg = [r[key] for r in rows if r["truth"] == 0 and r[key] is not None]
    if len(pos) < 5 or len(neg) < 5:
        return None
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda x: x[0])
    ranks, index = [0.0] * len(merged), 0
    while index < len(merged):
        stop = index
        while stop + 1 < len(merged) and merged[stop + 1][0] == merged[index][0]:
            stop += 1
        average = (index + stop + 2) / 2.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    positive = sum(ranks[i] for i in range(len(merged)) if merged[i][1] == 1)
    return (positive - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pr_auc(rows, key) -> Optional[float]:
    usable = [(r[key], r["truth"]) for r in rows if r[key] is not None]
    if not usable:
        return None
    usable.sort(key=lambda x: -x[0])
    positives = sum(t for _, t in usable)
    if not positives:
        return None
    hits = total = 0.0
    for index, (_, truth) in enumerate(usable, start=1):
        if truth:
            hits += 1
            total += hits / index
    return total / positives


def mcnemar_exact(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "discordant": 0, "p_value": 1.0,
                "method": "exact binomial; no discordant pairs, reported as p = 1"}
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return {"b": b, "c": c, "discordant": n, "p_value": min(1.0, 2.0 * tail),
            "one_sided_p_value": tail,
            "method": "two-sided exact binomial on discordant pairs (McNemar)"}


def paired_bootstrap(rows, base, other, clusters=False, resamples=10000, seed=42) -> dict:
    generator = np.random.default_rng(seed)
    base_ok = np.array([base[r["clip"]] == (r["truth"] == 1) for r in rows])
    other_ok = np.array([other[r["clip"]] == (r["truth"] == 1) for r in rows])
    if clusters:
        groups: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(row["source"], []).append(index)
        members = [np.array(v) for v in groups.values()]
    values = []
    for _ in range(resamples):
        if clusters:
            drawn = generator.integers(0, len(members), size=len(members))
            picked = np.concatenate([members[i] for i in drawn])
        else:
            picked = generator.integers(0, len(rows), size=len(rows))
        values.append(float(other_ok[picked].mean() - base_ok[picked].mean()))
    array = np.asarray(values)
    return {"difference": float(other_ok.mean() - base_ok.mean()),
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
            "resamples": resamples, "seed": seed,
            "level": "source video" if clusters else "clip",
            "method": "paired percentile bootstrap; clips resampled as pairs"}


def percentile_of(value, reference) -> Optional[float]:
    if value is None or not reference:
        return None
    low, high = 0, len(reference)
    while low < high:
        middle = (low + high) // 2
        if reference[middle] < value:
            low = middle + 1
        else:
            high = middle
    return low / len(reference)


def command_analyse(args) -> int:
    protocol = load_protocol()
    temporal = load_primary_temporal()
    if not SPATIAL_OUT.is_file():
        raise ConfirmatoryError(f"{SPATIAL_OUT} not found; run the score step first.")
    spatial_payload = json.loads(SPATIAL_OUT.read_text(encoding="utf-8"))
    if spatial_payload["protocol_sha256"] != EXPECTED_PROTOCOL_SHA:
        raise ConfirmatoryError("the spatial scores were produced under a different protocol.")

    normalisation = protocol["normalisation"]
    development = protocol["development_data"]
    temporal_reference = normalisation["temporal_reference_values"]
    spatial_reference = normalisation["spatial_reference_values"]
    if len(temporal_reference) != development["clips_used"]:
        raise ConfirmatoryError("temporal reference size does not match the development set.")
    if len(spatial_reference) != development["spatially_defined"]:
        raise ConfirmatoryError("spatial reference size does not match the development set.")

    spatial_by_clip = {r["clip"]: r.get("spatial_score") for r in spatial_payload["clips"]}
    failures = [r for r in spatial_payload["clips"] if "error" in r]
    if failures:
        raise ConfirmatoryError(f"{len(failures)} primary clips failed to process; STOP.")

    rows = []
    for clip, record in sorted(temporal.items()):
        rows.append({
            "clip": clip, "label": record["label"],
            "truth": 1 if record["label"] == "Fight" else 0,
            "temporal_max": max(record["windows"]),
            "spatial": spatial_by_clip.get(clip),
            "source": source_video_id(clip),
        })
    if len(rows) != EXPECTED_FIGHT + EXPECTED_NONFIGHT:
        raise ConfirmatoryError(f"{len(rows)} clips assembled, expected 394.")

    candidates = protocol["candidates"]
    theta_s = candidates["B1_spatial_only"]["parameters_calibrated"]["theta_s"]
    theta_or = candidates["F1_or_gate"]["parameters_calibrated"]["theta_or"]
    theta_f = candidates["F2_rank_sum"]["parameters_calibrated"]["theta_f"]
    weight = candidates["F2_rank_sum"]["parameters_frozen"]["weight"]
    temporal_threshold = candidates["B0_temporal_only"]["parameters_frozen"]["threshold"]

    def f2_value(row):
        t = percentile_of(row["temporal_max"], temporal_reference)
        s = percentile_of(row["spatial"], spatial_reference)
        if t is None:
            return None
        return weight * t + (1 - weight) * (s if s is not None else 0.0)

    for row in rows:
        row["f2_score"] = f2_value(row)

    decisions = {
        "B0_temporal_only": {r["clip"]: r["temporal_max"] >= temporal_threshold for r in rows},
        "B1_spatial_only": {r["clip"]: (r["spatial"] is not None and r["spatial"] >= theta_s)
                            for r in rows},
        "F1_or_gate": {r["clip"]: (r["temporal_max"] >= temporal_threshold)
                       or (r["spatial"] is not None and r["spatial"] >= theta_or) for r in rows},
        "F2_rank_sum": {r["clip"]: (r["f2_score"] is not None and r["f2_score"] >= theta_f)
                        for r in rows},
    }

    results = {}
    for name, decision in decisions.items():
        entry = metrics(confusion(rows, decision))
        if name == "B0_temporal_only":
            entry["roc_auc"] = roc_auc(rows, "temporal_max")
            entry["pr_auc"] = pr_auc(rows, "temporal_max")
            entry["score_used_for_auc"] = "temporal_max"
        elif name == "B1_spatial_only":
            entry["roc_auc"] = roc_auc(rows, "spatial")
            entry["pr_auc"] = pr_auc(rows, "spatial")
            entry["score_used_for_auc"] = "spatial"
        elif name == "F2_rank_sum":
            entry["roc_auc"] = roc_auc(rows, "f2_score")
            entry["pr_auc"] = pr_auc(rows, "f2_score")
            entry["score_used_for_auc"] = "f2_score (frozen rank sum)"
        else:
            entry["roc_auc"] = None
            entry["pr_auc"] = None
            entry["score_used_for_auc"] = "none; F1 is a boolean gate with no continuous score"
        results[name] = entry

    def compare(name):
        base, other = decisions["B0_temporal_only"], decisions[name]
        ob = oc = fb = fc = nb = nc = 0
        changed = {"fight_gained": [], "fight_lost": [], "nonfight_false_alarm_gained": [],
                   "nonfight_false_alarm_removed": []}
        for row in rows:
            positive = row["truth"] == 1
            base_ok = base[row["clip"]] == positive
            other_ok = other[row["clip"]] == positive
            if base_ok and not other_ok:
                ob += 1
            elif other_ok and not base_ok:
                oc += 1
            if positive:
                if base[row["clip"]] and not other[row["clip"]]:
                    fb += 1
                    changed["fight_lost"].append(row["clip"])
                elif other[row["clip"]] and not base[row["clip"]]:
                    fc += 1
                    changed["fight_gained"].append(row["clip"])
            else:
                if (not base[row["clip"]]) and other[row["clip"]]:
                    nb += 1
                    changed["nonfight_false_alarm_gained"].append(row["clip"])
                elif base[row["clip"]] and not other[row["clip"]]:
                    nc += 1
                    changed["nonfight_false_alarm_removed"].append(row["clip"])
        return {
            "overall": mcnemar_exact(ob, oc),
            "fight_detection": {**mcnemar_exact(fb, fc),
                                "reading": "b = Fight clips B0 detects and this candidate misses; "
                                           "c = Fight clips this candidate detects and B0 misses"},
            "nonfight_false_alarms": {**mcnemar_exact(nb, nc),
                                      "reading": "b = NonFight clips only this candidate "
                                                 "false-alarms on; c = only B0 does"},
            "accuracy_difference_clip_level": paired_bootstrap(rows, base, other),
            "accuracy_difference_source_level": paired_bootstrap(rows, base, other, True),
            "changed_clips": changed,
        }

    comparisons = {name: compare(name)
                   for name in ("F2_rank_sum", "F1_or_gate", "B1_spatial_only")}
    primary = comparisons["F2_rank_sum"]["overall"]
    difference = results["F2_rank_sum"]["accuracy"] - results["B0_temporal_only"]["accuracy"]
    if primary["p_value"] < 0.05:
        decision = "A_SIGNIFICANT_IMPROVEMENT" if primary["c"] > primary["b"] \
            else "D_SIGNIFICANT_DEGRADATION"
    elif difference > 0:
        decision = "B_DIRECTIONAL_INCONCLUSIVE"
    else:
        decision = "C_NO_MEANINGFUL_IMPROVEMENT"

    undefined = [r for r in rows if r["spatial"] is None]
    report = {
        "tool": "run_confirmatory_primary_evaluation.analyse",
        "artifact_kind": "FINAL CONFIRMATORY PRIMARY EVALUATION",
        "one_shot": True,
        "result_scope": "CLIP-LEVEL OFFLINE CONFIRMATORY CLASSIFICATION -- not a causal "
                        "mid-clip alarm, not real-time, not early warning",
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_version": protocol["protocol_version"],
        "primary_endpoint": "B0_temporal_only versus F2_rank_sum, exact two-sided McNemar, alpha 0.05",
        "decision": decision,
        "sample": {
            "clips": len(rows),
            "fight": sum(1 for r in rows if r["truth"] == 1),
            "nonfight": sum(1 for r in rows if r["truth"] == 0),
            "sources": len({r["source"] for r in rows}),
            "spatially_undefined": len(undefined),
            "spatially_undefined_by_class": {
                "Fight": sum(1 for r in undefined if r["truth"] == 1),
                "NonFight": sum(1 for r in undefined if r["truth"] == 0)},
            "processing_failures": 0,
        },
        "frozen_parameters": {
            "B0_threshold": temporal_threshold, "B1_theta_s": theta_s,
            "F1_theta_or": theta_or, "F2_theta_f": theta_f, "F2_weight": weight,
            "temporal_reference_n": len(temporal_reference),
            "spatial_reference_n": len(spatial_reference),
            "normalisation_source": "development carve values stored in the frozen protocol; "
                                    "no primary value was added to any reference",
        },
        "results": results,
        "paired_comparisons": comparisons,
        "descriptive_only": ["F1_or_gate", "B1_spatial_only"],
        "post_hoc_selection_performed": False,
        "provenance": {
            "primary_windows_sha256": file_sha256(PRIMARY_WINDOWS),
            "temporal_checkpoint_sha256": protocol["reproducibility"]["temporal_checkpoint_sha256"],
            "yolo_sha256": file_sha256(YOLO_WEIGHTS),
            "spatial_scores_sha256": file_sha256(SPATIAL_OUT),
            "source_grouping": "metric_intervals.source_video_id",
            "statistical_method": "exact McNemar on discordant pairs; paired percentile "
                                  "bootstrap, 10000 resamples, seed 42",
            "bootstrap_seed": 42, "bootstrap_resamples": 10000, "significance_level": 0.05,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool_sha256": file_sha256(Path(__file__).resolve()),
            "spatial_wall_seconds": spatial_payload["wall_seconds"],
        },
        "per_clip": rows,
    }
    RESULT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("FINAL CONFIRMATORY PRIMARY EVALUATION")
    print("=" * 78)
    print(f"  sample: {report['sample']}")
    print(f"\n  {'candidate':<20}{'tp':>5}{'fp':>5}{'tn':>5}{'fn':>5}{'recall':>9}"
          f"{'spec':>8}{'F1':>8}{'acc':>8}{'ROC':>8}")
    for name, entry in results.items():
        auc = f"{entry['roc_auc']:.4f}" if entry["roc_auc"] is not None else "   n/a"
        print(f"  {name:<20}{entry['tp']:>5}{entry['fp']:>5}{entry['tn']:>5}{entry['fn']:>5}"
              f"{entry['recall']:>9.4f}{entry['specificity']:>8.4f}"
              f"{entry['f1']:>8.4f}{entry['accuracy']:>8.4f}{auc:>8}")
    print(f"\n  PRIMARY ENDPOINT  B0 vs F2 (overall): b={primary['b']} c={primary['c']} "
          f"p={primary['p_value']:.6g}")
    fight = comparisons["F2_rank_sum"]["fight_detection"]
    nonfight = comparisons["F2_rank_sum"]["nonfight_false_alarms"]
    print(f"    Fight stratum   : b={fight['b']} c={fight['c']} p={fight['p_value']:.6g}")
    print(f"    NonFight stratum: b={nonfight['b']} c={nonfight['c']} p={nonfight['p_value']:.6g}")
    clip = comparisons["F2_rank_sum"]["accuracy_difference_clip_level"]
    source = comparisons["F2_rank_sum"]["accuracy_difference_source_level"]
    print(f"    acc diff {clip['difference']:+.4f}  clip CI [{clip['ci_low']:+.4f},"
          f"{clip['ci_high']:+.4f}]  source CI [{source['ci_low']:+.4f},{source['ci_high']:+.4f}]")
    print(f"\n  DECISION: {decision}")
    print(f"\nWrote {RESULT_OUT}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("score", "analyse"))
    parser.add_argument("--i-am-authorized", action="store_true", dest="authorized")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args(argv)

    if not args.authorized:
        print("REFUSED: the primary 394 split is protected. This tool runs the ONE-SHOT "
              "confirmatory evaluation and requires --i-am-authorized.", file=sys.stderr)
        return 3
    try:
        return command_score(args) if args.command == "score" else command_analyse(args)
    except ConfirmatoryError as error:
        print(f"CONFIRMATORY EVALUATION STOPPED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
