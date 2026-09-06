#!/usr/bin/env python3
"""
tools/score_fresh_carve.py

Score the fresh validation carve with BOTH branches, once, so that a frozen
fusion protocol can be fitted on the fit split and evaluated on the eval split.

WHAT IT PRODUCES PER CLIP
-------------------------
Temporal: the 17 sliding-window Fight probabilities from the recovered
clean-baseline R3D-18 checkpoint, using the same clip_length=16 / stride=8
regime and the same shared factory (``temporal_runtime.build_violence_engine``)
that the offline scorer and the live demo use, so nothing about the temporal
branch is re-implemented here.

Spatial: per-clip aggregates of the geometric quantities the Phase-4 runtime
already computes -- person counts, normalised centroid distances, group
diagonal, stationary frames, and per-track motion -- obtained by driving the
committed analyzers with the committed configuration (conf 0.25, SimpleTracker
defaults, ``active_tracks`` filtering).

TWO MEASUREMENT CAVEATS, RECORDED IN EVERY OUTPUT
-------------------------------------------------
ACCELERATION IS DERIVED. The runtime has no acceleration quantity. It is the
first difference of ``BehaviorAnalyzer._movement_pixels`` between consecutive
frames of the same track, computed here. It must never be described as a
runtime signal.

SPEED IS IN RAW PIXELS. ``_movement_pixels`` is centroid displacement in image
pixels, so it scales with apparent person size and with camera motion. A
scale-normalised variant (speed divided by the person-group diagonal) is
recorded alongside it, and the frozen protocol uses the normalised form
precisely because it is the more conservative of the two -- on the existing
carve the raw form scored HIGHER, so this is not a choice made to flatter the
result. Camera motion is NOT compensated and remains an unresolved confounder.

This script fits nothing and selects nothing. It reads the frozen manifest and
writes measurements.

Usage:
    python tools/score_fresh_carve.py --split fit
    python tools/score_fresh_carve.py --split eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MANIFEST = REPO_ROOT / "temporal_risk" / "fresh_validation_carve.json"
DATASET_ROOT = REPO_ROOT / "data" / "rwf2000" / "RWF-2000"
CHECKPOINT = REPO_ROOT / "models" / "temporal_violence" / "best.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "fusion"

RUNTIME_CONFIDENCE = 0.25          # exactly what tools/run_demo.py passes
CLIP_LENGTH, STRIDE = 16, 8


def centroid(track) -> tuple:
    x1, y1, x2, y2 = [float(v) for v in track.bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def score_clip(relative: str, model, engine, root: Path) -> dict:
    """One pass over a clip, producing both branches' measurements."""
    import cv2
    from tracker import SimpleTracker, active_tracks, build_tracking_snapshot
    from behavior_analyzer import BehaviorAnalyzer
    from abnormal_event_detector import AbnormalEventDetector
    from event_rules import (
        DEFAULT_MOVEMENT_THRESHOLD_PIXELS, DEFAULT_STATIONARY_FRAME_THRESHOLD,
    )

    path = root / relative
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"clip": relative, "error": "cannot open"}

    # One engine for the whole run, reset per clip: rebuilding it would reload a
    # 132 MB checkpoint for every clip and change nothing about the result.
    engine.reset()
    tracker, behaviour = SimpleTracker(), BehaviorAnalyzer()
    detector = AbnormalEventDetector()

    windows: List[float] = []
    counts, distances, diagonals, stationary = [], [], [], []
    speeds, accelerations = [], []
    previous_speed: Dict[int, float] = {}
    fired = {"Stationary Person": set(), "Restricted Area Entry": set(),
             "Crowding": set(), "Proximity/Interaction": set()}
    frames = zero_person = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1

        result = engine.add_frame(frame)
        if result is not None:
            windows.append(float(result.prediction.probabilities[0, 1]))

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
                speed = behaviour._movement_pixels(previous, track.bbox)
                speeds.append(speed)
                if track.id in previous_speed:
                    accelerations.append(abs(speed - previous_speed[track.id]))
                previous_speed[track.id] = speed

        if persons:
            xs = [float(t.bbox[0]) for t in persons] + [float(t.bbox[2]) for t in persons]
            ys = [float(t.bbox[1]) for t in persons] + [float(t.bbox[3]) for t in persons]
            diagonal = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
            diagonals.append(diagonal)
            for first, second in combinations(persons, 2):
                a, b = centroid(first), centroid(second)
                separation = float(np.hypot(a[0] - b[0], a[1] - b[1]))
                if diagonal > 0:
                    distances.append(separation / diagonal)

        for summary in behaviour.update_history(
            snapshot, DEFAULT_MOVEMENT_THRESHOLD_PIXELS, DEFAULT_STATIONARY_FRAME_THRESHOLD
        ):
            stationary.append(summary.stationary_frames)
        for event in detector.process_snapshot(snapshot):
            if event.event_type in fired:
                fired[event.event_type].add(frames - 1)

    capture.release()

    def stats(values):
        if not values:
            return {"mean": None, "median": None, "min": None, "max": None}
        return {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "min": float(np.min(values)), "max": float(np.max(values))}

    speed_stats, diagonal_stats = stats(speeds), stats(diagonals)
    normalised = (
        speed_stats["mean"] / diagonal_stats["mean"]
        if speed_stats["mean"] is not None and diagonal_stats["mean"] else None
    )
    return {
        "clip": relative,
        "frames": frames,
        "temporal_windows": windows,
        "temporal_max": max(windows) if windows else None,
        "zero_person_frames": zero_person,
        "person_count": stats(counts),
        "normalized_centroid_distance": stats(distances),
        "person_group_diagonal": diagonal_stats,
        "stationary_frames": stats(stationary),
        "speed_pixels": speed_stats,
        "acceleration_pixels_DERIVED": stats(accelerations),
        "speed_normalised_by_group_diagonal": normalised,
        "rule_fire_fraction": {
            name: len(hits) / max(frames, 1) for name, hits in fired.items()
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=("fit", "eval"), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    from checkpoint_identity import file_sha256
    from detection import load_yolo_model, DEFAULT_MODEL_PATH
    from temporal_runtime import build_violence_engine

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    members = manifest["splits"][args.split]["members"]
    labels = {m["clip"]: m["label"] for m in members}
    clips = [m["clip"] for m in members]

    model = load_yolo_model()
    engine, verdict = build_violence_engine(args.checkpoint, "cpu", CLIP_LENGTH, STRIDE)

    print(f"scoring {len(clips)} clips of the '{args.split}' split "
          f"({sum(1 for c in clips if labels[c]=='Fight')} Fight)")
    rows, started = [], time.perf_counter()
    for index, clip in enumerate(clips):
        record = score_clip(clip, model, engine, args.root)
        record["label"] = labels[clip]
        record["split"] = args.split
        rows.append(record)
        if (index + 1) % 10 == 0:
            elapsed = time.perf_counter() - started
            print(f"  {index+1}/{len(clips)}  {elapsed:.0f}s  "
                  f"({elapsed/(index+1):.1f}s/clip, eta {(len(clips)-index-1)*elapsed/(index+1)/60:.0f} min)",
                  flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "score_fresh_carve",
        "split": args.split,
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "yolo_sha256": file_sha256(Path(DEFAULT_MODEL_PATH)),
        "yolo_confidence": RUNTIME_CONFIDENCE,
        "clip_length": CLIP_LENGTH,
        "stride": STRIDE,
        "fitting_occurred": False,
        "primary_data_accessed": False,
        "caveats": {
            "acceleration": "DERIVED, not a runtime quantity: first difference of "
                            "BehaviorAnalyzer._movement_pixels",
            "speed": "raw image pixels; scales with apparent person size and camera "
                     "motion. Camera motion is NOT compensated.",
        },
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool_sha256": file_sha256(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "wall_seconds": time.perf_counter() - started,
        },
        "clips": rows,
    }
    out = args.output_dir / f"fresh_carve_{args.split}_scores.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
