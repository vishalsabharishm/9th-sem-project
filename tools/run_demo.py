#!/usr/bin/env python3
"""
tools/run_demo.py

End-to-end demo: video -> YOLO detection -> tracking -> Phase-4 abnormal-event
rules -> temporal violence signal -> risk assessment -> explanation, producing
an annotated video, a risk-assessment JSON, and a markdown explanation report.

This demo uses REAL RWF-2000 evaluation clips (data/rwf2000/RWF-2000/val/...)
and REAL, already-computed per-window violence probabilities
(temporal_risk/primary_window_scores.csv), scored by the clean-baseline
checkpoint on Kaggle -- see docs/clean_baseline_results.md and
temporal_risk/window_scoring_manifest.json for exact provenance. The
checkpoint itself (best.pt, 132.74 MB) has not been downloaded to this
machine, so live temporal-model inference is not run here; everything else
(YOLO detection, tracking, Phase-4 rules, frozen aggregation, risk mapping,
explanation generation) runs live, on this machine, on this video, for real.

The demo replays the precomputed window scores causally: the "TEMPORAL
VIOLENCE SIGNAL" overlay only turns on once enough of the clip has been
"seen" (frame-by-frame) for the frozen aggregation rule to have fired, the
same way a live deployment would only know once enough windows have scored.

Usage:
    python tools/run_demo.py --clip fight      # trtrhrt_1049.avi (Fight, correctly flagged)
    python tools/run_demo.py --clip nonfight   # ZCUy99AN_0.avi (NonFight, correctly not flagged)
    python tools/run_demo.py --clip fp         # 39BFeYnbu-I_0.avi (NonFight, a real false positive --
                                                #   included on purpose so the demo doesn't hide the
                                                #   system's honest failure modes)
    python tools/run_demo.py --video-path <path> --clip-key <key-in-primary_window_scores.csv>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from abnormal_event_detector import AbnormalEventDetector, EventDetection  # noqa: E402
from config import OUTPUTS_DIR, ensure_directories  # noqa: E402
from detection import DEFAULT_MODEL_PATH, load_yolo_model  # noqa: E402
from risk_assessment import RiskAssessor, write_risk_assessments  # noqa: E402
from temporal_event_adapter import (  # noqa: E402
    TEMPORAL_EVENT_TYPE,
    FrozenAggregationRule,
    PrecomputedWindowScoreSource,
    TemporalEventAdapter,
    WindowScore,
)
from tracker import SimpleTracker, active_tracks, build_tracking_snapshot, draw_tracks  # noqa: E402


DEMO_CLIPS = {
    "fight": {
        "video": REPO_ROOT / "data" / "rwf2000" / "RWF-2000" / "val" / "Val_Fight" / "trtrhrt_1049.avi",
        "clip_key": "val/Val_Fight/trtrhrt_1049.avi",
    },
    "nonfight": {
        "video": REPO_ROOT / "data" / "rwf2000" / "RWF-2000" / "val" / "Val_NonFight" / "ZCUy99AN_0.avi",
        "clip_key": "val/Val_NonFight/ZCUy99AN_0.avi",
    },
    "fp": {
        "video": REPO_ROOT / "data" / "rwf2000" / "RWF-2000" / "val" / "Val_NonFight" / "39BFeYnbu-I_0.avi",
        "clip_key": "val/Val_NonFight/39BFeYnbu-I_0.avi",
    },
}

FROZEN_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"
PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"


def current_window_probability(windows: List[WindowScore], frame_idx: int) -> Optional[float]:
    """Score to display for this frame: the highest of any window currently
    covering it, or the most recently completed window if none does yet."""
    active = [w for w in windows if w.first_frame <= frame_idx <= w.last_frame]
    if active:
        return max(w.fight_probability for w in active)
    seen = [w for w in windows if w.last_frame <= frame_idx]
    if seen:
        return seen[-1].fight_probability
    return None


def draw_hud(frame: np.ndarray, frame_idx: int, current_prob: Optional[float],
             temporal_fired: bool, risk_level: str) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay_h = 90
    cv2.rectangle(frame, (0, 0), (width, overlay_h), (20, 20, 20), -1)
    prob_text = (
        f"Windowed fight probability: {current_prob:.3f}"
        if current_prob is not None else "Windowed fight probability: n/a"
    )
    cv2.putText(frame, prob_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"frame {frame_idx}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if temporal_fired:
        cv2.putText(frame, "TEMPORAL VIOLENCE SIGNAL", (10, 68), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"risk: {risk_level}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


def write_explanation_report(path: Path, summary: dict, windows: List[WindowScore],
                              risk_assessments, frozen_rule: FrozenAggregationRule) -> None:
    lines = []
    lines.append(f"# Demo explanation: {summary['clip_key']}\n\n")
    lines.append(
        f"Ground-truth label (dataset, shown for transparency only -- **not** used by "
        f"the pipeline's decision): **{summary['ground_truth_label']}**\n\n"
    )
    lines.append(f"Frames processed: {summary['frames_processed']}\n\n")
    lines.append("## Temporal violence signal\n\n")
    lines.append(
        f"Frozen aggregation rule: `{frozen_rule.rule}` params=`{frozen_rule.params}` "
        f"(selected by `tools/select_temporal_aggregation.py` on the 240-clip carve split only; "
        f"see `temporal_risk/frozen_aggregation.json`).\n\n"
    )
    if summary["temporal_violence_signal_fired"]:
        lines.append(f"**FIRED** at frame {summary['temporal_signal_first_frame']}.\n\n")
    else:
        lines.append("Did not fire.\n\n")
    lines.append(
        "Per-window scores (`measured_model_probability`, from the clean-baseline checkpoint, "
        "precomputed on Kaggle -- see `temporal_risk/window_scoring_manifest.json` and "
        "`docs/clean_baseline_results.md`):\n\n"
    )
    lines.append("| window | frames | fight_probability |\n|---:|---|---:|\n")
    for w in windows:
        lines.append(f"| {w.window_index} | [{w.first_frame},{w.last_frame}] | {w.fight_probability:.4f} |\n")
    lines.append("\n## Risk assessments\n\n")
    if not risk_assessments:
        lines.append("No events were raised for this clip.\n")

    # The Phase-4 rule engine fires every frame a condition holds (e.g. a
    # person remains stationary), so a single clip can generate hundreds of
    # near-duplicate records. Summarize those by type/risk; give the
    # Temporal Violence Signal -- the one genuinely new, per-clip event this
    # integration adds -- full detail, since there is at most one of it.
    from collections import defaultdict

    by_type: dict = defaultdict(list)
    for ra in risk_assessments:
        by_type[(ra.event_type, ra.risk_level)].append(ra)

    lines.append(
        "Phase-4 rule-engine events fire every frame their condition holds "
        "(e.g. a person remaining stationary re-fires each frame), so counts "
        "below reflect frame-level persistence, not distinct incidents:\n\n"
    )
    lines.append("| event type | risk level | occurrences | example frames |\n|---|---|---:|---|\n")
    for (event_type, risk_level), items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        example_frames = ", ".join(str(ra.frame_number) for ra in items[:5])
        lines.append(f"| {event_type} | {risk_level} | {len(items)} | {example_frames} ... |\n")

    temporal_ras = [ra for ra in risk_assessments if ra.event_type == TEMPORAL_EVENT_TYPE]
    if temporal_ras:
        lines.append("\n### Temporal Violence Signal -- full detail\n\n")
        for ra in temporal_ras:
            lines.append(
                f"- risk **{ra.risk_level}**, confidence={ra.confidence}, "
                f"provenance={ra.confidence_provenance}\n  reason: {ra.reason}\n"
            )
            for ev in ra.evidence:
                lines.append(f"  - evidence: {ev}\n")

    Path(path).write_text("".join(lines), encoding="utf-8")


def run_demo(video_path: Path, clip_key: str, model_path: Path, output_dir: Path) -> dict:
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_rule = FrozenAggregationRule.load(FROZEN_JSON)
    score_source = PrecomputedWindowScoreSource(PRIMARY_CSV)
    windows = score_source.get(clip_key)
    if windows is None:
        raise SystemExit(f"No precomputed window scores found for clip '{clip_key}' in {PRIMARY_CSV}")
    true_label = score_source.true_label(clip_key)

    adapter = TemporalEventAdapter(frozen_rule)
    temporal_event = adapter.evaluate_clip(clip_key, windows)

    model = load_yolo_model(model_path)
    tracker = SimpleTracker()
    event_detector = AbnormalEventDetector()
    risk_assessor = RiskAssessor()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = Path(clip_key).stem
    out_video_path = output_dir / f"demo_{stem}.mp4"
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video writer: {out_video_path}")

    all_events: List[EventDetection] = []
    frame_idx = 0
    temporal_signal_frame: Optional[int] = None
    running_scores: List[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=0.25, verbose=False)
        r = results[0]
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes = np.empty((0, 4))
            confidences = np.array([])
            class_ids = np.array([])

        # SimpleTracker retains a track for max_age frames after its last
        # detection, with its bounding box frozen at the last observed
        # position. Those retained tracks must not reach the rule engine (a
        # frozen box reads as a stationary person) and must not be drawn (a
        # frozen box reads as a ghost detection). src/end_to_end_fixture.py
        # already filters them; the demo does the same here.
        tracks = active_tracks(tracker.update(boxes, class_ids, confidences), tracker.frame_idx)
        snapshot = build_tracking_snapshot(frame_idx + 1, tracks)
        frame_events = event_detector.process_snapshot(snapshot)
        all_events.extend(frame_events)

        # Causal replay: only count a window once its last frame has been "seen".
        newly_seen = [w for w in windows if w.last_frame == frame_idx]
        running_scores.extend(w.fight_probability for w in newly_seen)
        fired_so_far = frozen_rule.decide(running_scores) if running_scores else False
        if fired_so_far and temporal_signal_frame is None:
            temporal_signal_frame = frame_idx

        current_prob = current_window_probability(windows, frame_idx)
        abnormal_ids = {e.object_id for e in frame_events if e.object_id is not None}
        annotated = draw_tracks(frame, tracks, model.names, abnormal_ids)
        annotated = draw_hud(
            annotated, frame_idx, current_prob,
            temporal_fired=fired_so_far,
            risk_level="High" if fired_so_far else "-",
        )
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    if temporal_event is not None:
        temporal_event.frame_number = (
            (temporal_signal_frame + 1) if temporal_signal_frame is not None else temporal_event.frame_number
        )
        all_events.append(temporal_event)

    risk_assessments = risk_assessor.assess_events(all_events)
    risk_json_path = output_dir / f"demo_{stem}_risk_assessment.json"
    write_risk_assessments(risk_assessments, risk_json_path)

    summary = {
        "clip_key": clip_key,
        "video_path": str(video_path),
        "ground_truth_label": true_label,
        "frames_processed": frame_idx,
        "n_phase4_rule_events": len([e for e in all_events if e.event_type != TEMPORAL_EVENT_TYPE]),
        "temporal_violence_signal_fired": temporal_event is not None,
        "temporal_signal_first_frame": temporal_signal_frame,
        "temporal_event_evidence": temporal_event.evidence if temporal_event is not None else None,
        "annotated_video": str(out_video_path),
        "risk_assessment_json": str(risk_json_path),
    }

    report_path = output_dir / f"demo_{stem}_explanation.md"
    write_explanation_report(report_path, summary, windows, risk_assessments, frozen_rule)
    summary["explanation_report"] = str(report_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end surveillance demo pipeline.")
    parser.add_argument("--clip", choices=sorted(DEMO_CLIPS.keys()), default="fight",
                         help="Which built-in demo clip to run (default: fight).")
    parser.add_argument("--video-path", type=Path, default=None, help="Override: path to a video file.")
    parser.add_argument("--clip-key", type=str, default=None,
                         help="Override: clip key as it appears in temporal_risk/primary_window_scores.csv.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "demo")
    args = parser.parse_args()

    if args.video_path and args.clip_key:
        video_path, clip_key = args.video_path, args.clip_key
    else:
        preset = DEMO_CLIPS[args.clip]
        video_path, clip_key = preset["video"], preset["clip_key"]

    if not Path(video_path).exists():
        raise SystemExit(f"Demo video not found: {video_path}")

    summary = run_demo(Path(video_path), clip_key, args.model_path, args.output_dir)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
