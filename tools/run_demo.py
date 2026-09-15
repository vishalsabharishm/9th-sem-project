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
from checkpoint_identity import file_sha256  # noqa: E402
from temporal_event_adapter import (  # noqa: E402
    PROVENANCE_LIVE_INFERENCE,
    TEMPORAL_EVENT_TYPE,
    FrozenAggregationRule,
    PrecomputedWindowScoreSource,
    TemporalEventAdapter,
    WindowScore,
    window_score_from,
)
from operational_fusion import (  # noqa: E402
    FrozenFusionProtocol,
    FusionProtocolError,
    SpatialFeatureAccumulator,
)
from temporal_runtime import (  # noqa: E402
    RUNTIME_CLIP_LENGTH,
    RUNTIME_STRIDE,
    build_violence_engine,
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

# Shown when a clip has no row in the evaluation CSV and therefore no dataset
# label -- which is the normal case for live inference on arbitrary footage.
# The absence is preserved separately in ``ground_truth_available`` so a
# consumer can distinguish "no label exists" from a label that happens to read
# like one; only the display string changes.
GROUND_TRUTH_UNAVAILABLE = "Not available"


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


# --------------------------------------------------------------------------
# Window-score sources
#
# The demo used to load every window score up front and call
# ``evaluate_clip`` BEFORE the frame loop, then replay when the decision would
# have fired. That made causality a display property rather than a
# computation, and left no seam a live model could occupy.
#
# A source now yields window scores DURING the loop, and the clip-level
# decision is taken afterwards from what the source actually produced. The CSV
# replay source and a future live-inference source differ only in where the
# numbers come from; everything downstream is identical.
# --------------------------------------------------------------------------

SOURCE_CSV = "csv"
SOURCE_LIVE = "live"
SUPPORTED_TEMPORAL_SOURCES = (SOURCE_CSV, SOURCE_LIVE)


class WindowScoreSource:
    """Yields sliding-window scores as frames are processed."""

    name = "abstract"

    def windows_completed_at(self, frame_idx: int, frame: np.ndarray) -> List[WindowScore]:
        """Return the windows that complete at this frame, in window order."""
        raise NotImplementedError

    def hud_probability(
        self, frame_idx: int, observed: List[WindowScore]
    ) -> Optional[float]:
        """The probability to display on the overlay for this frame."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Provenance of the scores this source produced, for the run record."""
        return {"temporal_source": self.name}


class PrecomputedReplaySource(WindowScoreSource):
    """Replays real scores transcribed from Kaggle, keyed by clip.

    Behaviour is deliberately identical to the pre-inversion demo, including
    the HUD's look-ahead: ``current_window_probability`` reports the highest
    score of any window *covering* the current frame, which for a window not
    yet complete is information a live system could not have. That is a known
    property of the replay (audit finding E2), and preserving it exactly is
    what keeps this refactor byte-identical. A live source cannot look ahead
    and will necessarily differ here.
    """

    name = SOURCE_CSV

    def __init__(self, clip_key: str, windows: List[WindowScore]) -> None:
        self.clip_key = clip_key
        self._windows = list(windows)

    def windows_completed_at(self, frame_idx: int, frame: np.ndarray) -> List[WindowScore]:
        return [w for w in self._windows if w.last_frame == frame_idx]

    def hud_probability(self, frame_idx: int, observed: List[WindowScore]) -> Optional[float]:
        return current_window_probability(self._windows, frame_idx)

    def describe(self) -> dict:
        return {
            "temporal_source": self.name,
            "scores_from": str(PRIMARY_CSV),
            "clip_key": self.clip_key,
            "note": (
                "real per-window probabilities from the clean-baseline checkpoint, "
                "precomputed on Kaggle and replayed here; no model ran in this process"
            ),
        }


class LiveInferenceSource(WindowScoreSource):
    """Scores windows by running R3D-18 on the frames as they arrive.

    Every frame is handed to ``TemporalInferenceEngine``, which buffers it and
    runs a forward pass whenever a 16-frame window completes at stride 8. The
    result is converted by ``temporal_event_adapter.window_score_from``, which
    refuses a prediction whose weights were not trained for this task.

    Model construction, preprocessing and provenance validation are NOT
    reimplemented here: they come from ``temporal_runtime.build_violence_engine``,
    the same factory ``tools/score_temporal_windows.py`` uses, so the runtime
    and the offline scorer cannot drift apart on the regime that produced the
    committed evidence.

    CAUSALITY. Unlike the replay source, this one cannot look ahead: a score
    exists only once its window's sixteenth frame has been observed and the
    forward pass has run. ``hud_probability`` therefore reports the most
    recently *completed* window, never one still filling. This is a visible
    behavioural difference from CSV mode and it is the correct one -- the
    replay's look-ahead is an artifact of having the whole file up front.
    """

    name = SOURCE_LIVE

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cpu",
        clip_length: int = RUNTIME_CLIP_LENGTH,
        stride: int = RUNTIME_STRIDE,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.clip_length = clip_length
        self.stride = stride

        # Raises CheckpointIntegrityError on a missing, random-init, smoke-test
        # or otherwise unverifiable checkpoint. There is no fallback path.
        self.engine, self.verdict = build_violence_engine(
            self.checkpoint, device=device, clip_length=clip_length, stride=stride
        )
        self.checkpoint_sha256 = file_sha256(self.checkpoint)
        self.checkpoint_bytes = self.checkpoint.stat().st_size
        self.model_provenance = self.engine.model.provenance

        self._window_index = 0
        self._inference_seconds: List[float] = []

    def windows_completed_at(self, frame_idx: int, frame: np.ndarray) -> List[WindowScore]:
        """Feed one frame to the model; return a score only if a window closed."""
        result = self.engine.add_frame(frame, frame_number=frame_idx)
        if result is None:
            return []
        score = window_score_from(result, self._window_index)
        self._window_index += 1
        self._inference_seconds.append(result.inference_seconds)
        return [score]

    def hud_probability(self, frame_idx: int, observed: List[WindowScore]) -> Optional[float]:
        """Strictly causal: the most recently completed window, or nothing yet."""
        return observed[-1].fight_probability if observed else None

    @property
    def total_inference_seconds(self) -> float:
        return float(sum(self._inference_seconds))

    def describe(self) -> dict:
        per_window = self._inference_seconds
        return {
            "temporal_source": self.name,
            "score_provenance": PROVENANCE_LIVE_INFERENCE,
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "model_provenance": self.model_provenance,
            "device": self.device,
            "window_geometry": {
                "clip_length": self.clip_length,
                "stride": self.stride,
                "windows_completed": self._window_index,
            },
            "inference_seconds_total": round(self.total_inference_seconds, 4),
            "inference_seconds_per_window": [round(value, 4) for value in per_window],
            "inference_seconds_mean_per_window": (
                round(self.total_inference_seconds / len(per_window), 4) if per_window else None
            ),
            "checkpoint_warnings": list(self.verdict.warnings),
            "note": (
                "every score was produced by an R3D-18 forward pass in this "
                "process on frames decoded from this video; no CSV was read"
            ),
        }


def build_window_source(
    temporal_source: str,
    clip_key: str,
    windows: Optional[List[WindowScore]],
    checkpoint: Optional[Path] = None,
    device: str = "cpu",
) -> WindowScoreSource:
    """Select a window-score source by name. Never falls back silently.

    A failure in live mode raises. Degrading to replay would produce an output
    indistinguishable from a genuine live run, which is exactly the confusion
    this whole integration exists to prevent.
    """
    if temporal_source == SOURCE_CSV:
        if windows is None:
            raise SystemExit(
                f"No precomputed window scores found for clip '{clip_key}' in "
                f"{PRIMARY_CSV}. Use --temporal-source live --checkpoint <path> "
                "to score an unrecorded clip."
            )
        return PrecomputedReplaySource(clip_key, windows)

    if temporal_source == SOURCE_LIVE:
        if checkpoint is None:
            raise SystemExit(
                "--temporal-source live requires --checkpoint <path to best.pt>. "
                "There is no default and no fallback: an unverified or absent "
                "checkpoint must never silently become a CSV replay or an "
                "untrained model."
            )
        return LiveInferenceSource(checkpoint, device=device)

    raise SystemExit(
        f"Unknown temporal source {temporal_source!r}; expected one of "
        f"{list(SUPPORTED_TEMPORAL_SOURCES)}."
    )


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


def run_demo(
    video_path: Path,
    clip_key: str,
    model_path: Path,
    output_dir: Path,
    *,
    temporal_source: str = SOURCE_CSV,
    checkpoint: Optional[Path] = None,
    device: str = "cpu",
) -> dict:
    """Run the full pipeline on one clip.

    ``temporal_source`` selects where sliding-window scores come from. It is
    keyword-only with a CSV default so existing callers -- the CLI and
    ``web/server.py`` -- are unaffected.

    Window scores are produced DURING the frame loop by the selected source and
    accumulated; the clip-level decision is taken afterwards from what was
    actually observed. Previously the decision was computed before the loop
    from the whole CSV, and the loop only replayed when it would have fired.
    """
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_rule = FrozenAggregationRule.load(FROZEN_JSON)

    # A CSV row is required for replay, but NOT for live inference: live mode
    # must be able to score a clip that was never in the committed file, which
    # is also the strongest available evidence that a model actually ran.
    #
    # The CSV is opened here for two different reasons, and only one of them is
    # load-bearing in live mode. Replay needs the SCORES. Both modes consult it
    # for an optional dataset GROUND-TRUTH LABEL, which is display-only and
    # never reaches a decision. Constructing the source unconditionally made a
    # live run fail outright when the CSV was absent -- a dependency on a file
    # whose contents live mode does not use. A missing CSV now simply means no
    # label is available, which is a state the summary already represents.
    # Replay still fails loudly, in build_window_source, because for replay the
    # file genuinely is the evidence.
    windows = None
    true_label = None
    try:
        score_source = PrecomputedWindowScoreSource(PRIMARY_CSV)
        windows = score_source.get(clip_key)
        true_label = score_source.true_label(clip_key)
    except Exception:
        if temporal_source != SOURCE_LIVE:
            raise

    window_source = build_window_source(
        temporal_source, clip_key, windows, checkpoint=checkpoint, device=device
    )
    adapter = TemporalEventAdapter(frozen_rule)

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
    # Windows the source actually produced, in completion order.
    observed_windows: List[WindowScore] = []
    # The frozen spatial feature, measured from THIS video as it is decoded.
    # It keeps its own BehaviorAnalyzer so that driving it cannot perturb the
    # rule engine's analyzer, which holds separate state for a different job.
    spatial_feature = SpatialFeatureAccumulator()

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

        # Fed BEFORE the rule engine, because the frozen spatial scorer reads
        # get_previous_bbox before update_history and that ordering is part of
        # what produced the locked artifact. Separate analyzer, so the order of
        # these two calls relative to each other cannot matter.
        spatial_feature.observe(frame_idx, tracks)

        snapshot = build_tracking_snapshot(frame_idx + 1, tracks)
        frame_events = event_detector.process_snapshot(snapshot)
        all_events.extend(frame_events)

        # Windows complete as frames arrive; the decision is re-evaluated on
        # everything observed so far, exactly as a streaming system would.
        newly_seen = window_source.windows_completed_at(frame_idx, frame)
        observed_windows.extend(newly_seen)
        running_scores.extend(w.fight_probability for w in newly_seen)
        fired_so_far = frozen_rule.decide(running_scores) if running_scores else False
        if fired_so_far and temporal_signal_frame is None:
            temporal_signal_frame = frame_idx

        current_prob = window_source.hud_probability(frame_idx, observed_windows)
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

    # The clip-level decision, taken after the loop from the windows the source
    # actually produced -- not from a file read before the video was opened.
    temporal_event = adapter.evaluate_clip(clip_key, observed_windows)

    if temporal_event is not None:
        temporal_event.frame_number = (
            (temporal_signal_frame + 1) if temporal_signal_frame is not None else temporal_event.frame_number
        )
        all_events.append(temporal_event)

    risk_assessments = risk_assessor.assess_events(all_events)
    risk_json_path = output_dir / f"demo_{stem}_risk_assessment.json"
    write_risk_assessments(risk_assessments, risk_json_path)

    # Evidence fusion, evaluated once the whole video has been seen. The
    # spatial feature's denominator is a mean over every frame containing a
    # person, so it does not exist until now -- this is an offline whole-video
    # judgement and is labelled as one. A missing or edited protocol makes
    # fusion explicitly unavailable; it never falls back to default parameters.
    temporal_max = (
        max(w.fight_probability for w in observed_windows) if observed_windows else None
    )
    try:
        fusion = FrozenFusionProtocol.load().evaluate(temporal_max, spatial_feature.score)
    except FusionProtocolError as error:
        fusion = {
            "available": False,
            "reason": "protocol_unavailable",
            "detail": str(error),
        }

    summary = {
        "clip_key": clip_key,
        "video_path": str(video_path),
        "ground_truth_label": (
            true_label if true_label is not None else GROUND_TRUTH_UNAVAILABLE
        ),
        "ground_truth_available": true_label is not None,
        "frames_processed": frame_idx,
        "n_phase4_rule_events": len([e for e in all_events if e.event_type != TEMPORAL_EVENT_TYPE]),
        "temporal_violence_signal_fired": temporal_event is not None,
        "temporal_signal_first_frame": temporal_signal_frame,
        "temporal_event_evidence": temporal_event.evidence if temporal_event is not None else None,
        "annotated_video": str(out_video_path),
        "risk_assessment_json": str(risk_json_path),
        "temporal_provenance": window_source.describe(),
        "windows_observed": len(observed_windows),
        # The windows this run actually observed, in completion order. Live
        # mode has no CSV to read a timeline from, and a consumer must never
        # have to reconstruct one; replay yields the same list it replayed.
        "window_scores": [
            {
                "window_index": w.window_index,
                "first_frame": w.first_frame,
                "last_frame": w.last_frame,
                "fight_probability": w.fight_probability,
            }
            for w in observed_windows
        ],
        "temporal_max_probability": temporal_max,
        "temporal_evidence_available": bool(observed_windows),
        # With zero completed windows there is no temporal evidence at all, and
        # calling that "NonFight" would assert a negative the system never
        # measured -- the honest answer is that the question was not decidable.
        # A video needs 16 frames to close one window; replay clips always
        # close 17, so this state is reachable only on very short live input.
        "final_temporal_decision": (
            "Undetermined" if not observed_windows
            else "Fight" if temporal_event is not None
            else "NonFight"
        ),
        "undetermined_reason": (
            None if observed_windows else
            "No 16-frame window completed, so the temporal model never ran on "
            "this video. No temporal decision is claimed."
        ),
        "alert_risk_level": ("High" if temporal_event is not None else None),
        "spatial_fusion_feature": spatial_feature.as_dict(),
        "fusion": fusion,
    }

    report_path = output_dir / f"demo_{stem}_explanation.md"
    write_explanation_report(
        report_path, summary, observed_windows, risk_assessments, frozen_rule
    )
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
    parser.add_argument(
        "--temporal-source",
        choices=list(SUPPORTED_TEMPORAL_SOURCES),
        default=SOURCE_CSV,
        help="Where sliding-window violence scores come from. 'csv' (default) "
        "replays the committed per-window probabilities and runs no temporal "
        "model. 'live' runs R3D-18 on the decoded frames and requires "
        "--checkpoint; it never falls back to csv or to untrained weights.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Verified 2-class R3D-18 checkpoint. Required by "
        "--temporal-source live. Validated by src/checkpoint_identity.py "
        "before any inference runs.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for temporal inference (cpu or cuda)."
    )
    args = parser.parse_args()

    if args.video_path and args.clip_key:
        video_path, clip_key = args.video_path, args.clip_key
    elif args.video_path and args.temporal_source == SOURCE_LIVE:
        # Live inference needs no CSV row, so a clip key is only a label here.
        video_path, clip_key = args.video_path, str(args.video_path)
    else:
        preset = DEMO_CLIPS[args.clip]
        video_path, clip_key = preset["video"], preset["clip_key"]

    if not Path(video_path).exists():
        raise SystemExit(f"Demo video not found: {video_path}")

    summary = run_demo(
        Path(video_path),
        clip_key,
        args.model_path,
        args.output_dir,
        temporal_source=args.temporal_source,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
