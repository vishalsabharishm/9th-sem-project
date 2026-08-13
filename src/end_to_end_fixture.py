"""End-to-end verification runner for a user-supplied person-video fixture.

The repository deliberately does not ship a real-person fixture. Place a
short, lawfully usable surveillance-style video at ``data/person_fixture.mp4``
to exercise the existing Phase 2--4 components together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from abnormal_event_detector import AbnormalEventDetector, EventDetection
    from config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_directories
    from detection import DetectionRecord, detect_frame, load_yolo_model
    from risk_assessment import RiskAssessment, RiskAssessor, write_risk_assessments
    from tracker import SimpleTracker, Track, build_tracking_snapshot
    from video_loader import open_video
except ImportError:  # pragma: no cover - supports package execution
    from src.abnormal_event_detector import AbnormalEventDetector, EventDetection
    from src.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_directories
    from src.detection import DetectionRecord, detect_frame, load_yolo_model
    from src.risk_assessment import RiskAssessment, RiskAssessor, write_risk_assessments
    from src.tracker import SimpleTracker, Track, build_tracking_snapshot
    from src.video_loader import open_video


PERSON_FIXTURE_PATH = DATA_DIR / "person_fixture.mp4"
DEFAULT_MODEL_PATH = MODELS_DIR / "yolov8s.pt"
DEFAULT_RESULTS_PATH = OUTPUTS_DIR / "tracking_results.json"
DEFAULT_RISK_OUTPUT_PATH = OUTPUTS_DIR / "risk_assessment.json"


@dataclass(frozen=True)
class EndToEndFixtureResult:
    """Summary returned after processing a real-person fixture."""

    frames_processed: int
    detection_count: int
    person_detected: bool
    tracked_object_count: int
    persistent_ids_produced: bool
    events: List[EventDetection]
    results_json_path: Path
    risk_assessments: List[RiskAssessment]
    risk_assessment_path: Path


def _class_ids(records: List[DetectionRecord], model_names) -> np.ndarray:
    """Map detection labels back to the model's class IDs for ``SimpleTracker``."""
    names = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
    ids_by_name = {str(name): int(class_id) for class_id, name in names}
    return np.asarray([ids_by_name[record.class_name] for record in records], dtype=int)


def _tracking_record(frame_number: int, track: Track, model_names) -> dict:
    """Return a JSON-native record compatible with the ByteTrack export schema."""
    name = model_names.get(track.class_id, str(track.class_id)) if isinstance(model_names, dict) else model_names[track.class_id]
    return {
        "frame_number": frame_number,
        "track_id": track.id,
        "class_name": str(name),
        "confidence": float(track.conf),
        "bounding_box": [int(value) for value in track.bbox],
    }


def run_person_fixture(
    video_path: Path = PERSON_FIXTURE_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    results_json_path: Path = DEFAULT_RESULTS_PATH,
    risk_assessment_path: Path = DEFAULT_RISK_OUTPUT_PATH,
    confidence_threshold: float = 0.25,
    max_frames: Optional[int] = 120,
) -> EndToEndFixtureResult:
    """Run loader -> YOLO -> SimpleTracker -> event rules on a supplied video."""
    ensure_directories()
    fixture_path = Path(video_path)
    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"Person fixture not found: {fixture_path}. "
            "Provide data/person_fixture.mp4 before running this verification."
        )

    model = load_yolo_model(model_path)
    tracker = SimpleTracker()
    event_detector = AbnormalEventDetector()
    risk_assessor = RiskAssessor()
    capture, metadata = open_video(fixture_path)
    all_events: List[EventDetection] = []
    risk_assessments: List[RiskAssessment] = []
    tracking_records: List[dict] = []
    detection_count = 0
    person_detected = False
    frames_processed = 0
    id_occurrences: dict[int, int] = {}
    tracked_ids: set[int] = set()

    try:
        while max_frames is None or frames_processed < max_frames:
            success, frame = capture.read()
            if not success:
                break
            frames_processed += 1
            _, records = detect_frame(model, frame, frames_processed, confidence_threshold)
            detection_count += len(records)
            person_detected = person_detected or any(record.class_name == "person" for record in records)

            boxes = np.asarray([record.bounding_box for record in records], dtype=float)
            if not records:
                boxes = np.empty((0, 4), dtype=float)
            class_ids = _class_ids(records, model.names) if records else np.empty(0, dtype=int)
            confidences = np.asarray([record.confidence for record in records], dtype=float)
            tracks = tracker.update(boxes, class_ids, confidences)
            # A retained stale track is useful for visual continuity, but it
            # must not count as a fresh association or create a false
            # stationary event when this frame has no matching detection.
            current_tracks = [track for track in tracks if track.last_seen == tracker.frame_idx]
            frame_events = event_detector.process_snapshot(
                build_tracking_snapshot(frames_processed, current_tracks)
            )
            all_events.extend(frame_events)
            risk_assessments.extend(
                risk_assessor.assess_events(
                    frame_events,
                    frame_number=frames_processed,
                    timestamp_seconds=(frames_processed - 1) / metadata.fps if metadata.fps > 0 else None,
                )
            )

            for track in current_tracks:
                tracked_ids.add(track.id)
                id_occurrences[track.id] = id_occurrences.get(track.id, 0) + 1
                tracking_records.append(_tracking_record(frames_processed, track, model.names))
    finally:
        capture.release()

    results_path = Path(results_json_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(tracking_records, indent=2), encoding="utf-8")
    risk_output_path = write_risk_assessments(risk_assessments, risk_assessment_path)
    return EndToEndFixtureResult(
        frames_processed=frames_processed,
        detection_count=detection_count,
        person_detected=person_detected,
        tracked_object_count=len(tracked_ids),
        persistent_ids_produced=any(count >= 2 for count in id_occurrences.values()),
        events=all_events,
        results_json_path=results_path,
        risk_assessments=risk_assessments,
        risk_assessment_path=risk_output_path,
    )
