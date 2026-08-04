"""Entry point for the Ultralytics multi-object-tracking demonstration."""

import logging

try:
    from config import DEFAULT_VIDEO_PATH
    from detection import DEFAULT_MODEL_PATH
    from tracker import TrackingError, run_object_tracking
except ImportError:  # pragma: no cover - supports package execution
    from src.config import DEFAULT_VIDEO_PATH
    from src.detection import DEFAULT_MODEL_PATH
    from src.tracker import TrackingError, run_object_tracking


def main() -> None:
    """Run ByteTrack-based YOLOv8 tracking on the configured sample video."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Starting YOLOv8 multi-object tracking...")
    try:
        result = run_object_tracking(DEFAULT_VIDEO_PATH, DEFAULT_MODEL_PATH)
    except TrackingError as error:
        print(f"Object tracking failed: {error}")
        return

    print(f"Tracker used: {result.tracker_name}")
    print(f"Tracking records: {len(result.records)}")
    print(f"Tracked video: {result.tracked_video_path}")
    print(f"Tracking JSON: {result.results_json_path}")


if __name__ == "__main__":
    main()
