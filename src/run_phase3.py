"""Run Phase 3 object tracking demo: uses YOLOv8 detections + tracker."""

from pathlib import Path

from config import ensure_directories, DATA_DIR, OUTPUTS_DIR, DEFAULT_VIDEO_PATH
from yolo_utils import load_yolo_model
from tracker import track_video


def main() -> None:
    ensure_directories()

    print("Loading YOLOv8s model (this may download weights)...")
    model = load_yolo_model("yolov8s.pt")

    print("Running tracking on sample video...")
    output_video = OUTPUTS_DIR / "tracked_video.mp4"
    track_video(model, DEFAULT_VIDEO_PATH, output_video)

    print("Phase 3 tracking demo finished.")


if __name__ == "__main__":
    main()
