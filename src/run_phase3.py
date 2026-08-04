"""Run the Phase 4 abnormal-event demo: YOLOv8 detections + tracking + rules."""

from config import ensure_directories, OUTPUTS_DIR, DEFAULT_VIDEO_PATH
from yolo_utils import load_yolo_model
from tracker import track_video


def main() -> None:
    ensure_directories()

    print("Loading YOLOv8s model (this may download weights)...")
    model = load_yolo_model("yolov8s.pt")

    print("Running abnormal-event detection on sample video...")
    output_video = OUTPUTS_DIR / "phase4_abnormal_events.mp4"
    track_video(model, DEFAULT_VIDEO_PATH, output_video)

    print("Phase 4 abnormal-event detection demo finished.")


if __name__ == "__main__":
    main()
