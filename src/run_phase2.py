"""Run Phase 2 object detection demo: sample image and sample video."""

from pathlib import Path

from config import ensure_directories, DATA_DIR, OUTPUTS_DIR, DEFAULT_VIDEO_PATH
from yolo_utils import load_yolo_model
from detector import detect_image, detect_video


def create_sample_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np

    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (280, 200), (0, 255, 0), 2)
    cv2.putText(img, "Sample", (90, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(path), img)
    return path


def main() -> None:
    ensure_directories()

    print("Loading YOLOv8s model (this may download weights)...")
    model = load_yolo_model("yolov8s.pt")

    sample_image = DATA_DIR / "sample_image.jpg"
    create_sample_image(sample_image)

    print("Running detection on sample image...")
    detect_image(model, sample_image)

    print("Running detection on sample video...")
    output_video = OUTPUTS_DIR / "annotated_video.mp4"
    detect_video = globals().get("detect_video")
    if not detect_video:
        from detector import detect_video as _dv

        _dv(model, DEFAULT_VIDEO_PATH, output_video)
    else:
        detect_video(model, DEFAULT_VIDEO_PATH, output_video)

    print("Phase 2 object detection demo finished.")


if __name__ == "__main__":
    main()
