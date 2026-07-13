"""Utilities for opening and previewing surveillance videos."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from config import DEFAULT_VIDEO_PATH, OUTPUTS_DIR, WINDOW_NAME, ensure_directories
except ImportError:  # pragma: no cover - supports package execution
    from src.config import DEFAULT_VIDEO_PATH, OUTPUTS_DIR, WINDOW_NAME, ensure_directories


def create_sample_video(video_path: Path) -> None:
    """Create a simple sample video if no real video file exists."""
    video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (320, 240))

    if not writer.isOpened():
        raise RuntimeError(f"Unable to create sample video at {video_path}")

    for index in range(60):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (280, 200), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Frame {index + 1}",
            (70, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    writer.release()
    print(f"Created sample video: {video_path}")


def load_video(video_path: Optional[str] = None) -> None:
    """Open a video, print its dimensions, and display frames until quit."""
    ensure_directories()

    video_file = Path(video_path) if video_path else DEFAULT_VIDEO_PATH

    if not video_file.exists():
        create_sample_video(video_file)

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_file}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0

    print(f"Video loaded from: {video_file}")
    print(f"Frame dimensions: {width}x{height}")
    print(f"Estimated FPS: {fps}")

    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_count += 1
        print(f"Displaying frame {frame_count}")

        try:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit key pressed. Closing preview.")
                break
        except cv2.error:
            output_path = OUTPUTS_DIR / f"frame_{frame_count:04d}.png"
            cv2.imwrite(str(output_path), frame)
            print(f"Display window unavailable. Saved frame preview to {output_path}")
            break

    capture.release()
    cv2.destroyAllWindows()
    print("Video preview finished.")


if __name__ == "__main__":
    load_video()
