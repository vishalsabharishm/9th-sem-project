"""Video loading, metadata inspection, and frame-preview utilities.

This module deliberately has no detection or analysis dependencies.  Future
pipeline stages can use :func:`open_video` and :func:`iter_frames` directly,
while the current Phase 1 workflow can use :func:`load_video` for a preview.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from config import DEFAULT_VIDEO_PATH, WINDOW_NAME
except ImportError:  # pragma: no cover - supports package execution
    from src.config import DEFAULT_VIDEO_PATH, WINDOW_NAME


PathLike = Union[str, Path]
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}


class VideoLoadError(ValueError):
    """Raised when a local video cannot be validated or opened."""


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable metadata required by later video-processing modules."""

    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def long_path(path: PathLike) -> str:
    """Return a path string this platform can stat and open past MAX_PATH.

    On Windows a path of 260 or more characters is rejected by ``stat``
    and ``open`` unless it carries the extended-length prefix, so a file
    that genuinely exists is reported as missing purely because its name
    is long. Dataset clips can exceed the limit on name length alone, so
    every filesystem call routes through this. Other platforms are
    returned unchanged.

    This is the single path-handling helper for the project: dataset
    tooling imports it from here rather than reimplementing it, so the
    rule lives in exactly one place.
    """
    text = os.path.abspath(str(path))
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):  # UNC share: \\server\share -> \\?\UNC\server\share
        return "\\\\?\\UNC" + text[1:]
    return "\\\\?\\" + text


def validate_video_file(video_path: PathLike) -> Path:
    """Validate a local video path and return its resolved ``Path``.

    File existence and type are checked before OpenCV is asked to decode the
    content. OpenCV-specific decoding errors are handled by :func:`open_video`.
    """
    if not video_path:
        raise VideoLoadError("A local video file path is required.")

    path = Path(video_path).expanduser()
    target = long_path(path)
    if not os.path.exists(target):
        raise VideoLoadError(f"Video file does not exist: {path}")
    if not os.path.isfile(target):
        raise VideoLoadError(f"Video path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise VideoLoadError(
            f"Unsupported video format '{path.suffix}'. Supported formats: {supported}"
        )

    try:
        return path.resolve()
    except OSError:
        # ``resolve`` can fail on an over-long Windows path; the absolute
        # path is still correct and is what OpenCV is handed.
        return Path(os.path.abspath(str(path)))


def read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Read metadata from an already-open OpenCV video capture."""
    if not capture.isOpened():
        raise VideoLoadError("Cannot read metadata from a closed video capture.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = frame_count / fps if fps > 0 else 0.0

    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )


def open_video(video_path: PathLike) -> Tuple[cv2.VideoCapture, VideoMetadata]:
    """Open a validated video and return its capture object and metadata.

    The caller owns the returned capture and must call ``release()`` when it is
    no longer needed. This makes the function suitable for future inference
    pipelines as well as the preview interface below.
    """
    path = validate_video_file(video_path)
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()
        raise VideoLoadError(f"OpenCV could not decode video file: {path}")

    metadata = read_video_metadata(capture)
    if metadata.width <= 0 or metadata.height <= 0:
        capture.release()
        raise VideoLoadError(f"Video has invalid frame dimensions: {path}")

    return capture, metadata


def resize_frame(
    frame: np.ndarray,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> np.ndarray:
    """Resize ``frame`` to fit within bounds without changing its aspect ratio.

    A frame is never enlarged: this keeps previewing efficient and prevents
    interpolated pixels from being passed to later computer-vision modules.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Cannot resize an empty frame.")
    if max_width is not None and max_width <= 0:
        raise ValueError("max_width must be a positive integer.")
    if max_height is not None and max_height <= 0:
        raise ValueError("max_height must be a positive integer.")

    if max_width is None and max_height is None:
        return frame

    height, width = frame.shape[:2]
    width_scale = max_width / width if max_width is not None else 1.0
    height_scale = max_height / height if max_height is not None else 1.0
    scale = min(width_scale, height_scale, 1.0)

    if scale == 1.0:
        return frame

    dimensions = (round(width * scale), round(height * scale))
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def iter_frames(
    capture: cv2.VideoCapture,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield sequential, optionally resized frames as ``(index, frame)`` pairs."""
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            return

        yield frame_index, resize_frame(frame, max_width, max_height)
        frame_index += 1


def print_video_metadata(video_path: Path, metadata: VideoMetadata) -> None:
    """Print metadata in a concise format suitable for command-line demos."""
    print(f"Video loaded: {video_path}")
    print(f"FPS: {metadata.fps:.2f}")
    print(f"Frame count: {metadata.frame_count}")
    print(f"Dimensions: {metadata.width}x{metadata.height}")
    print(f"Duration: {metadata.duration_seconds:.2f} seconds")


def preview_video(
    video_path: PathLike,
    max_width: Optional[int] = 960,
    max_height: Optional[int] = 540,
    window_name: str = WINDOW_NAME,
) -> bool:
    """Display a processed video preview and return ``False`` on load errors.

    Press ``q`` while the preview window is active to exit early. A display
    backend may be unavailable on headless systems; that condition is reported
    cleanly instead of producing an unhandled OpenCV exception.
    """
    capture: Optional[cv2.VideoCapture] = None
    try:
        validated_path = validate_video_file(video_path)
        capture, metadata = open_video(validated_path)
        print_video_metadata(validated_path, metadata)

        delay_ms = max(1, round(1000 / metadata.fps)) if metadata.fps > 0 else 1
        for _, frame in iter_frames(capture, max_width, max_height):
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                print("Preview closed by user.")
                break

        return True
    except VideoLoadError as error:
        print(f"Unable to load video: {error}")
        return False
    except cv2.error as error:
        print(f"Unable to display video preview: {error}")
        return False
    finally:
        if capture is not None:
            capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def load_video(video_path: Optional[PathLike] = None) -> bool:
    """Run the Phase 1 preview workflow using the configured default if needed."""
    return preview_video(video_path or DEFAULT_VIDEO_PATH)


if __name__ == "__main__":
    load_video()
