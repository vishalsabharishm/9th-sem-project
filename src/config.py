"""Configuration settings for the surveillance video analysis project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_VIDEO_PATH = DATA_DIR / "sample_video.mp4"
WINDOW_NAME = "Surveillance Preview"


def ensure_directories() -> None:
    """Create required directories if they do not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
