"""Entry point for the Phase 1 video loader demo."""

try:
    from video_loader import load_video
except ImportError:  # pragma: no cover - supports package execution
    from src.video_loader import load_video


def main() -> None:
    """Run the basic video loading workflow."""
    print("Starting Phase 1 video loader...")
    load_video()


if __name__ == "__main__":
    main()
