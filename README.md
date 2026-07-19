# Explainable AI-Based Abnormal Event Detection and Risk Assessment in Surveillance Videos

## Project Overview
This project builds an explainable AI-based surveillance pipeline for abnormal event detection and risk assessment. The work so far focuses on a clear, modular foundation that can later support event reasoning, explainability, and risk scoring without overcomplicating the early architecture.

## Current Project Structure
```text
.
├── data/                           # Input media and sample assets
├── docs/                           # Project notes and documentation
├── models/                         # Model weights and checkpoints
├── outputs/                        # Generated images and videos
├── src/                            # Source code
│   ├── abnormal_event_detector.py  # Phase 4 scaffolding for event detection
│   ├── behavior_analyzer.py       # Phase 4 scaffolding for behavior analysis
│   ├── config.py                   # Project configuration and paths
│   ├── detector.py                 # YOLOv8 detection routines
│   ├── event_rules.py              # Phase 4 scaffolding for event rules
│   ├── main.py                     # Phase 1 entry point
│   ├── run_phase2.py               # Phase 2 demo runner
│   ├── run_phase3.py               # Phase 3 demo runner
│   ├── tracker.py                  # Lightweight IoU-based tracking implementation
│   ├── video_loader.py             # OpenCV video loading and preview logic
│   └── yolo_utils.py               # YOLO loading and annotation helpers
├── tests/                          # Test files
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Demos
From the project root:

### Phase 1
```bash
python src/main.py
```
This loads the sample video and previews it until the user presses q.

### Phase 2
```bash
python src/run_phase2.py
```
This runs YOLOv8-based object detection on a sample image and the sample video and writes outputs to the outputs folder.

### Phase 3
```bash
python src/run_phase3.py
```
This runs YOLOv8 detections, assigns persistent object IDs through the lightweight tracker, and writes a tracked output video to outputs/tracked_video.mp4.

## Completed Phases
### Phase 1 - Project Setup and Video Loader
Completed:
- Created a clean project structure
- Added dependency management and environment support
- Implemented OpenCV-based video loading and previewing

### Phase 2 - YOLOv8 Object Detection
Completed:
- Integrated YOLOv8-based object detection
- Added image and video annotation utilities
- Verified demo outputs are written successfully

### Phase 3 - Lightweight Tracking
Completed:
- Implemented a simple IoU-based tracker
- Assigned persistent IDs across frames
- Drew bounding boxes and motion trails
- Produced a tracked video output for downstream analysis

### Phase 4 Preparation
Added scaffolding for the next stage:
- abnormal_event_detector.py for future abnormal event detection
- event_rules.py for rule definitions and evaluation hooks
- behavior_analyzer.py for motion and interaction analysis structure

## Project Architecture
The current pipeline is intentionally simple and modular:
1. Video input is loaded using OpenCV.
2. YOLOv8 detects objects in each frame.
3. The tracker maintains object identity over time using IoU matching.
4. Tracking snapshots can now flow into future abnormal-event analysis modules.

This design keeps detection, tracking, and future event reasoning separated so the system remains easy to extend.

## Remaining Roadmap
Planned next steps:
- Implement rule-based abnormal event detection using tracking snapshots
- Add behavior analysis features such as trajectory, proximity, and persistence metrics
- Introduce explainability layers that describe why an event was flagged
- Extend the system toward risk scoring and reporting

## Notes
- The current tracker is intentionally lightweight and easy to inspect.
- No abnormal detection, risk assessment, dashboard, database, or web interface has been implemented yet.
