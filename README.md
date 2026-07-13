# Explainable AI-Based Abnormal Event Detection and Risk Assessment in Surveillance Videos

## Project Overview
This project focuses on building an explainable AI-based system for detecting abnormal events in surveillance videos and assessing their potential risk level. The initial Phase 1 establishes the project foundation, including a clean folder structure, dependency setup, and a working video-loading pipeline.

## Current Project Structure
```text
.
├── data/                # Input media files
├── docs/                # Documentation and reports
├── models/              # Trained models and checkpoints
├── outputs/             # Generated outputs and previews
├── src/                 # Source code
│   ├── config.py        # Project configuration and paths
│   ├── main.py          # Entry point for the application
│   └── video_loader.py  # Video loading and preview logic
├── tests/               # Test files
├── requirements.txt     # Python dependencies for Phase 1
└── README.md            # Project documentation
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

## How to Run the Project
From the project root, run:
```bash
python src/main.py
```

This will load the default video, print the frame dimensions, and display the video preview until the user presses q.

## Phase 1 Completion Summary
Completed in Phase 1:
- Created a professional project structure
- Set up a Python virtual environment
- Added the required Phase 1 dependencies
- Implemented a working video loader using OpenCV
- Added clean, documented Python source files
- Verified the application runs successfully
