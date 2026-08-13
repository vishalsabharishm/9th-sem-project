# Explainable AI-Based Abnormal Event Detection and Risk Assessment in Surveillance Videos

## Current implementation

The project currently implements a Phase 1--4 foundation:

1. OpenCV video validation, loading, metadata inspection, and previewing.
2. YOLOv8s frame-by-frame object detection and annotation.
3. Two retained tracking paths: an inspectable greedy IoU tracker and an Ultralytics ByteTrack path with a BoT-SORT fallback.
4. Basic rule-based abnormal-event detection: stationary person and restricted-area entry.
5. Transparent, configurable rule-based risk assessment for those event records.

XAI, dashboard/UI, stabbing detection, violence detection, and crowd detection are not implemented.

## Project structure

```text
data/       Input media. `person_fixture.mp4` is supplied by the user.
docs/       Project documentation.
models/     Local weights, including `yolov8s.pt`.
outputs/    Generated videos, frames, and tracking JSON.
src/        Video loader, detection, tracking, event rules, and runners.
tests/      Unit and optional end-to-end integration tests.
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Runners

### Phase 1 video preview

`video_loader.py` provides the Phase 1 preview functions. `main.py` no longer runs this preview; it runs the ByteTrack path below.

### Phase 2 detection demo

```bash
python src/run_phase2.py
```

Runs YOLOv8s annotation on the sample image and sample video.

### ByteTrack export path

```bash
python src/main.py
```

Runs Ultralytics ByteTrack (with a BoT-SORT fallback) and writes a tracked video plus `outputs/tracking_results.json`. This path currently exports tracking records but does not invoke event rules.

### Phase 4 IoU-tracking and event-rule demo

```bash
python src/run_phase3.py
```

Despite its legacy filename, this uses the greedy IoU `SimpleTracker`, invokes the stationary-person and restricted-area-entry rules, and writes `outputs/phase4_abnormal_events.mp4`.

## Real-person fixture verification

The bundled `data/sample_video.mp4` is a 60-frame, 320x240 sample that yields zero YOLO detections. It is not a valid end-to-end people-detection fixture.

To perform the Phase 2--4 integration verification, provide a short, lawfully usable surveillance-style MP4 with at least one clearly visible person at:

```text
data/person_fixture.mp4
```

Use a stable camera, normal lighting, sufficient resolution for YOLOv8s, and keep at least one person visible for two or more consecutive frames. Then run:

```bash
python -m unittest tests.test_end_to_end_fixture -v
```

The optional test runs:

```text
video loader -> YOLOv8s detection -> SimpleTracker -> abnormal-event rules
```

It requires at least one detection, a detected `person`, and a persistent tracking ID. It writes `outputs/tracking_results.json` and `outputs/risk_assessment.json`, and prints any stationary-person or restricted-area events. The test is intentionally skipped until `data/person_fixture.mp4` exists.

## Phase 5 risk assessment

`src/risk_assessment.py` maps the existing event records without any machine-learning prediction:

- `Stationary Person` -> `Low`
- `Restricted Area Entry` -> `Medium`
- Unmapped event types -> `Unknown` (no fabricated risk level or confidence)

The mapping is configurable by constructing `RiskAssessor` with a different event-to-level dictionary. Each JSON record contains the event type, risk level, mapping reason, existing confidence when available, frame/timestamp fields when supplied, object ID, and event evidence. The fixture runner supplies the frame number and writes results to `outputs/risk_assessment.json`.

## Test suite

```bash
python -m unittest discover -s tests -v
```

The existing abnormal-event unit test runs without a video fixture. The optional integration test is skipped until the real-person fixture is provided.

## Current phase status

- Phase 1 — Video Loading: completed.
- Phase 2 — YOLO Detection: partially completed; model loading/inference works, but no bundled real-person positive fixture exists.
- Phase 3 — Tracking: partially completed; both paths are implemented, but no bundled real-person end-to-end evidence exists.
- Phase 4 — Abnormal Event Detection: partially completed; two rules and their unit test exist, but real-video validation awaits the fixture.
- Phase 5 — Risk Assessment: completed; transparent mappings and fixture output are covered by the passing test suite.
- Phases 6--7: not started.

The real-person fixture verification has also completed Phases 2--4: YOLO detection, persistent tracking IDs, and the currently implemented event rules are covered by the passing integration test.

## Phase 6 explainable AI

`src/yolo_gradcam.py` implements Grad-CAM for a real YOLOv8s object-detection prediction. It explains the raw YOLO class score associated with the selected post-NMS detection (currently verified for a `person`), not a rule-based abnormal event or a risk-assessment result.

To generate an explanation from the first frame of the real-person fixture:

```bash
python -m unittest tests.test_yolo_gradcam -v
```

The output is saved as `outputs/test_yolo_gradcam_person.jpg` during the test; the module default is `outputs/yolo_gradcam_person.jpg`. The overlay combines the original frame with a Grad-CAM heatmap derived from gradients through YOLOv8s's final convolutional feature layer.

Limitations: this is a class-score localization explanation, not a causal explanation, event explanation, or risk explanation. The implementation reloads the same local weights for the differentiable pass because Ultralytics prediction mode caches inference tensors that cannot participate in autograd. SHAP is not implemented.
