"""Startup preflight for the demo application.

WHY THIS EXISTS
---------------
The failures that ruin a demonstration are almost never interesting ones. They
are a missing dependency, a moved file, or a checkpoint that was never copied to
this machine -- and they surface halfway through an examiner's first click
rather than at startup. This checks the things that are genuinely required, and
says which of them are required *for what*.

THREE TIERS, DELIBERATELY SEPARATE
----------------------------------
    REPLAY   the default dashboard path: templates, static assets, the
             committed temporal CSV, YOLO weights for detection and tracking
    LIVE     optional, explicitly selected: the R3D checkpoint
    SALIENCY on-demand Grad-CAM: also the R3D checkpoint

The distinction is the point. The documented replay design does NOT need the
R3D checkpoint -- window probabilities are replayed from the committed CSV --
so a missing checkpoint must never be reported as a blocker for the standard
demo. It blocks live inference and saliency, and the preflight says exactly
that.

WHAT IT WILL NOT DO
-------------------
It reports; it does not repair, and it never falls back. A missing live
checkpoint degrading silently into replay would make an output ambiguous about
which mode produced it, which is precisely the confusion the provenance fields
exist to prevent.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

TIER_REPLAY = "replay"
TIER_LIVE = "live_inference"
TIER_SALIENCY = "saliency"

# Third-party modules the dashboard imports at runtime, with the pip name to
# suggest when one is missing -- an examiner should not have to work out that
# "cv2" is installed as "opencv-python".
REPLAY_IMPORTS = {
    "flask": "flask",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
}
MODEL_IMPORTS = {"torch": "torch", "torchvision": "torchvision"}

EXPECTED_TEMPORAL_COLUMNS = (
    "split", "clip", "window_index", "first_frame", "last_frame",
    "true_label", "fight_probability",
)
EXPECTED_TEMPORAL_CLIPS = 394
EXPECTED_WINDOWS_PER_CLIP = 17


@dataclass
class Check:
    name: str
    tier: str
    ok: bool
    detail: str
    remedy: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "tier": self.tier, "ok": self.ok,
                "detail": self.detail, "remedy": self.remedy}


@dataclass
class PreflightResult:
    checks: List[Check] = field(default_factory=list)

    def failures(self, tier: Optional[str] = None) -> List[Check]:
        return [c for c in self.checks
                if not c.ok and (tier is None or c.tier == tier)]

    @property
    def replay_ready(self) -> bool:
        return not self.failures(TIER_REPLAY)

    @property
    def live_ready(self) -> bool:
        return self.replay_ready and not self.failures(TIER_LIVE)

    @property
    def saliency_ready(self) -> bool:
        return self.replay_ready and not self.failures(TIER_SALIENCY)

    def as_dict(self) -> dict:
        return {
            "replay_ready": self.replay_ready,
            "live_inference_ready": self.live_ready,
            "saliency_ready": self.saliency_ready,
            "checks": [c.as_dict() for c in self.checks],
            "note": (
                "The standard dashboard demo uses REPLAY: temporal probabilities "
                "come from the committed CSV and no model is run for them. A "
                "missing R3D checkpoint therefore blocks live inference and "
                "saliency, not the standard demo."
            ),
        }

    def render(self) -> str:
        lines = ["Demo preflight", "=" * 58]
        for tier, label in ((TIER_REPLAY, "REPLAY (standard demo)"),
                            (TIER_LIVE, "LIVE INFERENCE (optional)"),
                            (TIER_SALIENCY, "SALIENCY / Grad-CAM (on demand)")):
            lines.append(f"\n{label}")
            for check in [c for c in self.checks if c.tier == tier]:
                lines.append(f"  [{'OK ' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
                if not check.ok and check.remedy:
                    lines.append(f"         -> {check.remedy}")
        lines += [
            "",
            f"replay ready   : {self.replay_ready}",
            f"live ready     : {self.live_ready}",
            f"saliency ready : {self.saliency_ready}",
        ]
        if not self.replay_ready:
            lines.append("\nThe standard demo CANNOT run until the REPLAY failures above are fixed.")
        elif not self.saliency_ready:
            lines.append("\nStandard demo is ready. Saliency is unavailable; see above.")
        return "\n".join(lines)


def _check_imports(mapping: Dict[str, str], tier: str) -> List[Check]:
    checks = []
    for module, package in mapping.items():
        try:
            importlib.import_module(module)
            checks.append(Check(f"import {module}", tier, True, "available"))
        except Exception as error:
            checks.append(Check(
                f"import {module}", tier, False, f"{type(error).__name__}: {error}",
                f"pip install {package}",
            ))
    return checks


def _check_file(path: Path, name: str, tier: str, remedy: str) -> Check:
    if path.is_file():
        return Check(name, tier, True, f"{path.name} ({path.stat().st_size} bytes)")
    return Check(name, tier, False, f"missing: {path}", remedy)


def _check_temporal_csv(path: Path) -> List[Check]:
    """Presence, schema and expected size of the replay score source."""
    if not path.is_file():
        return [Check("temporal score CSV", TIER_REPLAY, False, f"missing: {path}",
                      "the committed replay scores are required for the standard demo")]
    checks = []
    import csv as _csv
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = _csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            clips = set()
            rows = 0
            for row in reader:
                clips.add(row["clip"])
                rows += 1
    except Exception as error:
        return [Check("temporal score CSV", TIER_REPLAY, False,
                      f"unreadable: {type(error).__name__}: {error}",
                      "the file exists but could not be parsed")]

    missing = [c for c in EXPECTED_TEMPORAL_COLUMNS if c not in columns]
    checks.append(Check(
        "temporal CSV schema", TIER_REPLAY, not missing,
        "all expected columns present" if not missing else f"missing columns: {missing}",
        "the replay source does not match the expected schema",
    ))
    expected_rows = EXPECTED_TEMPORAL_CLIPS * EXPECTED_WINDOWS_PER_CLIP
    ok = len(clips) == EXPECTED_TEMPORAL_CLIPS and rows == expected_rows
    checks.append(Check(
        "temporal CSV contents", TIER_REPLAY, ok,
        f"{len(clips)} clips, {rows} rows"
        + ("" if ok else f" (expected {EXPECTED_TEMPORAL_CLIPS} clips, {expected_rows} rows)"),
        "the replay source is not the committed artifact",
    ))
    return checks


def run_preflight(
    repo_root: Path = REPO_ROOT,
    temporal_csv: Optional[Path] = None,
    yolo_weights: Optional[Path] = None,
    r3d_checkpoint: Optional[Path] = None,
    demo_videos: Optional[Dict[str, Path]] = None,
) -> PreflightResult:
    """Inspect the environment. Reports only; repairs nothing, falls back never."""
    temporal_csv = temporal_csv or repo_root / "temporal_risk" / "primary_window_scores.csv"
    yolo_weights = yolo_weights or repo_root / "models" / "yolov8s.pt"
    r3d_checkpoint = r3d_checkpoint or repo_root / "models" / "temporal_violence" / "best.pt"

    checks: List[Check] = []
    checks += _check_imports(REPLAY_IMPORTS, TIER_REPLAY)
    checks.append(_check_file(
        repo_root / "web" / "templates" / "index.html", "dashboard template",
        TIER_REPLAY, "web/templates/index.html is missing"))
    checks.append(_check_file(
        repo_root / "web" / "static" / "app.js", "dashboard script",
        TIER_REPLAY, "web/static/app.js is missing"))
    checks.append(_check_file(
        repo_root / "web" / "static" / "style.css", "dashboard stylesheet",
        TIER_REPLAY, "web/static/style.css is missing"))
    checks.append(_check_file(
        yolo_weights, "YOLO weights", TIER_REPLAY,
        "detection and tracking run live in every mode and need these weights"))
    checks += _check_temporal_csv(temporal_csv)

    if demo_videos:
        missing = sorted(key for key, path in demo_videos.items() if not Path(path).is_file())
        checks.append(Check(
            "demo preset videos", TIER_REPLAY, not missing,
            "all present" if not missing else f"missing presets: {missing}",
            "the dataset is not available at the expected path",
        ))

    checks += _check_imports(MODEL_IMPORTS, TIER_LIVE)
    checkpoint_remedy = (
        "live inference and saliency need the R3D-18 checkpoint; the standard "
        "replay demo does NOT, and will still run without it"
    )
    live_checkpoint = _check_file(r3d_checkpoint, "R3D checkpoint", TIER_LIVE, checkpoint_remedy)
    checks.append(live_checkpoint)
    checks.append(Check("R3D checkpoint", TIER_SALIENCY, live_checkpoint.ok,
                        live_checkpoint.detail, checkpoint_remedy))
    checks.append(Check(
        "torch available for saliency", TIER_SALIENCY,
        all(c.ok for c in checks if c.tier == TIER_LIVE and c.name.startswith("import")),
        "torch and torchvision are importable"
        if all(c.ok for c in checks if c.tier == TIER_LIVE and c.name.startswith("import"))
        else "torch or torchvision is missing",
        "pip install torch torchvision",
    ))
    return PreflightResult(checks=checks)
