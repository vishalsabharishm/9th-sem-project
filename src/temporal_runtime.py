"""One place that builds the violence model for every runtime consumer.

WHY THIS EXISTS
---------------
Before this module, ``tools/score_temporal_windows.py`` constructed the 2-class
R3D-18 itself, and the demo would have needed a second copy of the same
construction to run live inference. Two copies of "how the violence model is
built and validated" is two chances to drift from the regime the committed
window scores were produced under -- and a drift there would silently
invalidate every number in the evaluation.

So the parameters that define that regime live here, once:

    num_classes   2               the violence head, never Kinetics' 400
    class_labels  ("NonFight", "Fight")
    clip_length   16              }  matches
    stride        8               }  temporal_risk/window_scoring_manifest.json

``TemporalInferenceConfig()`` defaults to ``num_classes=400`` (torchvision's
Kinetics head), so a caller who builds an engine without overriding it gets a
model that cannot load the violence checkpoint at all. The factories below
never leave that to the caller.

THE GUARD IS NOT OPTIONAL
-------------------------
Every builder here runs ``checkpoint_identity.verify_trained_checkpoint``
first. ``R3D18TemporalModel.load_checkpoint`` sets provenance to
``checkpoint:<filename>`` for any loadable file, so ``is_task_specific`` alone
would accept the randomly-initialised checkpoint that
``temporal_training.py --validate-pipeline`` writes -- to the canonical
``models/temporal_violence/best.pt`` path, at almost exactly the real file's
size. Verifying the recorded training evidence is the only thing that
distinguishes them. See ``docs/CHECKPOINT_INTEGRITY.md``.

Nothing here trains, downloads, or writes a checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:
    from checkpoint_identity import CheckpointVerdict, verify_trained_checkpoint
    from rwf2000_config import CANONICAL_LABELS, LABEL_TO_INDEX
except ImportError:  # pragma: no cover - supports package execution
    from src.checkpoint_identity import CheckpointVerdict, verify_trained_checkpoint
    from src.rwf2000_config import CANONICAL_LABELS, LABEL_TO_INDEX


# The sliding-window regime, matching temporal_risk/window_scoring_manifest.json.
# Changing either of these changes which windows exist, and therefore makes new
# scores incomparable with the committed ones.
RUNTIME_CLIP_LENGTH = 16
RUNTIME_STRIDE = 8

RUNTIME_NUM_CLASSES = len(CANONICAL_LABELS)
RUNTIME_CLASS_LABELS: Tuple[str, ...] = tuple(CANONICAL_LABELS)
RUNTIME_POSITIVE_CLASS_INDEX = LABEL_TO_INDEX["Fight"]

DEFAULT_DEVICE = "cpu"


class TemporalRuntimeError(RuntimeError):
    """Raised when a violence model cannot be built for runtime use."""


def verified_checkpoint(checkpoint: Path) -> CheckpointVerdict:
    """Run the Step-3 provenance guard, raising unless the file is acceptable.

    Returns the verdict so callers can surface its warnings (for example a
    legacy checkpoint whose initialisation had to be inferred rather than read).
    """
    path = Path(checkpoint)
    verdict = verify_trained_checkpoint(path)
    verdict.raise_if_rejected()
    return verdict


def violence_model_config(checkpoint: Path, device: str = DEFAULT_DEVICE):
    """Return the ``TemporalModelConfig`` every runtime consumer must use.

    ``pretrained_backbone=False`` because the checkpoint supplies all weights;
    fetching Kinetics-400 first would download ~120 MB and then discard it.
    """
    from temporal_model import TemporalModelConfig  # local: keeps torch out of import time

    return TemporalModelConfig(
        num_classes=RUNTIME_NUM_CLASSES,
        device=device,
        pretrained_backbone=False,
        checkpoint_path=Path(checkpoint),
        class_labels=RUNTIME_CLASS_LABELS,
    )


def build_violence_model(checkpoint: Path, device: str = DEFAULT_DEVICE):
    """Build the verified 2-class R3D-18 for whole-window scoring.

    Used by ``tools/score_temporal_windows.py``, which does its own windowing
    and calls ``model.predict`` directly.
    """
    from temporal_model import R3D18TemporalModel, TemporalModelError

    verdict = verified_checkpoint(checkpoint)
    try:
        model = R3D18TemporalModel(violence_model_config(checkpoint, device))
    except TemporalModelError as error:
        raise TemporalRuntimeError(
            f"Could not build the violence model from {checkpoint}: {error}"
        ) from error

    if not model.is_task_specific:
        raise TemporalRuntimeError(
            f"Model provenance is {model.provenance!r} after loading "
            f"{checkpoint}; refusing to use non-task-specific weights."
        )
    return model, verdict


def build_violence_engine(
    checkpoint: Path,
    device: str = DEFAULT_DEVICE,
    clip_length: int = RUNTIME_CLIP_LENGTH,
    stride: int = RUNTIME_STRIDE,
):
    """Build a verified ``TemporalInferenceEngine`` for frame-by-frame scoring.

    This is what a live runtime feeds frames to. The engine owns the buffer, the
    preprocessor and the model; nothing about their implementations is changed
    here, only their configuration.

    ``clip_length`` and ``stride`` default to the committed regime. Overriding
    them produces windows that do not correspond to any committed score, so a
    caller doing that is on their own for comparability.
    """
    from temporal_inference import TemporalInferenceConfig, TemporalInferenceEngine

    verdict = verified_checkpoint(checkpoint)
    config = TemporalInferenceConfig(
        clip_length=clip_length,
        stride=stride,
        model=violence_model_config(checkpoint, device),
    )
    engine = TemporalInferenceEngine(config)

    if not engine.model.is_task_specific:
        raise TemporalRuntimeError(
            f"Engine model provenance is {engine.model.provenance!r}; refusing "
            "to score with non-task-specific weights."
        )
    return engine, verdict


def describe_runtime() -> dict:
    """The regime constants, for run records and provenance reporting."""
    return {
        "clip_length": RUNTIME_CLIP_LENGTH,
        "stride": RUNTIME_STRIDE,
        "num_classes": RUNTIME_NUM_CLASSES,
        "class_labels": list(RUNTIME_CLASS_LABELS),
        "positive_class_index": RUNTIME_POSITIVE_CLASS_INDEX,
        "matches_manifest": "temporal_risk/window_scoring_manifest.json",
    }
