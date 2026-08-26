"""R3D-18 model construction and forward inference.

This module owns model building, checkpoint loading, and the forward
pass. It does not buffer frames and does not preprocess them; it consumes
the ``(1, C, T, H, W)`` tensors produced by ``temporal_preprocessing``.

RESEARCH INTEGRITY
------------------
An R3D-18 backbone is not a violence classifier. Depending on how it was
constructed, its output means one of three different things, and this
module records which one on every prediction it returns:

``RANDOM_INIT``
    Untrained weights. Output is meaningless noise, useful only for shape
    and plumbing tests.
``KINETICS400_PRETRAINED``
    Output is 400 Kinetics-400 *action recognition* logits (``abseiling``,
    ``air drumming``, ...). Kinetics-400 contains no violence class, so
    these logits must never be read as a violence score.
``CHECKPOINT``
    Output is whatever the loaded checkpoint was trained to produce. Only
    a checkpoint actually fine-tuned and evaluated on a violence dataset
    yields a violence prediction.

``TemporalPrediction.is_task_specific`` is False for the first two cases.
Downstream code must refuse to treat a non-task-specific prediction as an
abnormal-event or violence signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

try:
    from config import MODELS_DIR
except ImportError:  # pragma: no cover - supports package execution
    from src.config import MODELS_DIR


KINETICS400_NUM_CLASSES = 400
R3D18_FEATURE_DIM = 512

RANDOM_INIT = "random_init"
KINETICS400_PRETRAINED = "kinetics400_pretrained"
CHECKPOINT = "checkpoint"

DEFAULT_CHECKPOINT_DIR = MODELS_DIR


class TemporalModelError(RuntimeError):
    """Raised when the temporal model cannot be built, loaded, or run."""


@dataclass(frozen=True)
class TemporalModelConfig:
    """Explicit model construction and execution parameters.

    ``pretrained_backbone`` defaults to False so that constructing a model
    never triggers a network download as a side effect. Enabling it
    fetches the Kinetics-400 checkpoint through torchvision's cache.
    """

    num_classes: int = KINETICS400_NUM_CLASSES
    device: str = "cpu"
    pretrained_backbone: bool = False
    checkpoint_path: Optional[Path] = None
    class_labels: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise TemporalModelError("num_classes must be a positive integer.")
        if self.class_labels is not None and len(self.class_labels) != self.num_classes:
            raise TemporalModelError(
                f"class_labels has {len(self.class_labels)} entries but "
                f"num_classes is {self.num_classes}."
            )


@dataclass(frozen=True, eq=False)
class TemporalPrediction:
    """One forward pass result, tagged with what it actually means.

    ``eq=False`` because the tensor fields make generated equality and
    hashing raise rather than behave usefully.
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_index: int
    num_classes: int
    provenance: str
    is_task_specific: bool
    predicted_label: Optional[str] = None
    evidence: Sequence[str] = field(default_factory=tuple)

    def describe(self) -> str:
        """Return an honest one-line description of this prediction."""
        if not self.is_task_specific:
            return (
                f"Non-task-specific output from a '{self.provenance}' model: "
                f"argmax index {self.predicted_index} of {self.num_classes}"
                f"{f' ({self.predicted_label})' if self.predicted_label else ''}. "
                "This is NOT a violence or abnormal-event prediction."
            )
        return (
            f"Task-specific prediction from '{self.provenance}': index "
            f"{self.predicted_index} of {self.num_classes}"
            f"{f' ({self.predicted_label})' if self.predicted_label else ''}."
        )


def resolve_device(device: str) -> torch.device:
    """Resolve a device string, failing loudly instead of silently downgrading.

    A caller that explicitly asks for CUDA on a CPU-only machine has a
    configuration bug; silently running on CPU would hide it.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise TemporalModelError(
            "CUDA was requested but torch reports no CUDA device is available. "
            "Use device='cpu', or install a CUDA-enabled torch build."
        )
    return resolved


class R3D18TemporalModel:
    """Wrapper around torchvision's R3D-18 video classification network."""

    def __init__(self, config: TemporalModelConfig = TemporalModelConfig()) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.provenance = RANDOM_INIT
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

        if config.checkpoint_path is not None:
            self.load_checkpoint(config.checkpoint_path)

    def _build_model(self) -> nn.Module:
        """Build the backbone and attach a head sized for ``num_classes``."""
        weights = None
        if self.config.pretrained_backbone:
            weights = R3D_18_Weights.KINETICS400_V1

        try:
            model = r3d_18(weights=weights)
        except Exception as error:
            raise TemporalModelError(
                f"Could not construct R3D-18 (pretrained={weights is not None}): {error}"
            ) from error

        self.provenance = (
            KINETICS400_PRETRAINED if weights is not None else RANDOM_INIT
        )

        if self.config.num_classes != KINETICS400_NUM_CLASSES:
            # Replacing the head discards the Kinetics classifier while
            # retaining the pretrained spatio-temporal features. The new
            # head is randomly initialized and therefore untrained until a
            # task-specific checkpoint is loaded or fine-tuning is run.
            model.fc = nn.Linear(R3D18_FEATURE_DIM, self.config.num_classes)
            self.provenance = (
                f"{self.provenance}+untrained_head"
                if weights is not None
                else RANDOM_INIT
            )

        return model

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a task-specific state dict, replacing the current weights."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise TemporalModelError(f"Checkpoint not found: {path}")

        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as error:
            raise TemporalModelError(
                f"Could not read checkpoint {path}: {error}"
            ) from error

        # Accept either a bare state dict or a training checkpoint that
        # nests one, which is the common convention for saved epochs.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        try:
            self.model.load_state_dict(state)
        except Exception as error:
            raise TemporalModelError(
                f"Checkpoint {path} is not compatible with this R3D-18 "
                f"configuration (num_classes={self.config.num_classes}): {error}"
            ) from error

        self.model.to(self.device)
        self.model.eval()
        self.provenance = f"{CHECKPOINT}:{path.name}"

    @property
    def feature_layer(self) -> nn.Module:
        """Final convolutional block -- the hook point for 3D Grad-CAM.

        ``layer4`` is the last module producing spatio-temporal feature
        maps ``(N, 512, T', H', W')`` before global pooling collapses them,
        so it is where activations and gradients must be captured to
        localise *when* and *where* the evidence was. Exposed now so a
        later Grad-CAM implementation attaches to a documented seam
        instead of reaching into the torchvision internals.

        Nothing here registers a hook or computes a CAM; that is a
        separate, later step, and the existing YOLO Grad-CAM is untouched.
        """
        return self.model.layer4

    @property
    def is_task_specific(self) -> bool:
        """Whether the current weights were trained for our actual task.

        Only a loaded checkpoint qualifies. Random initialization and the
        stock Kinetics-400 classifier do not.
        """
        return self.provenance.startswith(CHECKPOINT)

    def predict(self, clip_tensor: torch.Tensor) -> TemporalPrediction:
        """Run one forward pass over a preprocessed clip tensor."""
        self._validate_input(clip_tensor)

        with torch.no_grad():
            logits = self.model(clip_tensor.to(self.device))

        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(torch.argmax(probabilities, dim=1)[0].item())
        predicted_label = None
        if self.config.class_labels is not None:
            predicted_label = self.config.class_labels[predicted_index]

        return TemporalPrediction(
            logits=logits.detach().cpu(),
            probabilities=probabilities.detach().cpu(),
            predicted_index=predicted_index,
            num_classes=self.config.num_classes,
            provenance=self.provenance,
            is_task_specific=self.is_task_specific,
            predicted_label=predicted_label,
            evidence=(
                f"provenance={self.provenance}",
                f"num_classes={self.config.num_classes}",
                f"device={self.device.type}",
            ),
        )

    def _validate_input(self, clip_tensor: torch.Tensor) -> None:
        if not isinstance(clip_tensor, torch.Tensor):
            raise TemporalModelError(
                f"Model input must be a torch.Tensor, got {type(clip_tensor)!r}."
            )
        if clip_tensor.ndim != 5:
            raise TemporalModelError(
                "Model input must have 5 dimensions (N, C, T, H, W), got shape "
                f"{tuple(clip_tensor.shape)}."
            )
        if clip_tensor.shape[1] != 3:
            raise TemporalModelError(
                "Model input must have 3 colour channels at dim 1, got shape "
                f"{tuple(clip_tensor.shape)}."
            )


def kinetics400_labels() -> Tuple[str, ...]:
    """Return torchvision's published Kinetics-400 category names.

    Reading the metadata does not download the checkpoint. These are
    action-recognition categories; none of them denote violence.
    """
    return tuple(R3D_18_Weights.KINETICS400_V1.meta["categories"])
