"""Grad-CAM for the R3D-18 temporal violence decision.

WHY THIS EXISTS
---------------
The repository already had ``src/yolo_gradcam.py``, but that explains a YOLO
*person detection* -- it says where a person is, not why the system called a
clip violent. The Fight/NonFight decision is made entirely by the temporal
branch, so a saliency map over YOLO features cannot explain it, and presenting
one as if it did would be misleading. ``R3D18TemporalModel.feature_layer``
already exposed ``layer4`` with a test noting it was "for future gradcam"; this
is that future.

WHAT IT COMPUTES
----------------
Standard Grad-CAM against the *task* logit:

    A       = layer4 activations,         shape (1, 512, T', H', W')
    g       = d(logit_class) / dA,        same shape
    w_c     = mean over (T', H', W') of g[0, c]
    cam     = ReLU(sum_c w_c * A[0, c]),  shape (T', H', W')

For a 16-frame clip at 112x112, layer4 emits (1, 512, 2, 7, 7): the backbone
downsamples 8x temporally and 16x spatially. The map is then trilinearly
upsampled to (16, 112, 112) so every input frame receives its own heatmap.

TEMPORAL ALIGNMENT -- THE PART THAT IS EASY TO GET WRONG
--------------------------------------------------------
A CAM slice is meaningless unless you can say which source frames it covers.
Two facts make that exact here. The sliding-window regime feeds 16 CONSECUTIVE
source frames, so tensor index i corresponds to source frame
``first_frame + i``. And the temporal stride of 8 means each of the two raw CAM
slices summarises 8 input frames, which is recorded per slice rather than
hidden by the upsampling. ``frame_numbers`` is returned alongside the heatmaps
so a caller can never mis-attribute a map to the wrong moment.

This is NOT valid for the whole-clip evaluation regime, which samples 16 frames
spread across all 150. That regime is rejected rather than silently mapped, see
``explain_window``.

SPATIAL ALIGNMENT
-----------------
Preprocessing resizes to (128, 171) then centre-crops to 112x112, so a heatmap
pixel does not correspond to the source pixel underneath it. ``to_source_frame``
inverts both steps, and reports the cropped-away border so a caller knows which
part of the original frame the model never saw.

WHAT THIS DOES AND DOES NOT LICENSE
-----------------------------------
Grad-CAM shows where in the input the gradient of the chosen logit concentrates.
That is a localisation diagnostic. It is NOT evidence that the explanation is
faithful -- Grad-CAM can be plausible and wrong, and no faithfulness test
(deletion, insertion, counterfactual) is implemented here. Nothing in this
module may be described as a validated explanation, and no explainability metric
is computed or claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ImportError as error:  # pragma: no cover - torch is a hard dependency here
    raise ImportError("temporal_gradcam requires torch") from error


class TemporalGradCAMError(RuntimeError):
    """Raised when a saliency map cannot be produced honestly."""


FIGHT_CLASS_INDEX = 1
RUNTIME_CLIP_LENGTH = 16


@dataclass(frozen=True)
class TemporalGradCAM:
    """A saliency map over one completed temporal window.

    ``heatmaps`` is (clip_length, crop_h, crop_w), normalised to [0, 1] over the
    whole window so that slices remain comparable with one another. ``frame_numbers``
    gives the source frame each slice belongs to.
    """

    heatmaps: np.ndarray
    frame_numbers: Tuple[int, ...]
    class_index: int
    class_label: str
    logit: float
    probability: float
    raw_temporal_positions: int
    frames_per_raw_position: int
    provenance: str
    faithfulness_tested: bool = False

    def peak_frame(self) -> int:
        """The source frame whose slice carries the most saliency mass."""
        return int(self.frame_numbers[int(np.argmax(self.heatmaps.sum(axis=(1, 2))))])

    def as_dict(self) -> dict:
        return {
            "class_index": self.class_index,
            "class_label": self.class_label,
            "logit": self.logit,
            "probability": self.probability,
            "frame_numbers": list(self.frame_numbers),
            "peak_frame": self.peak_frame(),
            "heatmap_shape": list(self.heatmaps.shape),
            "raw_temporal_positions": self.raw_temporal_positions,
            "frames_per_raw_position": self.frames_per_raw_position,
            "provenance": self.provenance,
            "faithfulness_tested": self.faithfulness_tested,
            "interpretation_limit": (
                "Grad-CAM localises where the gradient of the selected logit "
                "concentrates. It is a diagnostic, not evidence that the "
                "explanation is faithful; no deletion, insertion or "
                "counterfactual test has been run."
            ),
        }


def _require_task_specific(model) -> None:
    """Refuse to explain a head that was never trained for this task.

    A Kinetics-400 or randomly initialised head still produces a logit, and
    Grad-CAM will still draw a confident-looking map from it. Explaining such a
    model would manufacture a picture of reasoning that does not exist.
    """
    provenance = getattr(model, "provenance", None)
    if provenance is None:
        raise TemporalGradCAMError(
            "the model carries no provenance; refusing to explain a decision whose "
            "weights cannot be shown to be task-specific."
        )
    # R3D18TemporalModel exposes is_task_specific as a bool PROPERTY, while a
    # prediction object exposes it as a plain field; accept either shape rather
    # than assuming one and crashing on the other.
    flag = getattr(model, "is_task_specific", False)
    if callable(flag):
        flag = flag()
    if not bool(flag):
        raise TemporalGradCAMError(
            f"model provenance {provenance!r} is not task-specific; a saliency map "
            "over an untrained violence head would be meaningless."
        )


def explain_window(
    model,
    clip_tensor: "torch.Tensor",
    frame_numbers: Sequence[int],
    class_index: int = FIGHT_CLASS_INDEX,
    class_label: str = "Fight",
    require_task_specific: bool = True,
) -> TemporalGradCAM:
    """Grad-CAM for one completed sliding window.

    ``clip_tensor`` is the preprocessed (1, 3, T, H, W) tensor that produced the
    decision, and ``frame_numbers`` the source frames it was built from, in
    order. Both come from the runtime, so the map explains the decision that was
    actually taken rather than a re-derived one.
    """
    if require_task_specific:
        _require_task_specific(model)

    if clip_tensor.dim() != 5 or clip_tensor.shape[0] != 1:
        raise TemporalGradCAMError(
            f"expected a single (1, 3, T, H, W) clip, got {tuple(clip_tensor.shape)}."
        )
    frames = tuple(int(value) for value in frame_numbers)
    if len(frames) != clip_tensor.shape[2]:
        raise TemporalGradCAMError(
            f"{len(frames)} frame numbers for {clip_tensor.shape[2]} tensor frames; "
            "the map could not be aligned to source frames."
        )
    # Consecutive frames are what make index -> source-frame mapping exact. The
    # whole-clip regime samples 16 frames spread over 150 and is refused rather
    # than mapped as though its slices were contiguous.
    if len(frames) > 1 and any(b - a != 1 for a, b in zip(frames, frames[1:])):
        raise TemporalGradCAMError(
            "frame numbers are not consecutive. This explainer covers the sliding-"
            "window regime, where a window is 16 consecutive frames; a uniformly "
            "sampled whole-clip tensor cannot be aligned slice-to-frame."
        )

    network = getattr(model, "model", model)
    feature_layer = getattr(model, "feature_layer", None)
    if feature_layer is None:
        raise TemporalGradCAMError("the model exposes no feature_layer to hook.")

    activations: List["torch.Tensor"] = []
    gradients: List["torch.Tensor"] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(gradients.append)

    handle = feature_layer.register_forward_hook(forward_hook)
    was_training = network.training
    try:
        network.eval()
        clip = clip_tensor.detach().clone().requires_grad_(True)
        network.zero_grad(set_to_none=True)
        # Gradients are required, so no torch.no_grad() here.
        logits = network(clip)
        if logits.dim() != 2 or class_index >= logits.shape[1]:
            raise TemporalGradCAMError(
                f"class index {class_index} is outside logits of shape "
                f"{tuple(logits.shape)}."
            )
        probability = float(torch.softmax(logits, dim=1)[0, class_index].detach())
        score = logits[0, class_index]
        score.backward()

        if not activations or not gradients:
            raise TemporalGradCAMError(
                "no activation or gradient was captured at the feature layer."
            )
        activation = activations[0].detach()[0]      # (C, T', H', W')
        gradient = gradients[-1].detach()[0]         # same
    finally:
        handle.remove()
        network.zero_grad(set_to_none=True)
        if was_training:
            network.train()

    weights = gradient.mean(dim=(1, 2, 3))                       # (C,)
    cam = torch.relu((weights[:, None, None, None] * activation).sum(dim=0))
    raw_positions = int(cam.shape[0])

    cam = functional.interpolate(
        cam[None, None], size=tuple(clip_tensor.shape[2:]),
        mode="trilinear", align_corners=False,
    )[0, 0]

    peak = float(cam.max())
    cam = cam / peak if peak > 0 else cam        # all-zero map stays all-zero

    return TemporalGradCAM(
        heatmaps=cam.cpu().numpy(),
        frame_numbers=frames,
        class_index=class_index,
        class_label=class_label,
        logit=float(score.detach()),
        probability=probability,
        raw_temporal_positions=raw_positions,
        frames_per_raw_position=int(clip_tensor.shape[2] // max(raw_positions, 1)),
        provenance=str(getattr(model, "provenance", "unknown")),
    )


def to_source_frame(
    heatmap: np.ndarray,
    source_height: int,
    source_width: int,
    resize_size: Tuple[int, int] = (128, 171),
    crop_size: Tuple[int, int] = (112, 112),
) -> dict:
    """Map one crop-space heatmap back onto original frame coordinates.

    Preprocessing resized to ``resize_size`` then centre-cropped to
    ``crop_size``, so heatmap pixel (r, c) does not sit above source pixel
    (r, c). This inverts both steps and reports the border the crop discarded --
    a region the model never saw, and therefore one no explanation may claim
    anything about.
    """
    resize_h, resize_w = int(resize_size[0]), int(resize_size[1])
    crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
    if heatmap.shape != (crop_h, crop_w):
        raise TemporalGradCAMError(
            f"heatmap is {heatmap.shape}, expected {(crop_h, crop_w)}."
        )

    top = max((resize_h - crop_h) // 2, 0)
    left = max((resize_w - crop_w) // 2, 0)
    scale_y = source_height / resize_h
    scale_x = source_width / resize_w

    return {
        "crop_origin_in_resized": (top, left),
        "scale_to_source": (scale_y, scale_x),
        "visible_region_in_source": (
            int(round(top * scale_y)), int(round(left * scale_x)),
            int(round((top + crop_h) * scale_y)), int(round((left + crop_w) * scale_x)),
        ),
        "discarded_border_pixels_in_resized": {
            "top": top, "bottom": max(resize_h - crop_h - top, 0),
            "left": left, "right": max(resize_w - crop_w - left, 0),
        },
        "caveat": (
            "the centre crop discards part of every frame; the model never saw "
            "the discarded border and an explanation must not claim anything "
            "about it"
        ),
    }
