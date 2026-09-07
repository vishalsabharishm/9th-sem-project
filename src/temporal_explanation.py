"""UI-ready explanation payloads for the temporal violence decision.

WHY A SEPARATE MODULE
---------------------
``TemporalInferenceResult`` carries the prediction and the frame numbers but not
the preprocessed tensor, and Grad-CAM needs the tensor. Rather than widen that
result contract -- which the byte-identical CSV replay path depends on -- this
module re-runs preprocessing through the ENGINE'S OWN preprocessor and hooks the
engine's own model. Nothing in the runtime changes, and there is no second
preprocessing or geometry implementation to drift.

WHAT A PAYLOAD PROMISES
-----------------------
Each field is tagged with where it came from, because the dashboard shows four
very different kinds of number side by side and conflating them is exactly how
an honest system starts making dishonest claims:

    measured_model_probability   the R3D softmax output
    gradient_saliency            Grad-CAM over the task logit
    declared_rule_constant       an engineer's hardcoded 0.9 / 0.95
    configured_interpretation    the risk dictionary's Low / Medium / High

REPLAY CANNOT BE EXPLAINED, AND SAYS SO
---------------------------------------
CSV replay has no model and no tensor -- only a recorded probability. A saliency
map therefore cannot exist for it. ``unavailable`` returns an explicit reason
rather than a blank panel, so the UI can distinguish "no explanation here" from
"explanation still loading".

COST
----
Grad-CAM needs a forward AND a backward pass, so it is more expensive than
inference alone. It must never run per frame; it is computed for one selected
completed window, on demand.
"""
from __future__ import annotations

from typing import Optional, Sequence

try:
    from temporal_gradcam import TemporalGradCAMError, explain_window, to_source_frame
except ImportError:  # pragma: no cover - package-style execution
    from src.temporal_gradcam import (  # type: ignore
        TemporalGradCAMError, explain_window, to_source_frame,
    )

# The exact wording the UI must use. Kept here, beside the code that produces the
# map, so a caller cannot invent a stronger claim and drift from what the method
# supports.
SALIENCY_HEADING = "Gradient-based temporal saliency"
SALIENCY_EXPLANATION = (
    "Highlighted regions indicate input areas contributing to the R3D task-logit "
    "gradient for this temporal window. This is a saliency visualization, not a "
    "causal or ground-truth explanation."
)
SALIENCY_NOT_CLAIMED = (
    "Not proof of causality, not a faithfulness-tested explanation, not attention, "
    "and not ground-truth localization."
)
RISK_STATUS_TEXT = (
    "Configured severity/risk interpretation - not validated as a risk model."
)

PROVENANCE_MEASURED_PROBABILITY = "measured_model_probability"
PROVENANCE_GRADIENT_SALIENCY = "gradient_saliency"
PROVENANCE_DECLARED_CONSTANT = "declared_rule_constant"
PROVENANCE_CONFIGURED = "configured_interpretation"


def unavailable(reason: str, detail: str = "") -> dict:
    """An explicit 'no explanation' payload, never a blank panel."""
    return {
        "available": False,
        "reason": reason,
        "detail": detail,
        "heading": SALIENCY_HEADING,
        "faithfulness_tested": False,
    }


def explain_completed_window(
    engine,
    clip,
    threshold: float,
    class_index: int = 1,
    class_label: str = "Fight",
    source_height: Optional[int] = None,
    source_width: Optional[int] = None,
) -> dict:
    """Build the explanation payload for ONE completed sliding window.

    ``clip`` is the engine's own assembled Clip, so the tensor explained here is
    produced by the same preprocessing that produced the decision. ``threshold``
    is the frozen decision threshold, passed in rather than imported, so this
    module cannot become a second place where an operating point is defined.
    """
    frame_numbers: Sequence[int] = list(getattr(clip, "frame_numbers", []))
    try:
        tensor = engine.preprocessor.preprocess(clip)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        saliency = explain_window(
            engine.model, tensor, frame_numbers,
            class_index=class_index, class_label=class_label,
        )
    except TemporalGradCAMError as error:
        return unavailable("saliency_unavailable", str(error))

    probability = saliency.probability
    payload = {
        "available": True,
        "temporal_evidence": {
            "fight_probability": probability,
            "fight_probability_provenance": PROVENANCE_MEASURED_PROBABILITY,
            "decision": class_label if probability >= threshold else "NonFight",
            "decision_threshold": threshold,
            "window_first_frame": int(frame_numbers[0]),
            "window_last_frame": int(frame_numbers[-1]),
            "window_frame_count": len(frame_numbers),
            "model_provenance": saliency.provenance,
        },
        "saliency": {
            "heading": SALIENCY_HEADING,
            "explanation": SALIENCY_EXPLANATION,
            "not_claimed": SALIENCY_NOT_CLAIMED,
            "provenance": PROVENANCE_GRADIENT_SALIENCY,
            "frame_numbers": list(saliency.frame_numbers),
            "peak_frame": saliency.peak_frame(),
            "displayed_temporal_slices": int(saliency.heatmaps.shape[0]),
            "raw_temporal_positions": saliency.raw_temporal_positions,
            "frames_per_raw_position": saliency.frames_per_raw_position,
            "temporal_resolution_caveat": (
                f"The backbone resolves only {saliency.raw_temporal_positions} temporal "
                f"positions across this window; the "
                f"{int(saliency.heatmaps.shape[0])} displayed slices are interpolated "
                f"from them and do not indicate frame-level temporal detail."
            ),
            "faithfulness_tested": saliency.faithfulness_tested,
        },
        "heatmaps": saliency.heatmaps,
    }
    if source_height and source_width:
        # Reuses the single geometry implementation rather than inverting the
        # resize and crop a second time.
        payload["saliency"]["source_geometry"] = to_source_frame(
            saliency.heatmaps[0], int(source_height), int(source_width)
        )
    return payload


def explain_frame_window(
    engine,
    frames,
    frame_numbers: Sequence[int],
    threshold: float,
    **kwargs,
) -> dict:
    """Explain a window assembled from frames the caller already holds.

    This is the shape an on-demand UI action actually needs.
    ``TemporalInferenceEngine.add_frame`` CONSUMES its buffer when a window
    completes, so the clip that produced a decision cannot be retrieved
    afterwards. A dashboard asking "explain the window at frames 24-39"
    therefore re-reads those frames and rebuilds the clip here, which also keeps
    Grad-CAM off the per-frame path by construction: there is no way to trigger
    it except by asking for one specific window.
    """
    try:
        from temporal_clip_buffer import Clip
    except ImportError:  # pragma: no cover - package-style execution
        from src.temporal_clip_buffer import Clip  # type: ignore

    numbers = [int(value) for value in frame_numbers]
    if len(frames) != len(numbers):
        return unavailable(
            "frame_mismatch",
            f"{len(frames)} frames supplied for {len(numbers)} frame numbers.",
        )
    clip = Clip(frames=list(frames), frame_numbers=numbers)
    return explain_completed_window(engine, clip, threshold, **kwargs)


def risk_presentation(risk_level: Optional[str] = None) -> dict:
    """How a risk level must be shown: the label plus its validation status.

    ``risk_level`` is optional because some callers (the saliency endpoint, for
    one) know the status must be shown but do not know the level. Passing a
    placeholder string there would put a meaningless value like "--" on screen;
    omitting the key is honest.
    """
    payload = {
        "provenance": PROVENANCE_CONFIGURED,
        "risk_level_is_validated": False,
        "status_text": RISK_STATUS_TEXT,
        "forbidden_wording": ("risk probability", "risk confidence", "risk score"),
    }
    if risk_level is not None:
        payload["level"] = risk_level
    return payload


def describe_provenance_legend() -> dict:
    """The four kinds of number the dashboard shows, named apart."""
    return {
        PROVENANCE_MEASURED_PROBABILITY: (
            "R3D softmax output for the Fight class; a measured model probability"),
        PROVENANCE_GRADIENT_SALIENCY: (
            "Grad-CAM over the task logit; a localisation diagnostic, not a "
            "faithfulness-tested explanation"),
        PROVENANCE_DECLARED_CONSTANT: (
            "a fixed number written by an engineer (0.9 / 0.95); NOT model confidence"),
        PROVENANCE_CONFIGURED: (
            "a configured mapping from event type to severity; not calibrated, not "
            "learned, not validated"),
    }
