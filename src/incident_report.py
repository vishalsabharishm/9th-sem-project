"""Incident report for one analysed clip.

WHY THIS EXISTS
---------------
A surveillance system that produces no artefact is hard to evaluate. This gives
an examiner something tangible to take away: what fired, when, on what evidence,
and -- just as importantly -- what the system is not entitled to claim about it.

WHAT IT WILL NOT DO
-------------------
Every number here comes from an analysis that already happened. The builder
invents nothing:

  * no confidence value is manufactured for a rule that has none (Crowding and
    Proximity carry no confidence, and the report leaves them empty rather than
    filling in a plausible-looking number);
  * a configured severity label is never converted into a probability;
  * no causal claim is attached to a saliency map;
  * no real-time claim is made anywhere.

The risk disclaimer travels inside the report body, not in a footnote a reader
can drop when they quote it. That is the same rule the dashboard follows: a
severity label must never appear without its validation status.

PROVENANCE
----------
``temporal_source`` records whether the temporal probabilities were replayed
from the committed CSV or produced by live inference. The two are materially
different claims and the report distinguishes them explicitly, because a reader
who assumes "live" when the answer is "replay" has misunderstood the entire
demonstration.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

try:
    from temporal_explanation import (
        RISK_STATUS_TEXT,
        SALIENCY_EXPLANATION,
        SALIENCY_NOT_CLAIMED,
        describe_provenance_legend,
        risk_presentation,
    )
except ImportError:  # pragma: no cover - package-style execution
    from src.temporal_explanation import (  # type: ignore
        RISK_STATUS_TEXT,
        SALIENCY_EXPLANATION,
        SALIENCY_NOT_CLAIMED,
        describe_provenance_legend,
        risk_presentation,
    )

REPORT_VERSION = "incident-report-v1"

LIMITATIONS = (
    "The severity label is a configured mapping from event type. It is not "
    "calibrated, not learned, and not validated against any outcome.",
    "Spatial rules (stationary, restricted area, crowding, proximity) have no "
    "event-level ground truth in this dataset and were never evaluated as "
    "detectors.",
    "Saliency, where present, is a gradient-based localisation diagnostic. It "
    "has not undergone a faithfulness test.",
    "Processing is offline and roughly seven times slower than real time on "
    "CPU. No real-time deployment claim is made.",
    "Rule confidences of 0.9 and 0.95 are engineer-declared constants, not "
    "model confidences.",
)


def _events(risk_assessments: Sequence[dict]) -> List[dict]:
    """Group spatial events, preserving the absence of a confidence value."""
    grouped: Dict[tuple, dict] = {}
    for record in risk_assessments:
        event_type = record.get("event_type")
        key = (event_type, record.get("risk_level"))
        entry = grouped.setdefault(key, {
            "event_type": event_type,
            "risk_level": record.get("risk_level"),
            "occurrences": 0,
            # None is meaningful: Crowding and Proximity genuinely have no
            # confidence, and inventing one here would be a fabrication.
            "confidence": record.get("confidence"),
            "confidence_provenance": record.get("confidence_provenance"),
            "risk_level_is_validated": record.get("risk_level_is_validated", False),
        })
        entry["occurrences"] += 1
    return sorted(grouped.values(), key=lambda item: -item["occurrences"])


def build_incident_report(
    summary: Dict[str, Any],
    risk_assessments: Sequence[dict],
    window_scores: Sequence[dict],
    overall_risk: str,
    saliency: Optional[dict] = None,
    selected_window: Optional[dict] = None,
    generated_at: Optional[str] = None,
    fusion: Optional[dict] = None,
    spatial_feature: Optional[dict] = None,
) -> dict:
    """Assemble the report from an analysis that already ran.

    ``saliency`` is the optional payload from the explanation endpoint. When it
    is absent the report says saliency was not requested, which is different
    from saying it failed -- a distinction an examiner will ask about.
    """
    windows = list(window_scores or [])
    peak = max(windows, key=lambda w: w["fight_probability"]) if windows else None
    provenance = summary.get("temporal_provenance") or {}
    source = provenance.get("temporal_source")

    if saliency is None:
        saliency_block = {
            "requested": False,
            "available": False,
            "status": "not requested for this report",
        }
    elif not saliency.get("available"):
        saliency_block = {
            "requested": True,
            "available": False,
            "status": saliency.get("reason", "unavailable"),
            "detail": saliency.get("detail", ""),
        }
    else:
        block = saliency["saliency"]
        saliency_block = {
            "requested": True,
            "available": True,
            "window_frames": [
                saliency["temporal_evidence"]["window_first_frame"],
                saliency["temporal_evidence"]["window_last_frame"],
            ],
            "peak_frame": block["peak_frame"],
            "displayed_temporal_slices": block["displayed_temporal_slices"],
            "raw_temporal_positions": block["raw_temporal_positions"],
            "temporal_resolution_caveat": block["temporal_resolution_caveat"],
            "faithfulness_tested": block["faithfulness_tested"],
            "interpretation": SALIENCY_EXPLANATION,
            "not_claimed": SALIENCY_NOT_CLAIMED,
        }

    return {
        "report_version": REPORT_VERSION,
        "generated_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "clip": {
            "clip_key": summary.get("clip_key"),
            "frames_processed": summary.get("frames_processed"),
            "ground_truth_label": summary.get("ground_truth_label"),
            "ground_truth_available": summary.get("ground_truth_available"),
        },
        "temporal_evidence": {
            "decision": summary.get("final_temporal_decision"),
            "signal_fired": summary.get("temporal_violence_signal_fired"),
            "first_alarm_frame": summary.get("temporal_signal_first_frame"),
            "windows_observed": summary.get("windows_observed"),
            "peak_probability": peak["fight_probability"] if peak else None,
            "peak_window_index": peak["window_index"] if peak else None,
            "peak_window_frames": [peak["first_frame"], peak["last_frame"]] if peak else None,
            "probability_provenance": "measured_model_probability",
            "aggregation_rule": "max over completed windows >= 0.14 (frozen)",
        },
        "selected_window": (
            {
                "window_index": selected_window.get("window_index"),
                "first_frame": selected_window.get("first_frame"),
                "last_frame": selected_window.get("last_frame"),
                "fight_probability": selected_window.get("fight_probability"),
                "note": (
                    "the window the operator selected in the timeline; this is "
                    "not necessarily the peak window the frozen max rule used"
                ),
            }
            if selected_window else None
        ),
        "spatial_events": _events(risk_assessments),
        # The measured fusion feature, kept separate from the rule events above
        # because it is a continuous measurement, not a fired rule.
        "spatial_fusion_feature": spatial_feature,
        "fusion": fusion,
        "risk_interpretation": {
            **risk_presentation(overall_risk),
            "disclaimer": RISK_STATUS_TEXT,
        },
        "saliency": saliency_block,
        "provenance": {
            "temporal_source": source,
            "mode": "replay (precomputed scores)" if source == "csv"
                    else "live inference" if source == "live"
                    else "unknown",
            "note": provenance.get("note"),
            "scores_from": provenance.get("scores_from"),
            # Present only for a live run: which weights actually produced
            # these numbers, and how long they took. A replayed report has no
            # checkpoint because no model ran in that process.
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "model_provenance": provenance.get("model_provenance"),
            "device": provenance.get("device"),
            "inference_seconds_total": provenance.get("inference_seconds_total"),
            "legend": describe_provenance_legend(),
        },
        "limitations": list(LIMITATIONS),
        "not_claimed": {
            "calibrated_risk": False,
            "faithful_explanation": False,
            "real_time": False,
            "causality": False,
        },
    }


def render_incident_report_text(report: dict) -> str:
    """A plain-text rendering for someone who will not open JSON."""
    clip = report["clip"]
    temporal = report["temporal_evidence"]
    risk = report["risk_interpretation"]
    lines = [
        "INCIDENT REPORT",
        "=" * 68,
        f"Clip            : {clip['clip_key']}",
        f"Generated (UTC) : {report['generated_utc']}",
        f"Frames processed: {clip['frames_processed']}",
        f"Ground truth    : {clip['ground_truth_label']}",
        "",
        "TEMPORAL EVIDENCE",
        "-" * 68,
        f"Decision          : {temporal['decision']}",
        f"Signal fired      : {temporal['signal_fired']}"
        + (f" at frame {temporal['first_alarm_frame']}" if temporal["signal_fired"] else ""),
        f"Peak probability  : {temporal['peak_probability']}"
        f" (window {temporal['peak_window_index']}, frames {temporal['peak_window_frames']})",
        f"Windows observed  : {temporal['windows_observed']}",
        f"Aggregation rule  : {temporal['aggregation_rule']}",
        f"Value provenance  : {temporal['probability_provenance']}",
        "",
    ]
    selected = report.get("selected_window")
    if selected:
        lines += [
            "SELECTED WINDOW (operator choice, not necessarily the peak)",
            "-" * 68,
            f"  Window {selected['window_index']}, frames "
            f"{selected['first_frame']}-{selected['last_frame']}, "
            f"p = {selected['fight_probability']}",
            "",
        ]
    lines += [
        "SPATIAL EVENTS",
        "-" * 68,
    ]
    if report["spatial_events"]:
        for event in report["spatial_events"]:
            confidence = (
                f"{event['confidence']} ({event['confidence_provenance']})"
                if event["confidence"] is not None else "none recorded"
            )
            lines.append(
                f"  {event['event_type']:<26} x{event['occurrences']:<5} "
                f"risk {event['risk_level']:<8} confidence {confidence}"
            )
    else:
        lines.append("  none")

    feature = report.get("spatial_fusion_feature")
    if feature:
        lines += ["", "SPATIAL FUSION FEATURE (measured)", "-" * 68]
        if feature.get("defined"):
            lines += [
                f"  {feature['feature']} = {feature['spatial_score']:.6f}",
                f"  mean speed {feature['speed_mean_pixels']:.3f} px / "
                f"mean group diagonal {feature['group_diagonal_mean']:.3f} px "
                f"({feature['form']})",
                f"  persons/frame {feature['person_count_mean']}, "
                f"{feature['zero_person_frames']} frame(s) with no person",
            ]
        else:
            lines.append(f"  undefined -- {feature.get('note')}")

    fusion = report.get("fusion")
    if fusion:
        lines += ["", "EVIDENCE FUSION", "-" * 68]
        if not fusion.get("available"):
            lines.append(f"  Not available : {fusion.get('reason')} -- {fusion.get('detail','')}")
        else:
            lines.append(f"  Mode      : {fusion['mode']}")
            for name, candidate in fusion["candidates"].items():
                score = candidate.get("score")
                suffix = f"  score {score:.6f}" if isinstance(score, float) else ""
                lines.append(f"    {name:18} {candidate['decision']:<9}{suffix}")
            lines += [
                f"  System    : {fusion['system_decision']['decision']} "
                f"(from {fusion['system_decision']['from']})",
                f"  Causality : {fusion['causality']}",
            ]
        # The no-improvement statement travels inside the report body, not a
        # footnote, for the same reason the risk disclaimer does.
        lines.append(f"  Finding   : {fusion['interpretation']}")
        lines.append(f"  {fusion['not_claimed']}")

    lines += [
        "",
        "RISK INTERPRETATION",
        "-" * 68,
        f"Level  : {risk.get('level')}",
        f"Status : {risk['disclaimer']}",
        "",
        "SALIENCY",
        "-" * 68,
    ]
    saliency = report["saliency"]
    if saliency.get("available"):
        lines += [
            f"Window        : frames {saliency['window_frames'][0]}-{saliency['window_frames'][1]}",
            f"Peak frame    : {saliency['peak_frame']}",
            f"Resolution    : {saliency['displayed_temporal_slices']} slices displayed from "
            f"{saliency['raw_temporal_positions']} resolved temporal positions",
            f"Faithfulness  : tested = {saliency['faithfulness_tested']}",
            f"Interpretation: {saliency['interpretation']}",
        ]
    else:
        lines.append(f"Not available : {saliency.get('status')}")

    provenance = report["provenance"]
    lines += [
        "",
        "PROVENANCE",
        "-" * 68,
        f"Mode : {provenance['mode']}",
    ]
    if provenance.get("checkpoint_sha256"):
        lines.append(f"Model: {provenance['checkpoint_sha256']} on {provenance.get('device')}")
    if provenance.get("inference_seconds_total") is not None:
        lines.append(f"Time : {provenance['inference_seconds_total']} s of R3D inference")
    if provenance.get("note"):
        lines.append(f"Note : {provenance['note']}")

    lines += ["", "LIMITATIONS", "-" * 68]
    lines += [f"  - {item}" for item in report["limitations"]]
    lines.append("")
    return "\n".join(lines)
