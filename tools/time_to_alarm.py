#!/usr/bin/env python3
"""
tools/time_to_alarm.py

How long does the sliding-window temporal branch take to raise an alarm?

WHY THIS MEASUREMENT EXISTS
---------------------------
``docs/temporal_risk_integration.md`` argues that sliding-window aggregation
gives up accuracy (0.789 vs 0.832) in exchange for "a signal that updates
continuously as a live video streams in". The cost of that trade was measured;
the benefit never was. This script measures it, so the trade-off can be stated
with two numbers instead of one.

It reads only committed data. No model, no checkpoint, no video decoding.

DEFINITION OF "ALARM"
---------------------
An alarm is the moment the frozen aggregation rule
(``temporal_risk/frozen_aggregation.json``) would first fire if the clip were
streaming, using exactly the causal replay ``tools/run_demo.py`` performs:

- A window contributes only once its ``last_frame`` has been observed. A
  window covering frames [8,23] is not usable at frame 8; it is usable at
  frame 23, when the 16th of its frames has arrived.
- After each window completes, the frozen rule is applied to every window
  completed so far.
- The alarm frame is the ``last_frame`` of the earliest window at which the
  rule first holds.

For the frozen ``max >= 0.14`` rule that reduces to: the first window, in
index order, whose ``fight_probability >= 0.14``. The implementation applies
the rule generically via ``FrozenAggregationRule.decide`` rather than
special-casing ``max``, so it stays correct if the frozen rule is ever
re-selected as ``k_of_n`` or ``mean``.

Time is ``(alarm_frame + 1) / fps``: after observing frame index ``f``,
``f + 1`` frames have elapsed. Every RWF-2000 clip is exactly 150 frames at
exactly 30.00 fps (verified over all 2000 clips in
``data/rwf2000_metadata/rwf2000_metadata.json``), so one frame is 1/30 s and
the maximum possible alarm time is 144/30 = 4.800 s -- the last window ends at
frame 143, not 149, because 150 frames yield 17 complete 16-frame windows and
the final six frames never complete an eighteenth.

CENSORING -- READ THIS BEFORE QUOTING A MEAN
--------------------------------------------
A Fight clip the rule never flags has no alarm time. It is not a large alarm
time; it is a missing observation (right-censored at 4.800 s). Averaging only
over clips that did alarm therefore *understates* real-world latency, because
the hardest clips are exactly the ones excluded. This script reports the
detected and censored counts side by side and never blends them into one mean.

Usage:
    python tools/time_to_alarm.py
    python tools/time_to_alarm.py --split carve --json outputs/tta_carve.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal_event_adapter import (  # noqa: E402
    FrozenAggregationRule,
    PrecomputedWindowScoreSource,
    WindowScore,
)

FROZEN_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"
SPLIT_CSV = {
    "primary": REPO_ROOT / "temporal_risk" / "primary_window_scores.csv",
    "carve": REPO_ROOT / "temporal_risk" / "carve_window_scores.csv",
}
# Verified over all 2000 clips: fps_counts == {"30.00": 2000},
# frame_count_counts == {"150": 2000}.
SOURCE_FPS = 30.0


def alarm_frame(windows: Sequence[WindowScore], rule: FrozenAggregationRule) -> Optional[int]:
    """Return the frame index at which the rule first fires, or None.

    Replays the windows causally in index order, applying the frozen rule to
    the growing list of completed windows -- the same procedure
    ``tools/run_demo.py`` performs frame by frame, expressed here over windows
    because a window can only change the decision when it completes.
    """
    seen: List[float] = []
    for window in sorted(windows, key=lambda w: w.window_index):
        seen.append(window.fight_probability)
        if rule.decide(seen):
            return window.last_frame
    return None


def _summary(values: List[float]) -> Dict[str, Optional[float]]:
    """Distribution summary for a list of alarm times, in seconds."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    quantiles = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 3
    return {
        "n": len(ordered),
        "mean_seconds": round(statistics.fmean(ordered), 4),
        "median_seconds": round(statistics.median(ordered), 4),
        "stdev_seconds": round(statistics.stdev(ordered), 4) if len(ordered) > 1 else 0.0,
        "min_seconds": round(ordered[0], 4),
        "p25_seconds": round(quantiles[0], 4),
        "p75_seconds": round(quantiles[2], 4),
        "max_seconds": round(ordered[-1], 4),
    }


def compute(split: str = "primary", fps: float = SOURCE_FPS) -> dict:
    """Compute time-to-alarm over one split's committed window scores."""
    csv_path = SPLIT_CSV[split]
    rule = FrozenAggregationRule.load(FROZEN_JSON)
    source = PrecomputedWindowScoreSource(csv_path)

    per_clip: List[dict] = []
    for clip in source.clips():
        windows = source.get(clip) or []
        frame = alarm_frame(windows, rule)
        per_clip.append(
            {
                "clip": clip,
                "true_label": source.true_label(clip),
                "alarmed": frame is not None,
                "alarm_frame": frame,
                "alarm_window_index": next(
                    (w.window_index for w in sorted(windows, key=lambda x: x.window_index)
                     if w.last_frame == frame),
                    None,
                ) if frame is not None else None,
                "time_to_alarm_seconds": round((frame + 1) / fps, 4) if frame is not None else None,
            }
        )

    fights = [row for row in per_clip if row["true_label"] == "Fight"]
    nonfights = [row for row in per_clip if row["true_label"] == "NonFight"]
    detected = [row for row in fights if row["alarmed"]]
    censored = [row for row in fights if not row["alarmed"]]
    false_alarms = [row for row in nonfights if row["alarmed"]]

    last_window_end = max(
        (w.last_frame for clip in source.clips() for w in (source.get(clip) or [])),
        default=0,
    )
    return {
        "tool": "time_to_alarm",
        "split": split,
        "scores_csv": str(csv_path),
        "frozen_rule": {"rule": rule.rule, "params": rule.params},
        "alarm_definition": (
            "first frame at which the frozen aggregation rule holds over all "
            "windows completed so far; a window completes at its last_frame"
        ),
        "fps": fps,
        "source_frames_per_clip": 150,
        "last_window_end_frame": last_window_end,
        "max_possible_alarm_seconds": round((last_window_end + 1) / fps, 4),
        "fight_clips": {
            "total": len(fights),
            "detected": len(detected),
            "censored_never_alarmed": len(censored),
            "recall": round(len(detected) / len(fights), 6) if fights else None,
            "time_to_alarm_over_detected_only": _summary(
                [row["time_to_alarm_seconds"] for row in detected]
            ),
            "censoring_note": (
                f"{len(censored)} Fight clip(s) never alarmed. They are right-censored "
                f"at {round((last_window_end + 1) / fps, 4)} s, NOT included in the "
                "statistics above, and NOT equivalent to a large alarm time. The "
                "reported mean/median therefore describe latency among clips the "
                "system does catch, and understate end-to-end latency."
            ),
            "alarm_window_index_histogram": _histogram(
                [row["alarm_window_index"] for row in detected]
            ),
        },
        "nonfight_clips": {
            "total": len(nonfights),
            "false_alarms": len(false_alarms),
            "false_alarm_rate": round(len(false_alarms) / len(nonfights), 6) if nonfights else None,
            "time_to_false_alarm_over_false_alarms_only": _summary(
                [row["time_to_alarm_seconds"] for row in false_alarms]
            ),
        },
        "per_clip": per_clip,
    }


def _histogram(values: List[Optional[int]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {k: counts[k] for k in sorted(counts, key=lambda s: int(s))}


def render(report: dict) -> str:
    """Human-readable summary."""
    fight = report["fight_clips"]
    nonfight = report["nonfight_clips"]
    detected = fight["time_to_alarm_over_detected_only"]
    lines = [
        f"TIME TO ALARM -- split '{report['split']}'",
        "=" * 66,
        f"frozen rule           : {report['frozen_rule']['rule']} {report['frozen_rule']['params']}",
        f"alarm definition      : {report['alarm_definition']}",
        f"fps / frames per clip : {report['fps']} / {report['source_frames_per_clip']}",
        f"max possible alarm    : {report['max_possible_alarm_seconds']} s "
        f"(last window ends at frame {report['last_window_end_frame']})",
        "",
        f"Fight clips           : {fight['total']}",
        f"  detected (alarmed)  : {fight['detected']}   recall {fight['recall']}",
        f"  censored (no alarm) : {fight['censored_never_alarmed']}",
        "",
        "Time to alarm, over the DETECTED Fight clips only:",
        f"  mean   {detected.get('mean_seconds')} s        median {detected.get('median_seconds')} s",
        f"  stdev  {detected.get('stdev_seconds')} s",
        f"  min    {detected.get('min_seconds')} s        max    {detected.get('max_seconds')} s",
        f"  p25    {detected.get('p25_seconds')} s        p75    {detected.get('p75_seconds')} s",
        "",
        f"  {fight['censoring_note']}",
        "",
        "Alarm window index histogram (which sliding window fired first):",
    ]
    for index, count in fight["alarm_window_index_histogram"].items():
        bar = "#" * min(count, 60)
        lines.append(f"  window {int(index):2d}  {count:4d}  {bar}")
    false_alarm = nonfight["time_to_false_alarm_over_false_alarms_only"]
    lines += [
        "",
        f"NonFight clips        : {nonfight['total']}",
        f"  false alarms        : {nonfight['false_alarms']}   rate {nonfight['false_alarm_rate']}",
        f"  time to false alarm : mean {false_alarm.get('mean_seconds')} s, "
        f"median {false_alarm.get('median_seconds')} s (over false alarms only)",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(SPLIT_CSV), default="primary")
    parser.add_argument("--fps", type=float, default=SOURCE_FPS)
    parser.add_argument("--json", type=Path, default=None, help="Write the full report as JSON.")
    args = parser.parse_args(argv)

    report = compute(args.split, args.fps)
    print(render(report))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
