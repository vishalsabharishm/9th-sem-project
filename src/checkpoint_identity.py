"""Checkpoint identity and provenance: is this file actually a trained model?

THE HOLE THIS CLOSES
--------------------
``R3D18TemporalModel.load_checkpoint`` sets ``provenance`` to
``checkpoint:<filename>`` for *any* file it can load, and
``is_task_specific`` returns True whenever provenance starts with
``checkpoint``. So a checkpoint written by ``temporal_training.py
--validate-pipeline`` -- a randomly initialised network trained for one epoch
on four clips -- loads as "task-specific" and is indistinguishable from the
real fine-tuned model. It even lands at the canonical
``models/temporal_violence/best.pt`` path. Anything downstream would then
report metrics, or emit violence probabilities, from noise.

Three things it is NOT safe to rely on, and which this module never uses:

- the filename (``best.pt`` is a path convention, not evidence),
- the file size (an untrained R3D-18 is the same 132 MB as a trained one),
- ``torch.load`` succeeding (that proves the file is a tensor archive).

WHAT COUNTS AS EVIDENCE OF TRAINING
-----------------------------------
Only metadata the training loop recorded about the run that produced the
weights. ``Checkpointer.save`` writes ``epoch``, ``monitor``,
``monitored_value``, ``metrics``, ``config``, ``num_classes`` and
``class_labels``; since Step 3 it also writes ``training_provenance``,
``protocol`` and ``saved_at``. A checkpoint is accepted only when that record
describes a real training run:

- the network was not randomly initialised,
- it was evaluated (a monitored value and a metrics block exist),
- it trained on the full split, not a smoke-test subset,
- it has the project's 2-class violence head.

LEGACY CHECKPOINTS
------------------
The authentic ``best.pt`` predates Step 3 and therefore carries no explicit
``training_provenance`` field. Rejecting it for that would be absurd, so for
such files initialisation is *inferred* from ``config.pretrained_backbone``
(the smoke test sets it False; every real run sets it True) and the subset
limits. The verdict records that the judgement was inferred rather than read,
so a reader can tell the difference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rwf2000_config import CANONICAL_LABELS
except ImportError:  # pragma: no cover - supports package execution
    from src.rwf2000_config import CANONICAL_LABELS


# Metadata a training checkpoint must carry. Absence of any of these means the
# file was not written by this project's training loop.
REQUIRED_KEYS: Tuple[str, ...] = (
    "state_dict",
    "epoch",
    "monitor",
    "monitored_value",
    "metrics",
    "config",
    "num_classes",
    "class_labels",
)

EXPECTED_NUM_CLASSES = 2
RANDOM_INIT_MARKERS = ("random_init",)
HASH_CHUNK_BYTES = 1 << 20


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint cannot be inspected at all."""


@dataclass(frozen=True)
class CheckpointVerdict:
    """The outcome of verifying one checkpoint file."""

    path: str
    accepted: bool
    rejections: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    inferred_initialisation: bool = False

    def raise_if_rejected(self) -> "CheckpointVerdict":
        """Raise with every reason, or return self when accepted."""
        if self.accepted:
            return self
        reasons = "\n  - ".join(self.rejections)
        raise CheckpointIntegrityError(
            f"{self.path} is not an acceptable trained checkpoint:\n  - {reasons}\n"
            "See src/checkpoint_identity.py and models/temporal_violence/CHECKPOINT.md."
        )

    def as_dict(self) -> dict:
        """JSON-native representation."""
        return {
            "path": self.path,
            "accepted": self.accepted,
            "rejections": list(self.rejections),
            "warnings": list(self.warnings),
            "inferred_initialisation": self.inferred_initialisation,
            "metadata": self.metadata,
        }


def file_sha256(path: Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    """Stream a SHA-256 so a 132 MB checkpoint does not need to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path) -> Dict[str, Any]:
    """Read a checkpoint's payload without constructing a model.

    ``weights_only=True`` keeps this from executing arbitrary pickled objects
    in a file whose trustworthiness is exactly what is in question.
    """
    import torch  # local import: verification should not require torch at import time

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointIntegrityError(
            f"Could not read {path} as a torch checkpoint: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError(
            f"{path} does not contain a checkpoint dictionary (got "
            f"{type(payload).__name__}). A bare state dict carries no training "
            "record and cannot be verified."
        )
    return payload


def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    """Return the training metadata recorded in a checkpoint, without judging it."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointIntegrityError(f"Checkpoint not found: {path}")
    payload = _load_payload(path)
    config = payload.get("config") or {}
    state_dict = payload.get("state_dict")
    return {
        "filename": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "present_keys": sorted(payload),
        "missing_required_keys": [key for key in REQUIRED_KEYS if key not in payload],
        "tensor_count": len(state_dict) if isinstance(state_dict, dict) else None,
        "epoch": payload.get("epoch"),
        "monitor": payload.get("monitor"),
        "monitored_value": payload.get("monitored_value"),
        "metrics": payload.get("metrics"),
        "num_classes": payload.get("num_classes"),
        "class_labels": payload.get("class_labels"),
        # Written since Step 3; absent on legacy checkpoints.
        "training_provenance": payload.get("training_provenance"),
        "protocol": payload.get("protocol") or config.get("protocol"),
        "saved_at": payload.get("saved_at"),
        "config": config,
    }


def verify_trained_checkpoint(path: Path) -> CheckpointVerdict:
    """Decide whether a file is a checkpoint from a real training run.

    Every check is about recorded training evidence. Nothing depends on the
    filename, the file size, or ``torch.load`` merely succeeding.
    """
    path = Path(path)
    try:
        metadata = inspect_checkpoint(path)
    except CheckpointIntegrityError as error:
        return CheckpointVerdict(path=str(path), accepted=False, rejections=(str(error),))

    rejections: List[str] = []
    warnings: List[str] = []
    config = metadata["config"] or {}

    missing = metadata["missing_required_keys"]
    if missing:
        rejections.append(
            f"missing required training metadata: {missing}. This file was not "
            "written by this project's training loop."
        )

    if metadata["monitored_value"] is None:
        rejections.append(
            "monitored_value is None -- the checkpoint records no validation "
            "score, so nothing shows the weights were ever evaluated."
        )
    if metadata["metrics"] is None:
        rejections.append("metrics block is None -- no evaluation was recorded.")

    epoch = metadata["epoch"]
    if not isinstance(epoch, int) or epoch < 1:
        rejections.append(f"epoch is {epoch!r}; a trained checkpoint records epoch >= 1.")

    # Initialisation: explicit when recorded, inferred for legacy checkpoints.
    provenance = metadata["training_provenance"]
    inferred = False
    if provenance is not None:
        if any(marker in str(provenance) for marker in RANDOM_INIT_MARKERS):
            rejections.append(
                f"training_provenance is {provenance!r} -- the network was randomly "
                "initialised, so its outputs are noise."
            )
    else:
        inferred = True
        warnings.append(
            "no training_provenance field (checkpoint predates Step 3); "
            "initialisation inferred from config.pretrained_backbone."
        )
        pretrained = config.get("pretrained_backbone")
        if pretrained is False:
            rejections.append(
                "config.pretrained_backbone is False -- this is the "
                "--validate-pipeline smoke-test configuration, which starts from "
                "random weights and produces no usable model."
            )
        elif pretrained is None:
            rejections.append(
                "config.pretrained_backbone is absent, so the initialisation "
                "cannot be established either explicitly or by inference."
            )

    for key in ("limit_train_clips", "limit_eval_clips"):
        limit = config.get(key)
        if limit is not None:
            rejections.append(
                f"config.{key} is {limit} -- the run used a subset of the data, "
                "which is the smoke-test path, not a reportable training run."
            )

    num_classes = metadata["num_classes"]
    if num_classes != EXPECTED_NUM_CLASSES:
        rejections.append(
            f"num_classes is {num_classes!r}; this project's violence head has "
            f"{EXPECTED_NUM_CLASSES} classes."
        )
    labels = metadata["class_labels"]
    if labels is not None and list(labels) != list(CANONICAL_LABELS):
        rejections.append(
            f"class_labels {labels!r} do not match this project's "
            f"{list(CANONICAL_LABELS)!r}."
        )

    protocol = metadata["protocol"]
    if protocol is None:
        warnings.append(
            "no protocol recorded (checkpoint predates Step 2); which evaluation "
            "protocol produced it must be established from its documentation."
        )
    elif protocol == "experiment1_primary_monitor":
        warnings.append(
            "protocol is experiment1_primary_monitor: early stopping used the "
            "reporting set, so metrics from this checkpoint are optimistically "
            "biased. Valid as a historical baseline, not as a reported result."
        )

    return CheckpointVerdict(
        path=str(path),
        accepted=not rejections,
        rejections=tuple(rejections),
        warnings=tuple(warnings),
        metadata=metadata,
        inferred_initialisation=inferred,
    )


def checkpoint_record(path: Path, retrieved_from: Optional[str] = None) -> dict:
    """Build the full identity record for a checkpoint: hash plus provenance.

    Deterministic: the same file always produces the same record apart from
    ``recorded_at``, which is excluded from ``identity`` for that reason.
    """
    path = Path(path)
    verdict = verify_trained_checkpoint(path)
    metadata = verdict.metadata
    config = metadata.get("config") or {}
    return {
        "identity": {
            "filename": metadata.get("filename", path.name),
            "sha256": file_sha256(path),
            "bytes": metadata.get("bytes", path.stat().st_size),
            "modified_utc": metadata.get("modified_utc"),
            "retrieved_from": retrieved_from,
        },
        "verification": {
            "accepted": verdict.accepted,
            "rejections": list(verdict.rejections),
            "warnings": list(verdict.warnings),
            "inferred_initialisation": verdict.inferred_initialisation,
        },
        "training": {
            "training_provenance": metadata.get("training_provenance"),
            "protocol": metadata.get("protocol"),
            "monitor": metadata.get("monitor"),
            "monitored_value": metadata.get("monitored_value"),
            "epoch": metadata.get("epoch"),
            "saved_at": metadata.get("saved_at"),
            "num_classes": metadata.get("num_classes"),
            "class_labels": metadata.get("class_labels"),
            "tensor_count": metadata.get("tensor_count"),
        },
        "configuration": {
            key: config.get(key)
            for key in (
                "protocol", "seed", "epochs", "batch_size", "optimizer", "scheduler",
                "learning_rate_head", "learning_rate_backbone", "weight_decay",
                "frozen_modules", "pretrained_backbone", "use_amp", "device",
                "early_stopping_metric", "early_stopping_patience",
                "decision_threshold", "limit_train_clips", "limit_eval_clips",
            )
            if key in config
        },
        "metrics_at_save": metadata.get("metrics"),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def render_checkpoint_md(records: List[dict], missing: Optional[List[dict]] = None) -> str:
    """Render CHECKPOINT.md from identity records and known-absent checkpoints."""
    lines = [
        "# Temporal violence checkpoints",
        "",
        "Identity and provenance record for the R3D-18 violence checkpoints this",
        "project depends on. Generated by `tools/record_checkpoint.py`; do not edit",
        "by hand.",
        "",
        "**Checkpoint weights are not committed to this repository.** `.gitignore`",
        "excludes `*.pt`, and the RWF-2000 licence terms this project follows",
        "(`docs/cloud_setup_rwf2000.md`) discourage redistributing derived",
        "artifacts. This file records identity so a checkpoint obtained elsewhere",
        "can be *verified*, not so it can be distributed.",
        "",
        "Verification rules are in `src/checkpoint_identity.py`. Filename, file",
        "size, and `torch.load` succeeding are explicitly NOT accepted as evidence",
        "of training.",
        "",
    ]

    if records:
        lines += ["## Present and verified", ""]
        for record in records:
            identity = record["identity"]
            training = record["training"]
            verification = record["verification"]
            lines += [
                f"### `{identity['filename']}`",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| SHA-256 | `{identity['sha256']}` |",
                f"| Size | {identity['bytes']:,} bytes |",
                f"| Modified (UTC) | {identity['modified_utc']} |",
                f"| Retrieved from | {identity['retrieved_from'] or 'not recorded'} |",
                f"| Accepted | {'yes' if verification['accepted'] else '**NO**'} |",
                f"| Training provenance | {training['training_provenance'] or 'not recorded (legacy)'} |",
                f"| Protocol | {training['protocol'] or 'not recorded (pre-Step-2)'} |",
                f"| Monitor | {training['monitor']} = {training['monitored_value']} |",
                f"| Best epoch | {training['epoch']} |",
                f"| Classes | {training['num_classes']} {training['class_labels']} |",
                "",
            ]
            if verification["rejections"]:
                lines += ["**Rejected because:**", ""]
                lines += [f"- {reason}" for reason in verification["rejections"]] + [""]
            if verification["warnings"]:
                lines += ["**Warnings:**", ""]
                lines += [f"- {reason}" for reason in verification["warnings"]] + [""]
            if record.get("configuration"):
                lines += [
                    "<details><summary>Training configuration</summary>",
                    "",
                    "```json",
                    json.dumps(record["configuration"], indent=2),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
    else:
        lines += [
            "## Present and verified",
            "",
            "_None. No verified checkpoint is present in this working tree._",
            "",
        ]

    if missing:
        lines += ["## Known but absent", ""]
        for entry in missing:
            lines += [f"### `{entry['filename']}`", ""]
            for key, value in entry.items():
                if key == "filename":
                    continue
                label = key.replace("_", " ").capitalize()
                lines.append(f"- **{label}:** {value}")
            lines.append("")

    lines += [
        "## Verifying a checkpoint you have obtained",
        "",
        "```bash",
        "python tools/record_checkpoint.py --checkpoint models/temporal_violence/best.pt \\",
        "    --retrieved-from \"kaggle notebook7bb9a86555 v3, Output tab\"",
        "```",
        "",
        "The command verifies first and refuses to record an unacceptable file",
        "unless `--record-anyway` is passed, which marks it as rejected in this",
        "document rather than silently accepting it.",
        "",
    ]
    return "\n".join(lines)
