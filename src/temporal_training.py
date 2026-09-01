"""Training infrastructure for the R3D-18 violence classifier.

ISOLATION
---------
This is a standalone research pipeline for ONE temporal branch of the
wider surveillance project. It does not import, modify, or notify
``AbnormalEventDetector``, the tracker, the crowd/interaction analyzers,
the rule-based events, or ``RiskAssessor``. Wiring a trained checkpoint
into the event pipeline is a separate, later step, gated on the model
actually being trained and evaluated.

NOTHING RUNS ON IMPORT
----------------------
Importing this module builds no model, reads no clips, and downloads no
weights. Training starts only through :func:`run_training`, i.e. by
invoking this file as a script with explicit arguments.

LOCAL VALIDATION vs GPU TRAINING
--------------------------------
Two clearly separated modes:

``--validate-pipeline``
    A CPU smoke test over a handful of clips with a randomly initialised
    backbone. It proves the wiring -- data, sampling, preprocessing,
    forward, backward, metrics, checkpointing -- and produces NO usable
    model and NO reportable metric. This is the only mode intended for
    this 16 GB CPU-only machine.

``--train``
    The real fine-tuning run: Kinetics-400 pretrained initialisation on a
    Colab/Kaggle GPU runtime. Which clips are used depends on
    ``--protocol`` (see ``carved_validation``):

    ``carved_validation`` (default) trains on the 1360-clip remainder of
    the train split and monitors on the 240 clips carved out of it. The
    394-clip reporting set is not loaded during training.

    ``experiment1_primary_monitor`` reproduces the original baseline:
    train on all 1600, monitor on the 394 reporting clips. Retained for
    historical comparison; its metrics are optimistically biased.

Metrics produced by ``--validate-pipeline`` describe an untrained network
and must never be quoted as results.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    from config import MODELS_DIR, OUTPUTS_DIR
    from rwf2000_config import (
        CANONICAL_LABELS,
        LABEL_TO_INDEX,
        InitialTrainingConfig,
        ReproducibilityRecord,
        SamplingConfig,
    )
    from carved_validation import (
        PROTOCOL_CARVED,
        PROTOCOL_EXPERIMENT1,
        SUPPORTED_PROTOCOLS,
        load_carved_split,
        verify_disjoint,
    )
    from rwf2000_dataset import (
        DatasetConfig,
        RWF2000ClipDataset,
        build_carved_train_dataset,
        build_carved_validation_dataset,
        build_official_validation_dataset,
        build_primary_evaluation_dataset,
        build_training_dataset,
    )
    from temporal_metrics import BinaryMetrics, binary_metrics
    from temporal_model import (
        R3D18TemporalModel,
        TemporalModelConfig,
        TemporalModelError,
        resolve_device,
    )
    from temporal_preprocessing import PreprocessingConfig
except ImportError:  # pragma: no cover - supports package execution
    from src.config import MODELS_DIR, OUTPUTS_DIR
    from src.rwf2000_config import (
        CANONICAL_LABELS,
        LABEL_TO_INDEX,
        InitialTrainingConfig,
        ReproducibilityRecord,
        SamplingConfig,
    )
    from src.carved_validation import (
        PROTOCOL_CARVED,
        PROTOCOL_EXPERIMENT1,
        SUPPORTED_PROTOCOLS,
        load_carved_split,
        verify_disjoint,
    )
    from src.rwf2000_dataset import (
        DatasetConfig,
        RWF2000ClipDataset,
        build_carved_train_dataset,
        build_carved_validation_dataset,
        build_official_validation_dataset,
        build_primary_evaluation_dataset,
        build_training_dataset,
    )
    from src.temporal_metrics import BinaryMetrics, binary_metrics
    from src.temporal_model import (
        R3D18TemporalModel,
        TemporalModelConfig,
        TemporalModelError,
        resolve_device,
    )
    from src.temporal_preprocessing import PreprocessingConfig


NUM_CLASSES = 2
POSITIVE_CLASS_INDEX = LABEL_TO_INDEX["Fight"]
DEFAULT_CHECKPOINT_DIR = MODELS_DIR / "temporal_violence"
DEFAULT_HISTORY_DIR = OUTPUTS_DIR / "temporal_violence"


class TrainingError(RuntimeError):
    """Raised when the training pipeline is misconfigured."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRunConfig:
    """Everything needed to reproduce one run.

    Hyperparameter defaults are pulled from
    :class:`rwf2000_config.InitialTrainingConfig`, which documents them as
    INITIAL EXPERIMENTAL SETTINGS -- not tuned, not validated. They are a
    starting point, and any reported result must say so.
    """

    dataset_root: Path
    # Which evaluation protocol this run follows. See src/carved_validation.py
    # and docs/EXPERIMENT_REPRODUCIBILITY.md. 'carved_validation' is the
    # correct, leakage-safe protocol and the default for new runs;
    # 'experiment1_primary_monitor' reproduces the original biased baseline
    # and is retained only for historical comparison.
    protocol: str = PROTOCOL_CARVED
    seed: int = 42
    epochs: int = 30
    batch_size: int = 8
    learning_rate_head: float = 1e-3
    learning_rate_backbone: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"
    scheduler: str = "cosine_annealing"
    frozen_modules: Tuple[str, ...] = ("stem", "layer1", "layer2")
    early_stopping_metric: str = "roc_auc"
    early_stopping_patience: int = 5
    device: str = "cpu"
    num_workers: int = 0
    pretrained_backbone: bool = True
    # Mixed precision is opt-in: it roughly halves 3D-convolution step time on
    # a T4, but it is ignored on CPU, where autocast to float16 is not a win
    # and GradScaler has nothing to scale.
    use_amp: bool = False
    decision_threshold: float = 0.5
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    history_dir: Path = DEFAULT_HISTORY_DIR
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    # Caps used only by the local pipeline smoke test.
    limit_train_clips: Optional[int] = None
    limit_eval_clips: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise TrainingError("epochs must be a positive integer.")
        if self.batch_size <= 0:
            raise TrainingError("batch_size must be a positive integer.")
        if self.num_workers < 0:
            raise TrainingError("num_workers cannot be negative.")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise TrainingError("decision_threshold must be in [0, 1].")
        if self.optimizer not in ("AdamW", "Adam", "SGD"):
            raise TrainingError(f"Unsupported optimizer {self.optimizer!r}.")
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise TrainingError(
                f"Unsupported protocol {self.protocol!r}; expected one of "
                f"{list(SUPPORTED_PROTOCOLS)}."
            )

    @classmethod
    def from_initial_config(
        cls,
        dataset_root: Path,
        initial: Optional[InitialTrainingConfig] = None,
        **overrides,
    ) -> "TrainingRunConfig":
        """Build a run config from the project's documented defaults."""
        base = initial or InitialTrainingConfig()
        settings = dict(
            dataset_root=Path(dataset_root),
            seed=base.seed,
            epochs=base.max_epochs,
            batch_size=base.batch_size,
            learning_rate_head=base.learning_rate_head,
            learning_rate_backbone=base.learning_rate_backbone,
            weight_decay=base.weight_decay,
            optimizer=base.optimizer,
            scheduler=base.scheduler,
            frozen_modules=base.frozen_modules,
            early_stopping_patience=base.early_stopping_patience,
            pretrained_backbone=base.pretrained_backbone,
        )
        settings.update(overrides)
        return cls(**settings)

    def as_dict(self) -> dict:
        """Return a JSON-native representation for the run record."""
        document = asdict(self)
        for key in ("dataset_root", "checkpoint_dir", "history_dir"):
            document[key] = str(document[key])
        return document


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on this machine
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def build_model(config: TrainingRunConfig) -> R3D18TemporalModel:
    """Build a 2-class R3D-18, reusing the existing model wrapper.

    Construction, Kinetics initialisation, head replacement, and
    provenance tracking all live in ``temporal_model``; this only supplies
    the training-specific configuration so there is one model builder in
    the project rather than two.
    """
    return R3D18TemporalModel(
        TemporalModelConfig(
            num_classes=NUM_CLASSES,
            device=config.device,
            pretrained_backbone=config.pretrained_backbone,
            class_labels=CANONICAL_LABELS,
        )
    )


def freeze_modules(module: nn.Module, names: Sequence[str]) -> List[str]:
    """Disable gradients for the named top-level modules.

    Early R3D-18 stages encode generic motion and texture; with only 1600
    clips, fine-tuning them invites overfitting. Returns the names
    actually frozen so a typo cannot silently freeze nothing.
    """
    frozen: List[str] = []
    for name in names:
        target = getattr(module, name, None)
        if target is None:
            raise TrainingError(
                f"Cannot freeze unknown module {name!r}; available: "
                f"{[child for child, _ in module.named_children()]}"
            )
        for parameter in target.parameters():
            parameter.requires_grad = False
        frozen.append(name)
    return frozen


def parameter_groups(module: nn.Module, config: TrainingRunConfig) -> List[dict]:
    """Split parameters into head and backbone groups with separate rates.

    The randomly initialised head needs a larger step than the pretrained
    backbone, which only needs gentle adaptation.
    """
    head_parameters, backbone_parameters = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        (head_parameters if name.startswith("fc.") else backbone_parameters).append(
            parameter
        )
    groups = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config.learning_rate_head})
    if backbone_parameters:
        groups.append(
            {"params": backbone_parameters, "lr": config.learning_rate_backbone}
        )
    if not groups:
        raise TrainingError("No trainable parameters remain after freezing.")
    return groups


def build_optimizer(module: nn.Module, config: TrainingRunConfig):
    """Create the configured optimizer over the parameter groups."""
    groups = parameter_groups(module, config)
    if config.optimizer == "AdamW":
        return torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    if config.optimizer == "Adam":
        return torch.optim.Adam(groups, weight_decay=config.weight_decay)
    return torch.optim.SGD(groups, momentum=0.9, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config: TrainingRunConfig):
    """Create the LR schedule, or ``None`` when scheduling is disabled."""
    if config.scheduler == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler in ("none", None):
        return None
    raise TrainingError(f"Unsupported scheduler {config.scheduler!r}.")


# --------------------------------------------------------------------------
# Checkpointing and history
# --------------------------------------------------------------------------


@dataclass
class Checkpointer:
    """Saves the latest epoch and tracks the best one by a chosen metric."""

    directory: Path
    monitor: str = "roc_auc"
    mode: str = "max"
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise TrainingError("Checkpointer mode must be 'max' or 'min'.")
        self.directory = Path(self.directory)

    @property
    def best_path(self) -> Path:
        """Path of the best checkpoint seen so far."""
        return self.directory / "best.pt"

    @property
    def last_path(self) -> Path:
        """Path of the most recently written checkpoint."""
        return self.directory / "last.pt"

    def is_improvement(self, value: Optional[float]) -> bool:
        """Whether ``value`` beats the best seen. ``None`` never wins."""
        if value is None:
            return False
        if self.best_value is None:
            return True
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def save(
        self,
        module: nn.Module,
        epoch: int,
        metrics: Optional[BinaryMetrics],
        config: TrainingRunConfig,
    ) -> Dict[str, Optional[str]]:
        """Write the ``last`` checkpoint, and ``best`` when it improves."""
        self.directory.mkdir(parents=True, exist_ok=True)
        monitored = getattr(metrics, self.monitor, None) if metrics else None
        payload = {
            "state_dict": module.state_dict(),
            "epoch": epoch,
            "monitor": self.monitor,
            "monitored_value": monitored,
            "metrics": metrics.as_dict() if metrics else None,
            "config": config.as_dict(),
            "num_classes": NUM_CLASSES,
            "class_labels": list(CANONICAL_LABELS),
        }
        torch.save(payload, self.last_path)

        written = {"last": str(self.last_path), "best": None}
        if self.is_improvement(monitored):
            self.best_value = monitored
            self.best_epoch = epoch
            torch.save(payload, self.best_path)
            written["best"] = str(self.best_path)
        return written


@dataclass
class TrainingHistory:
    """Per-epoch record of the run."""

    epochs: List[dict] = field(default_factory=list)

    def append(self, record: dict) -> None:
        """Add one epoch's record."""
        self.epochs.append(record)

    def write(self, path: Path, config: TrainingRunConfig, extra: Optional[dict] = None) -> Path:
        """Persist history plus the configuration that produced it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "config": config.as_dict(),
            "reproducibility": ReproducibilityRecord().as_dict(),
            "epochs": self.epochs,
        }
        if extra:
            document.update(extra)
        target.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
        return target


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _maybe_subset(dataset: RWF2000ClipDataset, limit: Optional[int]):
    """Return a small class-balanced slice of ``dataset``, or it unchanged.

    Used only by the local smoke test. Selection is deterministic (the
    first clips of each class, in order) rather than random, so the
    validation run is reproducible. It is balanced rather than a plain
    head-slice because the clips are sorted by path: the first N would all
    share one label, leaving ROC-AUC undefined and the best-checkpoint
    path unexercised -- so the smoke test would not actually validate the
    metric and checkpointing wiring.
    """
    if limit is None or limit >= len(dataset):
        return dataset

    per_class = max(1, limit // len(LABEL_TO_INDEX))
    chosen: List[int] = []
    for wanted in sorted(set(dataset.labels)):
        taken = [i for i, label in enumerate(dataset.labels) if label == wanted]
        chosen.extend(taken[:per_class])
    return Subset(dataset, sorted(chosen)[:limit] or list(range(limit)))


def build_dataloaders(
    config: TrainingRunConfig,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Build the training and monitor loaders for the configured protocol.

    ``carved_validation`` (default, correct): trains on the 1360-clip
    remainder of the official train split and monitors on the 240-clip carve
    subset held out of train. The 394-clip primary evaluation set is never
    loaded here, so no training-time choice can be influenced by it.

    ``experiment1_primary_monitor`` (original/baseline, retained for
    historical comparison): trains on all 1600 train clips and monitors on
    the 394-clip primary set -- the same set headline metrics are reported
    on. This is the protocol whose threshold ``rwf2000_config`` documents as
    "optimistically biased". It is reproduced faithfully, not fixed.

    The 6 confirmed leaks never enter any loader under either protocol.
    """
    train_config = DatasetConfig(
        sampling=config.sampling,
        preprocessing=config.preprocessing,
        training=True,
        seed=config.seed,
    )
    eval_config = DatasetConfig(
        sampling=config.sampling,
        preprocessing=config.preprocessing,
        training=False,
        seed=config.seed,
    )

    if config.protocol == PROTOCOL_CARVED:
        split = load_carved_split(config.dataset_root)
        problems = verify_disjoint(split)
        if problems:
            raise TrainingError(
                "Carved-validation split failed its leakage check; refusing to "
                "train on it: " + "; ".join(problems)
            )
        train_dataset = build_carved_train_dataset(config.dataset_root, train_config)
        eval_dataset = build_carved_validation_dataset(config.dataset_root, eval_config)
        monitor_set = "carve_validation (240 clips carved from train)"
    else:
        train_dataset = build_training_dataset(config.dataset_root, train_config)
        eval_dataset = build_primary_evaluation_dataset(config.dataset_root, eval_config)
        monitor_set = "primary_evaluation (394 clips -- ALSO THE REPORTING SET)"

    sizes = {
        "protocol": config.protocol,
        "monitor_set": monitor_set,
        "train_clips": len(train_dataset),
        "monitor_clips": len(eval_dataset),
    }

    train_loader = DataLoader(
        _maybe_subset(train_dataset, config.limit_train_clips),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    eval_loader = DataLoader(
        _maybe_subset(eval_dataset, config.limit_eval_clips),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    return train_loader, eval_loader, sizes


# --------------------------------------------------------------------------
# Train / evaluate loops
# --------------------------------------------------------------------------


def train_one_epoch(
    module: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> float:
    """Run one training epoch and return the mean loss per sample.

    When ``scaler`` is a ``GradScaler`` the forward pass runs under CUDA
    autocast and the backward pass is loss-scaled; the loss returned is the
    unscaled one either way, so histories stay comparable across runs.
    """
    module.train()
    use_amp = scaler is not None and device.type == "cuda"
    total_loss, total_samples = 0.0, 0
    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = module(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = module(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_samples += int(labels.size(0))
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    module: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[BinaryMetrics, float]:
    """Evaluate and return metrics plus the mean loss.

    Sampling is deterministic (the evaluation dataset disables jitter), so
    repeating this on the same weights reproduces the same numbers.
    """
    module.eval()
    scores: List[float] = []
    truths: List[int] = []
    total_loss, total_samples = 0.0, 0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        logits = module(clips)
        loss = criterion(logits, labels)

        probabilities = torch.softmax(logits, dim=1)[:, POSITIVE_CLASS_INDEX]
        scores.extend(probabilities.detach().cpu().tolist())
        truths.extend(labels.detach().cpu().tolist())
        total_loss += float(loss.item()) * labels.size(0)
        total_samples += int(labels.size(0))

    return (
        binary_metrics(truths, scores, threshold=threshold),
        total_loss / max(total_samples, 1),
    )


def run_training(config: TrainingRunConfig, log=print) -> dict:
    """Execute a full run. Nothing calls this on import.

    Returns a summary document; the same content is written to the
    history file alongside the checkpoints.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)

    wrapper = build_model(config)
    module = wrapper.model
    frozen = freeze_modules(module, config.frozen_modules)
    optimizer = build_optimizer(module, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()

    # Evaluation deliberately stays in float32: the reported probabilities
    # feed ROC-AUC, and fp16 rounding there would change the metric itself.
    scaler = (
        torch.cuda.amp.GradScaler()
        if config.use_amp and device.type == "cuda"
        else None
    )

    train_loader, eval_loader, sizes = build_dataloaders(config)
    checkpointer = Checkpointer(config.checkpoint_dir, monitor=config.early_stopping_metric)
    history = TrainingHistory()

    log(f"provenance      : {wrapper.provenance}")
    log(f"device          : {device}")
    log(f"mixed precision : {'on' if scaler is not None else 'off'}")
    log(f"frozen modules  : {frozen}")
    log(f"train clips     : {len(train_loader.dataset)} (of {sizes['train_clips']})")
    log(f"protocol        : {config.protocol}")
    log(f"monitor set     : {sizes['monitor_set']}")
    log(
        f"monitor clips   : {len(eval_loader.dataset)} "
        f"(of {sizes['monitor_clips']})"
    )
    if config.protocol == PROTOCOL_EXPERIMENT1:
        log(
            "WARNING: this is the original/baseline protocol. Early stopping "
            "and checkpoint selection use the SAME 394 clips headline metrics "
            "are reported on, so any threshold or metric derived from this run "
            "is optimistically biased. Use --protocol carved_validation for "
            "results that will be reported."
        )

    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        if isinstance(train_loader.dataset, RWF2000ClipDataset):
            train_loader.dataset.set_epoch(epoch)

        epoch_started = time.perf_counter()
        train_loss = train_one_epoch(
            module, train_loader, optimizer, criterion, device, scaler=scaler
        )
        metrics, eval_loss = evaluate(
            module, eval_loader, criterion, device, threshold=config.decision_threshold
        )
        if scheduler is not None:
            scheduler.step()

        improved = checkpointer.is_improvement(getattr(metrics, config.early_stopping_metric, None))
        written = checkpointer.save(module, epoch, metrics, config)
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "metrics": metrics.as_dict(),
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "seconds": time.perf_counter() - epoch_started,
            "improved": improved,
            "checkpoints": written,
        }
        history.append(record)
        log(
            f"epoch {epoch:3d}/{config.epochs}  train_loss={train_loss:.4f}  "
            f"eval_loss={eval_loss:.4f}  {metrics.summary()}"
            f"{'  *best*' if improved else ''}"
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            log(
                f"early stopping: no improvement in {config.early_stopping_metric} "
                f"for {config.early_stopping_patience} epochs"
            )
            break

    summary = {
        "completed_epochs": len(history.epochs),
        "best_epoch": checkpointer.best_epoch,
        "best_value": checkpointer.best_value,
        "monitor": checkpointer.monitor,
        "best_checkpoint": str(checkpointer.best_path)
        if checkpointer.best_path.exists()
        else None,
        "total_seconds": time.perf_counter() - started,
        "provenance": wrapper.provenance,
        "dataset_sizes": sizes,
    }
    history_path = history.write(
        Path(config.history_dir) / "training_history.json", config, {"summary": summary}
    )
    summary["history_path"] = str(history_path)
    log(f"history         : {history_path}")
    return summary


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI. Training requires an explicit mode flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-pipeline",
        action="store_true",
        help="CPU smoke test on a few clips with an untrained backbone. "
        "Produces no usable model and no reportable metric.",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="Run real fine-tuning. Intended for a GPU runtime.",
    )
    parser.add_argument("--root", default="data/rwf2000/RWF-2000", help="Dataset root.")
    parser.add_argument(
        "--protocol",
        choices=list(SUPPORTED_PROTOCOLS),
        default=PROTOCOL_CARVED,
        help="Evaluation protocol. 'carved_validation' (default) monitors on a "
        "240-clip subset carved out of train and never loads the reporting set "
        "during training. 'experiment1_primary_monitor' reproduces the original "
        "biased baseline, which monitors on the reporting set itself; kept for "
        "historical comparison only. See docs/EXPERIMENT_REPRODUCIBILITY.md.",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--history-dir", default=None)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA mixed precision. Ignored on CPU.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip the Kinetics-400 download (used by the smoke test).",
    )
    parser.add_argument("--limit-train-clips", type=int, default=None)
    parser.add_argument("--limit-eval-clips", type=int, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingRunConfig:
    """Turn parsed arguments into a run configuration."""
    overrides: dict = {}
    if args.validate_pipeline:
        # Deliberately tiny, CPU-only, and never downloads weights.
        overrides.update(
            epochs=args.epochs if args.epochs is not None else 1,
            batch_size=args.batch_size if args.batch_size is not None else 2,
            device=args.device or "cpu",
            pretrained_backbone=False,
            limit_train_clips=args.limit_train_clips
            if args.limit_train_clips is not None
            else 4,
            limit_eval_clips=args.limit_eval_clips
            if args.limit_eval_clips is not None
            else 4,
        )
    else:
        overrides.update(
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            pretrained_backbone=not args.no_pretrained,
        )
        for name, value in (
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("limit_train_clips", args.limit_train_clips),
            ("limit_eval_clips", args.limit_eval_clips),
        ):
            if value is not None:
                overrides[name] = value

    # Never in the smoke test: that mode is CPU-only, where AMP is a no-op.
    if args.amp and not args.validate_pipeline:
        overrides["use_amp"] = True
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = Path(args.checkpoint_dir)
    if args.history_dir:
        overrides["history_dir"] = Path(args.history_dir)
    if args.clip_length is not None:
        overrides["sampling"] = SamplingConfig(clip_length=args.clip_length)
    overrides["protocol"] = args.protocol

    return TrainingRunConfig.from_initial_config(Path(args.root), **overrides)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Training never starts without an explicit flag."""
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)

    if args.validate_pipeline:
        print("=" * 68)
        print("PIPELINE VALIDATION ONLY -- untrained backbone, few clips.")
        print("The metrics below describe an untrained network.")
        print("They are NOT results and must not be reported.")
        print("=" * 68)

    try:
        summary = run_training(config)
    except TemporalModelError as error:
        print(f"Model error: {error}", file=sys.stderr)
        return 2
    except TrainingError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, default=str))
    if args.validate_pipeline:
        print("\nPipeline validated. No usable model was produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
