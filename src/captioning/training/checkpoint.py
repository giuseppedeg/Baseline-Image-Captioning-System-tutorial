"""Saving and restoring a run.

A checkpoint records enough to resume training *and* enough to interpret the
result months later: the model weights, the optimiser and scheduler state, the
position in the schedule, and the configuration the run was launched with. The
last of these is the one people omit and then regret, because a weights file
whose architecture and preprocessing are unknown is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from captioning.utils.config import Config
from captioning.utils.logging import get_logger

__all__ = ["CheckpointState", "save_checkpoint", "load_checkpoint", "restore_config"]

logger = get_logger(__name__)


@dataclass
class CheckpointState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    #: Name of the quantity in ``best_metric``, and whether lower is better.
    metric_name: str = "val_cross_entropy"
    lower_is_better: bool = True

    def improves(self, value: float) -> bool:
        return value < self.best_metric if self.lower_is_better else value > self.best_metric


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    state: CheckpointState,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "state": vars(state),
        "config": config or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Restore a checkpoint, returning its full payload.

    ``weights_only=False`` is required because the payload carries the
    configuration dictionary alongside the tensors. Only load checkpoints you
    produced or otherwise trust.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
        if missing:
            logger.warning("checkpoint is missing %d parameter tensor(s): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("checkpoint has %d unexpected tensor(s): %s", len(unexpected), unexpected[:5])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])

    payload["state"] = CheckpointState(**payload.get("state", {}))
    return payload


def restore_config(config: "Config", payload: Dict[str, Any]) -> "Config":
    """Take the architecture from the checkpoint and the rest from the file.

    A checkpoint's weights are only interpretable under the architecture that
    produced them, so ``model`` and ``tokenizer`` must come from the run that
    wrote it -- not from whatever the configuration file happens to say now.
    A single ``--decoder lstm`` override at training time is enough to make the
    two disagree, and the resulting error is a wall of missing state-dict keys
    rather than an explanation.

    Everything else -- which split to read, how to decode, where to write --
    legitimately belongs to the caller and is preserved.
    """
    import dataclasses

    stored = payload.get("config") or {}
    if not stored:
        logger.warning(
            "this checkpoint carries no configuration; falling back to the file, which may "
            "describe a different architecture"
        )
        return config
    try:
        restored = Config.from_dict(stored)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not parse the checkpoint's configuration (%s); using the file", exc)
        return config

    if restored.model != config.model:
        logger.info(
            "architecture taken from the checkpoint: %s decoder, %s encoder",
            restored.model.decoder.kind,
            restored.model.encoder.name,
        )
    return dataclasses.replace(config, model=restored.model, tokenizer=restored.tokenizer)
