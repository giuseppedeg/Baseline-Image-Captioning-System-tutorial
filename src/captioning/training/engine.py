"""The training and validation loops.

Kept deliberately small and free of experiment bookkeeping: the loop advances
the model over a loader and returns numbers. Checkpointing, early stopping and
logging policy belong to the entry point, which is where a reader looks for
them.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from captioning.training.losses import CaptioningLoss
from captioning.utils.logging import get_logger

__all__ = ["EpochMetrics", "train_one_epoch", "validate"]

logger = get_logger(__name__)


@dataclass
class EpochMetrics:
    """Averages over one pass through a loader."""

    loss: float = 0.0
    cross_entropy: float = 0.0
    perplexity: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    seconds: float = 0.0
    steps: int = 0
    tokens: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render(self, prefix: str = "") -> str:
        return (
            f"{prefix}loss {self.loss:.4f} | ce {self.cross_entropy:.4f} | "
            f"ppl {self.perplexity:.2f} | acc {self.accuracy:.3f} | "
            f"{self.tokens / max(self.seconds, 1e-9):,.0f} tok/s"
        )


class _Accumulator:
    """Token-weighted averaging.

    Batches contain different numbers of supervised tokens, so a plain mean
    over batch losses is not the corpus loss. Weighting by token count is.
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.weight = 0.0

    def add(self, weight: float, **values: float) -> None:
        self.weight += weight
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + value * weight

    def mean(self, key: str) -> float:
        return self.totals.get(key, 0.0) / self.weight if self.weight else 0.0


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CaptioningLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    grad_clip: float = 1.0,
    log_every: int = 25,
    epoch: int = 0,
) -> EpochMetrics:
    model.train()
    accumulator = _Accumulator()
    grad_norms = 0.0
    started = time.perf_counter()
    amp_enabled = scaler is not None and scaler.is_enabled()

    for step, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, amp_enabled):
            output = model(batch.images, batch.decoder_input, padding_mask=batch.padding_mask)
            result = loss_fn(output.logits, batch.target)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(result.loss).backward()
            # Gradients must be unscaled before they can be clipped by norm.
            scaler.unscale_(optimizer)
            norm = _clip(model, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            result.loss.backward()
            norm = _clip(model, grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        grad_norms += norm
        accumulator.add(
            result.n_tokens,
            loss=float(result.loss.detach()),
            cross_entropy=float(result.cross_entropy),
            accuracy=float(result.accuracy),
        )

        if log_every and step % log_every == 0:
            logger.info(
                "epoch %d | step %d/%d | loss %.4f | ppl %.2f | lr %.2e",
                epoch,
                step,
                len(loader),
                float(result.loss.detach()),
                float(torch.exp(result.cross_entropy)),
                optimizer.param_groups[0]["lr"],
            )

    steps = max(1, len(loader))
    cross_entropy = accumulator.mean("cross_entropy")
    return EpochMetrics(
        loss=accumulator.mean("loss"),
        cross_entropy=cross_entropy,
        perplexity=float(torch.exp(torch.tensor(cross_entropy))),
        accuracy=accumulator.mean("accuracy"),
        learning_rate=optimizer.param_groups[0]["lr"],
        grad_norm=grad_norms / steps,
        seconds=time.perf_counter() - started,
        steps=steps,
        tokens=int(accumulator.weight),
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CaptioningLoss,
    device: torch.device,
    amp: bool = False,
) -> EpochMetrics:
    """Teacher-forced validation.

    This measures how well the model predicts the reference *given the
    reference prefix*. It is cheap, it is the right quantity for early
    stopping, and it is not caption quality -- for that the model has to
    generate its own prefixes, which is what ``scripts/evaluate.py`` does.
    """
    model.eval()
    accumulator = _Accumulator()
    started = time.perf_counter()

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with _autocast(device, amp):
            output = model(batch.images, batch.decoder_input, padding_mask=batch.padding_mask)
            result = loss_fn(output.logits, batch.target)
        accumulator.add(
            result.n_tokens,
            loss=float(result.loss),
            cross_entropy=float(result.cross_entropy),
            accuracy=float(result.accuracy),
        )

    cross_entropy = accumulator.mean("cross_entropy")
    return EpochMetrics(
        loss=accumulator.mean("loss"),
        cross_entropy=cross_entropy,
        perplexity=float(torch.exp(torch.tensor(cross_entropy))),
        accuracy=accumulator.mean("accuracy"),
        seconds=time.perf_counter() - started,
        steps=max(1, len(loader)),
        tokens=int(accumulator.weight),
    )


def _clip(model: nn.Module, max_norm: float) -> float:
    if max_norm and max_norm > 0:
        return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
    total = torch.sqrt(
        sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
    )
    return float(total)
