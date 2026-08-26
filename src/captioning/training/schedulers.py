"""Learning-rate schedule: linear warmup, then cosine decay.

Warmup matters disproportionately for a decoder trained from scratch. In the
first steps the embeddings are random, the gradients are large and poorly
conditioned, and a full-size step moves the parameters somewhere the model does
not recover from. Ramping the learning rate from zero over a few hundred steps
costs nothing and removes an entire class of failed runs.

Cosine decay afterwards is a default rather than a discovery: it works, it has
no extra hyper-parameters beyond the floor, and it removes the temptation to
tune a step schedule by hand.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from captioning.utils.config import SchedulerConfig

__all__ = ["build_scheduler", "warmup_cosine"]


def warmup_cosine(step: int, warmup_steps: int, total_steps: int, min_factor: float) -> float:
    """Multiplicative factor applied to the base learning rate at ``step``."""
    if warmup_steps > 0 and step < warmup_steps:
        # +1 so that the very first step is not exactly zero.
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig, total_steps: int) -> LambdaLR:
    """Return a per-step scheduler; call ``scheduler.step()`` after every
    optimiser step, not once per epoch."""
    warmup = min(config.warmup_steps, max(0, total_steps - 1))
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine(step, warmup, total_steps, config.min_lr_factor),
    )
