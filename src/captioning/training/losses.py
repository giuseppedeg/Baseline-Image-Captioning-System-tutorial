"""The training objective.

Captioning is trained as token-level classification: at each position the model
predicts the next token, and the loss is cross-entropy against the reference.
Two details separate a working implementation from a subtly broken one.

**Padded positions must be excluded.** They carry :data:`IGNORE_INDEX` in the
target, which ``CrossEntropyLoss`` skips. Averaging over padded positions
instead makes the loss depend on the batch's length distribution rather than on
the model.

**Label smoothing is reported separately from cross-entropy.** Smoothing helps
here for a specific reason: the reference caption is one of many acceptable
descriptions, so a target distribution placing all mass on it asserts something
false. But the smoothed value is not a log-likelihood, and exponentiating it
does not give perplexity. This module therefore returns both quantities: the
smoothed loss to optimise, and the true cross-entropy to report.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from captioning.data.collate import IGNORE_INDEX

__all__ = ["CaptioningLoss", "LossOutput", "token_accuracy"]


@dataclass
class LossOutput:
    #: The quantity that is back-propagated.
    loss: Tensor
    #: Unsmoothed cross-entropy, in nats per token. Report this one.
    cross_entropy: Tensor
    #: Fraction of non-padded positions whose arg-max matches the reference.
    accuracy: Tensor
    #: Number of supervised positions in the batch.
    n_tokens: int

    @property
    def perplexity(self) -> Tensor:
        return torch.exp(self.cross_entropy)


class CaptioningLoss(nn.Module):
    """Label-smoothed cross-entropy over non-padded positions."""

    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smoothed = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )
        self.plain = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> LossOutput:
        """``logits`` is ``[B, T, V]`` and ``targets`` is ``[B, T]``."""
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)

        loss = self.smoothed(flat_logits, flat_targets)
        with torch.no_grad():
            cross_entropy = (
                loss.detach()
                if self.label_smoothing == 0.0
                else self.plain(flat_logits, flat_targets)
            )
            accuracy = token_accuracy(flat_logits, flat_targets, self.ignore_index)
            n_tokens = int((flat_targets != self.ignore_index).sum())
        return LossOutput(loss, cross_entropy, accuracy, n_tokens)


@torch.no_grad()
def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = IGNORE_INDEX) -> Tensor:
    """Next-token accuracy over supervised positions.

    A useful sanity signal, not a measure of caption quality: a model that
    predicts common function words correctly scores well while saying nothing.
    """
    mask = targets != ignore_index
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    correct = (logits.argmax(dim=-1) == targets) & mask
    return correct.sum().float() / mask.sum().float()
