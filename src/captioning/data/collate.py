"""Batching variable-length captions.

Captions in a batch differ in length, so they are padded to the longest member
and the padding is masked out of the loss. Two conventions in this module are
worth stating explicitly because both are routinely got wrong.

**The teacher-forcing shift.** During training the decoder is given the
reference prefix and asked to predict the next token. A padded sequence
``[<bos>, w1, w2, <eos>]`` therefore yields

    decoder input  : ``<bos> w1   w2``
    target         : ``w1    w2   <eos>``

both of length ``T - 1``. Omitting the shift trains the model to copy its own
input, which produces a spectacularly low training loss and a useless model.

**Mask polarity.** ``padding_mask`` and the causal mask returned by
:func:`causal_mask` both use ``True`` to mean *this position must not be
attended to*, matching the convention of ``torch.nn.MultiheadAttention``'s
``attn_mask`` and ``key_padding_mask`` arguments. Inverting either mask trains
a model that attends only to padding.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

__all__ = ["CaptionBatch", "build_collate", "causal_mask", "IGNORE_INDEX"]

#: Target value excluded from the loss; matches the default of
#: ``torch.nn.CrossEntropyLoss(ignore_index=-100)``.
IGNORE_INDEX = -100


@dataclass
class CaptionBatch:
    """One collated batch."""

    images: Tensor  # [B, 3, H, W] float
    decoder_input: Tensor  # [B, T-1] long
    target: Tensor  # [B, T-1] long, IGNORE_INDEX at padded positions
    padding_mask: Tensor  # [B, T-1] bool, True where padded
    lengths: Tensor  # [B] long, unpadded length of decoder_input
    century: Tensor  # [B] long, IGNORE_INDEX where unknown
    century_known: Tensor  # [B] bool
    ids: List[str]
    raw_captions: List[str]

    def to(self, device: torch.device, non_blocking: bool = False) -> "CaptionBatch":
        moved: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = (
                value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value
            )
        return CaptionBatch(**moved)

    def __len__(self) -> int:
        return self.images.shape[0]


def build_collate(pad_id: int = 0) -> Callable[[Sequence[Dict[str, Any]]], CaptionBatch]:
    """Return a ``collate_fn`` closed over the padding identifier."""

    def collate(samples: Sequence[Dict[str, Any]]) -> CaptionBatch:
        if not samples:
            raise ValueError("received an empty batch")

        images = torch.stack([s["image"] for s in samples], dim=0)
        sequences = [torch.as_tensor(s["tokens"], dtype=torch.long) for s in samples]
        longest = max(int(seq.numel()) for seq in sequences)

        batch = len(samples)
        padded = torch.full((batch, longest), pad_id, dtype=torch.long)
        for row, seq in enumerate(sequences):
            padded[row, : seq.numel()] = seq

        decoder_input = padded[:, :-1]
        target = padded[:, 1:].clone()
        # A position is padding if the *target* is padding; the shifted input
        # at that position is then irrelevant to the loss.
        padding_mask = padded[:, 1:] == pad_id
        target[padding_mask] = IGNORE_INDEX

        lengths = torch.tensor([seq.numel() - 1 for seq in sequences], dtype=torch.long)

        centuries = [s.get("century") for s in samples]
        century_known = torch.tensor([c is not None for c in centuries], dtype=torch.bool)
        century = torch.tensor(
            [int(c) if c is not None else IGNORE_INDEX for c in centuries], dtype=torch.long
        )

        return CaptionBatch(
            images=images,
            decoder_input=decoder_input,
            target=target,
            padding_mask=padding_mask,
            lengths=lengths,
            century=century,
            century_known=century_known,
            ids=[str(s.get("id", "")) for s in samples],
            raw_captions=[str(s.get("raw_caption", "")) for s in samples],
        )

    return collate


def causal_mask(size: int, device: Optional[torch.device] = None) -> Tensor:
    """Upper-triangular mask forbidding attention to future positions.

    Returns a ``[size, size]`` boolean tensor in which ``True`` marks a
    forbidden (query, key) pair, so that position ``i`` may attend only to
    positions ``j <= i``.
    """
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
