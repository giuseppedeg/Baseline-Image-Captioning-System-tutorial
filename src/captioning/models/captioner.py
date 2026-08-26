"""The composition root: an encoder, a decoder, and the wiring between them.

Everything model-specific lives in :mod:`captioning.models.encoders` and
:mod:`captioning.models.decoders`. This module only decides how they are joined
and how their parameters are exposed to an optimiser -- which turns out to be
the part most often got wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from captioning.models.decoders import BaseDecoder, build_decoder
from captioning.models.encoders import EncoderOutput, VisualEncoder, build_encoder
from captioning.utils.config import Config, OptimizerConfig

__all__ = ["Captioner", "CaptionerOutput"]


@dataclass
class CaptionerOutput:
    #: ``[B, T, V]``
    logits: Tensor
    #: ``[B, T, N]`` cross-attention over the encoder grid, or ``None``.
    attention: Optional[Tensor] = None
    #: ``(height, width)`` of that grid, for reshaping the attention.
    grid: Tuple[int, int] = (0, 0)
    #: Auxiliary head outputs. Empty in Stage 1; populated in Stage 2.
    aux: Dict[str, Tensor] = field(default_factory=dict)


class Captioner(nn.Module):
    """A visual encoder feeding a caption decoder."""

    def __init__(self, encoder: VisualEncoder, decoder: BaseDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, config: Config, vocab_size: int, pad_id: int = 0) -> "Captioner":
        encoder = build_encoder(config.model.encoder)
        decoder = build_decoder(
            config.model.decoder,
            vocab_size=vocab_size,
            memory_dim=encoder.output_dim,
            # The decoder must accommodate the longest sequence the collate
            # function can produce, plus the tokens generation may add.
            max_length=max(config.data.max_caption_length, config.inference.max_new_tokens) + 2,
            pad_id=pad_id,
        )
        return cls(encoder, decoder)

    # -- forward -----------------------------------------------------------

    def encode(self, images: Tensor) -> EncoderOutput:
        return self.encoder(images)

    def forward(
        self, images: Tensor, tokens: Tensor, padding_mask: Optional[Tensor] = None
    ) -> CaptionerOutput:
        """Teacher-forced pass.

        ``tokens`` is the decoder input -- the reference caption shifted right,
        as produced by :func:`captioning.data.collate.build_collate`. The
        corresponding targets are compared against ``logits`` by the loss.
        """
        encoded = self.encode(images)
        decoded = self.decoder(encoded.tokens, tokens, padding_mask=padding_mask)
        return CaptionerOutput(
            logits=decoded.logits, attention=decoded.attention, grid=encoded.grid
        )

    @torch.no_grad()
    def generate(self, images: Tensor, tokenizer, config: Config, **overrides: Any):
        """Convenience wrapper over :mod:`captioning.inference.decoding`."""
        from captioning.inference.decoding import generate

        memory = self.encode(images)
        return generate(self.decoder, memory, tokenizer, config.inference, **overrides)

    # -- optimisation ------------------------------------------------------

    def parameter_groups(self, config: OptimizerConfig) -> List[Dict[str, Any]]:
        """Split parameters into groups by weight decay and by learning rate.

        Two conventions are applied, both standard and both consequential.

        *No weight decay on one-dimensional parameters.* Biases, layer-norm
        gains and the like have no scale to regularise; decaying them costs
        accuracy for no benefit.

        *A separate, lower learning rate for the encoder.* Once the backbone is
        unfrozen in Stage 2, training it at the decoder's learning rate erases
        the pre-training it was chosen for. Stage 1 freezes the encoder, so the
        group is simply empty.
        """
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        groups: Dict[Tuple[str, bool], List[nn.Parameter]] = {
            ("encoder", True): [],
            ("encoder", False): [],
            ("decoder", True): [],
            ("decoder", False): [],
        }
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            where = "encoder" if id(parameter) in encoder_ids else "decoder"
            groups[(where, parameter.ndim >= 2)].append(parameter)

        encoder_lr = config.encoder_lr if config.encoder_lr is not None else config.lr
        out: List[Dict[str, Any]] = []
        for (where, decayed), params in groups.items():
            if not params:
                continue
            out.append(
                {
                    "params": params,
                    "lr": encoder_lr if where == "encoder" else config.lr,
                    "weight_decay": config.weight_decay if decayed else 0.0,
                    "name": f"{where}_{'decay' if decayed else 'no_decay'}",
                }
            )
        return out

    # -- diagnostics -------------------------------------------------------

    def parameter_counts(self) -> Dict[str, int]:
        def count(module: nn.Module, trainable: bool) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad == trainable)

        return {
            "encoder_total": sum(p.numel() for p in self.encoder.parameters()),
            "encoder_trainable": count(self.encoder, True),
            "decoder_total": sum(p.numel() for p in self.decoder.parameters()),
            "decoder_trainable": count(self.decoder, True),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def describe(self) -> str:
        counts = self.parameter_counts()
        return (
            f"{type(self.encoder).__name__} -> {type(self.decoder).__name__} | "
            f"trainable {counts['trainable'] / 1e6:.1f}M of {counts['total'] / 1e6:.1f}M "
            f"(encoder {counts['encoder_trainable'] / 1e6:.1f}M trainable)"
        )
