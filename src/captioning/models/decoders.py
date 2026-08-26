"""Caption decoders, written from scratch.

Two architectures are implemented, and training both is the point of Stage 1.

:class:`AttentionalLSTMDecoder`
    The *Show, Attend and Tell* design (Xu et al., 2015). A recurrent state is
    carried forward one token at a time; at each step an additive attention
    module reads the encoder grid conditioned on that state. The recurrence is
    written out as an explicit Python loop over ``LSTMCell``, because the point
    of implementing it is to see it.

:class:`TransformerCaptionDecoder`
    Masked self-attention over the caption, cross-attention over the encoder
    grid, position-wise feed-forward. Built from ``nn.MultiheadAttention``
    rather than ``nn.TransformerDecoder`` for two reasons: the residual and
    normalisation structure stays visible, and the cross-attention weights
    remain accessible, which is what makes the attention maps of
    :mod:`captioning.inference.attention` possible at all.

Both expose the same three-method interface used by
:mod:`captioning.inference.decoding`:

``forward(memory, tokens)``
    Teacher-forced training pass over a whole caption at once.
``init_state(memory)`` / ``step(state, tokens)``
    Incremental generation, one token at a time.
``reorder_state(state, index)``
    Permutation of the batch dimension, which is what beam search does when
    hypotheses are re-ranked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from captioning.data.collate import causal_mask
from captioning.utils.config import DecoderConfig

__all__ = [
    "DecoderOutput",
    "BaseDecoder",
    "AdditiveAttention",
    "AttentionalLSTMDecoder",
    "TransformerCaptionDecoder",
    "build_decoder",
]

State = Dict[str, Tensor]


@dataclass
class DecoderOutput:
    #: ``[B, T, V]`` unnormalised scores over the vocabulary.
    logits: Tensor
    #: ``[B, T, N]`` attention over the encoder grid, or ``None``.
    attention: Optional[Tensor] = None


class BaseDecoder(nn.Module):
    """Interface shared by the two decoders."""

    def forward(self, memory: Tensor, tokens: Tensor, padding_mask: Optional[Tensor] = None) -> DecoderOutput:
        raise NotImplementedError  # pragma: no cover - abstract

    def init_state(self, memory: Tensor) -> State:
        raise NotImplementedError  # pragma: no cover - abstract

    def step(self, state: State, tokens: Tensor) -> Tuple[Tensor, State, Optional[Tensor]]:
        """Advance one position.

        ``tokens`` is the full prefix generated so far, ``[B, t]``. A recurrent
        decoder consumes only its last column; an attention decoder re-reads
        all of it.
        """
        raise NotImplementedError  # pragma: no cover - abstract

    @staticmethod
    def reorder_state(state: State, index: Tensor) -> State:
        """Select and reorder the batch dimension of every tensor in ``state``."""
        return {key: value.index_select(0, index) for key, value in state.items()}


# ---------------------------------------------------------------------------
# Additive attention
# ---------------------------------------------------------------------------


class AdditiveAttention(nn.Module):
    r"""Bahdanau-style attention.

    Scores each memory slot against the current query with a small feed-forward
    network, :math:`e_i = w^\top \tanh(W_m m_i + W_q q)`, and returns the
    softmax-weighted average of the memory. The projection of the memory does
    not depend on the query, so it is computed once per sequence rather than
    once per time step.
    """

    def __init__(self, memory_dim: int, query_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.memory_proj = nn.Linear(memory_dim, attention_dim)
        self.query_proj = nn.Linear(query_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1)

    def project_memory(self, memory: Tensor) -> Tensor:
        return self.memory_proj(memory)

    def forward(
        self, memory: Tensor, query: Tensor, projected_memory: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Return ``(context [B, M], weights [B, N])``."""
        projected = self.project_memory(memory) if projected_memory is None else projected_memory
        scores = self.score(torch.tanh(projected + self.query_proj(query).unsqueeze(1))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights


# ---------------------------------------------------------------------------
# Recurrent decoder
# ---------------------------------------------------------------------------


class AttentionalLSTMDecoder(BaseDecoder):
    """Show, Attend and Tell.

    The gate on the context vector (``f_beta`` in the original paper) lets the
    model decide how much visual evidence to admit at each step. Function words
    need none; content words need a great deal. Inspecting the gate over a
    generated caption is one of the more revealing diagnostics available at
    this stage.
    """

    def __init__(
        self,
        vocab_size: int,
        memory_dim: int,
        d_model: int = 512,
        hidden_dim: int = 512,
        attention_dim: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.init_h = nn.Linear(memory_dim, hidden_dim)
        self.init_c = nn.Linear(memory_dim, hidden_dim)
        self.attention = AdditiveAttention(memory_dim, hidden_dim, attention_dim)
        self.context_gate = nn.Linear(hidden_dim, memory_dim)
        self.cell = nn.LSTMCell(d_model + memory_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            if hidden_dim != d_model:
                raise ValueError(
                    "tying the output projection to the embedding requires "
                    f"hidden_dim == d_model, got {hidden_dim} and {d_model}"
                )
            self.output.weight = self.embedding.weight

        self.hidden_dim = hidden_dim

    # -- training ----------------------------------------------------------

    def init_state(self, memory: Tensor) -> State:
        pooled = memory.mean(dim=1)
        return {
            "h": torch.tanh(self.init_h(pooled)),
            "c": torch.tanh(self.init_c(pooled)),
            "memory": memory,
            "projected": self.attention.project_memory(memory),
        }

    def forward(self, memory: Tensor, tokens: Tensor, padding_mask: Optional[Tensor] = None) -> DecoderOutput:
        embedded = self.embedding(tokens)
        state = self.init_state(memory)
        logits, attentions = [], []
        for position in range(tokens.shape[1]):
            logit, state, weights = self._advance(state, embedded[:, position])
            logits.append(logit)
            attentions.append(weights)
        return DecoderOutput(torch.stack(logits, dim=1), torch.stack(attentions, dim=1))

    # -- generation --------------------------------------------------------

    def step(self, state: State, tokens: Tensor) -> Tuple[Tensor, State, Optional[Tensor]]:
        return self._advance(state, self.embedding(tokens[:, -1]))

    def _advance(self, state: State, embedded: Tensor) -> Tuple[Tensor, State, Tensor]:
        h, c = state["h"], state["c"]
        context, weights = self.attention(state["memory"], h, state["projected"])
        context = context * torch.sigmoid(self.context_gate(h))
        h, c = self.cell(torch.cat([embedded, context], dim=-1), (h, c))
        logits = self.output(self.dropout(h))
        return logits, {**state, "h": h, "c": c}, weights


# ---------------------------------------------------------------------------
# Transformer decoder
# ---------------------------------------------------------------------------


class TransformerDecoderLayer(nn.Module):
    """Pre-norm decoder layer: masked self-attention, cross-attention, feed-forward.

    Pre-normalisation (normalise the input of each sublayer, add the residual
    afterwards) is used rather than the post-norm of the original paper: it
    trains without a carefully tuned warmup and is what every modern
    implementation does.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        h = self.norm_self(x)
        attended, _ = self.self_attn(
            h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.dropout(attended)

        h = self.norm_cross(x)
        attended, weights = self.cross_attn(
            h, memory, memory, need_weights=True, average_attn_weights=True
        )
        x = x + self.dropout(attended)

        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x, weights


class TransformerCaptionDecoder(BaseDecoder):
    """Masked self-attention over the caption, cross-attention over the image."""

    def __init__(
        self,
        vocab_size: int,
        memory_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_length: int = 128,
        pad_id: int = 0,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_length, d_model)
        self.memory_proj = nn.Linear(memory_dim, d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, vocab_size)
        if tie_weights:
            self.output.weight = self.embedding.weight

        self.d_model = d_model
        self.max_length = max_length
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, std=self.d_model**-0.5)
        nn.init.normal_(self.position.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].zero_()

    # -- shared body -------------------------------------------------------

    def _encode_memory(self, memory: Tensor) -> Tensor:
        return self.memory_norm(self.memory_proj(memory))

    def _run(
        self, memory: Tensor, tokens: Tensor, padding_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        length = tokens.shape[1]
        if length > self.max_length:
            raise ValueError(
                f"sequence of length {length} exceeds the decoder's max_length "
                f"({self.max_length}); raise data.max_caption_length or shorten the input"
            )
        positions = self.position.weight[:length].unsqueeze(0)
        # Scaling the embeddings keeps their magnitude comparable to the
        # positional encodings; without it position dominates content early in
        # training.
        x = self.dropout(self.embedding(tokens) * math.sqrt(self.d_model) + positions)

        mask = causal_mask(length, device=tokens.device)
        attention = None
        for layer in self.layers:
            x, attention = layer(x, memory, attn_mask=mask, key_padding_mask=padding_mask)
        return self.output(self.norm(x)), attention

    # -- training ----------------------------------------------------------

    def forward(self, memory: Tensor, tokens: Tensor, padding_mask: Optional[Tensor] = None) -> DecoderOutput:
        logits, attention = self._run(self._encode_memory(memory), tokens, padding_mask)
        return DecoderOutput(logits, attention)

    # -- generation --------------------------------------------------------

    def init_state(self, memory: Tensor) -> State:
        # The projected memory is the only state worth carrying: it does not
        # change between steps, and recomputing it every step is pure waste.
        return {"memory": self._encode_memory(memory)}

    def step(self, state: State, tokens: Tensor) -> Tuple[Tensor, State, Optional[Tensor]]:
        # The whole prefix is re-encoded at every step. Caching the keys and
        # values of previous positions would make this linear instead of
        # quadratic; it is omitted deliberately, because the cache is an
        # optimisation and this implementation is meant to be read.
        logits, attention = self._run(state["memory"], tokens, padding_mask=None)
        last_attention = attention[:, -1] if attention is not None else None
        return logits[:, -1], state, last_attention


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_decoder(
    config: DecoderConfig,
    vocab_size: int,
    memory_dim: int,
    max_length: int = 128,
    pad_id: int = 0,
) -> BaseDecoder:
    if config.kind == "lstm":
        return AttentionalLSTMDecoder(
            vocab_size=vocab_size,
            memory_dim=memory_dim,
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            attention_dim=config.attention_dim,
            dropout=config.dropout,
            pad_id=pad_id,
            tie_weights=config.tie_weights,
        )
    return TransformerCaptionDecoder(
        vocab_size=vocab_size,
        memory_dim=memory_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        max_length=max_length,
        pad_id=pad_id,
        tie_weights=config.tie_weights,
    )
