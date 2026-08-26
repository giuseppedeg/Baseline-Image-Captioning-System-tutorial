"""Turning a trained decoder into captions.

Training and generation are different problems. During training the model is
given the reference prefix and predicts one token; during generation it is
given its own previous predictions, and a single early mistake propagates. The
gap between the two -- *exposure bias* -- is why validation loss can improve
while generated captions get worse, and why this stage is evaluated by
generating, not only by measuring loss.

Three strategies are implemented.

``greedy``
    Take the arg-max at every step. Fast, deterministic, and myopic: a token
    that is locally optimal can make the rest of the sentence impossible.

``beam``
    Carry the ``k`` highest-scoring prefixes forward and expand all of them.
    Better sequences, at ``k`` times the compute. Requires length
    normalisation: each additional token multiplies in a probability below one,
    so an unnormalised beam always prefers the shortest hypothesis.

``nucleus``
    Sample from the smallest set of tokens whose cumulative probability exceeds
    ``top_p``. Produces varied captions and is the appropriate choice when
    diversity is the goal; it is not the appropriate choice when the metric is
    similarity to a reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from captioning.models.decoders import BaseDecoder
from captioning.models.encoders import EncoderOutput
from captioning.utils.config import InferenceConfig

__all__ = ["GenerationOutput", "generate", "greedy_search", "beam_search", "nucleus_sampling"]


@dataclass
class GenerationOutput:
    """Generated captions and the traces needed to inspect them."""

    #: ``[B, T]`` including the leading ``<bos>``.
    tokens: Tensor
    #: Decoded strings, with special tokens stripped.
    captions: List[str]
    #: ``[B, T, N]`` attention over the encoder grid, or ``None``.
    attention: Optional[Tensor] = None
    #: ``[B]`` sequence log-probability; length-normalised for beam search.
    scores: Optional[Tensor] = None


def _memory_tensor(memory: Union[Tensor, EncoderOutput]) -> Tensor:
    return memory.tokens if isinstance(memory, EncoderOutput) else memory


@torch.no_grad()
def generate(
    decoder: BaseDecoder,
    memory: Union[Tensor, EncoderOutput],
    tokenizer,
    config: InferenceConfig,
    **overrides: Any,
) -> GenerationOutput:
    """Dispatch to the configured strategy, with per-call overrides."""
    settings = {
        "max_new_tokens": config.max_new_tokens,
        "beam_size": config.beam_size,
        "length_penalty": config.length_penalty,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    strategy = overrides.pop("strategy", config.strategy)
    settings.update(overrides)
    memory = _memory_tensor(memory)

    if strategy == "greedy":
        return greedy_search(decoder, memory, tokenizer, settings["max_new_tokens"])
    if strategy == "beam":
        return beam_search(
            decoder,
            memory,
            tokenizer,
            settings["max_new_tokens"],
            settings["beam_size"],
            settings["length_penalty"],
        )
    if strategy == "nucleus":
        return nucleus_sampling(
            decoder,
            memory,
            tokenizer,
            settings["max_new_tokens"],
            settings["temperature"],
            settings["top_p"],
        )
    raise ValueError(f"unknown decoding strategy {strategy!r}")


def _decode_all(tokenizer, tokens: Tensor) -> List[str]:
    return [tokenizer.decode(row.tolist()) for row in tokens]


# ---------------------------------------------------------------------------
# Greedy and sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_search(
    decoder: BaseDecoder, memory: Tensor, tokenizer, max_new_tokens: int = 48
) -> GenerationOutput:
    return _sequential_search(decoder, memory, tokenizer, max_new_tokens, sampler=None)


@torch.no_grad()
def nucleus_sampling(
    decoder: BaseDecoder,
    memory: Tensor,
    tokenizer,
    max_new_tokens: int = 48,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> GenerationOutput:
    def sampler(logits: Tensor) -> Tensor:
        logits = logits / max(temperature, 1e-6)
        ordered, order = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        # Keep the first token that crosses the threshold, so the kept set is
        # never empty even when one token holds more than top_p of the mass.
        remove = cumulative - torch.softmax(ordered, dim=-1) > top_p
        ordered[remove] = float("-inf")
        filtered = torch.full_like(logits, float("-inf")).scatter(-1, order, ordered)
        return torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1).squeeze(-1)

    return _sequential_search(decoder, memory, tokenizer, max_new_tokens, sampler=sampler)


def _sequential_search(
    decoder: BaseDecoder, memory: Tensor, tokenizer, max_new_tokens: int, sampler=None
) -> GenerationOutput:
    device = memory.device
    batch = memory.shape[0]
    state = decoder.init_state(memory)

    tokens = torch.full((batch, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    scores = torch.zeros(batch, device=device)
    attentions: List[Tensor] = []

    for _ in range(max_new_tokens):
        logits, state, attention = decoder.step(state, tokens)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        chosen = log_probs.argmax(dim=-1) if sampler is None else sampler(logits.float())

        # Finished sequences emit padding and stop accumulating score.
        chosen = torch.where(finished, torch.full_like(chosen, tokenizer.pad_id), chosen)
        scores = scores + torch.where(
            finished, torch.zeros_like(scores), log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
        )

        tokens = torch.cat([tokens, chosen.unsqueeze(1)], dim=1)
        if attention is not None:
            attentions.append(attention)
        finished = finished | (chosen == tokenizer.eos_id)
        if bool(finished.all()):
            break

    stacked = torch.stack(attentions, dim=1) if attentions else None
    return GenerationOutput(tokens, _decode_all(tokenizer, tokens), stacked, scores)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@torch.no_grad()
def beam_search(
    decoder: BaseDecoder,
    memory: Tensor,
    tokenizer,
    max_new_tokens: int = 48,
    beam_size: int = 3,
    length_penalty: float = 0.7,
) -> GenerationOutput:
    """Batched beam search.

    The batch and beam dimensions are flattened into one of size ``B * k`` so
    that a single decoder call advances every hypothesis. Row ``b * k + i`` is
    beam ``i`` of example ``b``; the ordering is what makes the reordering
    arithmetic below correct, and it is the part to check first if the output
    looks scrambled.
    """
    if beam_size < 2:
        return greedy_search(decoder, memory, tokenizer, max_new_tokens)

    device = memory.device
    batch, _, _ = memory.shape
    k = beam_size
    vocab = None

    expanded = memory.repeat_interleave(k, dim=0)
    state = decoder.init_state(expanded)

    tokens = torch.full((batch * k, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    # All beams of an example start identical. Giving every beam but the first
    # a score of -inf stops the first expansion from producing k copies of the
    # same hypothesis.
    beam_scores = torch.full((batch, k), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(-1)
    finished = torch.zeros(batch * k, dtype=torch.bool, device=device)
    attention_history: Optional[Tensor] = None

    for _ in range(max_new_tokens):
        logits, state, attention = decoder.step(state, tokens)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        vocab = log_probs.shape[-1]

        # A finished hypothesis may only be extended by padding, at no cost.
        done = finished.nonzero(as_tuple=True)[0]
        if done.numel():
            log_probs[done] = float("-inf")
            log_probs[done, tokenizer.pad_id] = 0.0

        candidates = (beam_scores.unsqueeze(1) + log_probs).view(batch, k * vocab)
        beam_scores, flat_index = candidates.topk(k, dim=-1)

        source_beam = torch.div(flat_index, vocab, rounding_mode="floor")  # [B, k]
        next_token = (flat_index % vocab).view(-1)  # [B * k]
        reorder = (torch.arange(batch, device=device).unsqueeze(1) * k + source_beam).view(-1)

        tokens = torch.cat([tokens.index_select(0, reorder), next_token.unsqueeze(1)], dim=1)
        state = decoder.reorder_state(state, reorder)
        finished = finished.index_select(0, reorder) | (next_token == tokenizer.eos_id)
        beam_scores = beam_scores.view(-1)

        if attention is not None:
            attention = attention.unsqueeze(1)  # [B*k, 1, N]
            attention_history = (
                attention
                if attention_history is None
                else torch.cat([attention_history.index_select(0, reorder), attention], dim=1)
            )

        if bool(finished.all()):
            break

    # Length normalisation. Without it the shortest hypothesis almost always
    # wins, because every extra token adds a negative log-probability.
    lengths = (tokens != tokenizer.pad_id).sum(dim=1).clamp(min=1).float()
    normalised = (beam_scores / lengths.pow(length_penalty)).view(batch, k)
    best = normalised.argmax(dim=-1)
    winners = torch.arange(batch, device=device) * k + best

    tokens = tokens.index_select(0, winners)
    scores = normalised.gather(1, best.unsqueeze(1)).squeeze(1)
    selected_attention = (
        attention_history.index_select(0, winners) if attention_history is not None else None
    )
    return GenerationOutput(tokens, _decode_all(tokenizer, tokens), selected_attention, scores)
