"""Tests for the model, the training objective and the decoding strategies.

These use a randomly initialised, deliberately tiny model. They check
structural properties -- shapes, gradient flow, causality, the arithmetic of
beam search -- none of which depend on the model being any good, and all of
which are silently wrong in a surprising number of published implementations.

    pytest tests/test_models.py
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from captioning.data.collate import IGNORE_INDEX, build_collate, causal_mask  # noqa: E402
from captioning.inference.decoding import beam_search, greedy_search  # noqa: E402
from captioning.models.captioner import Captioner  # noqa: E402
from captioning.models.decoders import BaseDecoder  # noqa: E402
from captioning.training.losses import CaptioningLoss  # noqa: E402
from captioning.training.schedulers import warmup_cosine  # noqa: E402
from captioning.utils.config import Config  # noqa: E402

VOCAB = 40
PAD, BOS, EOS, A, B = 0, 1, 2, 3, 4


@pytest.fixture()
def config() -> Config:
    return Config.from_dict(
        {
            "data": {"image_root": "data", "train_csv": "t.csv", "image_size": 64, "max_caption_length": 16},
            "model": {
                "encoder": {"name": "resnet18", "pretrained": False, "freeze": True},
                "decoder": {"kind": "transformer", "d_model": 32, "n_heads": 4, "n_layers": 2, "ffn_dim": 64, "dropout": 0.0},
            },
            "inference": {"max_new_tokens": 8},
        }
    )


def _model(config: Config, kind: str) -> Captioner:
    config = replace(config, model=replace(config.model, decoder=replace(config.model.decoder, kind=kind)))
    return Captioner.from_config(config, VOCAB, PAD).eval()


class TestCaptioner:
    @pytest.mark.parametrize("kind", ["transformer", "lstm"])
    def test_output_shapes(self, config, kind):
        model = _model(config, kind)
        images = torch.randn(2, 3, 64, 64)
        tokens = torch.randint(4, VOCAB, (2, 5))
        output = model(images, tokens)
        assert output.logits.shape == (2, 5, VOCAB)
        assert output.attention.shape[:2] == (2, 5)
        assert output.attention.shape[2] == output.grid[0] * output.grid[1]

    @pytest.mark.parametrize("kind", ["transformer", "lstm"])
    def test_frozen_encoder_receives_no_gradient(self, config, kind):
        model = _model(config, kind)
        model(torch.randn(1, 3, 64, 64), torch.randint(4, VOCAB, (1, 4))).logits.sum().backward()
        assert all(p.grad is None for p in model.encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())

    def test_frozen_encoder_stays_in_eval_mode(self, config):
        model = _model(config, "transformer")
        model.train()
        batchnorms = [m for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert batchnorms and all(not m.training for m in batchnorms)

    def test_decoder_cannot_read_the_future(self, config):
        """Changing the last input token must not change earlier logits."""
        model = _model(config, "transformer")
        images = torch.randn(1, 3, 64, 64)
        a = torch.randint(4, VOCAB, (1, 6))
        b = a.clone()
        b[0, -1] = (b[0, -1] + 1) % VOCAB
        with torch.no_grad():
            first, second = model(images, a).logits, model(images, b).logits
        assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5)

    def test_parameter_groups_exclude_norms_from_decay(self, config):
        model = _model(config, "transformer")
        groups = model.parameter_groups(config.training.optimizer)
        assert {g["name"] for g in groups} == {"decoder_decay", "decoder_no_decay"}
        assert all(p.ndim == 1 for g in groups if g["name"].endswith("no_decay") for p in g["params"])


class TestMasksAndLoss:
    def test_causal_mask_marks_the_future(self):
        mask = causal_mask(3)
        assert mask[0].tolist() == [False, True, True]
        assert mask[2].tolist() == [False, False, False]

    def test_collate_shifts_by_one(self):
        collate = build_collate(PAD)
        samples = [
            {"image": torch.zeros(3, 4, 4), "tokens": [BOS, A, B, EOS], "century": 17, "id": "1"},
            {"image": torch.zeros(3, 4, 4), "tokens": [BOS, A, EOS], "century": None, "id": "2"},
        ]
        batch = collate(samples)
        assert batch.decoder_input[0].tolist() == [BOS, A, B]
        assert batch.target[0].tolist() == [A, B, EOS]
        assert batch.target[1].tolist() == [A, EOS, IGNORE_INDEX]
        assert batch.padding_mask[1].tolist() == [False, False, True]
        assert batch.century_known.tolist() == [True, False]

    def test_loss_ignores_padded_positions(self):
        loss_fn = CaptioningLoss(label_smoothing=0.0)
        logits = torch.randn(2, 4, VOCAB)
        targets = torch.randint(4, VOCAB, (2, 4))
        targets[1, 2:] = IGNORE_INDEX
        first = loss_fn(logits, targets)
        # Changing the logits at ignored positions must not change the loss.
        logits[1, 2:] = torch.randn(2, VOCAB)
        assert torch.allclose(first.loss, loss_fn(logits, targets).loss, atol=1e-6)
        assert first.n_tokens == 6

    def test_smoothing_is_reported_separately(self):
        logits = torch.randn(1, 3, VOCAB)
        targets = torch.randint(4, VOCAB, (1, 3))
        result = CaptioningLoss(label_smoothing=0.1)(logits, targets)
        assert not torch.allclose(result.loss, result.cross_entropy)
        assert torch.allclose(result.perplexity, torch.exp(result.cross_entropy))


class TestScheduler:
    def test_warmup_then_decay(self):
        assert warmup_cosine(0, 10, 100, 0.05) == pytest.approx(0.1)
        assert warmup_cosine(9, 10, 100, 0.05) == pytest.approx(1.0)
        assert warmup_cosine(10, 10, 100, 0.05) == pytest.approx(1.0)
        assert warmup_cosine(100, 10, 100, 0.05) == pytest.approx(0.05)
        assert warmup_cosine(55, 10, 100, 0.05) < warmup_cosine(30, 10, 100, 0.05)


class _ScriptedDecoder(BaseDecoder):
    """A decoder with a known distribution, used to test search itself.

    From ``<bos>``: ``A`` with probability 0.6, ``B`` with 0.4.
    After ``A``: ``<eos>`` or ``A``, each with probability 0.5.
    After ``B``: ``<eos>`` with probability 1.

    Greedy therefore takes ``A`` and ends with log-probability
    ``log 0.6 + log 0.5 = -1.20``; the better sequence is ``B <eos>`` at
    ``log 0.4 = -0.92``, which only a search that keeps both prefixes finds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vocab = 5

    def init_state(self, memory):
        return {"memory": memory}

    def step(self, state, tokens):
        last = tokens[:, -1]
        probs = torch.zeros(tokens.shape[0], self.vocab)
        probs[last == BOS] = torch.tensor([0.0, 0.0, 0.0, 0.6, 0.4])
        probs[last == A] = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0])
        probs[last == B] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        probs[last == EOS] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        return torch.log(probs.clamp(min=1e-12)), state, None


class _Tokenizer:
    pad_id, bos_id, eos_id, unk_id = PAD, BOS, EOS, 3

    def decode(self, ids, skip_special=True):
        names = {A: "a", B: "b"}
        return " ".join(names[i] for i in ids if i in names)


class TestSearch:
    @pytest.fixture()
    def pieces(self):
        return _ScriptedDecoder(), torch.zeros(1, 4, 8), _Tokenizer()

    def test_greedy_is_myopic(self, pieces):
        decoder, memory, tokenizer = pieces
        output = greedy_search(decoder, memory, tokenizer, max_new_tokens=4)
        assert output.tokens[0].tolist()[:3] == [BOS, A, EOS]

    def test_beam_finds_the_better_sequence(self, pieces):
        decoder, memory, tokenizer = pieces
        output = beam_search(decoder, memory, tokenizer, max_new_tokens=4, beam_size=2, length_penalty=0.0)
        assert output.tokens[0].tolist()[:3] == [BOS, B, EOS]
        assert float(output.scores[0]) == pytest.approx(-0.9163, abs=1e-3)

    def test_beam_of_one_matches_greedy(self, pieces):
        decoder, memory, tokenizer = pieces
        greedy = greedy_search(decoder, memory, tokenizer, max_new_tokens=4)
        beam = beam_search(decoder, memory, tokenizer, max_new_tokens=4, beam_size=1)
        assert greedy.tokens.tolist() == beam.tokens.tolist()

    def test_batch_items_stay_independent(self, pieces):
        """Beam search flattens batch and beam into one axis; a mistake in that
        arithmetic mixes hypotheses between examples."""
        decoder, _, tokenizer = pieces
        memory = torch.zeros(3, 4, 8)
        output = beam_search(decoder, memory, tokenizer, max_new_tokens=4, beam_size=3, length_penalty=0.0)
        rows = output.tokens.tolist()
        assert len(rows) == 3 and all(row[:3] == [BOS, B, EOS] for row in rows)

    def test_generation_stops_at_eos(self, pieces):
        decoder, memory, tokenizer = pieces
        output = greedy_search(decoder, memory, tokenizer, max_new_tokens=10)
        tokens = output.tokens[0].tolist()
        assert EOS in tokens
        assert all(t == PAD for t in tokens[tokens.index(EOS) + 1 :])


@pytest.mark.parametrize("kind", ["transformer", "lstm"])
def test_end_to_end_generation_runs(config, kind):
    model = _model(config, kind)

    class Tok:
        pad_id, bos_id, eos_id = PAD, BOS, EOS

        def decode(self, ids, skip_special=True):
            return " ".join(str(i) for i in ids if i > EOS)

    with torch.no_grad():
        memory = model.encode(torch.randn(2, 3, 64, 64))
        from captioning.inference.decoding import generate

        for strategy in ("greedy", "beam", "nucleus"):
            output = generate(model.decoder, memory, Tok(), config.inference, strategy=strategy)
            assert output.tokens.shape[0] == 2
            assert len(output.captions) == 2
