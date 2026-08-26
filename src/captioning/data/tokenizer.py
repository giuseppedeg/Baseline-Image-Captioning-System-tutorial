"""Tokenisation, in two schemes behind one interface.

A caption decoder predicts a distribution over a finite vocabulary. How that
vocabulary is constructed determines what the model is able to say, and the two
schemes implemented here fail in instructively different ways.

``WordTokenizer``
    One entry per surface form above a frequency threshold. Simple, legible,
    and unable to represent anything it did not see during training: every
    unseen word collapses to ``<unk>``. On a corpus of descriptions the words
    it loses are overwhelmingly proper names -- the same content that
    :mod:`captioning.data.entities` removes, arriving at the problem from the
    other direction.

``BPETokenizer``
    Byte-pair encoding over subword units. Any string is representable, so the
    out-of-vocabulary rate is zero by construction, at the cost of sequences
    that are longer and units that are less interpretable.

Train both on your corpus and compare :meth:`BaseTokenizer.oov_rate` on the
validation split before choosing. The exercise is set in
``docs/01_problem_formulation.md``.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

__all__ = [
    "BaseTokenizer",
    "WordTokenizer",
    "BPETokenizer",
    "SPECIAL_TOKENS",
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "UNK_ID",
    "build_tokenizer",
    "load_tokenizer",
]

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

#: Words, contracted forms and hyphenated compounds; punctuation as single
#: tokens. ``16th-century`` is kept whole, which matters: it is one of the few
#: pieces of period information the grounding step preserves.
_WORD_RE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)


class BaseTokenizer(ABC):
    """Common interface. Special-token identifiers are fixed across schemes."""

    kind: str = "abstract"

    pad_id, bos_id, eos_id, unk_id = PAD_ID, BOS_ID, EOS_ID, UNK_ID

    @abstractmethod
    def encode_tokens(self, text: str) -> List[int]:
        """Encode ``text`` without special tokens."""

    @abstractmethod
    def decode_ids(self, ids: Sequence[int]) -> str:
        """Decode ordinary token identifiers back to a string."""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def save(self, path: Union[str, Path]) -> Path: ...

    def encode(
        self, text: str, add_special: bool = True, max_length: Optional[int] = None
    ) -> List[int]:
        """Encode a caption, optionally bracketing it with ``<bos>``/``<eos>``.

        Truncation reserves room for both special tokens, so that a sequence
        clipped at ``max_length`` still terminates with ``<eos>``. A decoder
        trained on sequences that sometimes lack a terminator learns not to
        stop.
        """
        ids = self.encode_tokens(text)
        if not add_special:
            return ids[:max_length] if max_length else ids
        if max_length is not None:
            ids = ids[: max(0, max_length - 2)]
        return [self.bos_id, *ids, self.eos_id]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        ids = list(ids)
        if skip_special:
            specials = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in specials]
        return self.decode_ids(ids)

    def oov_rate(self, texts: Sequence[str]) -> float:
        """Fraction of ordinary tokens that map to ``<unk>``.

        Zero by construction for byte-level subword schemes; the number of
        interest is the one measured on a *held-out* split with a word-level
        vocabulary fitted on training data only.
        """
        total = unknown = 0
        for text in texts:
            ids = self.encode_tokens(text or "")
            total += len(ids)
            unknown += sum(1 for i in ids if i == self.unk_id)
        return unknown / total if total else 0.0

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------------
# Word level
# ---------------------------------------------------------------------------


class WordTokenizer(BaseTokenizer):
    """Whitespace-and-punctuation tokenisation over a closed vocabulary."""

    kind = "word"

    def __init__(self, vocab: Sequence[str], lowercase: bool = True) -> None:
        if list(vocab[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(
                "the first four vocabulary entries must be the special tokens, "
                f"in the order {SPECIAL_TOKENS}"
            )
        self.itos: List[str] = list(vocab)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.lowercase = lowercase

    # -- construction ------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        lowercase: bool = True,
    ) -> "WordTokenizer":
        counter: Counter = Counter()
        for text in texts:
            counter.update(cls._split(text or "", lowercase))
        budget = max(0, vocab_size - len(SPECIAL_TOKENS))
        # Sort by descending frequency, then alphabetically, so that the
        # vocabulary is a deterministic function of the corpus.
        ordered = sorted(
            (t for t, c in counter.items() if c >= min_frequency),
            key=lambda t: (-counter[t], t),
        )
        return cls([*SPECIAL_TOKENS, *ordered[:budget]], lowercase=lowercase)

    @staticmethod
    def _split(text: str, lowercase: bool) -> List[str]:
        if lowercase:
            text = text.lower()
        return _WORD_RE.findall(text)

    # -- interface ---------------------------------------------------------

    def encode_tokens(self, text: str) -> List[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in self._split(text or "", self.lowercase)]

    def decode_ids(self, ids: Sequence[int]) -> str:
        tokens = [self.itos[i] if 0 <= i < len(self.itos) else UNK_TOKEN for i in ids]
        return _detokenise(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"kind": self.kind, "lowercase": self.lowercase, "vocab": self.itos}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WordTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocab"], lowercase=payload.get("lowercase", True))


_NO_SPACE_BEFORE = set(".,;:!?)]}%’'")
_NO_SPACE_AFTER = set("([{“")


def _detokenise(tokens: Sequence[str]) -> str:
    """Reassemble tokens into prose, respecting punctuation spacing."""
    out: List[str] = []
    for token in tokens:
        if out and (token in _NO_SPACE_BEFORE or out[-1] in _NO_SPACE_AFTER):
            out.append(token)
        else:
            out.append(" " + token if out else token)
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# Subword level
# ---------------------------------------------------------------------------


class BPETokenizer(BaseTokenizer):
    """Byte-pair encoding, delegated to the ``tokenizers`` library."""

    kind = "bpe"

    def __init__(self, backend) -> None:  # backend: tokenizers.Tokenizer
        self._backend = backend
        for token, expected in zip(SPECIAL_TOKENS, (PAD_ID, BOS_ID, EOS_ID, UNK_ID)):
            actual = backend.token_to_id(token)
            if actual != expected:
                raise ValueError(
                    f"special token {token!r} has id {actual} but {expected} was expected; "
                    "special tokens must be declared first when training the tokeniser"
                )

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        lowercase: bool = True,
    ) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "the 'tokenizers' package is required for BPE tokenisation:\n"
                "    pip install tokenizers"
            ) from exc

        backend = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        steps = [normalizers.NFKC()]
        if lowercase:
            steps.append(normalizers.Lowercase())
        backend.normalizer = normalizers.Sequence(steps)
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            # Declared first so that their identifiers are 0..3, matching the
            # constants used throughout the package.
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator((t or "" for t in texts), trainer=trainer)
        return cls(backend)

    def encode_tokens(self, text: str) -> List[int]:
        return self._backend.encode(text or "").ids

    def decode_ids(self, ids: Sequence[int]) -> str:
        return self._backend.decode(list(ids)).strip()

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size()

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._backend.save(str(path))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_REGISTRY = {"word": WordTokenizer, "bpe": BPETokenizer}


def build_tokenizer(config, texts: Iterable[str]) -> BaseTokenizer:
    """Fit a tokeniser on ``texts`` according to a :class:`TokenizerConfig`."""
    cls = _REGISTRY[config.kind]
    return cls.train(
        texts,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        lowercase=config.lowercase,
    )


def load_tokenizer(config) -> BaseTokenizer:
    """Load a previously fitted tokeniser described by a :class:`TokenizerConfig`."""
    path = config.artifact_path
    if not path.is_file():
        raise FileNotFoundError(
            f"no tokeniser at {path}. Fit one first:\n"
            f"    python scripts/build_tokenizer.py --config <your-config>.yaml"
        )
    return _REGISTRY[config.kind].load(path)
