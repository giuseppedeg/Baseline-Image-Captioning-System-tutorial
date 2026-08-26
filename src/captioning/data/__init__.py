"""Corpus access, tokenisation and caption grounding.

Only the text-processing modules are imported eagerly. ``dataset`` and
``transforms`` depend on PyTorch and torchvision, and are resolved lazily
through :pep:`562` so that the preprocessing entry points -- which never touch
a tensor -- run in an environment without them.
"""

from __future__ import annotations

from typing import Any

from captioning.data.corpus import load_table, parse_century, read_captions, resolve_caption_column
from captioning.data.entities import (
    DEFAULT_MASKED_LABELS,
    Entity,
    EntityDetector,
    GroundingReport,
    build_detector,
    ground_caption,
    ground_corpus,
)
from captioning.data.tokenizer import (
    BaseTokenizer,
    BPETokenizer,
    WordTokenizer,
    build_tokenizer,
    load_tokenizer,
)

_LAZY = {
    "CaptionDataset": "captioning.data.dataset",
    "CaptionRecord": "captioning.data.dataset",
    "CaptionBatch": "captioning.data.collate",
    "build_collate": "captioning.data.collate",
    "causal_mask": "captioning.data.collate",
    "build_transforms": "captioning.data.transforms",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    "BPETokenizer",
    "BaseTokenizer",
    "CaptionBatch",
    "CaptionDataset",
    "CaptionRecord",
    "DEFAULT_MASKED_LABELS",
    "Entity",
    "EntityDetector",
    "GroundingReport",
    "WordTokenizer",
    "build_collate",
    "build_detector",
    "build_tokenizer",
    "build_transforms",
    "causal_mask",
    "ground_caption",
    "ground_corpus",
    "load_table",
    "load_tokenizer",
    "parse_century",
    "read_captions",
    "resolve_caption_column",
]
