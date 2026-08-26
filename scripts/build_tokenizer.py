#!/usr/bin/env python3
"""Fit the caption tokeniser on the training split.

The tokeniser is fitted on training captions *only*. Fitting it on the union of
the splits leaks information: a word that appears solely in the validation set
would receive an identifier, and the measured out-of-vocabulary rate would then
understate the difficulty of the real task.

Examples
--------
Fit the scheme named in the configuration and persist it::

    python scripts/build_tokenizer.py --config configs/stage1_scratch.yaml

Compare word-level and subword vocabularies before committing to one::

    python scripts/build_tokenizer.py --config configs/stage1_scratch.yaml --compare
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Sequence

import _bootstrap  # noqa: F401

from captioning.data.corpus import read_captions
from captioning.data.tokenizer import build_tokenizer
from captioning.utils.config import Config
from captioning.utils.logging import get_logger
from captioning.utils.seed import set_seed

logger = get_logger("build_tokenizer")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("word", "bpe"), default=None, help="override tokenizer.kind")
    parser.add_argument("--vocab-size", type=int, default=None, help="override tokenizer.vocab_size")
    parser.add_argument(
        "--caption-field", default=None, help="override data.caption_field for this run"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="fit both schemes and report the comparison without saving either",
    )
    return parser.parse_args(argv)


def _statistics(tokenizer, captions: Sequence[str]) -> dict:
    lengths = [len(tokenizer.encode(text, add_special=True)) for text in captions]
    return {
        "vocab_size": tokenizer.vocab_size,
        "mean_length": statistics.mean(lengths) if lengths else 0.0,
        "median_length": statistics.median(lengths) if lengths else 0.0,
        "p95_length": sorted(lengths)[int(0.95 * (len(lengths) - 1))] if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def _render(name: str, stats: dict, oov_train: float, oov_val: float, has_val: bool) -> List[str]:
    val = f"{oov_val:.2%}" if has_val else "n/a"
    return [
        f"  {name:<6} {stats['vocab_size']:>8} {stats['mean_length']:>9.1f} "
        f"{stats['p95_length']:>7} {stats['max_length']:>7} {oov_train:>10.2%} {val:>10}"
    ]


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Config.from_yaml(args.config)
    set_seed(config.seed)

    tok_config = config.tokenizer
    if args.kind:
        tok_config = replace(tok_config, kind=args.kind)
    if args.vocab_size:
        tok_config = replace(tok_config, vocab_size=args.vocab_size)

    field = args.caption_field or config.data.caption_field
    train_captions = read_captions(config.data.train_csv, field, config.data.columns)
    logger.info("read %d training captions from column %r", len(train_captions), field)
    if not train_captions:
        logger.error("no usable captions found; nothing to fit")
        return 2

    val_captions: List[str] = []
    if config.data.val_csv is not None and Path(config.data.val_csv).is_file():
        val_captions = read_captions(config.data.val_csv, field, config.data.columns)
        logger.info("read %d validation captions for out-of-vocabulary measurement", len(val_captions))
    else:
        logger.warning(
            "no validation split configured; the out-of-vocabulary rate can only be measured "
            "on training data, where it is optimistic by construction"
        )

    kinds = ("word", "bpe") if args.compare else (tok_config.kind,)
    header = (
        f"  {'scheme':<6} {'vocab':>8} {'mean len':>9} {'p95':>7} {'max':>7} "
        f"{'OOV train':>10} {'OOV val':>10}"
    )
    lines = ["", "Tokeniser comparison" if args.compare else "Tokeniser summary", "=" * 64, header, "-" * 64]

    fitted = {}
    for kind in kinds:
        cfg = replace(tok_config, kind=kind)
        tokenizer = build_tokenizer(cfg, train_captions)
        fitted[kind] = (cfg, tokenizer)
        stats = _statistics(tokenizer, train_captions)
        lines += _render(
            kind,
            stats,
            tokenizer.oov_rate(train_captions),
            tokenizer.oov_rate(val_captions) if val_captions else 0.0,
            bool(val_captions),
        )
    lines += [
        "",
        "A word-level vocabulary trades coverage for interpretability: every token is a word,",
        "and every word it has not seen becomes <unk>. Subword units remove the second problem",
        "at the cost of longer sequences. Choose with the validation column above, not by habit.",
        "",
    ]
    print("\n".join(lines))

    if args.compare:
        logger.info("comparison only; no tokeniser was saved")
        return 0

    cfg, tokenizer = fitted[tok_config.kind]
    path = cfg.artifact_path
    tokenizer.save(path)
    logger.info("saved %s tokeniser (%d entries) to %s", cfg.kind, tokenizer.vocab_size, path)

    if config.data.max_caption_length < _statistics(tokenizer, train_captions)["p95_length"]:
        logger.warning(
            "data.max_caption_length (%d) truncates more than 5%% of the training captions; "
            "consider raising it",
            config.data.max_caption_length,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
