#!/usr/bin/env python3
"""Derive grounded captions from the raw ones.

The script adds two columns to the corpus table and leaves every existing
column untouched, under its original header:

``caption_raw``
    a verbatim copy of the source caption, kept for evaluation and inspection;
``caption_grounded``
    the caption with visually unsupported content removed.

Preserving the original headers means that the same ``data.columns`` mapping in
the configuration file applies to the input and to the output.

Examples
--------
Ground the training split with the default settings::

    python scripts/prepare_captions.py \
        --input-csv data/train.csv --output-csv data/train_processed.csv --report

Inspect the effect of a different policy without writing anything::

    python scripts/prepare_captions.py --input-csv data/train.csv \
        --strategy placeholder --date-policy mask_all --dry-run --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (side effect: makes `captioning` importable)
import pandas as pd

from captioning.data.entities import (
    DATE_POLICIES,
    DEFAULT_MASKED_LABELS,
    STRATEGIES,
    build_detector,
    ground_corpus,
)
from captioning.utils.config import Config
from captioning.utils.logging import get_logger

logger = get_logger("prepare_captions")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="corpus table to read")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="where to write the augmented table (default: <input>_processed.csv)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="configuration file, used only to look up data.columns.caption",
    )
    parser.add_argument(
        "--caption-column",
        default=None,
        help="source caption column; overrides the configuration",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "spacy", "rules"),
        default="auto",
        help="'auto' prefers spaCy and falls back to the rule-based detector",
    )
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--strategy", choices=STRATEGIES, default="remove")
    parser.add_argument("--date-policy", choices=DATE_POLICIES, default="keep_centuries")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=f"entity labels to remove (default: {' '.join(sorted(DEFAULT_MASKED_LABELS))})",
    )
    parser.add_argument("--report", action="store_true", help="print a grounding report")
    parser.add_argument(
        "--report-file", type=Path, default=None, help="also write the report to this path"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="compute everything but write no output file"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    caption_column = args.caption_column
    if caption_column is None and args.config is not None:
        caption_column = Config.from_yaml(args.config).data.columns.caption
    caption_column = caption_column or "caption"

    # Read without renaming so that the output preserves the original schema.
    frame = pd.read_csv(args.input_csv, dtype=str, keep_default_na=True)
    if caption_column not in frame.columns:
        logger.error(
            "column %r not found in %s; available columns are %s",
            caption_column,
            args.input_csv,
            sorted(frame.columns),
        )
        return 2

    raw = frame[caption_column].fillna("").astype(str).tolist()
    blank = sum(1 for text in raw if not text.strip())
    if blank:
        logger.warning("%d of %d rows have an empty caption", blank, len(raw))

    detector = build_detector(args.detector, args.spacy_model)
    if args.detector == "auto":
        logger.info("using the %s detector", detector.name)
    if detector.name == "rules":
        logger.warning(
            "the rule-based detector is a heuristic fallback; review the report before training"
        )

    labels = set(args.labels) if args.labels else DEFAULT_MASKED_LABELS
    grounded, report = ground_corpus(
        raw,
        detector,
        labels=labels,
        date_policy=args.date_policy,
        strategy=args.strategy,
    )

    frame["caption_raw"] = raw
    frame["caption_grounded"] = grounded

    empty_targets = sum(1 for text in grounded if not text.strip())
    if empty_targets:
        logger.warning(
            "%d grounded caption(s) are empty; those rows will be skipped during training",
            empty_targets,
        )

    text = report.render()
    if args.report:
        print(text)
    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(text, encoding="utf-8")
        logger.info("report written to %s", args.report_file)

    if args.dry_run:
        logger.info("dry run: no output written")
        return 0

    output = args.output_csv or args.input_csv.with_name(f"{args.input_csv.stem}_processed.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    logger.info("wrote %d rows to %s", len(frame), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
