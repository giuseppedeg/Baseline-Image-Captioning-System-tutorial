#!/usr/bin/env python3
"""Generate captions for a split and score them.

This is the evaluation that matters. Validation loss measures next-token
prediction *given the reference prefix*; this script makes the model produce
its own prefixes, which is the regime it will actually be used in and the one
where exposure bias shows up.

The row it writes is appended to a JSON file so that later stages can print
their results next to this one. Comparability across stages depends on running
this same script, with the same decoding settings, for every stage.

Examples
--------
    python scripts/evaluate.py --config configs/stage1_scratch.yaml \
        --checkpoint runs/stage1/best.pt --split val --name stage1-transformer

Add reference-free grounding metrics, and compare against earlier stages::

    python scripts/evaluate.py --config configs/stage1_scratch.yaml \
        --checkpoint runs/stage1/best.pt --clip \
        --compare runs/stage1_lstm/metrics.json runs/stage1/metrics.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List

import _bootstrap  # noqa: F401
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from captioning.data.collate import build_collate
from captioning.data.dataset import CaptionDataset
from captioning.data.entities import build_detector
from captioning.data.tokenizer import load_tokenizer
from captioning.evaluation.factual import compute_factual_metrics
from captioning.evaluation.grounding import compute_grounding_metrics
from captioning.evaluation.report import MetricTable
from captioning.evaluation.text_metrics import compute_text_metrics
from captioning.inference.decoding import generate
from captioning.models.captioner import Captioner
from captioning.training.checkpoint import load_checkpoint, restore_config
from captioning.training.engine import validate
from captioning.training.losses import CaptioningLoss
from captioning.utils.config import Config
from captioning.utils.logging import get_logger
from captioning.utils.seed import set_seed

logger = get_logger("evaluate")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--name", default=None, help="label for this system in the table")
    parser.add_argument("--strategy", choices=("greedy", "beam", "nucleus"), default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--clip", action="store_true", help="also compute CLIPScore and recall@k")
    parser.add_argument("--detector", choices=("auto", "spacy", "rules"), default="auto")
    parser.add_argument(
        "--compare", type=Path, nargs="*", default=(), help="earlier result files to print alongside"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Config.from_yaml(args.config)
    if args.strategy or args.beam_size:
        config = replace(
            config,
            inference=replace(
                config.inference,
                strategy=args.strategy or config.inference.strategy,
                beam_size=args.beam_size or config.inference.beam_size,
            ),
        )
    set_seed(config.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # The checkpoint is the authority on the architecture it was trained with.
    payload = load_checkpoint(args.checkpoint, map_location=device)
    config = restore_config(config, payload)

    tokenizer = load_tokenizer(config.tokenizer)
    dataset = CaptionDataset.from_config(config, args.split, tokenizer, train=False)
    if args.limit:
        dataset.records = dataset.records[: args.limit]

    batch_size = args.batch_size or config.training.batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=build_collate(tokenizer.pad_id),
    )

    model = Captioner.from_config(config, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    logger.info("evaluating %s on %s (%d images)", args.checkpoint, args.split, len(dataset))

    # Teacher-forced loss, for continuity with the training curves.
    teacher_forced = validate(model, loader, CaptioningLoss(label_smoothing=0.0), device)
    logger.info(teacher_forced.render("teacher forcing | "))

    predictions: List[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"generating ({config.inference.strategy})"):
            batch = batch.to(device)
            memory = model.encode(batch.images)
            output = generate(model.decoder, memory, tokenizer, config.inference)
            predictions.extend(output.captions)

    references = [record.caption for record in dataset.records]
    raw_references = [record.raw_caption for record in dataset.records]
    centuries = [record.century for record in dataset.records]

    text = compute_text_metrics(predictions, references)
    factual = compute_factual_metrics(predictions, build_detector(args.detector), centuries)
    grounding = None
    if args.clip:
        paths = [config.data.image_root / r.path for r in dataset.records]
        grounding = compute_grounding_metrics(paths, predictions)

    # Results belong beside the checkpoint they describe, not beside whatever
    # the configuration file's output_dir happens to point at.
    output_dir = Path(args.output_dir or args.checkpoint.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{output_dir.name}-{config.model.decoder.kind}"

    table = MetricTable().add(
        name,
        {"cross_entropy": teacher_forced.cross_entropy, "perplexity": teacher_forced.perplexity},
        text,
        factual,
        grounding,
        split=args.split,
        strategy=config.inference.strategy,
        checkpoint=str(args.checkpoint),
        n_images=len(dataset),
    )
    metrics_path = table.save(output_dir / "metrics.json")

    frame = pd.DataFrame(
        {
            "id": [r.id for r in dataset.records],
            "path": [str(r.path) for r in dataset.records],
            "prediction": predictions,
            "reference": references,
            "reference_raw": raw_references,
            "century": centuries,
        }
    )
    predictions_path = output_dir / f"predictions_{args.split}.csv"
    frame.to_csv(predictions_path, index=False)

    combined = MetricTable.collect(list(args.compare) + [metrics_path]) if args.compare else table
    print()
    print(combined.render(f"Generation results ({args.split}, {config.inference.strategy})"))
    print()
    print("Examples")
    print("-" * 72)
    for row in frame.head(5).itertuples():
        print(f"  reference : {row.reference}")
        print(f"  generated : {row.prediction}")
        print()

    logger.info("metrics -> %s | predictions -> %s", metrics_path, predictions_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
