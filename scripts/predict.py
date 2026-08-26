#!/usr/bin/env python3
"""Caption individual images, optionally saving the attention maps.

Useful for inspection rather than measurement: the attention figures are the
fastest way to tell whether a model that scores adequately is looking at the
right part of the image while it says so.

Examples
--------
    python scripts/predict.py --config configs/stage1_scratch.yaml \
        --checkpoint runs/stage1/best.pt --images data/img/001.jpg data/img/005.jpg

    python scripts/predict.py --config configs/stage1_scratch.yaml \
        --checkpoint runs/stage1/best.pt --images data/img --attention-dir runs/stage1/attention
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import _bootstrap  # noqa: F401
import torch
from PIL import Image

from captioning.data.tokenizer import load_tokenizer
from captioning.data.transforms import build_transforms
from captioning.inference.attention import attention_figure, tokens_of
from captioning.inference.decoding import generate
from captioning.models.captioner import Captioner
from captioning.training.checkpoint import load_checkpoint, restore_config
from captioning.utils.config import Config
from captioning.utils.logging import get_logger

logger = get_logger("predict")

_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="files or directories")
    parser.add_argument("--strategy", choices=("greedy", "beam", "nucleus"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--attention-dir",
        type=Path,
        default=None,
        help="write one attention figure per image into this directory",
    )
    return parser.parse_args(argv)


def _collect(paths) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in _SUFFIXES))
        elif path.is_file():
            files.append(path)
        else:
            logger.warning("skipping %s: not found", path)
    return files


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Config.from_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    files = _collect(args.images)
    if not files:
        logger.error("no images to caption")
        return 2

    payload = load_checkpoint(args.checkpoint, map_location=device)
    config = restore_config(config, payload)

    tokenizer = load_tokenizer(config.tokenizer)
    model = Captioner.from_config(config, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    transform = build_transforms(config.data.image_size, config.data.normalization, train=False)
    batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in files]).to(device)

    with torch.no_grad():
        encoded = model.encode(batch)
        # Attention maps are per-step and unambiguous only for a single
        # trajectory, so figures are produced from greedy decoding even when
        # the reported captions come from a beam.
        strategy = args.strategy or config.inference.strategy
        output = generate(model.decoder, encoded, tokenizer, config.inference, strategy=strategy)
        traced = (
            output
            if strategy == "greedy"
            else generate(model.decoder, encoded, tokenizer, config.inference, strategy="greedy")
        )

    for path, caption in zip(files, output.captions):
        print(f"{path.name}: {caption}")

    if args.attention_dir:
        if traced.attention is None:
            logger.warning("this decoder exposes no attention weights; no figures written")
            return 0
        args.attention_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(files):
            figure = attention_figure(
                batch[index].cpu(),
                tokens_of(tokenizer, traced.tokens[index].tolist()),
                traced.attention[index],
                encoded.grid,
                normalization=config.data.normalization,
            )
            destination = args.attention_dir / f"{path.stem}_attention.png"
            figure.savefig(destination, dpi=120, bbox_inches="tight")
            figure.clf()
            logger.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
