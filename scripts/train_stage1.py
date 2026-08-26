#!/usr/bin/env python3
"""Train the Stage 1 captioner: a frozen encoder and a decoder from scratch.

The encoder is frozen, so the only parameters receiving gradients are the
decoder's. This is what makes Stage 1 both fast and stable, and it is the
baseline every later stage is compared against.

Examples
--------
    python scripts/train_stage1.py --config configs/stage1_scratch.yaml

Train the recurrent decoder instead, into its own output directory::

    python scripts/train_stage1.py --config configs/stage1_scratch.yaml \
        --decoder lstm --output-dir runs/stage1_lstm

Check that the pipeline runs before committing a GPU to it::

    python scripts/train_stage1.py --config configs/stage1_scratch.yaml \
        --epochs 1 --limit-batches 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from captioning.data.collate import build_collate
from captioning.data.dataset import CaptionDataset
from captioning.data.tokenizer import load_tokenizer
from captioning.models.captioner import Captioner
from captioning.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from captioning.training.engine import train_one_epoch, validate
from captioning.training.losses import CaptioningLoss
from captioning.training.schedulers import build_scheduler
from captioning.utils.config import Config
from captioning.utils.logging import get_logger
from captioning.utils.seed import seed_worker, set_seed

logger = get_logger("train_stage1")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decoder", choices=("lstm", "transformer"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or a specific device")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument(
        "--limit-batches", type=int, default=0, help="stop each epoch early; for smoke tests"
    )
    return parser.parse_args(argv)


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    model, training = config.model, config.training
    if args.decoder:
        model = replace(model, decoder=replace(model.decoder, kind=args.decoder))
    if args.epochs:
        training = replace(training, epochs=args.epochs)
    if args.batch_size:
        training = replace(training, batch_size=args.batch_size)
    if args.lr:
        training = replace(training, optimizer=replace(training.optimizer, lr=args.lr))
    if args.output_dir:
        training = replace(training, output_dir=args.output_dir)
    return replace(config, model=model, training=training)


def _loader(dataset, config: Config, shuffle: bool, device: torch.device, pad_id: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        collate_fn=build_collate(pad_id),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        drop_last=False,
        persistent_workers=config.training.num_workers > 0,
    )


class _Truncated:
    """Wrap a loader so that only the first ``n`` batches are yielded."""

    def __init__(self, loader: DataLoader, n: int) -> None:
        self.loader, self.n = loader, n

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.n:
                return
            yield batch

    def __len__(self) -> int:
        return min(self.n, len(self.loader))


def main(argv=None) -> int:
    args = parse_args(argv)
    config = _apply_overrides(Config.from_yaml(args.config), args)
    set_seed(config.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("device %s | output %s", device, output_dir)

    tokenizer = load_tokenizer(config.tokenizer)
    logger.info("tokeniser: %s, %d entries", config.tokenizer.kind, tokenizer.vocab_size)

    train_set = CaptionDataset.from_config(config, "train", tokenizer, train=True)
    train_loader = _loader(train_set, config, True, device, tokenizer.pad_id)
    val_loader: Optional[DataLoader] = None
    if config.data.val_csv is not None:
        val_set = CaptionDataset.from_config(config, "val", tokenizer, train=False)
        val_loader = _loader(val_set, config, False, device, tokenizer.pad_id)
        logger.info("train %d images | val %d images", len(train_set), len(val_set))
    else:
        logger.warning(
            "no validation split configured; training will run for the full %d epochs with no "
            "early stopping and no model selection",
            config.training.epochs,
        )

    if args.limit_batches:
        train_loader = _Truncated(train_loader, args.limit_batches)
        if val_loader is not None:
            val_loader = _Truncated(val_loader, args.limit_batches)

    model = Captioner.from_config(config, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    logger.info(model.describe())

    loss_fn = CaptioningLoss(label_smoothing=config.training.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(config.training.optimizer),
        betas=tuple(config.training.optimizer.betas),
    )
    total_steps = max(1, len(train_loader)) * config.training.epochs
    scheduler = build_scheduler(optimizer, config.training.scheduler, total_steps)
    scaler = torch.amp.GradScaler(
        device.type, enabled=config.training.amp and device.type == "cuda"
    )

    state = CheckpointState()
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        state = payload["state"]
        logger.info("resumed from %s at epoch %d", args.resume, state.epoch)

    history = []
    stale_epochs = 0
    for epoch in range(state.epoch, config.training.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=config.training.grad_clip,
            log_every=config.training.log_every,
            epoch=epoch,
        )
        logger.info(train_metrics.render(f"epoch {epoch} train | "))

        record = {"epoch": epoch, "train": train_metrics.as_dict()}
        selection = train_metrics.cross_entropy

        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_fn, device, amp=config.training.amp)
            logger.info(val_metrics.render(f"epoch {epoch} val   | "))
            record["val"] = val_metrics.as_dict()
            selection = val_metrics.cross_entropy

        history.append(record)
        state.epoch = epoch + 1
        state.global_step += train_metrics.steps

        save_checkpoint(
            output_dir / "last.pt", model, state, optimizer, scheduler, config=config.as_dict()
        )
        if state.improves(selection):
            state.best_metric = selection
            stale_epochs = 0
            save_checkpoint(
                output_dir / "best.pt", model, state, config=config.as_dict(),
                extra={"tokenizer": str(config.tokenizer.artifact_path)},
            )
            logger.info("new best %s = %.4f", state.metric_name, selection)
        else:
            stale_epochs += 1
            patience = config.training.early_stopping_patience
            if patience and stale_epochs >= patience:
                logger.info("no improvement for %d epochs; stopping early", stale_epochs)
                break

        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info(
        "done. best %s = %.4f | checkpoints in %s", state.metric_name, state.best_metric, output_dir
    )
    logger.info("next: python scripts/evaluate.py --config %s --checkpoint %s",
                args.config, output_dir / "best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
