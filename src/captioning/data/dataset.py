"""The map-style dataset that feeds the captioning models.

One row of the corpus table becomes one sample: a preprocessed image tensor,
the caption encoded as token identifiers, and whatever auxiliary metadata the
row carries. Padding, the teacher-forcing shift and mask construction are not
done here but in :mod:`captioning.data.collate`, because they are properties of
a *batch* rather than of a sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image
from torch.utils.data import Dataset

from captioning.data.corpus import load_table, parse_century, resolve_caption_column
from captioning.data.transforms import build_transforms
from captioning.utils.config import ColumnMap, Config
from captioning.utils.logging import get_logger

__all__ = ["CaptionDataset", "CaptionRecord"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaptionRecord:
    """One row of the corpus, after parsing."""

    id: str
    path: Path
    caption: str
    raw_caption: str
    century: Optional[int] = None
    name: str = ""


class CaptionDataset(Dataset):
    """Images paired with captions and optional metadata.

    Parameters
    ----------
    csv_path:
        Corpus table for one split.
    image_root:
        Directory against which the ``path`` column is resolved.
    tokenizer:
        Any :class:`~captioning.data.tokenizer.BaseTokenizer`.
    caption_field:
        Column supervising the decoder. Defaults to the grounded captions;
        falls back to the raw ones with a warning if they are absent.
    transform:
        Image pipeline. When omitted, the deterministic evaluation pipeline is
        built, never the augmented one -- silently augmenting an evaluation set
        is a failure mode worth designing against.
    verify_paths:
        Check at construction time that every referenced file exists. Failing
        here costs a second; failing in epoch three costs an afternoon.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        image_root: Union[str, Path],
        tokenizer,
        *,
        columns: Optional[ColumnMap] = None,
        caption_field: str = "caption_grounded",
        raw_caption_field: str = "caption",
        transform=None,
        max_length: int = 64,
        century_unknown_values: Sequence[int] = (-1,),
        verify_paths: bool = True,
        image_size: int = 224,
        normalization: str = "imagenet",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform or build_transforms(image_size, normalization, train=False)

        frame = load_table(self.csv_path, columns)
        target_column = resolve_caption_column(frame, caption_field, raw_caption_field)
        if target_column != caption_field:
            logger.warning(
                "column %r not found in %s; supervising the decoder with %r instead. "
                "Run scripts/prepare_captions.py to derive grounded captions.",
                caption_field,
                self.csv_path.name,
                target_column,
            )
        else:
            logger.info("supervising the decoder with column %r", target_column)
        raw_column = raw_caption_field if raw_caption_field in frame.columns else target_column

        self.records: List[CaptionRecord] = []
        skipped = 0
        for _, row in frame.iterrows():
            caption = _as_text(row.get(target_column))
            if not caption:
                skipped += 1
                continue
            self.records.append(
                CaptionRecord(
                    id=_as_text(row.get("id")),
                    path=Path(_as_text(row.get("path"))),
                    caption=caption,
                    raw_caption=_as_text(row.get(raw_column)) or caption,
                    century=parse_century(row.get("century"), century_unknown_values),
                    name=_as_text(row.get("name")),
                )
            )
        if skipped:
            logger.warning("%d row(s) of %s had an empty caption and were skipped", skipped, self.csv_path.name)
        if not self.records:
            raise ValueError(f"{self.csv_path} yielded no usable rows")

        if verify_paths:
            self._verify_paths()

    # -- torch Dataset interface ------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        with Image.open(self.image_root / record.path) as handle:
            image = handle.convert("RGB")
            tensor = self.transform(image)
        tokens = self.tokenizer.encode(record.caption, add_special=True, max_length=self.max_length)
        return {
            "image": tensor,
            "tokens": tokens,
            "century": record.century,
            "id": record.id,
            "caption": record.caption,
            "raw_caption": record.raw_caption,
        }

    # -- construction helpers ---------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Config,
        split: str,
        tokenizer,
        train: Optional[bool] = None,
    ) -> "CaptionDataset":
        """Build the dataset for ``split`` ('train', 'val' or 'test')."""
        csv_path = {"train": config.data.train_csv, "val": config.data.val_csv, "test": config.data.test_csv}.get(split)
        if csv_path is None:
            raise ValueError(f"configuration defines no CSV for split {split!r}")
        augment = train if train is not None else (split == "train")
        return cls(
            csv_path,
            config.data.image_root,
            tokenizer,
            columns=config.data.columns,
            caption_field=config.data.caption_field,
            transform=build_transforms(
                config.data.image_size, config.data.normalization, train=augment
            ),
            max_length=config.data.max_caption_length,
            century_unknown_values=config.data.century_unknown_values,
            verify_paths=config.data.verify_paths,
        )

    # -- diagnostics -------------------------------------------------------

    def captions(self) -> List[str]:
        """Every training target, for fitting a tokeniser."""
        return [record.caption for record in self.records]

    def century_coverage(self) -> float:
        """Fraction of rows carrying a usable century.

        Consult this before enabling the period head in Stage 2: an auxiliary
        loss computed over a handful of labelled rows contributes noise, not
        supervision.
        """
        known = sum(1 for r in self.records if r.century is not None)
        return known / len(self.records) if self.records else 0.0

    def _verify_paths(self) -> None:
        missing = [r for r in self.records if not (self.image_root / r.path).is_file()]
        if not missing:
            return
        preview = "\n".join(f"  {r.id}: {self.image_root / r.path}" for r in missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} of {len(self.records)} image(s) referenced by "
            f"{self.csv_path.name} do not exist under {self.image_root}:\n{preview}{more}\n"
            f"Check data.image_root and the image-path column."
        )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text
