"""Reading the corpus table.

Kept separate from :mod:`captioning.data.dataset` so that the preprocessing
entry points -- which only ever touch text -- can read the corpus without
importing PyTorch.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd

from captioning.utils.config import ColumnMap

__all__ = [
    "CANONICAL_COLUMNS",
    "load_table",
    "parse_century",
    "resolve_caption_column",
    "read_captions",
]

#: Internal names the columns are mapped onto.
CANONICAL_COLUMNS = ("id", "path", "name", "century", "caption")

_LEADING_INT = re.compile(r"^\s*(-?\d+)")
_MISSING = {"", "nan", "none", "null", "na", "n/a", "unknown", "-"}


def load_table(
    csv_path: Union[str, Path],
    columns: Optional[ColumnMap] = None,
    required: Sequence[str] = ("id", "path", "caption"),
) -> pd.DataFrame:
    """Read a corpus CSV and rename its columns to the canonical names.

    Columns not named in the mapping are preserved unchanged, so derived
    columns such as ``caption_raw``, ``caption_grounded`` or ``typology``
    survive a round trip through this function.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"corpus file not found: {csv_path}")

    columns = columns or ColumnMap()
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=True)

    mapping = columns.canonical_mapping()
    missing_sources = [src for src in mapping if src not in frame.columns]
    rename = {src: dst for src, dst in mapping.items() if src in frame.columns}
    frame = frame.rename(columns=rename)

    absent = [name for name in required if name not in frame.columns]
    if absent:
        raise KeyError(
            f"{csv_path} is missing required column(s) {absent}. "
            f"Columns present: {sorted(frame.columns)}. "
            f"Column names not found through the configured mapping: {missing_sources}. "
            f"Adjust data.columns in the configuration file rather than editing the CSV."
        )
    return frame


def parse_century(value: object, unknown_values: Iterable[int] = (-1,)) -> Optional[int]:
    """Interpret a century cell, returning ``None`` when it carries no century.

    The parser reads a leading integer and ignores whatever follows. That
    tolerance is deliberate: a cell such as ``"19, A magnificent country
    house"`` -- the signature of a quoting error upstream -- still yields the
    century, while a cell with no leading digits yields ``None`` rather than an
    exception in the middle of an epoch.

    Sentinel values listed in ``unknown_values`` are mapped to ``None``, so
    that "unknown" never becomes a numeric class the model tries to predict.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in _MISSING:
        return None
    match = _LEADING_INT.match(text)
    if match is None:
        return None
    century = int(match.group(1))
    return None if century in set(unknown_values) else century


def resolve_caption_column(frame: pd.DataFrame, preferred: str, fallback: str = "caption") -> str:
    """Return ``preferred`` if the table has it, else ``fallback``.

    Training against the raw captions when the grounded ones were expected is a
    silent and consequential mistake, so the caller is expected to log the
    returned name.
    """
    if preferred in frame.columns:
        return preferred
    if fallback in frame.columns:
        return fallback
    raise KeyError(
        f"neither {preferred!r} nor {fallback!r} is present; columns are {sorted(frame.columns)}"
    )


def read_captions(
    csv_path: Union[str, Path],
    column: str = "caption",
    columns: Optional[ColumnMap] = None,
) -> List[str]:
    """Read a single caption column as a list of strings, dropping blanks."""
    frame = load_table(csv_path, columns)
    name = resolve_caption_column(frame, column)
    series = frame[name].fillna("")
    return [str(v) for v in series if str(v).strip()]
