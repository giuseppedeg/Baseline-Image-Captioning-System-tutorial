"""One table, printed identically by every stage.

The purpose of this module is comparability. Each stage writes a JSON row when
it finishes evaluating, and any stage can load the rows written by the ones
before it and print them together. Without that discipline, three stages
produce three sets of numbers computed at three slightly different moments with
three slightly different decoding settings, and the comparison between them
means nothing.

Every column carries its direction in the header. A metric whose direction the
reader has to remember is a metric the reader will misread.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

__all__ = ["StageResult", "MetricTable", "COLUMNS"]

#: ``(key, header, format, higher_is_better)``. ``None`` means neither.
COLUMNS: Sequence = (
    ("cross_entropy", "CE", "{:.3f}", False),
    ("perplexity", "PPL", "{:.2f}", False),
    ("bleu4", "BLEU-4", "{:.3f}", True),
    ("rouge_l", "ROUGE-L", "{:.3f}", True),
    ("distinct2", "dist-2", "{:.3f}", True),
    ("mean_length", "len", "{:.1f}", None),
    ("clip_score", "CLIP", "{:.3f}", True),
    ("recall_at_1", "R@1", "{:.3f}", True),
    ("entity_rate", "halluc.", "{:.3f}", False),
    ("century_agreement", "century", "{:.3f}", True),
)

_ARROWS = {True: "↑", False: "↓", None: ""}


@dataclass
class StageResult:
    """One row: a named system and the metrics measured for it."""

    name: str
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def value(self, key: str) -> Optional[float]:
        value = self.metrics.get(key)
        return None if value is None else float(value)


class MetricTable:
    """A collection of :class:`StageResult` rows with rendering and persistence."""

    def __init__(self, rows: Optional[Iterable[StageResult]] = None) -> None:
        self.rows: List[StageResult] = list(rows or [])

    # -- construction ------------------------------------------------------

    def add(self, name: str, *metric_groups: Any, **meta: Any) -> "MetricTable":
        """Append a row, merging any number of metric objects or dictionaries.

        Accepts the dataclasses returned by the other modules in this package
        (``TextMetrics``, ``FactualMetrics``, ``GroundingMetrics``,
        ``EpochMetrics``) as well as plain dictionaries. ``None`` groups are
        skipped, so an evaluation that could not compute CLIPScore still
        produces a complete row.
        """
        merged: Dict[str, Optional[float]] = {}
        for group in metric_groups:
            if group is None:
                continue
            if is_dataclass(group) and not isinstance(group, type):
                payload = asdict(group)
            elif hasattr(group, "as_dict"):
                payload = group.as_dict()
            elif isinstance(group, dict):
                payload = group
            else:
                raise TypeError(f"cannot merge metrics of type {type(group).__name__}")
            merged.update({k: v for k, v in payload.items() if _is_number(v)})
        self.rows.append(StageResult(name=name, metrics=merged, meta=dict(meta)))
        return self

    # -- rendering ---------------------------------------------------------

    def render(self, title: str = "Results") -> str:
        if not self.rows:
            return f"{title}\n(no results)"

        present = [c for c in COLUMNS if any(r.value(c[0]) is not None for r in self.rows)]
        name_width = max(len("system"), *(len(r.name) for r in self.rows))
        headers = [f"{h}{_ARROWS[d]}" for _, h, _, d in present]
        widths = [max(len(h), 7) for h in headers]

        lines = [title, "=" * (name_width + sum(w + 2 for w in widths))]
        lines.append(
            "  ".join([f"{'system':<{name_width}}"] + [f"{h:>{w}}" for h, w in zip(headers, widths)])
        )
        lines.append("-" * (name_width + sum(w + 2 for w in widths)))
        for row in self.rows:
            cells = []
            for (key, _, fmt, _), width in zip(present, widths):
                value = row.value(key)
                cells.append(f"{(fmt.format(value) if value is not None else '-'):>{width}}")
            lines.append("  ".join([f"{row.name:<{name_width}}"] + cells))
        lines.append("")
        lines.append("↑ higher is better   ↓ lower is better   - not measured")
        lines.append(
            "halluc. = fraction of captions containing a named entity the model cannot know."
        )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        present = [c for c in COLUMNS if any(r.value(c[0]) is not None for r in self.rows)]
        header = "| system | " + " | ".join(f"{h}{_ARROWS[d]}" for _, h, _, d in present) + " |"
        rule = "|---" * (len(present) + 1) + "|"
        lines = [header, rule]
        for row in self.rows:
            cells = [
                fmt.format(row.value(key)) if row.value(key) is not None else "-"
                for key, _, fmt, _ in present
            ]
            lines.append(f"| {row.name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    # -- persistence -------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"name": r.name, "metrics": r.metrics, "meta": r.meta} for r in self.rows]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MetricTable":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(StageResult(**entry) for entry in payload)

    @classmethod
    def collect(cls, paths: Iterable[Union[str, Path]]) -> "MetricTable":
        """Merge the result files of several stages into one table.

        Missing files are skipped silently: comparing the stages completed so
        far should not require having completed all of them.
        """
        table = cls()
        for path in paths:
            path = Path(path)
            if path.is_file():
                table.rows.extend(cls.load(path).rows)
        return table


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
