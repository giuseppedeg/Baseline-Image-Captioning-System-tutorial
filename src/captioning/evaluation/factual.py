"""Metrics for what a caption asserts, rather than how it is phrased.

Chapter 1 argued that a captioning model trained on descriptive text learns the
*syntactic frame* of an attribution without being able to learn its content,
and therefore fabricates. N-gram metrics do not see this: a fabricated name is
one wrong token among twenty. These metrics do.

``entity_rate``
    Fraction of generated captions containing at least one named entity of the
    kind :mod:`captioning.data.entities` removes from the targets. After
    grounding, this should be close to zero. A rise is the earliest and
    clearest symptom that something in the pipeline has regressed -- targets
    read from the wrong column, a detector that silently fell back, a
    checkpoint from before the change.

``century_agreement``
    Among captions that state a century, the fraction whose stated century
    matches the metadata. This is the one factual claim the model *is* expected
    to get right, because style carries it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from captioning.data.entities import DEFAULT_MASKED_LABELS, EntityDetector

__all__ = ["FactualMetrics", "compute_factual_metrics", "extract_century"]

_CENTURY_RE = re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)[\s-]*century\b", re.IGNORECASE)


@dataclass
class FactualMetrics:
    #: Fraction of captions containing at least one removable named entity.
    entity_rate: float = 0.0
    #: Mean number of such entities per caption.
    entities_per_caption: float = 0.0
    #: Fraction of captions that state a century at all.
    century_mentioned_rate: float = 0.0
    #: Among those, the fraction matching the metadata.
    century_agreement: Optional[float] = None
    #: Among those, mean absolute error in centuries.
    century_mae: Optional[float] = None
    n_samples: int = 0

    def as_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


def extract_century(text: str) -> Optional[int]:
    """Return the first century stated in ``text``, or ``None``."""
    match = _CENTURY_RE.search(text or "")
    return int(match.group(1)) if match else None


def compute_factual_metrics(
    predictions: Sequence[str],
    detector: Optional[EntityDetector] = None,
    centuries: Optional[Sequence[Optional[int]]] = None,
    labels: Iterable[str] = DEFAULT_MASKED_LABELS,
) -> FactualMetrics:
    """Score generated captions for fabricated content.

    Parameters
    ----------
    predictions:
        Generated captions.
    detector:
        Entity detector. When ``None``, entity statistics are reported as zero
        and only the century metrics are computed.
    centuries:
        Reference century per caption, aligned with ``predictions``. ``None``
        entries are skipped, so a corpus with partial coverage still yields a
        meaningful agreement rate over the rows that have one.
    """
    labels = frozenset(labels) - {"DATE"}  # dates are scored separately below
    n = len(predictions)
    if n == 0:
        return FactualMetrics()

    with_entity = 0
    entity_total = 0
    if detector is not None:
        detect_many = getattr(detector, "detect_many", None)
        batched = detect_many(list(predictions)) if callable(detect_many) else None
        for index, prediction in enumerate(predictions):
            found = batched[index] if batched is not None else detector.detect(prediction or "")
            relevant = [e for e in found if e.label in labels]
            entity_total += len(relevant)
            with_entity += bool(relevant)

    stated: List[int] = []
    matched = 0
    absolute_error = 0
    compared = 0
    for index, prediction in enumerate(predictions):
        century = extract_century(prediction)
        if century is None:
            continue
        stated.append(century)
        if centuries is None:
            continue
        reference = centuries[index] if index < len(centuries) else None
        if reference is None:
            continue
        compared += 1
        matched += int(reference == century)
        absolute_error += abs(reference - century)

    return FactualMetrics(
        entity_rate=with_entity / n,
        entities_per_caption=entity_total / n,
        century_mentioned_rate=len(stated) / n,
        century_agreement=matched / compared if compared else None,
        century_mae=absolute_error / compared if compared else None,
        n_samples=n,
    )
