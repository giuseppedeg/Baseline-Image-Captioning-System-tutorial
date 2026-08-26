"""N-gram metrics, implemented rather than imported.

BLEU and ROUGE-L are short enough to write out, and writing them out makes
their assumptions inspectable instead of hidden behind a package boundary.

A warning that applies to this corpus specifically: there is **one reference
caption per image**. BLEU was designed for multiple references, and CIDEr's
inverse-document-frequency weighting effectively requires them, which is why
CIDEr is not implemented here -- reporting a number computed outside the regime
it was defined for is worse than reporting none. With singleton references,
a correct caption phrased differently from the reference scores badly, and a
caption that copies the reference's function words while asserting something
false scores well. Read these metrics as a coarse fluency and overlap signal,
and read :mod:`captioning.evaluation.factual` alongside them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

__all__ = ["TextMetrics", "compute_text_metrics", "bleu", "rouge_l", "distinct_n"]

_TOKEN_RE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class TextMetrics:
    bleu1: float = 0.0
    bleu2: float = 0.0
    bleu3: float = 0.0
    bleu4: float = 0.0
    rouge_l: float = 0.0
    distinct1: float = 0.0
    distinct2: float = 0.0
    mean_length: float = 0.0
    n_samples: int = 0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(
    predictions: Sequence[str], references: Sequence[str], max_n: int = 4
) -> Tuple[float, ...]:
    """Corpus-level BLEU-1 to BLEU-``max_n``.

    Counts are accumulated over the corpus before the ratio is taken, which is
    what "corpus BLEU" means and what makes it differ from the average of
    per-sentence BLEU. Zero counts are floored at a small constant rather than
    producing a zero score: with short captions a missing 4-gram match is
    common and should not collapse the whole metric.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")

    numerators = [0.0] * max_n
    denominators = [0.0] * max_n
    prediction_length = 0
    reference_length = 0

    for prediction, reference in zip(predictions, references):
        hypothesis = tokenize(prediction)
        target = tokenize(reference)
        prediction_length += len(hypothesis)
        reference_length += len(target)
        for order in range(1, max_n + 1):
            hypothesis_grams = _ngrams(hypothesis, order)
            target_grams = _ngrams(target, order)
            # Clipping: an n-gram counts at most as often as it occurs in the
            # reference, so repeating a correct word does not raise the score.
            overlap = sum(min(count, target_grams[gram]) for gram, count in hypothesis_grams.items())
            numerators[order - 1] += overlap
            denominators[order - 1] += max(0, len(hypothesis) - order + 1)

    if prediction_length == 0:
        return tuple(0.0 for _ in range(max_n))

    brevity = 1.0 if prediction_length > reference_length else pow(
        2.718281828459045, 1.0 - reference_length / max(prediction_length, 1)
    )

    scores = []
    log_sum = 0.0
    for order in range(max_n):
        precision = (numerators[order] or 1e-9) / (denominators[order] or 1e-9)
        log_sum += _log(precision)
        scores.append(brevity * pow(2.718281828459045, log_sum / (order + 1)))
    return tuple(scores)


def _log(value: float) -> float:
    import math

    return math.log(max(value, 1e-12))


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[-1], previous[index + 1]))
        previous = current
    return previous[-1]


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean sentence-level ROUGE-L F1.

    Based on the longest common subsequence, so it rewards content words in the
    right order without requiring them to be contiguous. More forgiving than
    BLEU-4 on singleton references, which is why it is reported next to it.
    """
    total = 0.0
    for prediction, reference in zip(predictions, references):
        hypothesis = tokenize(prediction)
        target = tokenize(reference)
        if not hypothesis or not target:
            continue
        lcs = _lcs_length(hypothesis, target)
        if lcs == 0:
            continue
        precision = lcs / len(hypothesis)
        recall = lcs / len(target)
        total += 2 * precision * recall / (precision + recall)
    return total / len(predictions) if predictions else 0.0


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def distinct_n(predictions: Sequence[str], n: int = 1) -> float:
    """Ratio of unique n-grams to total n-grams across the corpus.

    A model that has collapsed onto one generic caption -- the characteristic
    failure of a captioner trained on too little data -- scores near zero here
    while its BLEU can look unremarkable rather than alarming.
    """
    seen: set = set()
    total = 0
    for prediction in predictions:
        tokens = tokenize(prediction)
        grams = list(_ngrams(tokens, n).elements())
        seen.update(grams)
        total += len(grams)
    return len(seen) / total if total else 0.0


def compute_text_metrics(predictions: Sequence[str], references: Sequence[str]) -> TextMetrics:
    scores = bleu(predictions, references, max_n=4)
    lengths = [len(tokenize(p)) for p in predictions]
    return TextMetrics(
        bleu1=scores[0],
        bleu2=scores[1],
        bleu3=scores[2],
        bleu4=scores[3],
        rouge_l=rouge_l(predictions, references),
        distinct1=distinct_n(predictions, 1),
        distinct2=distinct_n(predictions, 2),
        mean_length=sum(lengths) / len(lengths) if lengths else 0.0,
        n_samples=len(predictions),
    )
