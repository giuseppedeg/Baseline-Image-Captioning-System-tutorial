"""Turning raw captions into *grounded* captions.

Motivation
----------
Reference captions written by humans mix two kinds of statement. Some are
supported by the image -- material, form, scale, apparent period, structural
type. Others are not: the name of the subject, the person who commissioned it,
the exact year of completion. A model trained on the raw text receives gradient
signal for both, but only the first kind is predictable from pixels. The second
kind is, from the model's point of view, label noise with a strong syntactic
regularity: it learns that a proper noun follows ``built by``, and at inference
time it emits an arbitrary one, fluently and confidently.

This module removes that unlearnable content before training, producing a
*grounded* caption. The raw caption is retained for evaluation and inspection;
see :mod:`captioning.evaluation` and ``docs/01_problem_formulation.md``.

Design
------
Two detectors are provided behind one interface:

``SpacyEntityDetector``
    A transformer-based named-entity recogniser. Accurate, but adds a
    dependency and a model download.
``RuleBasedEntityDetector``
    Capitalisation and date heuristics. Transparent, dependency-free and
    demonstrably imperfect -- which makes it useful pedagogically as well as
    practically.

Dates receive their own policy, because they are the one category that is
partly grounded. An explicit year (``1754``, ``80 AD``) is not recoverable from
an image. A century expression (``17th-century``, ``late 19th century``) is a
coarse statement about style, and style is exactly what the model is being
asked to learn. The default policy therefore removes the former and keeps the
latter.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Entity",
    "EntityDetector",
    "GroundingReport",
    "MaskingResult",
    "RuleBasedEntityDetector",
    "SpacyEntityDetector",
    "build_detector",
    "ground_caption",
    "ground_corpus",
    "DEFAULT_MASKED_LABELS",
    "DATE_POLICIES",
    "STRATEGIES",
]


# ---------------------------------------------------------------------------
# Vocabulary of the module
# ---------------------------------------------------------------------------

#: Entity labels removed by default.
#:
#: ``NORP`` (nationalities and groups: "Mughal", "French") is deliberately
#: *kept*: such adjectives function as style descriptors and are weakly
#: recoverable from visual evidence. ``GPE`` and ``LOC`` are removed, since a
#: place name is not in general inferable from a photograph. Both choices are
#: defensible in the opposite direction and are exposed as configuration.
DEFAULT_MASKED_LABELS: frozenset = frozenset(
    {
        "PERSON",
        "ORG",
        "FAC",
        "GPE",
        "LOC",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "PRODUCT",
        "DATE",
        "PROPN",  # emitted by the rule-based detector
    }
)

DATE_POLICIES = ("keep_centuries", "mask_all", "keep_all")
STRATEGIES = ("remove", "placeholder", "drop_clause")

_PLACEHOLDERS = {
    "PERSON": "<person>",
    "ORG": "<organisation>",
    "GPE": "<place>",
    "LOC": "<place>",
    "FAC": "<name>",
    "WORK_OF_ART": "<name>",
    "PRODUCT": "<name>",
    "EVENT": "<event>",
    "LAW": "<name>",
    "DATE": "<date>",
    "PROPN": "<name>",
}

# A century expression: optionally qualified, always ordinal.
_CENTURY_EXPR = re.compile(
    r"\b(?:early|mid|late|first\s+half\s+of\s+the|second\s+half\s+of\s+the)?[\s-]*"
    r"\d{1,2}(?:st|nd|rd|th)[\s-]century\b",
    re.IGNORECASE,
)
# An explicit year: a bare three-or-four digit number, or any number followed
# by an era marker.
_EXPLICIT_YEAR = re.compile(r"\b\d{1,4}\s*(?:AD|BC|BCE|CE|A\.D\.|B\.C\.)\b|\b\d{3,4}\b", re.IGNORECASE)


@dataclass(frozen=True)
class Entity:
    """A character span identified as a named entity."""

    text: str
    start: int
    end: int
    label: str


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


class EntityDetector(ABC):
    """Interface shared by every entity detector."""

    name: str = "abstract"

    @abstractmethod
    def detect(self, text: str) -> List[Entity]:
        """Return the entities found in ``text``, in order of appearance."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


class SpacyEntityDetector(EntityDetector):
    """Named-entity recognition delegated to a spaCy pipeline."""

    name = "spacy"

    def __init__(self, model: str = "en_core_web_trf", batch_size: int = 32) -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "spaCy is not installed. Either install it with\n"
                "    pip install spacy && python -m spacy download en_core_web_trf\n"
                "or select the rule-based detector with --detector rules."
            ) from exc
        try:
            self._nlp = spacy.load(model, exclude=["lemmatizer", "textcat"])
        except OSError as exc:  # pragma: no cover - environment dependent
            raise OSError(
                f"the spaCy model {model!r} is not available. Download it with\n"
                f"    python -m spacy download {model}"
            ) from exc
        self.model = model
        self.batch_size = batch_size

    def detect(self, text: str) -> List[Entity]:
        doc = self._nlp(text)
        return [
            Entity(text=ent.text, start=ent.start_char, end=ent.end_char, label=ent.label_)
            for ent in doc.ents
        ]

    def detect_many(self, texts: Sequence[str]) -> List[List[Entity]]:
        """Vectorised variant; markedly faster on a full corpus."""
        out: List[List[Entity]] = []
        for doc in self._nlp.pipe(texts, batch_size=self.batch_size):
            out.append(
                [
                    Entity(text=e.text, start=e.start_char, end=e.end_char, label=e.label_)
                    for e in doc.ents
                ]
            )
        return out


#: Capitalised words that describe style, period or culture rather than
#: identity. The rule-based detector must not treat these as proper names.
#: The list is illustrative, not exhaustive -- extending it for a new corpus is
#: part of the exercise, and its incompleteness is precisely the argument for
#: preferring a statistical recogniser.
STYLE_ALLOWLIST: frozenset = frozenset(
    w.lower()
    for w in (
        "Renaissance Gothic Baroque Rococo Romanesque Byzantine Classical Neoclassical "
        "Neo-Renaissance Neo-Gothic Modernist Brutalist Art Deco Palladian Victorian "
        "Georgian Edwardian Tudor Moorish Mudejar Islamic Mughal Ottoman Roman Greek "
        "Hellenistic Medieval Colonial Revival Modern Contemporary "
        "French Italian English Spanish German Indian Persian Chinese Japanese Dutch "
        "Portuguese Russian Egyptian Mesopotamian Celtic Norman Saxon Viking "
        "January February March April May June July August September October November December "
        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday"
    ).split()
)

#: Lowercase words that may appear inside a multi-word proper name.
_NAME_CONNECTORS = frozenset(
    {"of", "the", "de", "du", "des", "del", "della", "di", "da", "van", "von", "el", "al", "bin", "ibn", "and"}
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


class RuleBasedEntityDetector(EntityDetector):
    """A dependency-free approximation of named-entity recognition.

    Two heuristics are applied:

    1. any run of capitalised words that does not begin a sentence and is not
       listed in :data:`STYLE_ALLOWLIST` is treated as a proper name;
    2. explicit years are treated as dates.

    The failure modes are instructive. Sentence-initial names are missed;
    unusual style vocabulary absent from the allowlist is masked spuriously;
    lowercase names are invisible. Compare its output against
    :class:`SpacyEntityDetector` on your own corpus before relying on it.
    """

    name = "rules"

    def __init__(self, allowlist: Optional[Iterable[str]] = None) -> None:
        self.allowlist = frozenset(w.lower() for w in allowlist) if allowlist else STYLE_ALLOWLIST

    def detect(self, text: str) -> List[Entity]:
        # Century expressions are emitted as dates even though the default
        # policy keeps them. Emitting them is what makes the policy meaningful:
        # a category that is never detected cannot be governed by a setting,
        # and the two detectors would then behave differently under
        # --date-policy mask_all.
        entities = [
            Entity(text=m.group(), start=m.start(), end=m.end(), label="DATE")
            for pattern in (_CENTURY_EXPR, _EXPLICIT_YEAR)
            for m in pattern.finditer(text)
        ]
        entities.extend(self._proper_names(text))
        return _deduplicate(entities)

    # -- internals ---------------------------------------------------------

    def _proper_names(self, text: str) -> List[Entity]:
        tokens = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
        found: List[Entity] = []
        i = 0
        while i < len(tokens):
            start, end, word = tokens[i]
            if not word[:1].isupper() or _is_sentence_start(text, start):
                i += 1
                continue
            # An allowlisted word may still open a name when a genuine proper
            # noun follows it: "Roman" alone is a style, "Roman Empire" is not.
            if word.lower() in self.allowlist and not self._opens_name(text, tokens, i):
                i += 1
                continue

            run_start, run_end = start, end
            j = i + 1
            while j < len(tokens):
                nxt_start, nxt_end, nxt_word = tokens[j]
                # Stop at anything other than plain whitespace between tokens,
                # so that a comma terminates the run.
                if text[run_end:nxt_start].strip():
                    break
                lowered = nxt_word.lower()
                if lowered in _NAME_CONNECTORS:
                    # A connector extends the run only if a capitalised word
                    # follows it: "Duke of Bedford", but not "Bedford and".
                    if j + 1 < len(tokens) and tokens[j + 1][2][:1].isupper():
                        run_end = tokens[j + 1][1]
                        j += 2
                        continue
                    break
                if nxt_word[:1].isupper() and lowered not in self.allowlist:
                    run_end = nxt_end
                    j += 1
                    continue
                break

            found.append(Entity(text[run_start:run_end], run_start, run_end, "PROPN"))
            i = max(j, i + 1)
        return found

    def _opens_name(self, text: str, tokens, i: int) -> bool:
        """True when the allowlisted token at ``i`` is immediately followed by
        a capitalised word that is itself not allowlisted."""
        if i + 1 >= len(tokens):
            return False
        _, prev_end, _ = tokens[i]
        nxt_start, _, nxt_word = tokens[i + 1]
        if text[prev_end:nxt_start].strip():
            return False
        lowered = nxt_word.lower()
        return nxt_word[:1].isupper() and lowered not in self.allowlist and lowered not in _NAME_CONNECTORS


def _is_sentence_start(text: str, index: int) -> bool:
    """True when position ``index`` opens the text or follows a full stop."""
    k = index - 1
    while k >= 0 and (text[k].isspace() or text[k] in "\"'([“‘"):
        k -= 1
    return k < 0 or text[k] in ".!?"


def build_detector(kind: str = "auto", spacy_model: str = "en_core_web_trf") -> EntityDetector:
    """Construct a detector.

    ``kind='auto'`` prefers spaCy and falls back to the rule-based detector if
    spaCy or its model is unavailable, so that the tutorial runs end to end on
    a machine without the optional dependency.
    """
    if kind == "spacy":
        return SpacyEntityDetector(spacy_model)
    if kind == "rules":
        return RuleBasedEntityDetector()
    if kind != "auto":
        raise ValueError(f"unknown detector kind {kind!r}; expected 'auto', 'spacy' or 'rules'")
    try:
        return SpacyEntityDetector(spacy_model)
    except (ImportError, OSError):
        return RuleBasedEntityDetector()


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskingResult:
    """The grounded caption together with what was taken out of it."""

    text: str
    masked: Tuple[Entity, ...] = ()
    strategy: str = "remove"

    @property
    def changed(self) -> bool:
        return bool(self.masked)


def _deduplicate(entities: Sequence[Entity]) -> List[Entity]:
    """Sort by position and drop spans contained in, or crossing, an earlier one."""
    ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    kept: List[Entity] = []
    for ent in ordered:
        if kept and ent.start < kept[-1].end:
            continue
        kept.append(ent)
    return kept


def _keep_date(entity: Entity, policy: str) -> bool:
    """Decide whether a ``DATE`` entity survives.

    Under ``keep_centuries`` a date is retained when it is a century
    expression and contains no explicit year: ``late 19th century`` is kept,
    ``1754`` and ``80 AD`` are not.
    """
    if policy == "keep_all":
        return True
    if policy == "mask_all":
        return False
    if policy != "keep_centuries":
        raise ValueError(f"unknown date policy {policy!r}; expected one of {DATE_POLICIES}")
    text = entity.text
    return bool(_CENTURY_EXPR.search(text)) and not _EXPLICIT_YEAR.search(text)


def ground_caption(
    text: str,
    detector: EntityDetector,
    *,
    entities: Optional[Sequence[Entity]] = None,
    labels: Iterable[str] = DEFAULT_MASKED_LABELS,
    date_policy: str = "keep_centuries",
    strategy: str = "remove",
) -> MaskingResult:
    """Remove visually ungrounded content from a single caption.

    Parameters
    ----------
    text:
        The raw caption.
    detector:
        Used when ``entities`` is not supplied.
    entities:
        Pre-computed entities, to avoid re-running the detector when a corpus
        has already been processed in batch.
    labels:
        Entity labels eligible for masking.
    date_policy:
        One of :data:`DATE_POLICIES`.
    strategy:
        ``remove`` deletes the span and repairs the surrounding text;
        ``placeholder`` substitutes a typed token such as ``<person>``, keeping
        the sentence grammatical at the cost of an artificial vocabulary;
        ``drop_clause`` deletes the whole comma-delimited clause, which yields
        the cleanest prose but discards grounded content along with the rest.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if not isinstance(text, str) or not text.strip():
        return MaskingResult(text="", masked=(), strategy=strategy)

    labels = frozenset(labels)
    detected = list(entities) if entities is not None else detector.detect(text)
    selected = [
        e
        for e in _deduplicate(detected)
        if e.label in labels and (e.label != "DATE" or not _keep_date(e, date_policy))
    ]
    if not selected:
        return MaskingResult(text=_repair(text), masked=(), strategy=strategy)

    if strategy == "drop_clause":
        grounded = _repair(_drop_clauses(text, selected))
        # A caption can consist of a single clause built around a name, in
        # which case clause deletion removes almost everything. Fall back
        # rather than emit a degenerate target.
        if len(grounded.split()) < 0.3 * len(text.split()):
            grounded = _repair(_splice(text, selected, mode="remove"))
    else:
        grounded = _repair(_splice(text, selected, mode=strategy))

    return MaskingResult(text=grounded, masked=tuple(selected), strategy=strategy)


def _splice(text: str, entities: Sequence[Entity], mode: str) -> str:
    """Rewrite ``text``, replacing each entity span. Applied right to left so
    that character offsets computed on the original string remain valid."""
    out = text
    for ent in sorted(entities, key=lambda e: e.start, reverse=True):
        replacement = "" if mode == "remove" else _PLACEHOLDERS.get(ent.label, "<name>")
        out = out[: ent.start] + replacement + out[ent.end :]
    return out


def _drop_clauses(text: str, entities: Sequence[Entity]) -> str:
    bounds = [0] + [m.end() for m in re.finditer(r"[,;:]", text)] + [len(text)]
    kept = []
    for start, end in zip(bounds, bounds[1:]):
        overlaps = any(not (e.end <= start or e.start >= end) for e in entities)
        if not overlaps:
            kept.append(text[start:end])
    return "".join(kept)


# Function words that become dangling once the phrase they governed is gone.
_DANGLING = r"(?:by|of|for|in|on|at|to|with|from|near|into|and|or|the|a|an|as|that|which)"
_DUPLICABLE = r"(?:in|of|by|for|the|a|an|to|at|on|with|and)"
# Prepositions only: determiners are excluded so that "of the" is left alone.
_PREPOSITION = r"(?:by|of|for|from|with|in|on|at|to|into|near|as)"


def _repair(text: str) -> str:
    """Restore well-formed prose after spans have been excised.

    Deleting a span leaves three kinds of damage: doubled or orphaned
    punctuation, repeated function words (``built in in the late style``), and
    prepositions that no longer govern anything (``for ,``). Each is repaired
    with a targeted substitution. The result is not guaranteed grammatical --
    no regular expression could guarantee that -- but it is close enough that
    the remaining noise is small relative to the noise it removes.
    """
    s = text.replace(" ", " ")
    s = re.sub(r"\(\s*\)|\[\s*\]|\"\s*\"|'\s*'", " ", s)
    s = re.sub(r"\s+", " ", s)
    # Repeated function words left adjacent by a deletion between them.
    s = re.sub(rf"\b({_DUPLICABLE})\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    # A preposition, optionally with its determiner, immediately followed by
    # another preposition: the phrase it governed was deleted from between
    # them ("built in the as a mausoleum" -> "built as a mausoleum").
    s = re.sub(
        rf"\b{_PREPOSITION}\s+(?:the|a|an)?\s*(?={_PREPOSITION}\s)", "", s, flags=re.IGNORECASE
    )
    # Prepositions and determiners with nothing left to govern.
    s = re.sub(rf"\s*\b{_DANGLING}(?:\s+{_DANGLING})*\s*(?=[,.;:]|$)", "", s, flags=re.IGNORECASE)
    # Punctuation hygiene.
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)
    s = re.sub(r"[,;:]\s*\.", ".", s)
    s = re.sub(r"\.(?:\s*\.)+", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[,;:.\s]+", "", s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


# ---------------------------------------------------------------------------
# Corpus level
# ---------------------------------------------------------------------------


@dataclass
class GroundingReport:
    """Aggregate diagnostics for a grounding pass over a corpus."""

    n_captions: int = 0
    n_changed: int = 0
    n_entities: int = 0
    by_label: Counter = field(default_factory=Counter)
    surfaces: Counter = field(default_factory=Counter)
    examples: List[Tuple[str, str]] = field(default_factory=list)
    detector: str = ""
    strategy: str = ""

    @property
    def changed_fraction(self) -> float:
        return self.n_changed / self.n_captions if self.n_captions else 0.0

    def render(self, n_examples: int = 5, n_surfaces: int = 15) -> str:
        lines = [
            "Caption grounding report",
            "=" * 64,
            f"detector                 : {self.detector}",
            f"strategy                 : {self.strategy}",
            f"captions processed       : {self.n_captions}",
            f"captions modified        : {self.n_changed} ({self.changed_fraction:.1%})",
            f"entity mentions removed  : {self.n_entities}",
            "",
            "Removed mentions by label",
            "-" * 64,
        ]
        for label, count in self.by_label.most_common():
            lines.append(f"  {label:<14} {count:>6}")
        if self.surfaces:
            lines += ["", f"Most frequently removed surface forms (top {n_surfaces})", "-" * 64]
            for surface, count in self.surfaces.most_common(n_surfaces):
                lines.append(f"  {count:>5}  {surface}")
        if self.examples:
            lines += ["", f"Examples (first {min(n_examples, len(self.examples))})", "-" * 64]
            for raw, grounded in self.examples[:n_examples]:
                lines += [f"  raw      : {raw}", f"  grounded : {grounded}", ""]
        return "\n".join(lines)


def ground_corpus(
    texts: Sequence[str],
    detector: EntityDetector,
    *,
    labels: Iterable[str] = DEFAULT_MASKED_LABELS,
    date_policy: str = "keep_centuries",
    strategy: str = "remove",
    collect_examples: int = 20,
) -> Tuple[List[str], GroundingReport]:
    """Ground every caption in ``texts`` and summarise what was removed.

    Inspecting the report before training is not optional. It is the only
    cheap opportunity to discover that the detector is deleting the content
    one intended to keep.
    """
    labels = frozenset(labels)
    report = GroundingReport(detector=detector.name, strategy=strategy)

    detect_many = getattr(detector, "detect_many", None)
    per_caption = detect_many(list(texts)) if callable(detect_many) else None

    grounded_texts: List[str] = []
    for index, raw in enumerate(texts):
        precomputed = per_caption[index] if per_caption is not None else None
        result = ground_caption(
            raw if isinstance(raw, str) else "",
            detector,
            entities=precomputed,
            labels=labels,
            date_policy=date_policy,
            strategy=strategy,
        )
        grounded_texts.append(result.text)

        report.n_captions += 1
        if result.changed:
            report.n_changed += 1
            report.n_entities += len(result.masked)
            for ent in result.masked:
                report.by_label[ent.label] += 1
                report.surfaces[ent.text] += 1
            if len(report.examples) < collect_examples:
                report.examples.append((str(raw), result.text))
    return grounded_texts, report
