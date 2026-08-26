"""Tests for the metrics and the shared report.

Metric code is the easiest place in a project for an error to survive: a wrong
number still looks like a number, and it looks like a result. These tests pin
the boundary cases where each metric is known analytically.
"""

from __future__ import annotations

import pytest

from captioning.data.entities import build_detector
from captioning.evaluation.factual import compute_factual_metrics, extract_century
from captioning.evaluation.report import MetricTable
from captioning.evaluation.text_metrics import bleu, compute_text_metrics, distinct_n, rouge_l

REFERENCE = "a grand 16th-century french renaissance castle with a symmetrical facade"


class TestTextMetrics:
    def test_identical_captions_score_one(self):
        metrics = compute_text_metrics([REFERENCE], [REFERENCE])
        assert metrics.bleu4 == pytest.approx(1.0, abs=1e-6)
        assert metrics.rouge_l == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_captions_score_near_zero(self):
        metrics = compute_text_metrics(["completely different words entirely here"], [REFERENCE])
        assert metrics.bleu4 < 1e-3
        assert metrics.rouge_l == 0.0

    def test_brevity_penalty_punishes_truncation(self):
        full = bleu([REFERENCE], [REFERENCE])[0]
        short = bleu(["a grand 16th-century"], [REFERENCE])[0]
        assert short < full

    def test_clipping_stops_repetition_from_paying(self):
        repeated = "castle castle castle castle"
        assert bleu([repeated], ["a castle"])[0] < 0.5

    def test_rouge_l_tolerates_reordering_more_than_bleu(self):
        shuffled = "a symmetrical facade with a grand renaissance castle"
        assert rouge_l([shuffled], [REFERENCE]) > bleu([shuffled], [REFERENCE], max_n=4)[3]

    def test_distinct_n_detects_collapse(self):
        """A model that has collapsed onto one caption scores near zero, while
        its overlap metrics can look unremarkable rather than alarming."""
        collapsed = ["a stone building"] * 20
        varied = [
            "a marble mausoleum with a bulbous dome",
            "a sandstone castle behind formal gardens",
            "an amphitheatre of weathered travertine arches",
            "a timber-framed hall under a slate roof",
            "a brick warehouse with iron window frames",
        ]
        assert distinct_n(collapsed, 2) < 0.2
        assert distinct_n(varied, 2) > 0.9
        assert distinct_n(varied, 1) > distinct_n(collapsed, 1)

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            bleu(["one"], ["one", "two"])


class TestFactualMetrics:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("a 17th century mausoleum", 17),
            ("built in the late 19th-century", 19),
            ("a 16th-century castle", 16),
            ("no period stated here", None),
            ("built in 1754", None),
        ],
    )
    def test_extract_century(self, text, expected):
        assert extract_century(text) == expected

    def test_hallucinated_names_are_counted(self):
        detector = build_detector("rules")
        clean = ["a white marble mausoleum in the 17th century"]
        dirty = ["a white marble mausoleum built by Shah Jahan"]
        assert compute_factual_metrics(clean, detector).entity_rate == 0.0
        assert compute_factual_metrics(dirty, detector).entity_rate == 1.0

    def test_century_agreement(self):
        predictions = ["a 17th century mausoleum", "a 19th century castle", "no period here"]
        metrics = compute_factual_metrics(predictions, None, centuries=[17, 16, 18])
        assert metrics.century_mentioned_rate == pytest.approx(2 / 3)
        assert metrics.century_agreement == pytest.approx(0.5)
        assert metrics.century_mae == pytest.approx(1.5)

    def test_missing_references_are_skipped_not_counted_wrong(self):
        metrics = compute_factual_metrics(["a 17th century mausoleum"], None, centuries=[None])
        assert metrics.century_agreement is None
        assert metrics.century_mae is None

    def test_empty_input(self):
        assert compute_factual_metrics([], None).n_samples == 0


class TestMetricTable:
    def test_merges_groups_and_renders(self):
        table = MetricTable().add(
            "stage1", {"cross_entropy": 2.0, "perplexity": 7.39}, {"bleu4": 0.12, "entity_rate": 0.0}
        )
        rendered = table.render()
        assert "stage1" in rendered and "BLEU-4↑" in rendered and "CE↓" in rendered

    def test_absent_metrics_render_as_dashes(self):
        rendered = MetricTable().add("a", {"bleu4": 0.1}).add("b", {}).render()
        assert " - " in rendered or rendered.rstrip().endswith("-") or "-" in rendered

    def test_none_groups_are_skipped(self):
        table = MetricTable().add("stage1", {"bleu4": 0.1}, None)
        assert table.rows[0].value("bleu4") == pytest.approx(0.1)

    def test_round_trip(self, tmp_path):
        table = MetricTable().add("stage1", {"bleu4": 0.25}, split="val")
        path = table.save(tmp_path / "metrics.json")
        loaded = MetricTable.load(path)
        assert loaded.rows[0].name == "stage1"
        assert loaded.rows[0].value("bleu4") == pytest.approx(0.25)
        assert loaded.rows[0].meta["split"] == "val"

    def test_collect_skips_missing_files(self, tmp_path):
        first = MetricTable().add("stage1", {"bleu4": 0.1}).save(tmp_path / "a.json")
        merged = MetricTable.collect([first, tmp_path / "absent.json"])
        assert [row.name for row in merged.rows] == ["stage1"]

    def test_markdown_has_one_row_per_system(self):
        markdown = MetricTable().add("a", {"bleu4": 0.1}).add("b", {"bleu4": 0.2}).to_markdown()
        assert markdown.count("\n") == 3  # header, rule, two rows
