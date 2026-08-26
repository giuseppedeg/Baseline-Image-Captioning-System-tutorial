"""Unit tests for the text-processing half of the data package.

Deliberately free of PyTorch: these run in a second and can be executed on any
machine, which is what makes them worth running before every commit.

    pytest tests/
"""

from __future__ import annotations

import pytest

from captioning.data.corpus import parse_century
from captioning.data.entities import RuleBasedEntityDetector, ground_caption
from captioning.data.tokenizer import SPECIAL_TOKENS, WordTokenizer
from captioning.utils.config import Config, ConfigError

CAPTION = (
    "Built in the 17th century by Mughal Emperor Shah Jahan as a mausoleum "
    "for his beloved wife, Mumtaz Mahal."
)


@pytest.fixture()
def detector() -> RuleBasedEntityDetector:
    return RuleBasedEntityDetector()


class TestGrounding:
    def test_personal_names_are_removed(self, detector):
        result = ground_caption(CAPTION, detector)
        assert "Shah Jahan" not in result.text
        assert "Mumtaz Mahal" not in result.text
        assert result.changed

    def test_century_expressions_survive(self, detector):
        assert "17th century" in ground_caption(CAPTION, detector).text

    def test_explicit_years_do_not(self, detector):
        text = "An imposing mausoleum built in 1754 in the late Mughal style."
        assert "1754" not in ground_caption(text, detector).text

    def test_mask_all_removes_centuries_too(self, detector):
        result = ground_caption(CAPTION, detector, date_policy="mask_all")
        assert "17th century" not in result.text

    def test_style_adjectives_are_preserved(self, detector):
        text = "A grand 16th-century French Renaissance castle."
        assert ground_caption(text, detector).text == text

    def test_repair_leaves_no_dangling_punctuation(self, detector):
        text = ground_caption(CAPTION, detector).text
        assert " ," not in text and ",." not in text and " ." not in text
        assert text[0].isupper()

    def test_placeholder_strategy_keeps_a_token(self, detector):
        result = ground_caption(CAPTION, detector, strategy="placeholder")
        assert "<name>" in result.text or "<person>" in result.text

    def test_empty_input_is_tolerated(self, detector):
        assert ground_caption("", detector).text == ""


class TestParseCentury:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("17", 17),
            (16, 16),
            ("19, A magnificent country house", 19),  # upstream quoting error
            ("-1", None),  # configured sentinel
            ("unknown", None),
            ("", None),
            (None, None),
            (float("nan"), None),
        ],
    )
    def test_values(self, value, expected):
        assert parse_century(value) == expected

    def test_sentinels_are_configurable(self):
        assert parse_century("0", unknown_values=(0,)) is None
        assert parse_century("0", unknown_values=(-1,)) == 0


class TestWordTokenizer:
    @pytest.fixture()
    def tokenizer(self) -> WordTokenizer:
        return WordTokenizer.train([CAPTION], vocab_size=100, min_frequency=1)

    def test_special_tokens_come_first(self, tokenizer):
        assert tokenizer.itos[:4] == SPECIAL_TOKENS
        assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id) == (0, 1, 2, 3)

    def test_round_trip(self, tokenizer):
        ids = tokenizer.encode("a mausoleum")
        assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
        assert tokenizer.decode(ids) == "a mausoleum"

    def test_truncation_preserves_the_terminator(self, tokenizer):
        ids = tokenizer.encode(CAPTION, max_length=8)
        assert len(ids) == 8 and ids[-1] == tokenizer.eos_id

    def test_unseen_words_become_unk(self, tokenizer):
        assert tokenizer.oov_rate(["travertine cupola"]) == 1.0
        assert tokenizer.oov_rate(["a mausoleum"]) == 0.0

    def test_hyphenated_periods_stay_whole(self):
        tok = WordTokenizer.train(["A 16th-century castle"], vocab_size=50, min_frequency=1)
        assert "16th-century" in tok.stoi

    def test_persistence(self, tokenizer, tmp_path):
        path = tokenizer.save(tmp_path / "word.json")
        assert WordTokenizer.load(path).itos == tokenizer.itos


class TestConfig:
    def test_unknown_keys_are_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("data:\n  image_root: d\n  train_csv: t.csv\n  image_sizes: 224\n")
        with pytest.raises(ConfigError, match="image_sizes"):
            Config.from_yaml(path)

    def test_unknown_sections_are_preserved(self, tmp_path):
        """Sections no stage consumes yet survive the round trip.

        This is how the configuration file grows across stages without
        breaking the entry points written for earlier ones.
        """
        path = tmp_path / "c.yaml"
        path.write_text("data:\n  image_root: d\n  train_csv: t.csv\ntracking:\n  project: caps\n")
        assert Config.from_yaml(path).section("tracking") == {"project": "caps"}

    def test_requesting_an_absent_section_is_explicit(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("data:\n  image_root: d\n  train_csv: t.csv\n")
        with pytest.raises(ConfigError, match="tracking"):
            Config.from_yaml(path).section("tracking")

    def test_invalid_normalisation_is_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("data:\n  image_root: d\n  train_csv: t.csv\n  normalization: pascal\n")
        with pytest.raises(ConfigError, match="normalization"):
            Config.from_yaml(path)

    def test_column_mapping_is_applied(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "data:\n  image_root: d\n  train_csv: t.csv\n  columns:\n    image_path: filename\n"
        )
        mapping = Config.from_yaml(path).data.columns.canonical_mapping()
        assert mapping["filename"] == "path"
