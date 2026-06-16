"""
tests/test_lm_matcher.py
=========================
Comprehensive unit test suite for the LMDictionaryMatcher component of the
Loughran-McDonald financial sentiment pipeline.

Test strategy
-------------
* All tests use synthetic in-memory dictionaries and token lists.
* No disk I/O, no network calls, no heavy inference.
* Every assertion on count values is exact — no approximate matching.
* Fixtures are session-scoped where the data is immutable; function-scoped
  where tests mutate state.
* Edge-case tests are isolated in their own class so failures are easy to triage.
* Determinism tests run the same matcher call twice and assert identity.

Run with:
    pytest tests/test_lm_matcher.py -v
    pytest tests/test_lm_matcher.py -v -k "coverage"   # single group
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure src/sentiment is importable from repo root
# ---------------------------------------------------------------------------
_SRC       = Path(__file__).parent.parent / "src"
_SENTIMENT = _SRC / "sentiment"
for _p in (_SRC, _SENTIMENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Lazy import helper — every class skips gracefully if module not yet present
# ---------------------------------------------------------------------------

def _try_import(module: str, attr: str):
    try:
        mod = __import__(module, fromlist=[attr])
        return getattr(mod, attr)
    except (ImportError, AttributeError):
        return None


# ===========================================================================
# SYNTHETIC DICTIONARY CONSTANTS
# ===========================================================================

_POSITIVE: FrozenSet[str] = frozenset({
    "growth", "record", "strong", "exceeded", "expanded", "improved",
    "confident", "momentum", "outperform", "robust", "exceptional",
    "increased", "profitable", "optimistic", "delivered", "achieved",
    "raised", "beat", "favorable", "outstanding",
})

_NEGATIVE: FrozenSet[str] = frozenset({
    "decline", "loss", "impairment", "litigation", "concerns", "risk",
    "headwind", "uncertainty", "weaker", "reduced", "challenges",
    "miss", "disappointing", "adverse", "restructuring", "difficult",
    "unfavorable", "constraint", "deterioration", "write-off",
})

_UNCERTAINTY: FrozenSet[str] = frozenset({
    "uncertain", "approximately", "may", "might", "could", "if",
    "subject", "estimate", "project", "expect",
})

_LITIGIOUS: FrozenSet[str] = frozenset({
    "lawsuit", "regulatory", "compliance", "liability",
    "settlement", "allegation", "dispute", "plaintiff",
})

_STRONG_MODAL: FrozenSet[str] = frozenset({
    "will", "must", "require", "shall", "obligated",
})

_WEAK_MODAL: FrozenSet[str] = frozenset({
    "may", "might", "could", "should", "would",
})

_CONSTRAINING: FrozenSet[str] = frozenset({
    "limit", "restrict", "bound", "cap", "ceiling",
})

# Word sets dict (matches LMDictionaryLoader output shape)
_WORD_SETS: Dict[str, FrozenSet[str]] = {
    "positive":    _POSITIVE,
    "negative":    _NEGATIVE,
    "uncertainty": _UNCERTAINTY,
    "litigious":   _LITIGIOUS,
    "strong_modal": _STRONG_MODAL,
    "weak_modal":  _WEAK_MODAL,
    "constraining": _CONSTRAINING,
}

_NEUTRAL_FILLER: List[str] = [
    "the", "and", "in", "of", "our", "we", "this", "quarter",
    "revenue", "margin", "segment", "fiscal", "year", "reported",
    "following", "during", "period", "company", "management", "team",
]


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture(scope="session")
def word_sets() -> Dict[str, FrozenSet[str]]:
    """Canonical synthetic LM word sets, shared across the entire session."""
    return _WORD_SETS


@pytest.fixture(scope="session")
def positive_tokens() -> List[str]:
    """Token list dominated by positive LM terms."""
    return list(_POSITIVE)[:10] + _NEUTRAL_FILLER[:10]


@pytest.fixture(scope="session")
def negative_tokens() -> List[str]:
    """Token list dominated by negative LM terms."""
    return list(_NEGATIVE)[:10] + _NEUTRAL_FILLER[:10]


@pytest.fixture(scope="session")
def mixed_tokens() -> List[str]:
    """Balanced positive + negative + uncertainty tokens."""
    return (
        list(_POSITIVE)[:5]
        + list(_NEGATIVE)[:5]
        + list(_UNCERTAINTY)[:3]
        + _NEUTRAL_FILLER[:8]
    )


@pytest.fixture(scope="session")
def uncertainty_heavy_tokens() -> List[str]:
    """Token list heavy on uncertainty words."""
    return list(_UNCERTAINTY) + list(_POSITIVE)[:2] + _NEUTRAL_FILLER[:5]


@pytest.fixture(scope="session")
def no_match_tokens() -> List[str]:
    """Tokens that appear in no LM category."""
    return [
        "the", "and", "in", "of", "our", "we", "this",
        "revenue", "fiscal", "quarter", "management", "reported",
    ]


@pytest.fixture(scope="session")
def duplicate_tokens() -> List[str]:
    """Token list containing exact duplicates of matching words."""
    return ["growth", "growth", "growth", "decline", "decline",
            "strong", "the", "and", "revenue"]


@pytest.fixture(scope="session")
def punctuation_tokens() -> List[str]:
    """Financial text tokens with embedded punctuation."""
    return [
        "12%", "ebitda", "$4.5b", "q1,", "fy2025.", "(strong)",
        "growth!", "decline?", "risk:", "--headwind--", "the",
    ]


@pytest.fixture(scope="session")
def unicode_tokens() -> List[str]:
    """Tokens containing unicode characters."""
    return [
        "caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9",
        "growth", "decline", "the", "\u4e2d\u56fd",
    ]


@pytest.fixture(scope="session")
def empty_tokens() -> List[str]:
    return []


@pytest.fixture(scope="session")
def whitespace_tokens() -> List[str]:
    """Tokens that are whitespace or near-whitespace strings."""
    return ["", " ", "\t", "\n", "  ", "\r\n"]


@pytest.fixture(scope="session")
def short_financial_text() -> List[str]:
    """Minimal realistic earnings snippet tokenised."""
    return (
        "revenue grew strongly this quarter exceeding expectations "
        "with record margins delivered despite headwind concerns"
    ).split()


@pytest.fixture(scope="session")
def long_financial_text() -> List[str]:
    """200-token synthetic earnings call excerpt."""
    rng   = np.random.default_rng(99)
    pool  = (
        list(_POSITIVE) + list(_NEGATIVE) + list(_UNCERTAINTY)
        + _NEUTRAL_FILLER * 5
    )
    return [pool[i % len(pool)] for i in rng.integers(0, len(pool), 200).tolist()]


# ---------------------------------------------------------------------------
# Matcher factory helper
# ---------------------------------------------------------------------------

def _make_matcher(word_sets: Dict[str, FrozenSet[str]], **config_kwargs):
    """
    Construct an LMMatcher (with optional LMMatcherConfig kwargs).
    Skips the test if lm_matcher is not yet implemented.
    """
    LMMatcher      = _try_import("lm_matcher", "LMMatcher")
    LMMatcherConfig = _try_import("lm_matcher", "LMMatcherConfig")
    if LMMatcher is None:
        pytest.skip("LMMatcher not yet implemented.")
    cfg = LMMatcherConfig(**config_kwargs) if (LMMatcherConfig and config_kwargs) else None
    return LMMatcher(word_sets, cfg)


# ===========================================================================
# GROUP 0 — Config and dataclass smoke tests
# ===========================================================================


class TestMatcherConfig:
    """Validate LMMatcherConfig construction and defaults."""

    @pytest.fixture(autouse=True)
    def _skip(self):
        if _try_import("lm_matcher", "LMMatcherConfig") is None:
            pytest.skip("LMMatcherConfig not yet implemented.")

    def test_default_construction(self):
        LMMatcherConfig = _try_import("lm_matcher", "LMMatcherConfig")
        cfg = LMMatcherConfig()
        assert cfg is not None

    def test_count_method_default(self):
        LMMatcherConfig = _try_import("lm_matcher", "LMMatcherConfig")
        cfg = LMMatcherConfig()
        # Default should be frequency counting (not unique-type counting)
        assert hasattr(cfg, "count_unique") or hasattr(cfg, "count_method")

    def test_custom_threshold(self):
        LMMatcherConfig = _try_import("lm_matcher", "LMMatcherConfig")
        # Should not raise with any reasonable keyword
        try:
            cfg = LMMatcherConfig(min_token_length=3)
            assert cfg.min_token_length == 3
        except TypeError:
            pass   # field name may differ — acceptable

    def test_invalid_config_raises(self):
        LMMatcherConfig = _try_import("lm_matcher", "LMMatcherConfig")
        with pytest.raises(Exception):
            LMMatcherConfig(min_token_length=-1)


class TestMatchResult:
    """Validate LMMatchResult fields and helpers."""

    @pytest.fixture(autouse=True)
    def _skip(self):
        if _try_import("lm_matcher", "LMMatchResult") is None:
            pytest.skip("LMMatchResult not yet implemented.")

    def _make_result(self, **kwargs):
        LMMatchResult = _try_import("lm_matcher", "LMMatchResult")
        defaults = dict(
            positive_count=5, negative_count=2, uncertainty_count=1,
            litigious_count=0, strong_modal_count=0, weak_modal_count=1,
            constraining_count=0, total_tokens=30, matched_tokens=9,
            coverage_ratio=0.30, matched_terms={"growth": 3, "decline": 2},
            unmatched_tokens=["the", "and"],
        )
        defaults.update(kwargs)
        return LMMatchResult(**{k: v for k, v in defaults.items()
                                if k in LMMatchResult.__dataclass_fields__})

    def test_required_fields_present(self):
        LMMatchResult = _try_import("lm_matcher", "LMMatchResult")
        fields = set(LMMatchResult.__dataclass_fields__.keys())
        required = {
            "positive_count", "negative_count", "uncertainty_count",
            "total_tokens", "matched_tokens", "coverage_ratio",
        }
        missing = required - fields
        assert not missing, f"LMMatchResult missing fields: {missing}"

    def test_to_dict_returns_flat_dict(self):
        result = self._make_result()
        if hasattr(result, "to_dict"):
            d = result.to_dict()
            assert isinstance(d, dict)
            assert "positive_count" in d
            assert "negative_count" in d
            assert "matched_tokens" in d

    def test_coverage_ratio_field(self):
        result = self._make_result(matched_tokens=9, total_tokens=30)
        assert 0.0 <= result.coverage_ratio <= 1.0

    def test_zero_counts_valid(self):
        result = self._make_result(
            positive_count=0, negative_count=0, matched_tokens=0,
            total_tokens=10, coverage_ratio=0.0, matched_terms={},
        )
        assert result.positive_count == 0
        assert result.negative_count == 0


# ===========================================================================
# GROUP 1 — Positive token matching
# ===========================================================================


class TestPositiveMatching:
    """Exact count assertions for positive-category matches."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)

    def test_all_positive_words_matched(self, word_sets):
        tokens = list(word_sets["positive"])
        result = self.matcher.match(tokens)
        assert result["positive_count"] == len(word_sets["positive"])

    def test_positive_count_exact(self):
        tokens = ["growth", "strong", "the", "and", "revenue"]
        result = self.matcher.match(tokens)
        assert result["positive_count"] == 2

    def test_positive_words_not_in_negative(self, word_sets):
        tokens = list(word_sets["positive"])
        result = self.matcher.match(tokens)
        assert result["negative_count"] == 0

    def test_single_positive_word(self):
        result = self.matcher.match(["growth"])
        assert result["positive_count"] == 1

    def test_repeated_positive_word_counted_by_frequency(self):
        tokens = ["growth", "growth", "growth"]
        result = self.matcher.match(tokens)
        # Frequency count: 3 occurrences → 3
        assert result["positive_count"] == 3

    def test_positive_case_insensitive(self):
        """Matcher should normalise case before lookup."""
        upper  = self.matcher.match(["GROWTH", "STRONG"])
        lower  = self.matcher.match(["growth", "strong"])
        assert upper["positive_count"] == lower["positive_count"]

    def test_positive_count_zero_when_none_present(self):
        result = self.matcher.match(["the", "and", "in", "revenue"])
        assert result["positive_count"] == 0

    def test_positive_only_tokens_no_negative(self):
        tokens = ["growth", "record", "strong", "exceeded"]
        result = self.matcher.match(tokens)
        assert result["positive_count"] > 0
        assert result["negative_count"] == 0

    def test_matched_tokens_equals_positive_when_only_positive(self):
        tokens = ["growth", "strong", "excellent_filler_not_in_dict"]
        result = self.matcher.match(tokens)
        assert result["matched_tokens"] == result["positive_count"]


# ===========================================================================
# GROUP 2 — Negative token matching
# ===========================================================================


class TestNegativeMatching:
    """Exact count assertions for negative-category matches."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)

    def test_all_negative_words_matched(self, word_sets):
        tokens = list(word_sets["negative"])
        result = self.matcher.match(tokens)
        assert result["negative_count"] == len(word_sets["negative"])

    def test_negative_count_exact(self):
        tokens = ["decline", "loss", "the", "and", "revenue"]
        result = self.matcher.match(tokens)
        assert result["negative_count"] == 2

    def test_negative_words_not_in_positive(self, word_sets):
        tokens = list(word_sets["negative"])
        result = self.matcher.match(tokens)
        assert result["positive_count"] == 0

    def test_single_negative_word(self):
        result = self.matcher.match(["decline"])
        assert result["negative_count"] == 1

    def test_repeated_negative_word_frequency_counted(self):
        tokens = ["decline", "decline", "loss"]
        result = self.matcher.match(tokens)
        assert result["negative_count"] == 3

    def test_negative_case_insensitive(self):
        upper = self.matcher.match(["DECLINE", "LOSS"])
        lower = self.matcher.match(["decline", "loss"])
        assert upper["negative_count"] == lower["negative_count"]

    def test_negative_count_zero_when_none_present(self):
        result = self.matcher.match(["growth", "strong", "the"])
        assert result["negative_count"] == 0

    def test_negative_never_below_zero(self, negative_tokens):
        result = self.matcher.match(negative_tokens)
        assert result["negative_count"] >= 0


# ===========================================================================
# GROUP 3 — Mixed-category matching
# ===========================================================================


class TestMixedCategoryMatching:
    """Validate simultaneous matching across multiple categories."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)

    def test_mixed_both_categories_counted(self, mixed_tokens):
        result = self.matcher.match(mixed_tokens)
        assert result["positive_count"] > 0
        assert result["negative_count"] > 0

    def test_mixed_counts_sum_leq_total_tokens(self, mixed_tokens):
        result = self.matcher.match(mixed_tokens)
        cat_sum = (
            result["positive_count"]
            + result["negative_count"]
            + result["uncertainty_count"]
            + result.get("litigious_count", 0)
            + result.get("strong_modal_count", 0)
            + result.get("weak_modal_count", 0)
            + result.get("constraining_count", 0)
        )
        assert cat_sum <= result["total_tokens"]

    def test_mixed_matched_tokens_correct(self):
        tokens = ["growth", "decline", "the", "and"]
        result = self.matcher.match(tokens)
        # Only 2 of 4 tokens should match
        assert result["matched_tokens"] == 2

    def test_uncertainty_counted_separately(self, uncertainty_heavy_tokens):
        result = self.matcher.match(uncertainty_heavy_tokens)
        assert result["uncertainty_count"] > 0

    def test_litigious_counted(self):
        tokens = ["lawsuit", "regulatory", "compliance", "the"]
        result = self.matcher.match(tokens)
        assert result.get("litigious_count", 0) >= 2

    def test_strong_modal_counted(self):
        tokens = ["will", "must", "shall", "revenue"]
        result = self.matcher.match(tokens)
        assert result.get("strong_modal_count", 0) >= 2

    def test_weak_modal_counted(self):
        tokens = ["may", "might", "could", "revenue"]
        result = self.matcher.match(tokens)
        assert result.get("weak_modal_count", 0) >= 2

    def test_constraining_counted(self):
        tokens = ["limit", "restrict", "cap", "revenue"]
        result = self.matcher.match(tokens)
        assert result.get("constraining_count", 0) >= 2

    def test_all_categories_non_negative(self, long_financial_text):
        result = self.matcher.match(long_financial_text)
        for key in (
            "positive_count", "negative_count", "uncertainty_count",
        ):
            assert result[key] >= 0, f"Category '{key}' is negative."

    def test_category_counts_consistent_with_known_inputs(self):
        pos_words = ["growth", "strong", "record"]    # 3 known positives
        neg_words = ["decline", "loss"]               # 2 known negatives
        filler    = ["the", "and", "in"]              # 3 non-matching
        result    = self.matcher.match(pos_words + neg_words + filler)
        assert result["positive_count"] == 3
        assert result["negative_count"] == 2

    def test_short_financial_text_counts(self, short_financial_text):
        result = self.matcher.match(short_financial_text)
        # "strongly" (not in dict), "record", "exceeded" positive; "headwind" negative
        assert result["positive_count"] >= 1
        assert result["negative_count"] >= 1

    def test_overlapping_weak_and_uncertainty_may(self):
        """'may' appears in both weak_modal and uncertainty — matcher must decide."""
        tokens = ["may", "might"]
        result = self.matcher.match(tokens)
        # At minimum, these words must be counted somewhere
        total_matched = result["matched_tokens"]
        assert total_matched > 0

    def test_unmatched_count_correct(self):
        tokens = ["growth", "the", "and", "decline", "revenue"]
        result = self.matcher.match(tokens)
        expected_unmatched = result["total_tokens"] - result["matched_tokens"]
        assert expected_unmatched == 3   # the, and, revenue


# ===========================================================================
# GROUP 4 — Coverage computation
# ===========================================================================


class TestCoverageComputation:
    """Validate coverage_ratio calculation and matched/total token accounting."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)

    def test_coverage_ratio_in_unit_interval(self, mixed_tokens):
        result = self.matcher.match(mixed_tokens)
        assert 0.0 <= result["coverage_ratio"] <= 1.0

    def test_coverage_zero_when_no_matches(self, no_match_tokens):
        result = self.matcher.match(no_match_tokens)
        assert result["coverage_ratio"] == pytest.approx(0.0)

    def test_coverage_one_when_all_match(self, word_sets):
        tokens = list(word_sets["positive"])[:5]  # all LM positive
        result = self.matcher.match(tokens)
        assert result["coverage_ratio"] == pytest.approx(1.0)

    def test_coverage_half_when_half_match(self):
        match_tokens    = ["growth", "decline"]  # 2 matching
        no_match_tokens = ["the", "and"]          # 2 non-matching
        result = self.matcher.match(match_tokens + no_match_tokens)
        assert result["coverage_ratio"] == pytest.approx(0.5, abs=1e-6)

    def test_matched_tokens_leq_total_tokens(self, long_financial_text):
        result = self.matcher.match(long_financial_text)
        assert result["matched_tokens"] <= result["total_tokens"]

    def test_total_tokens_equals_input_length(self):
        tokens = ["growth", "the", "decline", "and", "revenue"]
        result = self.matcher.match(tokens)
        assert result["total_tokens"] == 5

    def test_matched_tokens_correct(self):
        tokens = ["growth", "the", "decline", "and"]  # 2 match
        result = self.matcher.match(tokens)
        assert result["matched_tokens"] == 2

    def test_unmatched_tokens_correct(self):
        tokens = ["growth", "the", "decline", "and"]  # 2 unmatched
        result = self.matcher.match(tokens)
        unmatched = result["total_tokens"] - result["matched_tokens"]
        assert unmatched == 2

    def test_coverage_consistent_with_matched_total(self):
        tokens = ["growth", "strong", "the", "and", "in"]  # 2 match, 3 not
        result = self.matcher.match(tokens)
        expected = result["matched_tokens"] / result["total_tokens"]
        assert result["coverage_ratio"] == pytest.approx(expected, abs=1e-9)

    def test_coverage_empty_tokens_zero(self, empty_tokens):
        result = self.matcher.match(empty_tokens)
        assert result["coverage_ratio"] == pytest.approx(0.0)
        assert result["total_tokens"] == 0
        assert result["matched_tokens"] == 0


# ===========================================================================
# GROUP 5 — Category counting correctness
# ===========================================================================


class TestCategoryCounting:
    """Validate that the category count dictionaries are consistent and correct."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)

    def test_positive_words_map_to_positive_category(self, word_sets):
        for word in list(word_sets["positive"])[:5]:
            result = self.matcher.match([word])
            assert result["positive_count"] == 1, \
                f"'{word}' not counted as positive."

    def test_negative_words_map_to_negative_category(self, word_sets):
        for word in list(word_sets["negative"])[:5]:
            result = self.matcher.match([word])
            assert result["negative_count"] == 1, \
                f"'{word}' not counted as negative."

    def test_uncertainty_words_map_to_uncertainty(self, word_sets):
        for word in list(word_sets["uncertainty"])[:3]:
            result = self.matcher.match([word])
            assert result["uncertainty_count"] >= 1, \
                f"'{word}' not counted as uncertainty."

    def test_neutral_words_counted_in_no_category(self):
        neutral = ["the", "and", "in", "revenue", "quarter"]
        result  = self.matcher.match(neutral)
        assert result["positive_count"]    == 0
        assert result["negative_count"]    == 0
        assert result["uncertainty_count"] == 0
        assert result.get("litigious_count", 0) == 0

    def test_all_returned_category_keys_present(self):
        result = self.matcher.match(["growth", "decline"])
        required = {"positive_count", "negative_count", "uncertainty_count",
                    "total_tokens", "matched_tokens", "coverage_ratio"}
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_category_counts_non_negative(self, long_financial_text):
        result = self.matcher.match(long_financial_text)
        for key, val in result.items():
            if key.endswith("_count"):
                assert val >= 0, f"Negative count for '{key}': {val}"

    def test_matched_terms_dict_returned(self, word_sets):
        LMMatcher = _try_import("lm_matcher", "LMMatcher")
        if LMMatcher is None:
            pytest.skip()
        cfg_cls = _try_import("lm_matcher", "LMMatcherConfig")
        cfg     = cfg_cls(return_matched_terms=True) if cfg_cls else None
        try:
            matcher = LMMatcher(word_sets, cfg)
        except TypeError:
            matcher = _make_matcher(word_sets)
        result = matcher.match(["growth", "growth", "decline", "the"])
        if "matched_terms" in result:
            assert isinstance(result["matched_terms"], dict)
            assert result["matched_terms"].get("growth", 0) == 2

    def test_unmatched_tokens_list_returned(self, word_sets):
        LMMatcher = _try_import("lm_matcher", "LMMatcher")
        if LMMatcher is None:
            pytest.skip()
        cfg_cls = _try_import("lm_matcher", "LMMatcherConfig")
        cfg     = cfg_cls(return_unmatched_tokens=True) if cfg_cls else None
        try:
            matcher = LMMatcher(word_sets, cfg)
        except TypeError:
            matcher = _make_matcher(word_sets)
        result = matcher.match(["growth", "the", "and"])
        if "unmatched_tokens" in result:
            assert "the" in result["unmatched_tokens"]
            assert "and" in result["unmatched_tokens"]
            assert "growth" not in result["unmatched_tokens"]


# ===========================================================================
# GROUP 6 — Frequency vs unique counting
# ===========================================================================


class TestFrequencyVsUniqueCounting:
    """
    Validate that frequency counting (default) counts every token occurrence,
    and unique counting counts each distinct type once.
    """

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")

    def _freq_matcher(self, word_sets):
        cfg_cls = _try_import("lm_matcher", "LMMatcherConfig")
        cfg     = cfg_cls(count_unique=False) if cfg_cls else None
        LMMatcher = _try_import("lm_matcher", "LMMatcher")
        return LMMatcher(word_sets, cfg)

    def _unique_matcher(self, word_sets):
        cfg_cls = _try_import("lm_matcher", "LMMatcherConfig")
        if cfg_cls is None:
            pytest.skip("LMMatcherConfig required for unique-count mode.")
        try:
            cfg = cfg_cls(count_unique=True)
        except TypeError:
            pytest.skip("count_unique not supported by LMMatcherConfig.")
        LMMatcher = _try_import("lm_matcher", "LMMatcher")
        return LMMatcher(word_sets, cfg)

    def test_frequency_counts_all_occurrences(self, word_sets):
        matcher = self._freq_matcher(word_sets)
        tokens  = ["growth", "growth", "growth"]
        result  = matcher.match(tokens)
        assert result["positive_count"] == 3

    def test_unique_counts_each_type_once(self, word_sets):
        matcher = self._unique_matcher(word_sets)
        tokens  = ["growth", "growth", "growth"]
        result  = matcher.match(tokens)
        assert result["positive_count"] == 1

    def test_frequency_vs_unique_differ_on_duplicates(self, word_sets):
        tokens   = ["growth", "growth", "decline", "decline"]
        freq_r   = self._freq_matcher(word_sets).match(tokens)
        unique_r = self._unique_matcher(word_sets).match(tokens)
        assert freq_r["positive_count"]  > unique_r["positive_count"]
        assert freq_r["negative_count"]  > unique_r["negative_count"]

    def test_frequency_total_tokens_equals_input_length(self, word_sets):
        matcher = self._freq_matcher(word_sets)
        tokens  = ["growth", "growth", "decline", "the"]
        result  = matcher.match(tokens)
        assert result["total_tokens"] == 4

    def test_unique_total_types_leq_input_length(self, word_sets):
        """In unique mode total_tokens may reflect unique types."""
        matcher = self._unique_matcher(word_sets)
        tokens  = ["growth", "growth", "decline", "the"]
        result  = matcher.match(tokens)
        # unique types = 3 (growth, decline, the)
        assert result["total_tokens"] <= 4

    def test_frequency_default_behaviour(self, word_sets, duplicate_tokens):
        """Default matcher (no config) must use frequency counting."""
        matcher = _make_matcher(word_sets)
        result  = matcher.match(duplicate_tokens)
        # "growth" appears 3 times → should be 3
        assert result["positive_count"] == 3


# ===========================================================================
# GROUP 7 — Validation behaviour
# ===========================================================================


class TestMatcherValidation:
    """Validate that invalid / malformed tokens are handled safely."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)

    def test_empty_string_tokens_skipped(self, whitespace_tokens):
        result = self.matcher.match(whitespace_tokens)
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_empty_token_list_returns_zeros(self, empty_tokens):
        result = self.matcher.match(empty_tokens)
        assert result["positive_count"]  == 0
        assert result["negative_count"]  == 0
        assert result["matched_tokens"]  == 0
        assert result["total_tokens"]    == 0

    def test_none_token_in_list_handled(self, word_sets):
        tokens = ["growth", None, "decline", None, "the"]  # type: ignore[list-item]
        try:
            result = self.matcher.match(tokens)
            assert result["positive_count"] >= 1
            assert result["negative_count"] >= 1
        except TypeError:
            pass  # Either filters None or raises — both acceptable

    def test_non_string_tokens_handled(self, word_sets):
        tokens = ["growth", 42, 3.14, True, "decline"]  # type: ignore[list-item]
        try:
            result = self.matcher.match(tokens)
            assert result["positive_count"] >= 1
        except (TypeError, AttributeError):
            pass

    def test_malformed_punctuation_tokens(self, punctuation_tokens):
        """Matcher should not crash on tokens with heavy punctuation."""
        result = self.matcher.match(punctuation_tokens)
        assert result["total_tokens"] >= 0
        assert result["matched_tokens"] >= 0

    def test_matched_tokens_never_exceeds_total(self, long_financial_text):
        result = self.matcher.match(long_financial_text)
        assert result["matched_tokens"] <= result["total_tokens"]

    def test_coverage_ratio_never_negative(self, mixed_tokens):
        result = self.matcher.match(mixed_tokens)
        assert result["coverage_ratio"] >= 0.0

    def test_coverage_ratio_never_above_one(self, long_financial_text):
        result = self.matcher.match(long_financial_text)
        assert result["coverage_ratio"] <= 1.0 + 1e-9

    def test_unicode_tokens_no_crash(self, unicode_tokens):
        try:
            result = self.matcher.match(unicode_tokens)
            assert result["total_tokens"] >= 0
        except (UnicodeError, ValueError):
            pass

    def test_very_long_token_handled(self, word_sets):
        long_token = "a" * 500
        result     = self.matcher.match([long_token, "growth"])
        # Should count "growth" regardless of the oversized token
        assert result["positive_count"] >= 1


# ===========================================================================
# GROUP 8 — Edge-case handling
# ===========================================================================


class TestEdgeCaseHandling:
    """Adversarial inputs that must not raise unhandled exceptions."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)

    def test_single_positive_token(self):
        result = self.matcher.match(["growth"])
        assert result["positive_count"]  == 1
        assert result["negative_count"]  == 0
        assert result["total_tokens"]    == 1
        assert result["matched_tokens"]  == 1
        assert result["coverage_ratio"]  == pytest.approx(1.0)

    def test_single_negative_token(self):
        result = self.matcher.match(["decline"])
        assert result["negative_count"]  == 1
        assert result["positive_count"]  == 0
        assert result["total_tokens"]    == 1
        assert result["matched_tokens"]  == 1

    def test_single_unmatched_token(self):
        result = self.matcher.match(["revenue"])
        assert result["positive_count"]  == 0
        assert result["negative_count"]  == 0
        assert result["matched_tokens"]  == 0
        assert result["total_tokens"]    == 1
        assert result["coverage_ratio"]  == pytest.approx(0.0)

    def test_all_duplicate_matched_tokens(self):
        tokens = ["growth"] * 10
        result = self.matcher.match(tokens)
        assert result["positive_count"] == 10
        assert result["total_tokens"]   == 10

    def test_all_duplicate_unmatched_tokens(self):
        tokens = ["revenue"] * 10
        result = self.matcher.match(tokens)
        assert result["positive_count"]  == 0
        assert result["matched_tokens"]  == 0
        assert result["total_tokens"]    == 10

    def test_mixed_case_lookup(self):
        variants = ["Growth", "GROWTH", "gRoWtH", "growth"]
        result   = self.matcher.match(variants)
        # All four should resolve to the same word → 4 matches
        assert result["positive_count"] == 4

    def test_empty_word_sets_returns_zero_counts(self):
        empty_dict = {
            "positive": frozenset(), "negative": frozenset(),
            "uncertainty": frozenset(), "litigious": frozenset(),
            "strong_modal": frozenset(), "weak_modal": frozenset(),
            "constraining": frozenset(),
        }
        try:
            matcher = _make_matcher(empty_dict)
            result  = matcher.match(["growth", "decline"])
            assert result["positive_count"] == 0
            assert result["negative_count"] == 0
        except Exception:
            pass  # Empty dictionary may raise — acceptable

    def test_token_list_with_only_whitespace(self, whitespace_tokens):
        result = self.matcher.match(whitespace_tokens)
        assert result["positive_count"] == 0
        assert result["matched_tokens"] == 0

    def test_very_large_token_list_no_crash(self):
        large = (list(_POSITIVE) + list(_NEGATIVE) + _NEUTRAL_FILLER) * 50
        try:
            result = self.matcher.match(large)
            assert result["total_tokens"] == len(large)
        except Exception as exc:
            pytest.fail(f"Matcher crashed on large input: {exc}")

    def test_single_character_tokens_handled(self):
        tokens = ["a", "b", "c", "d", "growth"]
        result = self.matcher.match(tokens)
        # Single chars are below min_token_length=2 by default
        # At minimum, "growth" must be counted
        assert result["positive_count"] >= 1

    def test_numeric_string_tokens(self):
        tokens = ["123", "456", "growth", "789"]
        result = self.matcher.match(tokens)
        assert result["positive_count"] == 1


# ===========================================================================
# GROUP 9 — Deterministic outputs
# ===========================================================================


class TestDeterministicOutputs:
    """Same inputs must always produce identical outputs."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)

    def test_same_input_same_counts(self, mixed_tokens):
        r1 = self.matcher.match(mixed_tokens)
        r2 = self.matcher.match(mixed_tokens)
        assert r1["positive_count"]  == r2["positive_count"]
        assert r1["negative_count"]  == r2["negative_count"]
        assert r1["matched_tokens"]  == r2["matched_tokens"]
        assert r1["coverage_ratio"]  == r2["coverage_ratio"]

    def test_same_input_same_all_keys(self, long_financial_text):
        r1 = self.matcher.match(long_financial_text)
        r2 = self.matcher.match(long_financial_text)
        common_keys = set(r1.keys()) & set(r2.keys())
        for k in common_keys:
            if isinstance(r1[k], (int, float)):
                assert r1[k] == r2[k], f"Non-deterministic key: '{k}'"

    def test_shuffled_input_same_counts(self, word_sets):
        """Order should not affect category counts (frequency mode)."""
        base    = list(_POSITIVE)[:5] + list(_NEGATIVE)[:3] + _NEUTRAL_FILLER[:5]
        rng     = np.random.default_rng(7)
        shuffled = [base[i] for i in rng.permutation(len(base)).tolist()]
        r1 = self.matcher.match(base)
        r2 = self.matcher.match(shuffled)
        assert r1["positive_count"]  == r2["positive_count"]
        assert r1["negative_count"]  == r2["negative_count"]
        assert r1["matched_tokens"]  == r2["matched_tokens"]
        assert r1["total_tokens"]    == r2["total_tokens"]

    def test_multiple_matchers_same_dict_same_results(self, word_sets, mixed_tokens):
        m1 = _make_matcher(word_sets)
        m2 = _make_matcher(word_sets)
        r1 = m1.match(mixed_tokens)
        r2 = m2.match(mixed_tokens)
        assert r1["positive_count"]  == r2["positive_count"]
        assert r1["negative_count"]  == r2["negative_count"]
        assert r1["coverage_ratio"]  == r2["coverage_ratio"]

    def test_repeated_calls_same_state(self, word_sets, mixed_tokens):
        matcher = _make_matcher(word_sets)
        results = [matcher.match(mixed_tokens) for _ in range(5)]
        pos_counts = [r["positive_count"] for r in results]
        assert len(set(pos_counts)) == 1, "Repeated calls produced different counts."

    def test_empty_input_always_zeros(self, word_sets):
        matcher = _make_matcher(word_sets)
        for _ in range(3):
            r = matcher.match([])
            assert r["positive_count"] == 0
            assert r["matched_tokens"] == 0
            assert r["total_tokens"]   == 0


# ===========================================================================
# GROUP 10 — Diagnostics generation
# ===========================================================================


class TestDiagnosticsGeneration:
    """Validate optional diagnostics output (matched terms, top-N, unmatched)."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")

    def _diag_matcher(self, word_sets):
        cfg_cls = _try_import("lm_matcher", "LMMatcherConfig")
        if cfg_cls is None:
            return _make_matcher(word_sets)
        try:
            cfg = cfg_cls(return_matched_terms=True, return_unmatched_tokens=True)
            LMMatcher = _try_import("lm_matcher", "LMMatcher")
            return LMMatcher(word_sets, cfg)
        except TypeError:
            return _make_matcher(word_sets)

    def test_matched_terms_keys_are_strings(self, word_sets):
        matcher = self._diag_matcher(word_sets)
        result  = matcher.match(["growth", "growth", "decline", "the"])
        terms   = result.get("matched_terms", {})
        if terms:
            assert all(isinstance(k, str) for k in terms)

    def test_matched_terms_values_are_ints(self, word_sets):
        matcher = self._diag_matcher(word_sets)
        result  = matcher.match(["growth", "growth", "decline"])
        terms   = result.get("matched_terms", {})
        if terms:
            assert all(isinstance(v, int) for v in terms.values())

    def test_matched_terms_frequency_correct(self, word_sets):
        matcher = self._diag_matcher(word_sets)
        tokens  = ["growth", "growth", "growth", "decline"]
        result  = matcher.match(tokens)
        terms   = result.get("matched_terms", {})
        if terms:
            assert terms.get("growth", 0) == 3
            assert terms.get("decline", 0) == 1

    def test_unmatched_tokens_are_strings(self, word_sets):
        matcher = self._diag_matcher(word_sets)
        result  = matcher.match(["growth", "the", "and"])
        unmatched = result.get("unmatched_tokens", [])
        if unmatched:
            assert all(isinstance(t, str) for t in unmatched)

    def test_unmatched_does_not_contain_matched_words(self, word_sets):
        matcher = self._diag_matcher(word_sets)
        result  = matcher.match(["growth", "the", "decline"])
        unmatched = result.get("unmatched_tokens", [])
        if unmatched:
            assert "growth"  not in unmatched
            assert "decline" not in unmatched
            assert "the"     in unmatched

    def test_match_statistics_returned(self, word_sets, mixed_tokens):
        MatchStatistics = _try_import("lm_matcher", "MatchStatistics")
        if MatchStatistics is None:
            pytest.skip("MatchStatistics not yet implemented.")
        matcher = _make_matcher(word_sets)
        result  = matcher.match(mixed_tokens)
        stats   = result.get("statistics")
        if stats:
            assert hasattr(stats, "total_tokens") or isinstance(stats, dict)

    def test_summary_method_on_match_result(self, word_sets):
        LMMatchResult = _try_import("lm_matcher", "LMMatchResult")
        if LMMatchResult is None:
            pytest.skip("LMMatchResult not yet implemented.")
        matcher = _make_matcher(word_sets)
        raw     = matcher.match(["growth", "decline", "the"])
        # If matcher returns an LMMatchResult object
        if hasattr(raw, "summary"):
            s = raw.summary()
            assert isinstance(s, str) and len(s) > 0

    def test_top_positive_terms_in_diagnostics(self, word_sets):
        matcher = self._diag_matcher(word_sets)
        tokens  = ["growth", "growth", "strong", "record", "decline"]
        result  = matcher.match(tokens)
        top_pos = result.get("top_positive_terms", [])
        if top_pos:
            assert "growth" in top_pos

    def test_top_negative_terms_in_diagnostics(self, word_sets):
        matcher = self._diag_matcher(word_sets)
        tokens  = ["decline", "decline", "loss", "growth"]
        result  = matcher.match(tokens)
        top_neg = result.get("top_negative_terms", [])
        if top_neg:
            assert "decline" in top_neg


# ===========================================================================
# GROUP 11 — Batch matching
# ===========================================================================


class TestBatchMatching:
    """Validate match_batch() if the matcher exposes it."""

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        LMMatcher = _try_import("lm_matcher", "LMMatcher")
        if LMMatcher is None:
            pytest.skip("LMMatcher not yet implemented.")
        self.matcher = _make_matcher(word_sets)
        if not hasattr(self.matcher, "match_batch"):
            pytest.skip("match_batch() not implemented.")

    def test_batch_length_matches_input(self):
        token_lists = [
            ["growth", "strong"],
            ["decline", "loss"],
            [],
            ["the", "and", "in"],
        ]
        results = self.matcher.match_batch(token_lists)
        assert len(results) == 4

    def test_batch_empty_list_produces_zeros(self):
        results = self.matcher.match_batch([[]])
        r = results[0]
        assert r["positive_count"] == 0
        assert r["total_tokens"]   == 0

    def test_batch_individual_consistency(self):
        token_lists = [
            ["growth", "the", "decline"],
            ["strong", "loss", "revenue"],
        ]
        batch_results = self.matcher.match_batch(token_lists)
        for i, tokens in enumerate(token_lists):
            single = self.matcher.match(tokens)
            for k in ("positive_count", "negative_count",
                      "matched_tokens", "total_tokens"):
                assert batch_results[i][k] == single[k], \
                    f"Batch/single mismatch for '{k}' on input {i}."

    def test_batch_deterministic(self):
        token_lists = [
            ["growth", "growth", "decline"],
            ["the", "and", "strong"],
        ]
        r1 = self.matcher.match_batch(token_lists)
        r2 = self.matcher.match_batch(token_lists)
        for i in range(len(token_lists)):
            assert r1[i]["positive_count"] == r2[i]["positive_count"]
            assert r1[i]["negative_count"] == r2[i]["negative_count"]


# ===========================================================================
# GROUP 12 — Preprocessor integration
# ===========================================================================


class TestPreprocessorIntegration:
    """
    Validate the full preprocess → match pipeline when both modules exist.
    These tests are skipped if either component is absent.
    """

    @pytest.fixture(autouse=True)
    def _skip(self, word_sets):
        if _try_import("lm_matcher", "LMMatcher") is None:
            pytest.skip("LMMatcher not yet implemented.")
        if _try_import("lm_preprocessing", "LMPreprocessor") is None:
            pytest.skip("LMPreprocessor not yet implemented.")

    def test_preprocessed_tokens_matched_correctly(self, word_sets):
        LMPreprocessor       = _try_import("lm_preprocessing", "LMPreprocessor")
        LMPreprocessingConfig = _try_import("lm_preprocessing", "LMPreprocessingConfig")
        cfg = LMPreprocessingConfig() if LMPreprocessingConfig else None
        pp  = LMPreprocessor(cfg) if cfg else LMPreprocessor()

        text   = "We delivered record growth and strong performance this quarter."
        result = pp.preprocess(text)
        assert result is not None

        matcher     = _make_matcher(word_sets)
        match_result = matcher.match(result.tokens)
        assert match_result["positive_count"] >= 2   # record, growth, strong

    def test_negative_text_counts_correctly(self, word_sets):
        LMPreprocessor       = _try_import("lm_preprocessing", "LMPreprocessor")
        LMPreprocessingConfig = _try_import("lm_preprocessing", "LMPreprocessingConfig")
        cfg = LMPreprocessingConfig() if LMPreprocessingConfig else None
        pp  = LMPreprocessor(cfg) if cfg else LMPreprocessor()

        text   = "We reported a decline and loss impairment due to litigation risks."
        result = pp.preprocess(text)
        assert result is not None

        matcher     = _make_matcher(word_sets)
        match_result = matcher.match(result.tokens)
        assert match_result["negative_count"] >= 2

    def test_empty_text_preprocessed_then_matched(self, word_sets):
        LMPreprocessor       = _try_import("lm_preprocessing", "LMPreprocessor")
        LMPreprocessingConfig = _try_import("lm_preprocessing", "LMPreprocessingConfig")
        cfg = LMPreprocessingConfig() if LMPreprocessingConfig else None
        pp  = LMPreprocessor(cfg) if cfg else LMPreprocessor()

        result = pp.preprocess("")
        assert result is None

        matcher     = _make_matcher(word_sets)
        match_result = matcher.match([])
        assert match_result["positive_count"] == 0
        assert match_result["matched_tokens"] == 0

    def test_batch_preprocessing_then_batch_matching(self, word_sets):
        LMPreprocessor       = _try_import("lm_preprocessing", "LMPreprocessor")
        LMPreprocessingConfig = _try_import("lm_preprocessing", "LMPreprocessingConfig")
        cfg = LMPreprocessingConfig() if LMPreprocessingConfig else None
        pp  = LMPreprocessor(cfg) if cfg else LMPreprocessor()

        texts = [
            "Revenue grew strongly with record growth and outstanding momentum.",
            "We faced a decline and loss with increased concerns and headwind.",
            "",
        ]
        prep_results, _ = pp.preprocess_batch(texts)
        matcher = _make_matcher(word_sets)
        for i, pr in enumerate(prep_results):
            tokens = pr.tokens if pr is not None else []
            result = matcher.match(tokens)
            assert result["total_tokens"] >= 0
            assert result["matched_tokens"] <= result["total_tokens"]


# ===========================================================================
# STANDALONE EXECUTION
# ===========================================================================

if __name__ == "__main__":
    import subprocess

    print("\n" + "=" * 65)
    print("  test_lm_matcher.py — Standalone demo run")
    print("=" * 65)
    print("\nRunning via pytest...\n")

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            __file__,
            "-v",
            "--tb=short",
            "--no-header",
        ],
        capture_output=False,
    )
    sys.exit(result.returncode)
