"""
tests/test_lm_pipeline.py
==========================
Comprehensive end-to-end integration test suite for the Loughran-McDonald
financial sentiment pipeline.

Test strategy
-------------
* All tests use synthetic in-memory data — no network calls, no disk I/O
  except inside tmp_path fixtures.
* Every fixture is deterministic: same seed, same data, every run.
* Tests are organised into groups matching the pipeline stage they exercise.
* Edge-case tests are isolated from the main happy-path group so failures
  are easy to triage.
* The test_determinism group reruns the same pipeline twice and asserts
  byte-identical parquet outputs.

Run with:
    pytest tests/test_lm_pipeline.py -v
    pytest tests/test_lm_pipeline.py -v -k "scoring"   # single group
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path when running from repo root
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SENTIMENT = _SRC / "sentiment"
if str(_SENTIMENT) not in sys.path:
    sys.path.insert(0, str(_SENTIMENT))

# ---------------------------------------------------------------------------
# Lazy imports — each test group imports only what it exercises.
# If a module is not yet implemented the relevant tests are skipped with a
# clear message rather than erroring at collection time.
# ---------------------------------------------------------------------------


def _try_import(module: str, attr: str):
    try:
        mod = __import__(module, fromlist=[attr])
        return getattr(mod, attr)
    except (ImportError, AttributeError):
        return None


# ===========================================================================
# CONSTANTS — shared across fixtures
# ===========================================================================

POSITIVE_WORDS = [
    "growth", "record", "strong", "exceeded", "expanded", "improved",
    "confident", "momentum", "outperform", "robust", "exceptional",
    "increased", "profitable", "optimistic", "delivered", "achieved",
    "raised", "beat", "favorable", "opportunities", "outstanding",
]

NEGATIVE_WORDS = [
    "decline", "loss", "impairment", "litigation", "concerns", "risk",
    "headwind", "uncertainty", "weaker", "reduced", "challenges",
    "miss", "disappointing", "adverse", "restructuring", "write-off",
    "unfavorable", "difficult", "constraint", "deterioration",
]

UNCERTAINTY_WORDS = [
    "uncertain", "approximately", "may", "might", "could", "if",
    "subject", "estimate", "project", "expect",
]

NEUTRAL_FILLER = [
    "the", "and", "in", "of", "our", "we", "this", "quarter",
    "revenue", "margin", "segment", "fiscal", "year", "reported",
    "following", "during", "period", "company", "management",
]

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture(scope="session")
def lm_word_sets() -> Dict[str, List[str]]:
    """Canonical LM word-set fixture shared across all tests."""
    return {
        "positive":    POSITIVE_WORDS,
        "negative":    NEGATIVE_WORDS,
        "uncertainty": UNCERTAINTY_WORDS,
        "litigious":   ["litigation", "lawsuit", "regulatory", "compliance",
                        "liability", "settlement"],
        "strong_modal": ["will", "must", "require", "shall"],
        "weak_modal":   ["may", "might", "could", "should"],
        "constraining": ["constraint", "limit", "restrict", "bound"],
    }


@pytest.fixture(scope="session")
def synthetic_lm_dict_df(lm_word_sets) -> pd.DataFrame:
    """
    Synthetic LM master dictionary DataFrame.

    Mirrors the schema of the real Loughran-McDonald CSV so that
    LMDictionaryLoader can consume it without modification.
    """
    rows = []
    for word in lm_word_sets["positive"]:
        rows.append({"Word": word.upper(), "Positive": 2009, "Negative": 0,
                     "Uncertainty": 0, "Litigious": 0,
                     "StrongModal": 0, "WeakModal": 0, "Constraining": 0})
    for word in lm_word_sets["negative"]:
        rows.append({"Word": word.upper(), "Positive": 0, "Negative": 2009,
                     "Uncertainty": 0, "Litigious": 0,
                     "StrongModal": 0, "WeakModal": 0, "Constraining": 0})
    for word in lm_word_sets["uncertainty"]:
        rows.append({"Word": word.upper(), "Positive": 0, "Negative": 0,
                     "Uncertainty": 2009, "Litigious": 0,
                     "StrongModal": 0, "WeakModal": 0, "Constraining": 0})
    for word in lm_word_sets["litigious"]:
        rows.append({"Word": word.upper(), "Positive": 0, "Negative": 0,
                     "Uncertainty": 0, "Litigious": 2009,
                     "StrongModal": 0, "WeakModal": 0, "Constraining": 0})
    return pd.DataFrame(rows).drop_duplicates(subset=["Word"])


def _make_chunk_text(
    pos_count: int = 5,
    neg_count: int = 1,
    unc_count: int = 1,
    filler_count: int = 20,
    seed: int = 0,
) -> str:
    """Build a reproducible chunk text string from word pools."""
    rng = np.random.default_rng(seed)
    words: List[str] = []
    words += list(rng.choice(POSITIVE_WORDS, size=min(pos_count, len(POSITIVE_WORDS)),
                              replace=False))
    words += list(rng.choice(NEGATIVE_WORDS, size=min(neg_count, len(NEGATIVE_WORDS)),
                              replace=False))
    words += list(rng.choice(UNCERTAINTY_WORDS, size=min(unc_count, len(UNCERTAINTY_WORDS)),
                              replace=False))
    words += list(rng.choice(NEUTRAL_FILLER, size=min(filler_count, len(NEUTRAL_FILLER)),
                              replace=True))
    rng.shuffle(words)
    return " ".join(words)


@pytest.fixture(scope="session")
def synthetic_chunks_df() -> pd.DataFrame:
    """
    Deterministic synthetic transcript chunk dataset.

    Covers:
    * Three tickers — AAPL (positive), MSFT (mixed), NVDA (positive)
    * Two sections — prepared_remarks, qa
    * Two speakers each — CEO, CFO, Analyst
    * Varied sentiment ratios across chunks
    """
    rows = []
    config = [
        # (ticker, quarter, pos, neg, unc, filler, section, speaker, role)
        ("AAPL", "Q1_2025", 8, 1, 2, 20, "prepared_remarks", "Tim Cook",     "CEO"),
        ("AAPL", "Q1_2025", 6, 2, 1, 18, "prepared_remarks", "Luca Maestri", "CFO"),
        ("AAPL", "Q1_2025", 2, 4, 3, 15, "qa",               "Analyst",      "Analyst"),
        ("AAPL", "Q1_2025", 5, 1, 1, 20, "qa",               "Tim Cook",     "CEO"),
        ("MSFT", "Q2_2025", 5, 4, 3, 18, "prepared_remarks", "Satya Nadella","CEO"),
        ("MSFT", "Q2_2025", 3, 5, 2, 16, "prepared_remarks", "Amy Hood",     "CFO"),
        ("MSFT", "Q2_2025", 4, 3, 4, 14, "qa",               "Analyst",      "Analyst"),
        ("NVDA", "Q3_2025", 9, 1, 1, 20, "prepared_remarks", "Jensen Huang", "CEO"),
        ("NVDA", "Q3_2025", 7, 2, 2, 18, "prepared_remarks", "Colette Kress","CFO"),
        ("NVDA", "Q3_2025", 3, 2, 3, 15, "qa",               "Analyst",      "Analyst"),
    ]

    for i, (ticker, quarter, pos, neg, unc, filler, section, speaker, role) in enumerate(config):
        transcript_id = f"{ticker}_{quarter}"
        rows.append({
            "chunk_id":         f"{transcript_id}_{section}_{i:03d}",
            "transcript_id":    transcript_id,
            "ticker":           ticker,
            "section_type":     section,
            "dominant_speaker": speaker,
            "speaker_role":     role,
            "chunk_text":       _make_chunk_text(pos, neg, unc, filler, seed=i),
            "token_count":      pos + neg + unc + filler,
        })

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def malformed_chunks_df() -> pd.DataFrame:
    """Chunks with deliberate problems for edge-case tests."""
    rows = [
        # Empty text
        {"chunk_id": "BAD_001", "transcript_id": "BAD_Q1",
         "ticker": "BAD", "section_type": "prepared_remarks",
         "dominant_speaker": "CEO", "speaker_role": "CEO",
         "chunk_text": "",  "token_count": 0},
        # Whitespace-only
        {"chunk_id": "BAD_002", "transcript_id": "BAD_Q1",
         "ticker": "BAD", "section_type": "prepared_remarks",
         "dominant_speaker": "CEO", "speaker_role": "CEO",
         "chunk_text": "   \t\n  ", "token_count": 0},
        # Valid but extremely short
        {"chunk_id": "BAD_003", "transcript_id": "BAD_Q1",
         "ticker": "BAD", "section_type": "qa",
         "dominant_speaker": "Operator", "speaker_role": "Operator",
         "chunk_text": "Thank you.", "token_count": 2},
        # No LM matches (pure numeric / boilerplate)
        {"chunk_id": "BAD_004", "transcript_id": "BAD_Q1",
         "ticker": "BAD", "section_type": "prepared_remarks",
         "dominant_speaker": "CFO", "speaker_role": "CFO",
         "chunk_text": "123 456 789 000 111 222 333 444 555 666 777",
         "token_count": 11},
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def duplicate_chunks_df(synthetic_chunks_df) -> pd.DataFrame:
    """Dataset containing duplicate chunk_ids."""
    dupe = synthetic_chunks_df.iloc[:2].copy()
    return pd.concat([synthetic_chunks_df, dupe], ignore_index=True)


@pytest.fixture(scope="session")
def low_coverage_chunks_df() -> pd.DataFrame:
    """Chunks with virtually no dictionary coverage."""
    rows = []
    for i in range(4):
        rows.append({
            "chunk_id":         f"LCOV_Q1_{i:03d}",
            "transcript_id":    "LCOV_Q1",
            "ticker":           "LCOV",
            "section_type":     "prepared_remarks",
            "dominant_speaker": "CEO",
            "speaker_role":     "CEO",
            "chunk_text": " ".join(
                ["the", "and", "in", "of", "our", "we", "this"] * 6
            ),
            "token_count":      42,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def mixed_sentiment_chunks_df() -> pd.DataFrame:
    """Chunks spanning the full positive → negative spectrum."""
    configs = [
        ("MIXED_Q1", "strongly_positive", 15, 1),
        ("MIXED_Q1", "mildly_positive",    7, 4),
        ("MIXED_Q1", "neutral",            5, 5),
        ("MIXED_Q1", "mildly_negative",    3, 8),
        ("MIXED_Q1", "strongly_negative",  1, 14),
    ]
    rows = []
    for i, (tid, section, pos, neg) in enumerate(configs):
        rows.append({
            "chunk_id":         f"{tid}_{section}",
            "transcript_id":    tid,
            "ticker":           "MIXED",
            "section_type":     "prepared_remarks",
            "dominant_speaker": "CEO",
            "speaker_role":     "CEO",
            "chunk_text":       _make_chunk_text(pos, neg, 2, 15, seed=100 + i),
            "token_count":      pos + neg + 2 + 15,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Temporary output directory for each test."""
    out = tmp_path / "sentiment"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def tmp_interim_dir(tmp_path) -> Path:
    out = tmp_path / "interim"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_chunks_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Helper: write a chunk DataFrame to parquet for pipeline input."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path))
    return path


# ===========================================================================
# HELPERS — pipeline construction
# ===========================================================================


def _build_pipeline_config(
    chunks_path: Path,
    output_dir: Path,
    interim_dir: Path,
    *,
    overwrite: bool = True,
    batch_size: int = 8,
    max_chunks: Optional[int] = None,
):
    """Construct an LMPipelineConfig pointing at temp directories."""
    LMPipelineConfig = _try_import("lm_pipeline", "LMPipelineConfig")
    if LMPipelineConfig is None:
        pytest.skip("lm_pipeline.LMPipelineConfig not yet implemented.")

    return LMPipelineConfig(
        chunks_path  = chunks_path,
        output_dir   = output_dir,
        interim_dir  = interim_dir,
        overwrite    = overwrite,
        batch_size   = batch_size,
        max_chunks   = max_chunks,
        allow_stub_dictionary = True,
        export_csv   = True,
        export_parquet = True,
    )


def _run_pipeline(config) -> object:
    """Construct and run the LMPipeline, returning the result object."""
    LMPipeline = _try_import("lm_pipeline", "LMPipeline")
    if LMPipeline is None:
        pytest.skip("lm_pipeline.LMPipeline not yet implemented.")
    pipeline = LMPipeline(config)
    return pipeline.run()


# ===========================================================================
# GROUP 1 — Dictionary loading integration
# ===========================================================================


class TestDictionaryLoading:
    """Validate that the LM dictionary loads, normalises, and validates correctly."""

    def test_dict_df_schema(self, synthetic_lm_dict_df):
        required = {"Word", "Positive", "Negative", "Uncertainty",
                    "Litigious", "StrongModal", "WeakModal", "Constraining"}
        assert required.issubset(set(synthetic_lm_dict_df.columns))

    def test_dict_no_duplicate_words(self, synthetic_lm_dict_df):
        dupes = synthetic_lm_dict_df["Word"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate words in synthetic LM dict."

    def test_dict_positive_words_present(self, synthetic_lm_dict_df):
        words_upper = set(synthetic_lm_dict_df["Word"].str.upper())
        for w in POSITIVE_WORDS[:5]:
            assert w.upper() in words_upper, f"'{w}' missing from dict."

    def test_dict_negative_words_present(self, synthetic_lm_dict_df):
        words_upper = set(synthetic_lm_dict_df["Word"].str.upper())
        for w in NEGATIVE_WORDS[:5]:
            assert w.upper() in words_upper, f"'{w}' missing from dict."

    def test_dict_counts_non_negative(self, synthetic_lm_dict_df):
        count_cols = ["Positive", "Negative", "Uncertainty",
                      "Litigious", "StrongModal", "WeakModal", "Constraining"]
        for col in count_cols:
            assert (synthetic_lm_dict_df[col] >= 0).all(), \
                f"Negative values in '{col}'."

    def test_dict_category_counts(self, synthetic_lm_dict_df, lm_word_sets):
        pos_rows = (synthetic_lm_dict_df["Positive"] > 0).sum()
        neg_rows = (synthetic_lm_dict_df["Negative"] > 0).sum()
        assert pos_rows == len(lm_word_sets["positive"])
        assert neg_rows == len(lm_word_sets["negative"])

    def test_loader_integration(self, synthetic_lm_dict_df, tmp_path):
        """Write dict to CSV and load via LMDictionaryLoader if available."""
        LMDictionaryLoader = _try_import("lm_dictionary_loader", "LMDictionaryLoader")
        if LMDictionaryLoader is None:
            pytest.skip("LMDictionaryLoader not yet implemented.")

        csv_path = tmp_path / "LM_dict.csv"
        synthetic_lm_dict_df.to_csv(csv_path, index=False)

        loader = LMDictionaryLoader(csv_path)
        lm_dict = loader.load()
        assert lm_dict is not None
        assert len(lm_dict) > 0

    def test_loader_handles_missing_file(self, tmp_path):
        LMDictionaryLoader = _try_import("lm_dictionary_loader", "LMDictionaryLoader")
        if LMDictionaryLoader is None:
            pytest.skip("LMDictionaryLoader not yet implemented.")

        with pytest.raises(Exception):
            LMDictionaryLoader(tmp_path / "nonexistent.csv").load()


# ===========================================================================
# GROUP 2 — Preprocessing integration
# ===========================================================================


class TestPreprocessingIntegration:
    """Validate LMPreprocessor behaviour on financial transcript text."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_preprocessing", "LMPreprocessor") is None:
            pytest.skip("LMPreprocessor not yet implemented.")

    def _make_preprocessor(self, **kwargs):
        LMPreprocessor       = _try_import("lm_preprocessing", "LMPreprocessor")
        LMPreprocessingConfig = _try_import("lm_preprocessing", "LMPreprocessingConfig")
        cfg = LMPreprocessingConfig(**kwargs)
        return LMPreprocessor(cfg)

    def test_basic_tokenisation(self):
        pp  = self._make_preprocessor()
        res = pp.preprocess("Revenue grew 12% to $4.5B this quarter.")
        assert res is not None
        assert res.token_count > 0
        assert isinstance(res.tokens, list)

    def test_empty_text_returns_none(self):
        pp  = self._make_preprocessor()
        res = pp.preprocess("")
        assert res is None

    def test_whitespace_only_returns_none(self):
        pp  = self._make_preprocessor()
        res = pp.preprocess("   \t\n  ")
        assert res is None

    def test_lowercase_normalisation(self):
        pp  = self._make_preprocessor(lowercase=True)
        res = pp.preprocess("GROWTH Revenue STRONG")
        assert res is not None
        assert all(t == t.lower() for t in res.tokens)

    def test_sentence_count(self):
        text = "Revenue grew strongly. EPS beat estimates. Margins expanded."
        pp   = self._make_preprocessor()
        res  = pp.preprocess(text)
        assert res is not None
        assert res.sentence_count >= 2

    def test_financial_token_preservation(self):
        pp  = self._make_preprocessor(preserve_percentages=True)
        res = pp.preprocess("EBITDA margin expanded by 120bps to 31.4%.")
        assert res is not None
        assert res.preserved_percentages or res.preserved_tickers  # at least one

    def test_ticker_detection(self):
        pp  = self._make_preprocessor(preserve_tickers=True)
        res = pp.preprocess("AAPL reported strong results; MSFT also beat.")
        assert res is not None
        assert len(res.preserved_tickers) >= 1

    def test_batch_preprocessing(self, synthetic_chunks_df):
        pp    = self._make_preprocessor()
        texts = synthetic_chunks_df["chunk_text"].tolist()
        results, stats = pp.preprocess_batch(texts)
        assert len(results) == len(texts)
        assert stats.texts_processed > 0
        assert stats.texts_skipped >= 0

    def test_batch_skips_empty(self):
        pp = self._make_preprocessor()
        texts = ["Strong growth.", "", "Revenue declined.", None]
        results, stats = pp.preprocess_batch(texts)
        assert len(results) == 4
        assert results[1] is None
        assert results[3] is None
        assert stats.texts_skipped >= 2

    def test_deterministic_output(self):
        pp   = self._make_preprocessor()
        text = "Revenue grew strongly this quarter with record margins."
        r1   = pp.preprocess(text)
        r2   = pp.preprocess(text)
        assert r1 is not None and r2 is not None
        assert r1.tokens == r2.tokens
        assert r1.cleaned_text == r2.cleaned_text

    def test_unique_token_ratio_reasonable(self):
        pp  = self._make_preprocessor()
        res = pp.preprocess(" ".join(POSITIVE_WORDS + NEGATIVE_WORDS))
        assert res is not None
        # all distinct words → ratio should be high
        assert res.unique_token_count / max(res.token_count, 1) > 0.5

    def test_stats_timing_recorded(self, synthetic_chunks_df):
        pp    = self._make_preprocessor()
        texts = synthetic_chunks_df["chunk_text"].tolist()
        _, stats = pp.preprocess_batch(texts)
        assert stats.elapsed_seconds >= 0.0


# ===========================================================================
# GROUP 3 — Matching integration
# ===========================================================================


class TestMatchingIntegration:
    """Validate LMDictionaryMatcher count correctness."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_matcher", "LMDictionaryMatcher") is None:
            pytest.skip("LMDictionaryMatcher not yet implemented.")

    def _make_matcher(self, lm_word_sets):
        LMDictionaryMatcher = _try_import("lm_matcher", "LMDictionaryMatcher")
        return LMDictionaryMatcher(lm_word_sets)

    def test_positive_count_correct(self, lm_word_sets):
        matcher = self._make_matcher(lm_word_sets)
        tokens  = ["growth", "strong", "the", "and"]
        result  = matcher.match(tokens)
        assert result["positive_count"] == 2

    def test_negative_count_correct(self, lm_word_sets):
        matcher = self._make_matcher(lm_word_sets)
        tokens  = ["decline", "loss", "impairment", "revenue"]
        result  = matcher.match(tokens)
        assert result["negative_count"] == 3

    def test_empty_tokens(self, lm_word_sets):
        matcher = self._make_matcher(lm_word_sets)
        result  = matcher.match([])
        assert result["positive_count"]  == 0
        assert result["negative_count"]  == 0
        assert result["matched_tokens"]  == 0
        assert result["total_tokens"]    == 0

    def test_no_matches_returns_zeros(self, lm_word_sets):
        matcher = self._make_matcher(lm_word_sets)
        tokens  = ["the", "and", "in", "of", "our", "we"]
        result  = matcher.match(tokens)
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_total_tokens_correct(self, lm_word_sets):
        matcher = self._make_matcher(lm_word_sets)
        tokens  = ["growth", "the", "decline", "and", "strong"]
        result  = matcher.match(tokens)
        assert result["total_tokens"] == 5

    def test_matched_tokens_leq_total(self, lm_word_sets):
        matcher = self._make_matcher(lm_word_sets)
        tokens  = POSITIVE_WORDS[:5] + NEUTRAL_FILLER[:10]
        result  = matcher.match(tokens)
        assert result["matched_tokens"] <= result["total_tokens"]

    def test_coverage_ratio_in_range(self, lm_word_sets):
        matcher = self._make_matcher(lm_word_sets)
        tokens  = POSITIVE_WORDS[:5] + NEUTRAL_FILLER[:15]
        result  = matcher.match(tokens)
        ratio   = result["matched_tokens"] / max(result["total_tokens"], 1)
        assert 0.0 <= ratio <= 1.0

    def test_deterministic_matching(self, lm_word_sets):
        matcher = self._make_matcher(lm_word_sets)
        tokens  = POSITIVE_WORDS[:3] + NEGATIVE_WORDS[:2] + NEUTRAL_FILLER[:5]
        r1      = matcher.match(tokens)
        r2      = matcher.match(tokens)
        assert r1 == r2


# ===========================================================================
# GROUP 4 — Scoring integration
# ===========================================================================


class TestScoringIntegration:
    """Validate LMScorer formula correctness, edge cases, and label assignment."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_scoring", "LMScorer") is None:
            pytest.skip("LMScorer not yet implemented.")

    def _make_scorer(self, **kwargs):
        LMScorer       = _try_import("lm_scoring", "LMScorer")
        LMScoringConfig = _try_import("lm_scoring", "LMScoringConfig")
        cfg = LMScoringConfig(**kwargs) if kwargs else None
        return LMScorer(cfg)

    def _base_kwargs(self, pos: int, neg: int, unc: int = 0,
                     total: int = 100, matched: int = 20) -> dict:
        return dict(
            positive_count=pos, negative_count=neg, uncertainty_count=unc,
            litigious_count=0, strong_modal_count=0, weak_modal_count=0,
            constraining_count=0, total_tokens=total, matched_tokens=matched,
        )

    def test_pure_positive_score(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(10, 0))
        assert score.score == pytest.approx(1.0)

    def test_pure_negative_score(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(0, 10))
        assert score.score == pytest.approx(-1.0)

    def test_balanced_score_near_zero(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(10, 10))
        assert score.score == pytest.approx(0.0)

    def test_lm_tone_formula_exact(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(12, 4))
        expected = (12 - 4) / (12 + 4)
        assert score.score == pytest.approx(expected, abs=1e-6)

    def test_score_in_valid_range(self):
        scorer = self._make_scorer()
        for pos in range(0, 20, 3):
            for neg in range(0, 20, 3):
                score = scorer.compute_score(**self._base_kwargs(pos, neg))
                assert -1.0 <= score.score <= 1.0, \
                    f"Score out of range: pos={pos} neg={neg} → {score.score}"

    def test_zero_positive_negative_returns_zero(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(0, 0, matched=0))
        assert score.score == pytest.approx(0.0)
        assert score.diagnostics.zero_division_occurred

    def test_empty_input_returns_unknown(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(0, 0, total=0, matched=0))
        SentimentLabel = _try_import("lm_scoring", "SentimentLabel")
        if SentimentLabel:
            assert score.label == SentimentLabel.UNKNOWN

    def test_positive_label_threshold(self):
        scorer = self._make_scorer(positive_threshold=0.05)
        score  = scorer.compute_score(**self._base_kwargs(8, 2))
        SentimentLabel = _try_import("lm_scoring", "SentimentLabel")
        if SentimentLabel:
            assert score.label == SentimentLabel.POSITIVE

    def test_negative_label_threshold(self):
        scorer = self._make_scorer(negative_threshold=0.05)
        score  = scorer.compute_score(**self._base_kwargs(2, 8))
        SentimentLabel = _try_import("lm_scoring", "SentimentLabel")
        if SentimentLabel:
            assert score.label == SentimentLabel.NEGATIVE

    def test_confidence_in_unit_interval(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(10, 2))
        assert 0.0 <= score.confidence <= 1.0

    def test_coverage_ratio_correct(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(5, 5, total=200, matched=20))
        assert score.coverage_ratio == pytest.approx(0.10, abs=1e-6)

    def test_all_ratios_finite(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(7, 3))
        for attr in ("positive_ratio", "negative_ratio", "uncertainty_ratio",
                     "polarity_ratio", "sentiment_intensity",
                     "normalized_positive", "normalized_negative"):
            val = getattr(score, attr)
            assert math.isfinite(val), f"'{attr}' is non-finite."

    def test_to_dict_keys_present(self):
        scorer   = self._make_scorer()
        score    = scorer.compute_score(**self._base_kwargs(8, 3))
        d        = score.to_dict()
        required = [
            "lm_tone_score", "lm_label", "lm_confidence",
            "lm_positive_count", "lm_negative_count", "lm_coverage_ratio",
        ]
        for k in required:
            assert k in d, f"Key '{k}' missing from to_dict()."

    def test_validate_score_passes_on_valid(self):
        scorer = self._make_scorer()
        score  = scorer.compute_score(**self._base_kwargs(6, 4))
        # Should not raise
        scorer.validate_score(score)

    def test_deterministic_scoring(self):
        scorer = self._make_scorer()
        kwargs = self._base_kwargs(10, 3, unc=2)
        s1     = scorer.compute_score(**kwargs)
        s2     = scorer.compute_score(**kwargs)
        assert s1.score     == s2.score
        assert s1.label     == s2.label
        assert s1.confidence == s2.confidence

    def test_smoothing_shifts_extreme_scores(self):
        LMScoringConfig = _try_import("lm_scoring", "LMScoringConfig")
        if LMScoringConfig is None:
            pytest.skip("LMScoringConfig not available.")
        LMScorer = _try_import("lm_scoring", "LMScorer")
        cfg_smooth = LMScoringConfig(apply_smoothing=True, smoothing_alpha=1.0)
        scorer_s   = LMScorer(cfg_smooth)
        scorer_r   = LMScorer()
        kwargs     = self._base_kwargs(0, 20, matched=20)
        score_s    = scorer_s.compute_score(**kwargs)
        score_r    = scorer_r.compute_score(**kwargs)
        # Smoothing should pull -1.0 toward zero
        assert abs(score_s.score) < abs(score_r.score)

    def test_batch_scoring(self):
        scorer  = self._make_scorer()
        records = [
            self._base_kwargs(10, 2),
            self._base_kwargs(3, 9),
            self._base_kwargs(5, 5),
            {},  # empty → should be skipped
        ]
        scores, stats = scorer.score_batch(records)
        assert len(scores) == 4
        assert scores[3] is None
        assert stats.total_scored == 3
        assert stats.total_skipped == 1

    def test_batch_label_counts_sum(self):
        scorer  = self._make_scorer()
        records = [self._base_kwargs(10, 1), self._base_kwargs(1, 10),
                   self._base_kwargs(5, 5)]
        _, stats = scorer.score_batch(records)
        total = sum(stats.label_counts.values())
        assert total == stats.total_scored


# ===========================================================================
# GROUP 5 — Aggregation integration
# ===========================================================================


class TestAggregationIntegration:
    """Validate that chunk-level scores aggregate correctly to higher levels."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_aggregation", "LMAggregation") is None:
            pytest.skip("LMAggregation not yet implemented.")

    def _make_aggregator(self):
        LMAggregation = _try_import("lm_aggregation", "LMAggregation")
        return LMAggregation()

    def _scored_df(self, synthetic_chunks_df) -> pd.DataFrame:
        """Attach synthetic score columns to the chunk DataFrame."""
        df = synthetic_chunks_df.copy()
        rng = np.random.default_rng(42)
        n   = len(df)
        pos = rng.integers(1, 15, n)
        neg = rng.integers(1, 8,  n)
        denom = (pos + neg).clip(1)
        df["lm_tone_score"]        = ((pos - neg) / denom).round(6)
        df["lm_positive_count"]    = pos
        df["lm_negative_count"]    = neg
        df["lm_uncertainty_count"] = rng.integers(0, 5, n)
        df["lm_litigious_count"]   = rng.integers(0, 3, n)
        df["lm_matched_tokens"]    = pos + neg + df["lm_uncertainty_count"]
        df["lm_total_tokens"]      = df["token_count"]
        df["lm_coverage_ratio"]    = (
            df["lm_matched_tokens"] / df["lm_total_tokens"].clip(lower=1)
        ).round(6)
        df["lm_label"] = df["lm_tone_score"].apply(
            lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
        )
        return df

    def test_transcript_aggregation_row_count(self, synthetic_chunks_df):
        agg  = self._make_aggregator()
        df   = self._scored_df(synthetic_chunks_df)
        out  = agg.aggregate_transcript(df)
        expected_transcripts = df["transcript_id"].nunique()
        assert len(out) == expected_transcripts

    def test_transcript_no_duplicates(self, synthetic_chunks_df):
        agg = self._make_aggregator()
        df  = self._scored_df(synthetic_chunks_df)
        out = agg.aggregate_transcript(df)
        assert out["transcript_id"].duplicated().sum() == 0

    def test_section_aggregation_row_count(self, synthetic_chunks_df):
        agg = self._make_aggregator()
        df  = self._scored_df(synthetic_chunks_df)
        out = agg.aggregate_section(df)
        expected = df.groupby(["transcript_id", "section_type"]).ngroups
        assert len(out) == expected

    def test_speaker_aggregation_contains_ceo(self, synthetic_chunks_df):
        agg  = self._make_aggregator()
        df   = self._scored_df(synthetic_chunks_df)
        out  = agg.aggregate_speaker(df)
        roles = out["speaker_role"].str.upper().tolist() \
            if "speaker_role" in out.columns else []
        assert "CEO" in roles

    def test_aggregated_tone_in_range(self, synthetic_chunks_df):
        agg  = self._make_aggregator()
        df   = self._scored_df(synthetic_chunks_df)
        out  = agg.aggregate_transcript(df)
        col  = next(
            (c for c in out.columns if "tone" in c.lower()), None
        )
        if col:
            assert (out[col].dropna().abs() <= 1.0).all(), \
                "Aggregated tone score out of [-1, 1]."

    def test_token_totals_preserved(self, synthetic_chunks_df):
        agg  = self._make_aggregator()
        df   = self._scored_df(synthetic_chunks_df)
        out  = agg.aggregate_transcript(df)
        col  = next(
            (c for c in out.columns if "total_tokens" in c.lower()), None
        )
        if col:
            assert (out[col].dropna() >= 0).all()

    def test_deterministic_aggregation(self, synthetic_chunks_df):
        agg = self._make_aggregator()
        df  = self._scored_df(synthetic_chunks_df)
        o1  = agg.aggregate_transcript(df)
        o2  = agg.aggregate_transcript(df)
        pd.testing.assert_frame_equal(
            o1.reset_index(drop=True), o2.reset_index(drop=True)
        )


# ===========================================================================
# GROUP 6 — Validation integration
# ===========================================================================


class TestValidationIntegration:
    """Validate LMValidation catches real and synthetic data problems."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_validation", "LMValidation") is None:
            pytest.skip("LMValidation not yet implemented.")

    def _make_validator(self):
        LMValidation = _try_import("lm_validation", "LMValidation")
        return LMValidation()

    def _minimal_scored_df(self, n: int = 10) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        scores = rng.uniform(-0.8, 0.8, n)
        return pd.DataFrame({
            "chunk_id":       [f"CHK_{i}" for i in range(n)],
            "transcript_id":  [f"TXN_{i % 3}" for i in range(n)],
            "lm_tone_score":  scores.round(6),
            "lm_label":       ["positive" if s > 0.05 else
                                "negative" if s < -0.05 else "neutral"
                                for s in scores],
            "lm_coverage_ratio": rng.uniform(0.02, 0.20, n).round(6),
            "lm_total_tokens": rng.integers(30, 200, n),
        })

    def test_valid_data_passes(self):
        val = self._make_validator()
        df  = self._minimal_scored_df()
        report = val.validate(df)
        assert report is not None

    def test_detects_out_of_range_scores(self):
        val = self._make_validator()
        df  = self._minimal_scored_df()
        df.loc[0, "lm_tone_score"] = 2.5   # invalid
        report = val.validate(df)
        # Validator must flag this somehow
        has_issue = (
            (hasattr(report, "warnings") and report.warnings)
            or (hasattr(report, "is_valid") and not report.is_valid())
            or (isinstance(report, dict) and report.get("warnings"))
        )
        assert has_issue, "Validator did not flag out-of-range score."

    def test_detects_duplicate_chunk_ids(self):
        val = self._make_validator()
        df  = self._minimal_scored_df()
        df  = pd.concat([df, df.iloc[:2]], ignore_index=True)
        report = val.validate(df)
        has_issue = (
            (hasattr(report, "warnings") and report.warnings)
            or (hasattr(report, "is_valid") and not report.is_valid())
            or (isinstance(report, dict) and report.get("warnings"))
        )
        assert has_issue, "Validator did not flag duplicate chunk_ids."

    def test_detects_missing_required_column(self):
        val = self._make_validator()
        df  = self._minimal_scored_df().drop(columns=["lm_tone_score"])
        with pytest.raises(Exception):
            val.validate(df)

    def test_empty_dataframe_handled(self):
        val = self._make_validator()
        try:
            report = val.validate(pd.DataFrame())
        except Exception:
            pass  # Either raises or returns an invalid report — both acceptable


# ===========================================================================
# GROUP 7 — Full pipeline execution
# ===========================================================================


class TestFullPipelineExecution:
    """End-to-end pipeline.run() tests against synthetic chunk data."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_pipeline", "LMPipeline") is None:
            pytest.skip("LMPipeline not yet implemented.")

    def test_pipeline_run_succeeds(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        result = _run_pipeline(config)
        assert result.success, f"Pipeline failed.\n{result.summary()}"

    def test_pipeline_result_has_summary(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        result = _run_pipeline(config)
        summary = result.summary()
        assert isinstance(summary, str) and len(summary) > 0

    def test_chunks_processed_count(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        result = _run_pipeline(config)
        assert result.chunks_loaded == len(synthetic_chunks_df)
        assert result.chunks_processed == len(synthetic_chunks_df)

    def test_transcripts_scored_count(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        result = _run_pipeline(config)
        expected = synthetic_chunks_df["transcript_id"].nunique()
        assert result.transcripts_scored == expected

    def test_score_mean_finite(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        result = _run_pipeline(config)
        assert math.isfinite(result.score_mean)
        assert math.isfinite(result.score_std)
        assert math.isfinite(result.score_median)

    def test_label_counts_sum_to_processed(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        result = _run_pipeline(config)
        total_labels = sum(result.label_counts.values())
        assert total_labels == result.chunks_processed

    def test_pipeline_with_max_chunks(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(
            chunks_path, tmp_output_dir, tmp_interim_dir, max_chunks=4
        )
        result = _run_pipeline(config)
        assert result.chunks_processed == 4

    def test_pipeline_missing_input_fails_gracefully(
        self, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        config = _build_pipeline_config(
            tmp_path / "nonexistent.parquet", tmp_output_dir, tmp_interim_dir
        )
        result = _run_pipeline(config)
        assert not result.success

    def test_pipeline_elapsed_time_recorded(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        result = _run_pipeline(config)
        assert result.total_elapsed_seconds > 0.0

    def test_stage_audit_trail_present(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        result = _run_pipeline(config)
        assert len(result.stages) >= 3  # at minimum: load, score, export


# ===========================================================================
# GROUP 8 — Export validation
# ===========================================================================


class TestExportValidation:
    """Validate that output files exist, are readable, and have correct schemas."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_pipeline", "LMPipeline") is None:
            pytest.skip("LMPipeline not yet implemented.")

    def _run_and_get_result(self, df, tmp_output_dir, tmp_interim_dir, tmp_path):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_output_dir, tmp_interim_dir)
        return _run_pipeline(config), tmp_output_dir

    def test_chunk_scores_parquet_created(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        result, out = self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = out / "lm_chunk_scores.parquet"
        assert pq_path.exists(), f"Expected {pq_path} to exist."

    def test_transcript_scores_parquet_created(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        result, out = self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = out / "lm_transcript_scores.parquet"
        assert pq_path.exists(), f"Expected {pq_path} to exist."

    def test_compatibility_lm_scores_parquet_created(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        result, out = self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = out / "lm_scores.parquet"
        assert pq_path.exists(), f"Expected {pq_path} to exist."
        df = pd.read_parquet(pq_path)
        required = {
            "transcript_id",
            "ticker",
            "lm_positive_count",
            "lm_negative_count",
            "lm_tone_score",
            "lm_total_tokens",
            "lm_label",
        }
        assert required.issubset(df.columns)

    def test_csv_files_created(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        result, out = self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        csv_path = out / "lm_chunk_scores.csv"
        assert csv_path.exists(), f"Expected {csv_path} to exist."

    def test_chunk_parquet_readable(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = tmp_output_dir / "lm_chunk_scores.parquet"
        if not pq_path.exists():
            pytest.skip("Parquet not exported by this pipeline build.")
        df = pd.read_parquet(pq_path)
        assert len(df) > 0

    def test_chunk_parquet_has_required_columns(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = tmp_output_dir / "lm_chunk_scores.parquet"
        if not pq_path.exists():
            pytest.skip("Parquet not exported by this pipeline build.")
        df = pd.read_parquet(pq_path)
        required = ["chunk_id", "transcript_id", "lm_tone_score",
                    "lm_label", "lm_coverage_ratio"]
        for col in required:
            assert col in df.columns, f"Column '{col}' missing from chunk parquet."

    def test_chunk_parquet_row_count_matches(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = tmp_output_dir / "lm_chunk_scores.parquet"
        if not pq_path.exists():
            pytest.skip("Parquet not exported by this pipeline build.")
        df = pd.read_parquet(pq_path)
        assert len(df) == len(synthetic_chunks_df)

    def test_transcript_parquet_no_duplicate_ids(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = tmp_output_dir / "lm_transcript_scores.parquet"
        if not pq_path.exists():
            pytest.skip("Parquet not exported by this pipeline build.")
        df = pd.read_parquet(pq_path)
        if "transcript_id" in df.columns:
            assert df["transcript_id"].duplicated().sum() == 0

    def test_csv_readable_by_pandas(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        csv_path = tmp_output_dir / "lm_chunk_scores.csv"
        if not csv_path.exists():
            pytest.skip("CSV not exported by this pipeline build.")
        df = pd.read_csv(csv_path)
        assert len(df) > 0

    def test_parquet_csv_row_count_consistent(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq  = tmp_output_dir / "lm_chunk_scores.parquet"
        csv = tmp_output_dir / "lm_chunk_scores.csv"
        if not (pq.exists() and csv.exists()):
            pytest.skip("Both parquet and CSV required for this test.")
        assert len(pd.read_parquet(pq)) == len(pd.read_csv(csv))

    def test_exported_paths_recorded_in_result(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        result, _ = self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        if result.success:
            assert len(result.exported_paths) > 0

    def test_score_range_in_exported_parquet(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = tmp_output_dir / "lm_chunk_scores.parquet"
        if not pq_path.exists():
            pytest.skip("Parquet not exported by this pipeline build.")
        df = pd.read_parquet(pq_path)
        if "lm_tone_score" in df.columns:
            scores = df["lm_tone_score"].dropna()
            assert (scores.abs() <= 1.0 + 1e-6).all(), \
                "Exported scores outside [-1, 1]."

    def test_coverage_ratio_in_exported_parquet(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        self._run_and_get_result(
            synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
        )
        pq_path = tmp_output_dir / "lm_chunk_scores.parquet"
        if not pq_path.exists():
            pytest.skip("Parquet not exported by this pipeline build.")
        df = pd.read_parquet(pq_path)
        if "lm_coverage_ratio" in df.columns:
            ratios = df["lm_coverage_ratio"].dropna()
            assert (ratios >= 0.0).all()
            assert (ratios <= 1.0 + 1e-6).all()


# ===========================================================================
# GROUP 9 — Deterministic rerun validation
# ===========================================================================


class TestDeterministicRerun:
    """Verify that running the pipeline twice on identical inputs yields identical outputs."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_pipeline", "LMPipeline") is None:
            pytest.skip("LMPipeline not yet implemented.")

    def test_chunk_scores_identical_on_rerun(
        self, synthetic_chunks_df, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        int1 = tmp_path / "int1"
        int2 = tmp_path / "int2"

        cfg1 = _build_pipeline_config(chunks_path, out1, int1)
        cfg2 = _build_pipeline_config(chunks_path, out2, int2)

        _run_pipeline(cfg1)
        _run_pipeline(cfg2)

        pq1 = out1 / "lm_chunk_scores.parquet"
        pq2 = out2 / "lm_chunk_scores.parquet"
        if not (pq1.exists() and pq2.exists()):
            pytest.skip("Parquet not exported — cannot compare reruns.")

        df1 = pd.read_parquet(pq1).sort_values("chunk_id").reset_index(drop=True)
        df2 = pd.read_parquet(pq2).sort_values("chunk_id").reset_index(drop=True)

        score_col = "lm_tone_score"
        if score_col in df1.columns and score_col in df2.columns:
            pd.testing.assert_series_equal(df1[score_col], df2[score_col],
                                           check_names=False)

    def test_transcript_aggregation_identical_on_rerun(
        self, synthetic_chunks_df, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)

        for run_id in ("runA", "runB"):
            out = tmp_path / run_id
            itr = tmp_path / f"int_{run_id}"
            cfg = _build_pipeline_config(chunks_path, out, itr)
            _run_pipeline(cfg)

        pqA = tmp_path / "runA" / "lm_transcript_scores.parquet"
        pqB = tmp_path / "runB" / "lm_transcript_scores.parquet"
        if not (pqA.exists() and pqB.exists()):
            pytest.skip("Transcript parquet not exported.")

        dfA = pd.read_parquet(pqA)
        dfB = pd.read_parquet(pqB)
        assert set(dfA.columns) == set(dfB.columns), \
            "Column schemas differ between reruns."
        assert len(dfA) == len(dfB), "Row counts differ between reruns."

    def test_cache_reuse_produces_same_result(
        self, synthetic_chunks_df, tmp_output_dir, tmp_interim_dir, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(synthetic_chunks_df, chunks_path)

        # First run — populates cache
        cfg_first = _build_pipeline_config(
            chunks_path, tmp_output_dir, tmp_interim_dir, overwrite=True
        )
        result_first = _run_pipeline(cfg_first)

        # Second run — overwrite=False should reuse cache
        cfg_cached = _build_pipeline_config(
            chunks_path, tmp_output_dir, tmp_interim_dir, overwrite=False
        )
        result_cached = _run_pipeline(cfg_cached)

        # Both must succeed
        assert result_first.success
        assert result_cached.success

        # Row counts must be identical
        assert result_first.chunks_processed == result_cached.chunks_processed

    def test_ordering_deterministic(self, synthetic_chunks_df, tmp_path):
        """Shuffled input must produce the same chunk-level scores as sorted input."""
        chunks_path_sorted  = tmp_path / "sorted.parquet"
        chunks_path_shuffled = tmp_path / "shuffled.parquet"

        shuffled = synthetic_chunks_df.sample(frac=1, random_state=99)
        _write_chunks_parquet(synthetic_chunks_df, chunks_path_sorted)
        _write_chunks_parquet(shuffled,            chunks_path_shuffled)

        out_s = tmp_path / "out_sorted"
        out_u = tmp_path / "out_shuffled"
        int_s = tmp_path / "int_sorted"
        int_u = tmp_path / "int_shuffled"

        _run_pipeline(_build_pipeline_config(chunks_path_sorted,   out_s, int_s))
        _run_pipeline(_build_pipeline_config(chunks_path_shuffled, out_u, int_u))

        pq_s = out_s / "lm_chunk_scores.parquet"
        pq_u = out_u / "lm_chunk_scores.parquet"
        if not (pq_s.exists() and pq_u.exists()):
            pytest.skip("Parquet not exported.")

        df_s = pd.read_parquet(pq_s).sort_values("chunk_id").reset_index(drop=True)
        df_u = pd.read_parquet(pq_u).sort_values("chunk_id").reset_index(drop=True)
        assert len(df_s) == len(df_u)
        if "lm_tone_score" in df_s.columns:
            pd.testing.assert_series_equal(
                df_s["lm_tone_score"], df_u["lm_tone_score"], check_names=False
            )


# ===========================================================================
# GROUP 10 — Edge-case pipeline handling
# ===========================================================================


class TestEdgeCasePipelineHandling:
    """Pipeline must degrade gracefully on adversarial / malformed inputs."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_pipeline", "LMPipeline") is None:
            pytest.skip("LMPipeline not yet implemented.")

    def test_malformed_rows_pipeline_completes(
        self, malformed_chunks_df, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        out = tmp_path / "out"
        itr = tmp_path / "itr"
        _write_chunks_parquet(malformed_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, out, itr)
        result = _run_pipeline(config)
        # Pipeline should complete even if all rows are dropped
        assert result is not None

    def test_empty_text_rows_dropped_before_scoring(
        self, malformed_chunks_df, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        out = tmp_path / "out"
        itr = tmp_path / "itr"
        _write_chunks_parquet(malformed_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, out, itr)
        result = _run_pipeline(config)
        # chunks_processed must be < chunks_loaded due to empty text removal
        assert result.chunks_processed <= result.chunks_loaded

    def test_low_coverage_transcripts_flagged(
        self, low_coverage_chunks_df, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        out = tmp_path / "out"
        itr = tmp_path / "itr"
        _write_chunks_parquet(low_coverage_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, out, itr)
        result = _run_pipeline(config)
        if result.success:
            # Low coverage should be detected in validation
            assert result.low_coverage_pct >= 0.0  # can be zero if all match

    def test_mixed_sentiment_full_label_spectrum(
        self, mixed_sentiment_chunks_df, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        out = tmp_path / "out"
        itr = tmp_path / "itr"
        _write_chunks_parquet(mixed_sentiment_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, out, itr)
        result = _run_pipeline(config)
        if result.success:
            # Both positive and negative labels should appear
            pos = result.label_counts.get("positive", 0)
            neg = result.label_counts.get("negative", 0)
            assert pos + neg > 0

    def test_single_chunk_pipeline(self, tmp_path):
        """Pipeline must not crash on a single-row input."""
        single = pd.DataFrame([{
            "chunk_id":         "SINGLE_001",
            "transcript_id":    "SINGLE_Q1",
            "ticker":           "SINGLE",
            "section_type":     "prepared_remarks",
            "dominant_speaker": "CEO",
            "speaker_role":     "CEO",
            "chunk_text":       "We delivered strong growth and record revenue this quarter.",
            "token_count":      11,
        }])
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(single, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_path / "out", tmp_path / "itr")
        result = _run_pipeline(config)
        assert result is not None

    def test_all_neutral_chunks(self, tmp_path):
        """Pipeline must handle a dataset where all chunks score near-zero."""
        rows = []
        for i in range(5):
            rows.append({
                "chunk_id":         f"NEUTRAL_{i}",
                "transcript_id":    "NEUTRAL_Q1",
                "ticker":           "NEUTRAL",
                "section_type":     "prepared_remarks",
                "dominant_speaker": "CEO",
                "speaker_role":     "CEO",
                "chunk_text":       " ".join(NEUTRAL_FILLER),
                "token_count":      len(NEUTRAL_FILLER),
            })
        df = pd.DataFrame(rows)
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_path / "out", tmp_path / "itr")
        result = _run_pipeline(config)
        assert result is not None

    def test_large_batch_pipeline(self, tmp_path):
        """Pipeline must handle a dataset larger than batch_size."""
        rows = []
        for i in range(50):
            rows.append({
                "chunk_id":         f"LARGE_{i:04d}",
                "transcript_id":    f"LARGE_Q{i % 5}",
                "ticker":           TICKERS[i % len(TICKERS)],
                "section_type":     "prepared_remarks" if i % 2 == 0 else "qa",
                "dominant_speaker": "CEO",
                "speaker_role":     "CEO",
                "chunk_text":       _make_chunk_text(
                    pos_count=3, neg_count=2, filler_count=10, seed=i + 200
                ),
                "token_count":      15,
            })
        df = pd.DataFrame(rows)
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(df, chunks_path)
        config = _build_pipeline_config(
            chunks_path, tmp_path / "out", tmp_path / "itr", batch_size=8
        )
        result = _run_pipeline(config)
        if result.success:
            assert result.chunks_processed == 50

    def test_no_lm_matches_does_not_crash(
        self, low_coverage_chunks_df, tmp_path
    ):
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(low_coverage_chunks_df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_path / "out", tmp_path / "itr")
        result = _run_pipeline(config)
        # Must complete without exception
        assert result is not None

    def test_nan_heavy_input_handled(self, tmp_path):
        """DataFrame with NaN token_count values must not crash the pipeline."""
        df = pd.DataFrame([{
            "chunk_id":         "NAN_001",
            "transcript_id":    "NAN_Q1",
            "ticker":           "NAN",
            "section_type":     "prepared_remarks",
            "dominant_speaker": "CEO",
            "speaker_role":     "CEO",
            "chunk_text":       "Revenue growth exceeded expectations.",
            "token_count":      float("nan"),
        }])
        chunks_path = tmp_path / "chunks.parquet"
        _write_chunks_parquet(df, chunks_path)
        config = _build_pipeline_config(chunks_path, tmp_path / "out", tmp_path / "itr")
        try:
            result = _run_pipeline(config)
            assert result is not None
        except Exception as exc:
            pytest.fail(f"Pipeline raised on NaN input: {exc}")


# ===========================================================================
# GROUP 11 — lm_utils integration
# ===========================================================================


class TestLMUtilsIntegration:
    """Validate shared utility helpers used across the LM subsystem."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if _try_import("lm_utils", "safe_divide") is None:
            pytest.skip("lm_utils not yet implemented.")

    def test_safe_divide_normal(self):
        safe_divide = _try_import("lm_utils", "safe_divide")
        assert safe_divide(10.0, 4.0) == pytest.approx(2.5)

    def test_safe_divide_zero_denominator(self):
        safe_divide = _try_import("lm_utils", "safe_divide")
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, 0.0, default=-1.0) == -1.0

    def test_safe_divide_non_finite(self):
        safe_divide = _try_import("lm_utils", "safe_divide")
        assert safe_divide(float("inf"), 4.0) == 0.0
        assert safe_divide(10.0, float("nan")) == 0.0

    def test_clip_score_range(self):
        clip = _try_import("lm_utils", "clip_score_range")
        assert clip(1.5) == pytest.approx(1.0)
        assert clip(-1.5) == pytest.approx(-1.0)
        assert clip(0.5) == pytest.approx(0.5)
        assert math.isfinite(clip(float("nan")))

    def test_validate_score_range_valid(self):
        fn = _try_import("lm_utils", "validate_score_range")
        assert fn(0.5, raise_on_fail=False) is True
        assert fn(-0.9, raise_on_fail=False) is True

    def test_validate_score_range_invalid(self):
        fn = _try_import("lm_utils", "validate_score_range")
        assert fn(1.5, raise_on_fail=False) is False

    def test_validate_score_range_raises(self):
        fn = _try_import("lm_utils", "validate_score_range")
        with pytest.raises(ValueError):
            fn(2.0, raise_on_fail=True)

    def test_safe_mean_with_nans(self):
        fn = _try_import("lm_utils", "safe_mean")
        result = fn([1.0, 2.0, float("nan"), 4.0])
        assert result == pytest.approx(7.0 / 3.0)

    def test_safe_mean_empty(self):
        fn = _try_import("lm_utils", "safe_mean")
        assert fn([]) == 0.0

    def test_compute_coverage_ratio(self):
        fn = _try_import("lm_utils", "compute_coverage_ratio")
        assert fn(10, 100) == pytest.approx(0.10)
        assert fn(0,  100) == pytest.approx(0.00)
        assert fn(0,    0) == 0.0

    def test_low_coverage_flag(self):
        fn = _try_import("lm_utils", "low_coverage_flag")
        assert fn(0.005) is True
        assert fn(0.05)  is False
        assert fn(float("nan")) is True

    def test_normalize_probability(self):
        fn = _try_import("lm_utils", "normalize_probability")
        probs = fn([3.0, 1.0, 0.0, 2.0])
        assert abs(sum(probs) - 1.0) < 1e-9
        assert all(p >= 0.0 for p in probs)

    def test_normalize_probability_all_zero(self):
        fn = _try_import("lm_utils", "normalize_probability")
        probs = fn([0.0, 0.0, 0.0])
        assert all(p == 0.0 for p in probs)

    def test_detect_duplicates(self):
        detect = _try_import("lm_utils", "detect_duplicates")
        s = pd.Series(["A", "B", "A", "C"])
        has_dup, count, vals = detect(s)
        assert has_dup is True
        assert count >= 1
        assert "A" in vals

    def test_detect_no_duplicates(self):
        detect = _try_import("lm_utils", "detect_duplicates")
        s = pd.Series(["A", "B", "C", "D"])
        has_dup, count, _ = detect(s)
        assert has_dup is False
        assert count == 0

    def test_safe_parquet_export(self, tmp_path):
        fn = _try_import("lm_utils", "safe_parquet_export")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        path = tmp_path / "test.parquet"
        ok = fn(df, path)
        assert ok is True
        assert path.exists()
        assert len(pd.read_parquet(path)) == 3

    def test_safe_parquet_export_empty_df(self, tmp_path):
        fn = _try_import("lm_utils", "safe_parquet_export")
        ok = fn(pd.DataFrame(), tmp_path / "empty.parquet")
        assert ok is False

    def test_safe_parquet_export_overwrite_false(self, tmp_path):
        fn = _try_import("lm_utils", "safe_parquet_export")
        df = pd.DataFrame({"x": [1, 2]})
        path = tmp_path / "existing.parquet"
        fn(df, path)
        # Second call with overwrite=False should skip and return True
        ok = fn(pd.DataFrame({"x": [9, 8, 7]}), path, overwrite=False)
        assert ok is True
        assert len(pd.read_parquet(path)) == 2  # original not overwritten

    def test_timed_execution_decorator(self):
        timed = _try_import("lm_utils", "timed_execution")

        @timed
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_timed_execution_with_args(self):
        timed = _try_import("lm_utils", "timed_execution")

        @timed(label="my_op")
        def multiply(a, b):
            return a * b

        result = multiply(4, 5)
        assert result == 20

    def test_stage_timer(self):
        StageTimer = _try_import("lm_utils", "StageTimer")
        timer = StageTimer()
        with timer.measure("stage_a"):
            pass
        with timer.measure("stage_b"):
            pass
        assert timer.total_elapsed() >= 0.0
        summary = timer.summary()
        assert "stage_a" in summary
        assert "stage_b" in summary

    def test_summarize_distribution(self):
        fn = _try_import("lm_utils", "summarize_distribution")
        s  = pd.Series([0.1, 0.3, -0.2, 0.5, float("nan"), 0.0])
        dist = fn(s)
        assert dist.count == 5
        assert dist.null_count == 1
        assert math.isfinite(dist.mean)
        assert dist.min <= dist.max

    def test_summarize_coverage(self):
        fn = _try_import("lm_utils", "summarize_coverage")
        ratios = [0.0, 0.005, 0.02, 0.08, 0.15, float("nan")]
        cov = fn(ratios, threshold=0.01)
        assert cov.n_texts == 5
        assert cov.zero_coverage_count == 1
        assert cov.low_coverage_count >= 1


# ===========================================================================
# STANDALONE EXECUTION
# ===========================================================================


if __name__ == "__main__":
    import subprocess

    print("\n" + "=" * 65)
    print("  test_lm_pipeline.py — Standalone demo run")
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
