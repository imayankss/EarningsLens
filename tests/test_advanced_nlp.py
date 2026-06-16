"""
tests/test_advanced_nlp.py

Test suite for Day 14 — Advanced NLP Features.

Rules:
  * All tests use tmp_path only — zero reads from the real project tree.
  * Synthetic DataFrames are built inline or via fixtures.
  * Missing optional files must not crash; tests verify graceful degradation.
  * n=1 single-transcript dataset must produce correct, non-empty outputs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.nlp.advanced_nlp import (
    AdvancedNLPPipeline,
    _TranscriptTokens,
    _bigrams,
    _remove_stopwords,
    _tokenize,
    _TOPIC_LEXICON,
    _UNCERTAINTY_WORDS,
    build_features_summary,
    compute_speaker_sentiment,
    compute_topic_frequency,
    compute_uncertainty,
    extract_bigrams,
    extract_keywords,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = (
    "revenue grew significantly this quarter our guidance outlook remains "
    "strong we may face some risk uncertainty in the next period margin "
    "profitability improved cash flow was positive cost efficiency savings "
    "customer product demand accelerating expected approximately volatile"
)


def _make_records(text: str = _SAMPLE_TEXT, tid: str = "AAPL_Q1_2025") -> list[_TranscriptTokens]:
    tokens  = _tokenize(text)
    content = _remove_stopwords(tokens)
    return [_TranscriptTokens(transcript_id=tid, ticker="AAPL", tokens=tokens, content_tokens=content)]


@pytest.fixture()
def sample_records() -> list[_TranscriptTokens]:
    return _make_records()


@pytest.fixture()
def transcripts_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "transcript_id":       ["AAPL_Q1_2025"],
        "ticker":              ["AAPL"],
        "transcript_text_clean": [_SAMPLE_TEXT],
    })
    p = tmp_path / "transcripts_cleaned.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture()
def chunks_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "chunk_id":        ["c1", "c2"],
        "transcript_id":   ["AAPL_Q1_2025", "AAPL_Q1_2025"],
        "chunk_text":      [
            "revenue grew this quarter guidance outlook positive",
            "may face risk uncertainty margin improved cash",
        ],
        "section_type":    ["prepared_remarks", "qa"],
        "dominant_speaker": ["CEO", "Analyst"],
        "chunk_order":     [0, 1],
        "token_count":     [7, 7],
        "ticker":          ["AAPL", "AAPL"],
    })
    p = tmp_path / "chunks.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture()
def finbert_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "chunk_id":        ["c1", "c2", "c3"],
        "transcript_id":   ["AAPL_Q1_2025"] * 3,
        "sentiment_score": [0.45, -0.12, 0.08],
        "predicted_label": ["positive", "negative", "neutral"],
        "confidence":      [0.82, 0.65, 0.51],
        "section_type":    ["prepared_remarks", "qa", "prepared_remarks"],
        "dominant_speaker": ["CEO", "Analyst", "CEO"],
    })
    p = tmp_path / "finbert_chunk_scores.parquet"
    df.to_parquet(p, index=False)
    return p


def _pipeline(
    tmp_path: Path,
    *,
    transcripts: Path | None = None,
    chunks: Path | None = None,
    finbert: Path | None = None,
) -> AdvancedNLPPipeline:
    """Convenience factory that points all missing paths to nonexistent files."""
    def _ghost(name: str) -> Path:
        return tmp_path / name

    return AdvancedNLPPipeline(
        transcripts_path=transcripts or _ghost("no_transcripts.parquet"),
        chunks_path=chunks          or _ghost("no_chunks.parquet"),
        finbert_path=finbert        or _ghost("no_finbert.parquet"),
        lm_path=_ghost("no_lm.parquet"),
        master_path=_ghost("no_master.parquet"),
        nlp_out_dir=tmp_path / "nlp",
        tables_out_dir=tmp_path / "tables",
    )


# ---------------------------------------------------------------------------
# Low-level tokenisation helpers
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_lowercase(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_digits(self):
        tokens = _tokenize("revenue 123 grew")
        assert "123" not in tokens

    def test_min_length_two(self):
        tokens = _tokenize("a an revenue")
        # single-char tokens filtered by \b[a-z]{2,}\b
        assert all(len(t) >= 2 for t in tokens)

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_whitespace_only(self):
        assert _tokenize("   ") == []

    def test_non_string(self):
        assert _tokenize(None) == []  # type: ignore[arg-type]


class TestRemoveStopwords:
    def test_removes_the(self):
        assert "the" not in _remove_stopwords(["the", "revenue"])

    def test_keeps_financial_terms(self):
        result = _remove_stopwords(["revenue", "margin", "guidance"])
        assert result == ["revenue", "margin", "guidance"]

    def test_empty_input(self):
        assert _remove_stopwords([]) == []


class TestBigrams:
    def test_two_pairs(self):
        result = _bigrams(["a", "b", "c"])
        assert ("a", "b") in result
        assert ("b", "c") in result

    def test_single_token(self):
        assert _bigrams(["only"]) == []

    def test_empty(self):
        assert _bigrams([]) == []


# ---------------------------------------------------------------------------
# 1. Keyword extraction
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_returns_dataframe(self, sample_records):
        df = extract_keywords(sample_records)
        assert isinstance(df, pd.DataFrame)

    def test_not_empty(self, sample_records):
        assert not extract_keywords(sample_records).empty

    def test_required_columns(self, sample_records):
        df = extract_keywords(sample_records)
        assert {"transcript_id", "ticker", "keyword", "count", "frequency"}.issubset(df.columns)

    def test_transcript_id_preserved(self, sample_records):
        df = extract_keywords(sample_records)
        assert "AAPL_Q1_2025" in df["transcript_id"].values

    def test_frequency_bounds(self, sample_records):
        df = extract_keywords(sample_records)
        assert (df["frequency"] >= 0).all()
        assert (df["frequency"] <= 1).all()

    def test_count_positive(self, sample_records):
        df = extract_keywords(sample_records)
        assert (df["count"] > 0).all()

    def test_top_n_respected(self, sample_records):
        df = extract_keywords(sample_records, top_n=5)
        # ≤ 5 rows per transcript
        per_tid = df.groupby("transcript_id").size()
        assert (per_tid <= 5).all()

    def test_empty_tokens_warns(self):
        rec = _TranscriptTokens("T1", "X", [], [])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extract_keywords([rec])
        assert result.empty
        assert len(w) >= 1

    def test_stopwords_absent(self, sample_records):
        df = extract_keywords(sample_records)
        stopwords = {"the", "and", "of", "is", "in", "to", "a", "an"}
        assert not set(df["keyword"].tolist()).intersection(stopwords)


# ---------------------------------------------------------------------------
# 2. Uncertainty scoring
# ---------------------------------------------------------------------------

class TestComputeUncertainty:
    def test_returns_dataframe(self, sample_records):
        assert isinstance(compute_uncertainty(sample_records), pd.DataFrame)

    def test_required_columns(self, sample_records):
        df = compute_uncertainty(sample_records)
        assert {"transcript_id", "ticker", "total_tokens",
                "uncertainty_count", "uncertainty_ratio"}.issubset(df.columns)

    def test_one_row_per_transcript(self, sample_records):
        df = compute_uncertainty(sample_records)
        assert len(df) == len(sample_records)

    def test_ratio_in_unit_interval(self, sample_records):
        df = compute_uncertainty(sample_records)
        assert (df["uncertainty_ratio"] >= 0).all()
        assert (df["uncertainty_ratio"] <= 1).all()

    def test_count_nonneg(self, sample_records):
        df = compute_uncertainty(sample_records)
        assert (df["uncertainty_count"] >= 0).all()

    def test_detects_uncertainty_words(self):
        text   = "may might risk uncertain approximately"
        tokens = _tokenize(text)
        rec    = _TranscriptTokens("T1", "X", tokens, _remove_stopwords(tokens))
        df     = compute_uncertainty([rec])
        # "may", "might", "risk", "uncertain", "approximately" all in _UNCERTAINTY_WORDS
        assert df["uncertainty_count"].iloc[0] >= 4

    def test_zero_on_clean_text(self):
        text   = "revenue grew profit increased sales higher"
        tokens = _tokenize(text)
        rec    = _TranscriptTokens("T1", "X", tokens, _remove_stopwords(tokens))
        df     = compute_uncertainty([rec])
        assert df["uncertainty_count"].iloc[0] == 0

    def test_n1_single_transcript(self, sample_records):
        df = compute_uncertainty(sample_records)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# 3. Topic frequency
# ---------------------------------------------------------------------------

class TestComputeTopicFrequency:
    def test_returns_dataframe(self, sample_records):
        assert isinstance(compute_topic_frequency(sample_records), pd.DataFrame)

    def test_required_columns(self, sample_records):
        df = compute_topic_frequency(sample_records)
        assert {"transcript_id", "ticker", "topic", "count", "ratio"}.issubset(df.columns)

    def test_all_topics_present(self, sample_records):
        df     = compute_topic_frequency(sample_records)
        topics = set(df["topic"].unique())
        for t in _TOPIC_LEXICON:
            assert t in topics, f"Missing topic: {t}"

    def test_count_nonneg(self, sample_records):
        df = compute_topic_frequency(sample_records)
        assert (df["count"] >= 0).all()

    def test_ratio_nonneg(self, sample_records):
        df = compute_topic_frequency(sample_records)
        assert (df["ratio"] >= 0).all()

    def test_revenue_detected(self, sample_records):
        df  = compute_topic_frequency(sample_records)
        rev = df[df["topic"] == "revenue_growth"]
        assert not rev.empty
        assert rev["count"].iloc[0] > 0

    def test_rows_equal_topics_times_transcripts(self, sample_records):
        df = compute_topic_frequency(sample_records)
        expected = len(_TOPIC_LEXICON) * len(sample_records)
        assert len(df) == expected


# ---------------------------------------------------------------------------
# 4. Bigram analysis
# ---------------------------------------------------------------------------

class TestExtractBigrams:
    def test_returns_dataframe(self, sample_records):
        assert isinstance(extract_bigrams(sample_records), pd.DataFrame)

    def test_not_empty(self, sample_records):
        assert not extract_bigrams(sample_records).empty

    def test_required_columns(self, sample_records):
        df = extract_bigrams(sample_records)
        assert {"transcript_id", "ticker", "bigram", "count"}.issubset(df.columns)

    def test_bigrams_are_two_words(self, sample_records):
        df = extract_bigrams(sample_records)
        for bg in df["bigram"]:
            assert len(bg.split()) == 2, f"Not a bigram: {bg!r}"

    def test_count_positive(self, sample_records):
        df = extract_bigrams(sample_records)
        assert (df["count"] > 0).all()

    def test_top_n_respected(self, sample_records):
        df = extract_bigrams(sample_records, top_n=5)
        per_tid = df.groupby("transcript_id").size()
        assert (per_tid <= 5).all()

    def test_few_tokens_warns(self):
        rec = _TranscriptTokens("T1", "X", ["revenue"], ["revenue"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = extract_bigrams([rec])
        assert df.empty
        assert len(w) >= 1


# ---------------------------------------------------------------------------
# 5. Speaker / section sentiment
# ---------------------------------------------------------------------------

class TestComputeSpeakerSentiment:
    def test_none_returns_empty_with_issue(self):
        df, issues = compute_speaker_sentiment(None)
        assert df.empty
        assert len(issues) >= 1

    def test_empty_df_returns_empty_with_issue(self):
        df, issues = compute_speaker_sentiment(pd.DataFrame())
        assert df.empty
        assert len(issues) >= 1

    def test_no_speaker_col_returns_empty_with_issue(self):
        finbert = pd.DataFrame({
            "transcript_id":  ["T1"],
            "sentiment_score": [0.5],
        })
        df, issues = compute_speaker_sentiment(finbert)
        assert df.empty
        assert any("speaker" in i.lower() or "section" in i.lower() for i in issues)

    def test_no_score_col_returns_empty_with_issue(self):
        finbert = pd.DataFrame({
            "transcript_id":  ["T1"],
            "dominant_speaker": ["CEO"],
        })
        df, issues = compute_speaker_sentiment(finbert)
        assert df.empty
        assert any("score" in i.lower() or "sentiment" in i.lower() for i in issues)

    def test_valid_finbert_returns_dataframe(self, finbert_parquet):
        finbert = pd.read_parquet(finbert_parquet)
        df, issues = compute_speaker_sentiment(finbert)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_groups_by_dominant_speaker(self, finbert_parquet):
        finbert = pd.read_parquet(finbert_parquet)
        df, _   = compute_speaker_sentiment(finbert)
        assert "dominant_speaker" in df.columns

    def test_avg_sentiment_score_present(self, finbert_parquet):
        finbert = pd.read_parquet(finbert_parquet)
        df, _   = compute_speaker_sentiment(finbert)
        assert "avg_sentiment_score" in df.columns

    def test_chunk_count_correct(self, finbert_parquet):
        finbert = pd.read_parquet(finbert_parquet)
        df, _   = compute_speaker_sentiment(finbert)
        # CEO has 2 chunks (c1, c3), Analyst has 1 chunk
        totals  = df["chunk_count"].sum()
        assert totals == len(finbert)

    def test_positive_negative_neutral_counts(self, finbert_parquet):
        finbert = pd.read_parquet(finbert_parquet)
        df, _   = compute_speaker_sentiment(finbert)
        if "positive_count" in df.columns:
            totals = df[["positive_count", "negative_count", "neutral_count"]].sum().sum()
            assert totals == len(finbert)

    def test_section_type_fallback(self):
        finbert = pd.DataFrame({
            "transcript_id":  ["T1", "T1"],
            "section_type":   ["prepared_remarks", "qa"],
            "sentiment_score": [0.3, -0.1],
            "predicted_label": ["positive", "negative"],
        })
        df, issues = compute_speaker_sentiment(finbert)
        assert not df.empty
        assert "section_type" in df.columns


# ---------------------------------------------------------------------------
# Features summary (wide per-transcript)
# ---------------------------------------------------------------------------

class TestBuildFeaturesSummary:
    def test_returns_wide_dataframe(self, sample_records):
        unc   = compute_uncertainty(sample_records)
        topic = compute_topic_frequency(sample_records)
        kw    = extract_keywords(sample_records)
        df    = build_features_summary(unc, topic, kw)
        assert not df.empty

    def test_uncertainty_cols_present(self, sample_records):
        unc   = compute_uncertainty(sample_records)
        topic = compute_topic_frequency(sample_records)
        kw    = extract_keywords(sample_records)
        df    = build_features_summary(unc, topic, kw)
        assert "uncertainty_ratio" in df.columns
        assert "uncertainty_count" in df.columns

    def test_topic_cols_present(self, sample_records):
        unc   = compute_uncertainty(sample_records)
        topic = compute_topic_frequency(sample_records)
        kw    = extract_keywords(sample_records)
        df    = build_features_summary(unc, topic, kw)
        for t in _TOPIC_LEXICON:
            assert f"topic_{t}" in df.columns, f"Missing column topic_{t}"

    def test_top5_keywords_col(self, sample_records):
        unc   = compute_uncertainty(sample_records)
        topic = compute_topic_frequency(sample_records)
        kw    = extract_keywords(sample_records)
        df    = build_features_summary(unc, topic, kw)
        assert "top_5_keywords" in df.columns
        assert isinstance(df["top_5_keywords"].iloc[0], str)
        assert len(df["top_5_keywords"].iloc[0]) > 0

    def test_top5_has_at_most_5(self, sample_records):
        unc   = compute_uncertainty(sample_records)
        topic = compute_topic_frequency(sample_records)
        kw    = extract_keywords(sample_records)
        df    = build_features_summary(unc, topic, kw)
        for val in df["top_5_keywords"]:
            assert len(val.split(",")) <= 5

    def test_empty_uncertainty_returns_empty(self):
        df = build_features_summary(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert df.empty

    def test_one_row_per_transcript_n1(self, sample_records):
        unc   = compute_uncertainty(sample_records)
        topic = compute_topic_frequency(sample_records)
        kw    = extract_keywords(sample_records)
        df    = build_features_summary(unc, topic, kw)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Pipeline integration — end-to-end
# ---------------------------------------------------------------------------

class TestAdvancedNLPPipeline:
    def test_returns_dict_with_transcripts(self, tmp_path, transcripts_parquet):
        p = _pipeline(tmp_path, transcripts=transcripts_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outputs = p.run()
        assert isinstance(outputs, dict)
        assert len(outputs) > 0

    def test_all_output_keys_present(self, tmp_path, transcripts_parquet):
        p = _pipeline(tmp_path, transcripts=transcripts_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outputs = p.run()
        expected = {
            "features", "keyword_summary", "topic_frequency",
            "bigram_summary", "uncertainty_summary", "speaker_sentiment_summary",
        }
        assert expected.issubset(outputs.keys())

    def test_creates_parquet_file(self, tmp_path, transcripts_parquet):
        p = _pipeline(tmp_path, transcripts=transcripts_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p.run()
        assert (tmp_path / "nlp" / "advanced_nlp_features.parquet").exists()

    def test_creates_csv_files(self, tmp_path, transcripts_parquet):
        p = _pipeline(tmp_path, transcripts=transcripts_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p.run()
        tables = tmp_path / "tables"
        for name in [
            "keyword_summary.csv", "topic_frequency.csv", "bigram_summary.csv",
            "uncertainty_summary.csv", "speaker_sentiment_summary.csv",
        ]:
            assert (tables / name).exists(), f"Missing: {name}"
        assert (tmp_path / "nlp" / "advanced_nlp_features.csv").exists()

    def test_no_crash_all_inputs_missing(self, tmp_path):
        p = _pipeline(tmp_path)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outputs = p.run()
        assert isinstance(outputs, dict)

    def test_chunks_fallback(self, tmp_path, chunks_parquet):
        """chunks.parquet used when transcripts not available."""
        p = _pipeline(tmp_path, chunks=chunks_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outputs = p.run()
        assert not outputs["keyword_summary"].empty

    def test_n1_single_transcript(self, tmp_path, transcripts_parquet):
        p = _pipeline(tmp_path, transcripts=transcripts_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outputs = p.run()
        assert len(outputs["uncertainty_summary"]) == 1

    def test_with_finbert_speaker_summary_populated(
        self, tmp_path, transcripts_parquet, finbert_parquet
    ):
        p = _pipeline(tmp_path, transcripts=transcripts_parquet, finbert=finbert_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outputs = p.run()
        spk = outputs["speaker_sentiment_summary"]
        assert not spk.empty
        assert "avg_sentiment_score" in spk.columns

    def test_features_parquet_readable(self, tmp_path, transcripts_parquet):
        p = _pipeline(tmp_path, transcripts=transcripts_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p.run()
        parquet_path = tmp_path / "nlp" / "advanced_nlp_features.parquet"
        loaded = pd.read_parquet(parquet_path)
        assert not loaded.empty

    def test_keyword_csv_readable(self, tmp_path, transcripts_parquet):
        p = _pipeline(tmp_path, transcripts=transcripts_parquet)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p.run()
        loaded = pd.read_csv(tmp_path / "tables" / "keyword_summary.csv")
        assert not loaded.empty

    def test_multiple_transcripts(self, tmp_path):
        df = pd.DataFrame({
            "transcript_id":       ["AAPL_Q1_2025", "MSFT_Q2_2025"],
            "ticker":              ["AAPL", "MSFT"],
            "transcript_text_clean": [
                "revenue grew guidance positive outlook strong",
                "may face risk uncertainty volatile market challenging",
            ],
        })
        path = tmp_path / "transcripts_cleaned.parquet"
        df.to_parquet(path, index=False)

        p = _pipeline(tmp_path, transcripts=path)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outputs = p.run()

        assert len(outputs["uncertainty_summary"]) == 2

    def test_outputs_deterministic(self, tmp_path, transcripts_parquet):
        """Same run twice must produce identical keyword DataFrames."""
        p1 = _pipeline(tmp_path / "r1", transcripts=transcripts_parquet)
        p1.nlp_out_dir    = tmp_path / "r1" / "nlp"
        p1.tables_out_dir = tmp_path / "r1" / "tables"

        p2 = _pipeline(tmp_path / "r2", transcripts=transcripts_parquet)
        p2.nlp_out_dir    = tmp_path / "r2" / "nlp"
        p2.tables_out_dir = tmp_path / "r2" / "tables"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            o1 = p1.run()
            o2 = p2.run()

        kw1 = o1["keyword_summary"].sort_values(["transcript_id", "keyword"]).reset_index(drop=True)
        kw2 = o2["keyword_summary"].sort_values(["transcript_id", "keyword"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(kw1, kw2)
