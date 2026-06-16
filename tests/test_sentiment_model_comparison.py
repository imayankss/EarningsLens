from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.sentiment_model_comparison import (
    build_sentiment_model_comparison,
    run_comparison,
    summarize_comparison,
)


def _sentiment_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "earnings_date": "2020-10-29",
                "fiscal_quarter": "2020Q4",
                "finbert_score": 0.42,
                "finbert_positive": 0.70,
                "finbert_negative": 0.10,
                "finbert_neutral": 0.20,
                "chunk_count": 12,
                "mean_confidence": 0.84,
            },
            {
                "transcript_id": "MSFT_20210126",
                "ticker": "MSFT",
                "earnings_date": "2021-01-26",
                "fiscal_quarter": "2021Q2",
                "finbert_score": -0.20,
                "finbert_positive": 0.18,
                "finbert_negative": 0.46,
                "finbert_neutral": 0.36,
                "chunk_count": 10,
                "mean_confidence": 0.79,
            },
        ]
    )


def _lm_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "earnings_date": "2020-10-29",
                "lm_positive_count": 18,
                "lm_negative_count": 5,
                "lm_tone_score": 0.31,
                "lm_tone": "positive",
                "lm_word_count": 200,
                "lm_total_words": 200,
                "lm_total_tokens": 240,
                "lm_label": "positive",
            },
            {
                "transcript_id": "MSFT_20210126",
                "ticker": "MSFT",
                "earnings_date": "2021-01-26",
                "lm_positive_count": 7,
                "lm_negative_count": 12,
                "lm_tone_score": -0.16,
                "lm_tone": "negative",
                "lm_word_count": 190,
                "lm_total_words": 190,
                "lm_total_tokens": 225,
                "lm_label": "negative",
            },
        ]
    )


def _event_study() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "event_date": "2020-10-29",
                "finbert_score": 999.0,
                "lm_tone": "duplicate",
                "lm_tone_score": 999.0,
                "return_0": 0.012,
                "ar_0": 0.004,
                "car_0_1": 0.018,
            },
            {
                "transcript_id": "MSFT_20210126",
                "ticker": "MSFT",
                "event_date": "2021-01-26",
                "finbert_score": 999.0,
                "lm_tone": "duplicate",
                "lm_tone_score": 999.0,
                "return_0": -0.006,
                "ar_0": -0.002,
                "car_0_1": -0.009,
            },
        ]
    )


def test_build_comparison_merges_real_schema_on_transcript_id() -> None:
    comparison = build_sentiment_model_comparison(
        _sentiment_features(),
        _lm_scores(),
        _event_study(),
    )

    assert list(comparison["transcript_id"]) == ["AAPL_20201029", "MSFT_20210126"]
    assert "return_0" in comparison.columns
    assert "ar_0" in comparison.columns
    assert "car_0_1" in comparison.columns
    assert "finbert_score" in comparison.columns
    assert "lm_tone_score" in comparison.columns
    assert "finbert_score_x" not in comparison.columns
    assert "lm_tone_score_x" not in comparison.columns

    aapl = comparison.loc[comparison["transcript_id"] == "AAPL_20201029"].iloc[0]
    assert aapl["ticker"] == "AAPL"
    assert aapl["event_date"] == "2020-10-29"
    assert aapl["finbert_label"] == "positive"
    assert aapl["lm_label_normalized"] == "positive"
    assert aapl["sentiment_label_agree"]


def test_run_comparison_writes_only_requested_outputs_under_tmp_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    sentiment_path = input_dir / "sentiment_features.parquet"
    lm_path = input_dir / "lm_scores.parquet"
    event_path = input_dir / "event_study.parquet"

    _sentiment_features().to_parquet(sentiment_path, index=False)
    _lm_scores().to_parquet(lm_path, index=False)
    _event_study().to_parquet(event_path, index=False)

    comparison, summary, output_paths = run_comparison(
        sentiment_features_path=sentiment_path,
        lm_scores_path=lm_path,
        event_study_path=event_path,
        analysis_parquet_path=output_dir / "data/processed/analysis/sentiment_model_comparison.parquet",
        analysis_csv_path=output_dir / "data/processed/analysis/sentiment_model_comparison.csv",
        report_comparison_csv_path=output_dir / "reports/tables/sentiment_model_comparison.csv",
        report_summary_csv_path=output_dir / "reports/tables/sentiment_model_comparison_summary.csv",
        print_report=False,
    )

    assert len(comparison) == 2
    assert summary.loc[summary["metric"] == "observation_count", "value"].iloc[0] == 2.0
    assert all(path.exists() for path in output_paths)
    assert pd.read_parquet(output_paths[0]).shape[0] == 2
    assert pd.read_csv(output_paths[1]).shape[0] == 2
    assert pd.read_csv(output_paths[2]).shape[0] == 2
    assert "finbert_lm_correlation" in set(pd.read_csv(output_paths[3])["metric"])


def test_single_observation_summary_warns_and_uses_nan_correlation() -> None:
    comparison = build_sentiment_model_comparison(
        _sentiment_features().head(1),
        _lm_scores().head(1),
        _event_study().head(1),
    )

    with pytest.warns(RuntimeWarning, match="requires at least 2 observations"):
        summary = summarize_comparison(comparison)

    correlation = summary.loc[summary["metric"] == "finbert_lm_correlation", "value"].iloc[0]
    assert np.isnan(correlation)
