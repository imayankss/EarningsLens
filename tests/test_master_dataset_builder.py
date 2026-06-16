from pathlib import Path

import pandas as pd
import pytest

from src.analysis.master_dataset_builder import (
    aggregate_market_data,
    build_master_dataset,
    run_master_dataset_builder,
)


def _transcripts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "earnings_date": "2020-10-29",
                "fiscal_quarter": "2020Q4",
                "year": 2020,
                "company_name": "Apple Inc.",
                "word_count_clean": 8500,
            },
            {
                "transcript_id": "MSFT_20210126",
                "ticker": "MSFT",
                "earnings_date": "2021-01-26",
                "fiscal_quarter": "2021Q2",
                "year": 2021,
                "company_name": "Microsoft Corp.",
                "word_count_clean": 7900,
            },
        ]
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
            }
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
            }
        ]
    )


def _event_study() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "event_date": "2020-10-30",
                "return_1d": 0.012,
                "ar_1d": 0.004,
                "return_2d": 0.020,
                "ar_2d": 0.010,
                "return_3d": 0.030,
                "ar_3d": 0.020,
                "return_5d": 0.040,
                "ar_5d": 0.030,
                "return_10d": 0.050,
                "ar_10d": 0.040,
            }
        ]
    )


def _market_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "date": pd.Timestamp("2020-10-28"),
                "daily_return": 0.01,
                "abnormal_return": 0.004,
                "volume": 100,
                "low": 95.0,
                "high": 101.0,
            },
            {
                "transcript_id": "AAPL_20201029",
                "date": pd.Timestamp("2020-10-30"),
                "daily_return": 0.03,
                "abnormal_return": 0.008,
                "volume": 200,
                "low": 97.0,
                "high": 104.0,
            },
        ]
    )


def _comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "earnings_date": "2020-10-29",
                "lm_positive_count_x": 18,
                "lm_negative_count_x": 5,
                "lm_positive_count_y": 999,
                "lm_negative_count_y": 999,
                "finbert_label": "positive",
                "lm_label_normalized": "positive",
                "sentiment_label_agree": True,
                "sentiment_score_gap": 0.11,
            }
        ]
    )


def test_aggregate_market_data_creates_one_row_per_transcript() -> None:
    aggregated = aggregate_market_data(_market_data())

    assert aggregated.shape[0] == 1
    row = aggregated.iloc[0]
    assert row["market_row_count"] == 2
    assert row["market_window_start"] == pd.Timestamp("2020-10-28")
    assert row["market_window_end"] == pd.Timestamp("2020-10-30")
    assert row["mean_return"] == pytest.approx(0.02)
    assert row["mean_abnormal_return"] == pytest.approx(0.006)
    assert row["mean_volume"] == pytest.approx(150)
    assert row["price_min"] == pytest.approx(95)
    assert row["price_max"] == pytest.approx(104)


def test_build_master_dataset_left_joins_inputs_and_avoids_suffix_columns() -> None:
    master = build_master_dataset(
        _transcripts(),
        _sentiment_features(),
        _lm_scores(),
        _event_study(),
        _market_data(),
        _comparison(),
    )

    assert list(master["transcript_id"]) == ["AAPL_20201029", "MSFT_20210126"]
    assert not any(column.endswith("_x") or column.endswith("_y") for column in master.columns)

    aapl = master.loc[master["transcript_id"] == "AAPL_20201029"].iloc[0]
    assert aapl["ticker"] == "AAPL"
    assert aapl["company_name"] == "Apple Inc."
    assert aapl["finbert_score"] == pytest.approx(0.42)
    assert aapl["lm_positive_count"] == 18
    assert aapl["return_10d"] == pytest.approx(0.05)
    assert aapl["market_row_count"] == 2
    assert aapl["finbert_direction"] == "positive"
    assert aapl["lm_direction"] == "positive"
    assert aapl["directional_agreement"]
    assert aapl["score_difference"] == pytest.approx(0.11)
    assert aapl["absolute_difference"] == pytest.approx(0.11)
    assert not aapl["divergence_flag"]

    msft = master.loc[master["transcript_id"] == "MSFT_20210126"].iloc[0]
    assert pd.isna(msft["finbert_score"])
    assert pd.isna(msft["market_row_count"])


def test_build_master_dataset_rejects_duplicate_base_transcript_ids() -> None:
    transcripts = pd.concat([_transcripts(), _transcripts().head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate transcript_id"):
        build_master_dataset(
            transcripts,
            _sentiment_features(),
            _lm_scores(),
            _event_study(),
            _market_data(),
            _comparison(),
        )


def test_single_row_master_warns() -> None:
    with pytest.warns(RuntimeWarning, match="row_count is only 1"):
        build_master_dataset(
            _transcripts().head(1),
            _sentiment_features(),
            _lm_scores(),
            _event_study(),
            _market_data(),
            _comparison(),
        )


def test_run_master_dataset_builder_writes_only_requested_outputs_under_tmp_path(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    transcripts_path = input_dir / "transcripts_cleaned.parquet"
    sentiment_path = input_dir / "sentiment_features.parquet"
    lm_path = input_dir / "lm_scores.parquet"
    event_path = input_dir / "event_study.parquet"
    market_path = input_dir / "market_data.parquet"
    comparison_path = input_dir / "sentiment_model_comparison.parquet"

    _transcripts().to_parquet(transcripts_path, index=False)
    _sentiment_features().to_parquet(sentiment_path, index=False)
    _lm_scores().to_parquet(lm_path, index=False)
    _event_study().to_parquet(event_path, index=False)
    _market_data().to_parquet(market_path, index=False)
    _comparison().to_parquet(comparison_path, index=False)

    master, summary, output_paths = run_master_dataset_builder(
        transcripts_cleaned_path=transcripts_path,
        sentiment_features_path=sentiment_path,
        lm_scores_path=lm_path,
        event_study_path=event_path,
        market_data_path=market_path,
        comparison_path=comparison_path,
        master_parquet_path=output_dir / "data/processed/master_dataset.parquet",
        master_csv_path=output_dir / "data/processed/master_dataset.csv",
        summary_csv_path=output_dir / "reports/tables/master_dataset_summary.csv",
    )

    assert len(master) == 2
    assert summary.loc[summary["metric"] == "row_count", "value"].iloc[0] == 2.0
    assert all(path.exists() for path in output_paths)
    assert pd.read_parquet(output_paths[0]).shape[0] == 2
    assert pd.read_csv(output_paths[1]).shape[0] == 2
    assert "column_count" in set(pd.read_csv(output_paths[2])["metric"])
