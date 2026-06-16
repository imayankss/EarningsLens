from pathlib import Path

import pandas as pd

from src.visualization.day12_figures import FIGURE_FILENAMES, generate_day12_figures


def _comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "earnings_date": "2020-10-29",
                "fiscal_quarter": "2020Q4",
                "finbert_score": 0.42,
                "lm_tone_score": 0.31,
                "finbert_label": "positive",
                "lm_label_normalized": "positive",
                "sentiment_label_agree": True,
                "return_0": 0.012,
                "ar_0": 0.004,
                "car_0_1": 0.018,
            }
        ]
    )


def _event_study() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "event_date": "2020-10-29",
                "return_0": 0.012,
                "ar_-1": -0.003,
                "ar_0": 0.004,
                "ar_1": 0.007,
                "car_0_1": 0.018,
            }
        ]
    )


def test_generate_day12_figures_handles_single_event_without_trends(
    tmp_path: Path,
    capsys,
) -> None:
    comparison_path = tmp_path / "data/processed/analysis/sentiment_model_comparison.parquet"
    event_path = tmp_path / "data/processed/event_study/event_study.parquet"
    figures_dir = tmp_path / "reports/figures"
    comparison_path.parent.mkdir(parents=True)
    event_path.parent.mkdir(parents=True)

    _comparison().to_parquet(comparison_path, index=False)
    _event_study().to_parquet(event_path, index=False)

    output_paths = generate_day12_figures(
        comparison_path=comparison_path,
        event_study_path=event_path,
        lm_scores_path=tmp_path / "unused_lm_scores.parquet",
        sentiment_features_path=tmp_path / "unused_sentiment_features.parquet",
        figures_dir=figures_dir,
    )

    captured = capsys.readouterr()
    assert "n < 2" in captured.out
    assert len(output_paths) == 4
    assert {path.name for path in output_paths} == set(FIGURE_FILENAMES.values())
    assert all(path.exists() for path in output_paths)
    assert all(path.stat().st_size > 0 for path in output_paths)


def test_generate_day12_figures_can_build_comparison_when_analysis_input_is_missing(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "data/processed/event_study/event_study.parquet"
    lm_path = tmp_path / "data/processed/sentiment/lm_scores.parquet"
    sentiment_path = tmp_path / "data/processed/sentiment/sentiment_features.parquet"
    figures_dir = tmp_path / "reports/figures"

    for path in [event_path, lm_path, sentiment_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    sentiment = pd.DataFrame(
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
    lm_scores = pd.DataFrame(
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

    sentiment.to_parquet(sentiment_path, index=False)
    lm_scores.to_parquet(lm_path, index=False)
    _event_study().to_parquet(event_path, index=False)

    output_paths = generate_day12_figures(
        comparison_path=tmp_path / "missing_comparison.parquet",
        event_study_path=event_path,
        lm_scores_path=lm_path,
        sentiment_features_path=sentiment_path,
        figures_dir=figures_dir,
    )

    assert len(output_paths) == 4
    assert all(path.exists() for path in output_paths)
