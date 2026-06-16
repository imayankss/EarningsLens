from pathlib import Path

import pandas as pd

from src.visualization.professional_charts import (
    FIGURE_FILENAMES,
    SMALL_N_NOTE,
    generate_professional_charts,
)


def _master_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "finbert_score": 0.42,
                "lm_tone_score": 0.31,
                "score_difference": 0.11,
                "absolute_difference": 0.11,
                "directional_agreement": True,
                "divergence_flag": False,
                "finbert_direction": "positive",
                "lm_direction": "positive",
                "return_1d": 0.012,
                "ar_1d": 0.004,
                "return_2d": 0.020,
                "ar_2d": 0.010,
                "return_3d": 0.030,
                "ar_3d": 0.020,
                "return_5d": 0.040,
                "ar_5d": 0.030,
            }
        ]
    )


def _comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "finbert_score": 0.42,
                "lm_tone_score": 0.31,
                "sentiment_label_agree": True,
            }
        ]
    )


def _correlation_analysis() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_name": "finbert_score__return_1d",
                "x_column": "finbert_score",
                "y_column": "return_1d",
                "n_observations": 1,
                "pearson_r": None,
                "pearson_p_value": None,
                "spearman_rho": None,
                "spearman_p_value": None,
                "status": "insufficient_sample_size",
            }
        ]
    )


def _statistical_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "row_count", "value": 1},
            {"metric": "column_count", "value": 18},
            {"metric": "ticker_count", "value": 1},
        ]
    )


def test_generate_professional_charts_creates_all_expected_figures_with_tmp_inputs(
    tmp_path: Path,
) -> None:
    master_path = tmp_path / "data/processed/master_dataset.parquet"
    comparison_path = tmp_path / "data/processed/analysis/sentiment_model_comparison.parquet"
    correlation_path = tmp_path / "reports/tables/correlation_analysis.csv"
    summary_path = tmp_path / "reports/tables/statistical_summary.csv"
    figures_dir = tmp_path / "reports/figures"

    for path in [master_path, comparison_path, correlation_path, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    _master_dataset().to_parquet(master_path, index=False)
    _comparison().to_parquet(comparison_path, index=False)
    _correlation_analysis().to_csv(correlation_path, index=False)
    _statistical_summary().to_csv(summary_path, index=False)

    output_paths, warnings = generate_professional_charts(
        master_parquet_path=master_path,
        master_csv_path=tmp_path / "missing_master_dataset.csv",
        comparison_path=comparison_path,
        correlation_path=correlation_path,
        statistical_summary_path=summary_path,
        figures_dir=figures_dir,
    )

    assert len(output_paths) == 6
    assert {path.name for path in output_paths} == set(FIGURE_FILENAMES.values())
    assert all(path.exists() for path in output_paths)
    assert all(path.stat().st_size > 0 for path in output_paths)
    assert SMALL_N_NOTE in warnings
    assert any("Correlation Heatmap" in warning for warning in warnings)


def test_generate_professional_charts_falls_back_to_csv_and_uses_placeholders(
    tmp_path: Path,
) -> None:
    master_csv_path = tmp_path / "data/processed/master_dataset.csv"
    comparison_path = tmp_path / "data/processed/analysis/sentiment_model_comparison.parquet"
    correlation_path = tmp_path / "reports/tables/correlation_analysis.csv"
    summary_path = tmp_path / "reports/tables/statistical_summary.csv"
    figures_dir = tmp_path / "reports/figures"

    for path in [master_csv_path, comparison_path, correlation_path, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "directional_agreement": True,
            }
        ]
    ).to_csv(master_csv_path, index=False)
    _comparison().to_parquet(comparison_path, index=False)
    _correlation_analysis().to_csv(correlation_path, index=False)
    _statistical_summary().to_csv(summary_path, index=False)

    output_paths, warnings = generate_professional_charts(
        master_parquet_path=tmp_path / "missing_master_dataset.parquet",
        master_csv_path=master_csv_path,
        comparison_path=comparison_path,
        correlation_path=correlation_path,
        statistical_summary_path=summary_path,
        figures_dir=figures_dir,
    )

    assert len(output_paths) == 6
    assert all(path.exists() for path in output_paths)
    assert all(path.stat().st_size > 0 for path in output_paths)
    assert any("Missing numeric sentiment columns" in warning for warning in warnings)
    assert any("Missing required data" in warning for warning in warnings)
