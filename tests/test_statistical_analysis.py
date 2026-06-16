from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.statistical_analysis import (
    build_correlation_analysis,
    build_regression_results,
    run_statistical_analysis,
)


def _one_row_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "finbert_score": 0.40,
                "lm_tone_score": 0.30,
                "score_difference": 0.10,
                "absolute_difference": 0.10,
                "return_1d": 0.02,
                "ar_1d": 0.01,
                "directional_agreement": True,
                "divergence_flag": False,
                "finbert_direction": "positive",
            }
        ]
    )


def _three_row_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "finbert_score": 0.10,
                "lm_tone_score": 0.08,
                "score_difference": 0.02,
                "absolute_difference": 0.02,
                "return_1d": 0.01,
                "ar_1d": 0.005,
                "directional_agreement": True,
                "divergence_flag": False,
                "finbert_direction": "positive",
            },
            {
                "transcript_id": "MSFT_20210126",
                "ticker": "MSFT",
                "finbert_score": 0.20,
                "lm_tone_score": 0.18,
                "score_difference": 0.02,
                "absolute_difference": 0.02,
                "return_1d": 0.02,
                "ar_1d": 0.015,
                "directional_agreement": True,
                "divergence_flag": False,
                "finbert_direction": "positive",
            },
            {
                "transcript_id": "TSLA_20210127",
                "ticker": "TSLA",
                "finbert_score": 0.30,
                "lm_tone_score": 0.28,
                "score_difference": 0.02,
                "absolute_difference": 0.02,
                "return_1d": 0.03,
                "ar_1d": 0.025,
                "directional_agreement": True,
                "divergence_flag": False,
                "finbert_direction": "positive",
            },
        ]
    )


def test_n_equals_one_outputs_guarded_statistics_and_warnings(tmp_path: Path) -> None:
    master_path = tmp_path / "master_dataset.parquet"
    _one_row_master().to_parquet(master_path, index=False)

    summary, correlation, regression, warnings_df, output_paths = run_statistical_analysis(
        master_parquet_path=master_path,
        master_csv_path=tmp_path / "missing_master_dataset.csv",
        summary_path=tmp_path / "reports/tables/statistical_summary.csv",
        correlation_path=tmp_path / "reports/tables/correlation_analysis.csv",
        regression_path=tmp_path / "reports/tables/regression_results.csv",
        warnings_path=tmp_path / "reports/tables/statistical_warnings.csv",
    )

    assert summary.loc[summary["metric"] == "row_count", "value"].iloc[0] == 1
    assert set(correlation["status"]) == {"insufficient_sample_size"}
    assert set(regression["status"]) == {"insufficient_sample_size"}
    assert "sample_size" in set(warnings_df["category"])
    assert "t_test" in set(warnings_df["category"])
    assert all(path.exists() for path in output_paths)


def test_n_greater_equal_three_correlation_computes_coefficients() -> None:
    warnings_list: list[dict[str, str]] = []
    correlation = build_correlation_analysis(_three_row_master(), warnings_list)

    row = correlation.loc[correlation["pair_name"] == "finbert_score__return_1d"].iloc[0]
    assert row["status"] == "ok"
    assert row["n_observations"] == 3
    assert row["pearson_r"] == pytest.approx(1.0)
    assert row["spearman_rho"] == pytest.approx(1.0)
    assert "constant_input" in set(correlation["status"])


def test_regression_skips_when_sample_size_is_too_small() -> None:
    warnings_list: list[dict[str, str]] = []
    regression = build_regression_results(_three_row_master(), warnings_list)

    assert set(regression["status"]) == {"insufficient_sample_size"}
    assert regression["n_observations"].min() == 3
    assert all(np.isnan(value) for value in regression["r_squared"])
    assert any(warning["category"] == "regression" for warning in warnings_list)


def test_run_statistical_analysis_falls_back_to_csv_and_creates_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "master_dataset.csv"
    _three_row_master().to_csv(csv_path, index=False)

    summary, correlation, regression, warnings_df, output_paths = run_statistical_analysis(
        master_parquet_path=tmp_path / "missing_master_dataset.parquet",
        master_csv_path=csv_path,
        summary_path=tmp_path / "reports/tables/statistical_summary.csv",
        correlation_path=tmp_path / "reports/tables/correlation_analysis.csv",
        regression_path=tmp_path / "reports/tables/regression_results.csv",
        warnings_path=tmp_path / "reports/tables/statistical_warnings.csv",
    )

    assert summary.loc[summary["metric"] == "row_count", "value"].iloc[0] == 3
    assert not correlation.empty
    assert not regression.empty
    assert not warnings_df.empty
    assert all(path.exists() for path in output_paths)
    assert pd.read_csv(output_paths[0]).shape[0] == len(summary)
    assert pd.read_csv(output_paths[1]).shape[0] == len(correlation)
    assert pd.read_csv(output_paths[2]).shape[0] == len(regression)
    assert pd.read_csv(output_paths[3]).shape[0] == len(warnings_df)
