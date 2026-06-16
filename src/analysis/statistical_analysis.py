"""Run research-style statistical analysis on the Day 11 master dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - depends on the local environment
    stats = None

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover - depends on the local environment
    sm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_PARQUET_PATH = PROJECT_ROOT / "data/processed/master_dataset.parquet"
MASTER_CSV_PATH = PROJECT_ROOT / "data/processed/master_dataset.csv"
REPORT_TABLES_DIR = PROJECT_ROOT / "reports/tables"

STATISTICAL_SUMMARY_PATH = REPORT_TABLES_DIR / "statistical_summary.csv"
CORRELATION_ANALYSIS_PATH = REPORT_TABLES_DIR / "correlation_analysis.csv"
REGRESSION_RESULTS_PATH = REPORT_TABLES_DIR / "regression_results.csv"
STATISTICAL_WARNINGS_PATH = REPORT_TABLES_DIR / "statistical_warnings.csv"

SENTIMENT_COLUMNS = [
    "finbert_score",
    "lm_tone_score",
    "score_difference",
    "absolute_difference",
]

RETURN_COLUMNS = [
    "return_1d",
    "ar_1d",
    "return_2d",
    "ar_2d",
    "return_3d",
    "ar_3d",
    "return_5d",
    "ar_5d",
    "return_10d",
    "ar_10d",
]

AGREEMENT_COLUMNS = [
    "directional_agreement",
    "divergence_flag",
]

WARNING_COLUMNS = ["category", "item", "message"]
CORRELATION_COLUMNS = [
    "pair_name",
    "x_column",
    "y_column",
    "n_observations",
    "pearson_r",
    "pearson_p_value",
    "spearman_rho",
    "spearman_p_value",
    "status",
]
REGRESSION_COLUMNS = [
    "target_column",
    "feature_columns",
    "n_observations",
    "r_squared",
    "adj_r_squared",
    "coefficients",
    "p_values",
    "status",
]


def _warning(category: str, item: str, message: str) -> dict[str, str]:
    return {"category": category, "item": item, "message": message}


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _to_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(index=df.index)
    for column in columns:
        numeric[column] = pd.to_numeric(df[column], errors="coerce")
    return numeric


def _jsonify_mapping(values: dict[str, Any]) -> str:
    clean: dict[str, float | None] = {}
    for key, value in values.items():
        if pd.isna(value):
            clean[str(key)] = None
        else:
            clean[str(key)] = float(value)
    return json.dumps(clean, sort_keys=True)


def _mean_bool(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    if pd.api.types.is_bool_dtype(series):
        return float(series.mean())

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )
    return float(mapped.astype("boolean").mean())


def load_master_dataset(
    master_parquet_path: Path = MASTER_PARQUET_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
) -> pd.DataFrame:
    """Load the Day 11 master dataset from parquet, falling back to CSV."""

    if master_parquet_path.exists():
        return pd.read_parquet(master_parquet_path)
    if master_csv_path.exists():
        return pd.read_csv(master_csv_path)
    raise FileNotFoundError(
        f"Missing master dataset. Tried parquet={master_parquet_path} and csv={master_csv_path}"
    )


def build_statistical_summary(
    master: pd.DataFrame,
    warnings_list: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Summarize the available statistical analysis inputs."""

    warnings_list = warnings_list if warnings_list is not None else []
    available_sentiment = _available_columns(master, SENTIMENT_COLUMNS)
    available_returns = _available_columns(master, RETURN_COLUMNS)

    for column in SENTIMENT_COLUMNS + RETURN_COLUMNS + AGREEMENT_COLUMNS:
        if column not in master.columns:
            warnings_list.append(
                _warning("missing_optional_column", column, f"Optional column {column} is missing.")
            )

    if len(master) == 1:
        warnings_list.append(
            _warning(
                "sample_size",
                "master_dataset",
                "n=1; correlations, regressions, and t-tests will be skipped where underpowered.",
            )
        )

    directional_agreement_rate = (
        _mean_bool(master["directional_agreement"])
        if "directional_agreement" in master.columns
        else np.nan
    )
    divergence_rate = (
        _mean_bool(master["divergence_flag"]) if "divergence_flag" in master.columns else np.nan
    )

    summary = pd.DataFrame(
        [
            {"metric": "row_count", "value": len(master)},
            {"metric": "column_count", "value": len(master.columns)},
            {
                "metric": "ticker_count",
                "value": master["ticker"].nunique() if "ticker" in master.columns else np.nan,
            },
            {
                "metric": "available_sentiment_columns",
                "value": "|".join(available_sentiment),
            },
            {"metric": "available_return_columns", "value": "|".join(available_returns)},
            {"metric": "missing_value_count", "value": int(master.isna().sum().sum())},
            {"metric": "directional_agreement_rate", "value": directional_agreement_rate},
            {"metric": "divergence_rate", "value": divergence_rate},
        ]
    )
    return summary


def build_correlation_analysis(
    master: pd.DataFrame,
    warnings_list: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Compute guarded Pearson and Spearman correlations for sentiment/return pairs."""

    warnings_list = warnings_list if warnings_list is not None else []
    available_sentiment = _available_columns(master, SENTIMENT_COLUMNS)
    available_returns = _available_columns(master, RETURN_COLUMNS)
    rows: list[dict[str, Any]] = []

    if not available_sentiment or not available_returns:
        warnings_list.append(
            _warning(
                "correlation",
                "all_pairs",
                "No available sentiment/return column pairs for correlation analysis.",
            )
        )
        return pd.DataFrame(columns=CORRELATION_COLUMNS)

    numeric = _to_numeric_frame(master, available_sentiment + available_returns)
    for x_column in available_sentiment:
        for y_column in available_returns:
            pair_name = f"{x_column}__{y_column}"
            paired = numeric[[x_column, y_column]].dropna()
            n_observations = int(len(paired))
            row: dict[str, Any] = {
                "pair_name": pair_name,
                "x_column": x_column,
                "y_column": y_column,
                "n_observations": n_observations,
                "pearson_r": np.nan,
                "pearson_p_value": np.nan,
                "spearman_rho": np.nan,
                "spearman_p_value": np.nan,
                "status": "ok",
            }

            if n_observations < 2:
                row["status"] = "insufficient_sample_size"
                warnings_list.append(
                    _warning(
                        "correlation",
                        pair_name,
                        f"Skipped correlation for {pair_name}; n={n_observations} < 2.",
                    )
                )
            elif paired[x_column].nunique() < 2 or paired[y_column].nunique() < 2:
                row["status"] = "constant_input"
                warnings_list.append(
                    _warning(
                        "correlation",
                        pair_name,
                        f"Skipped correlation for {pair_name}; at least one input is constant.",
                    )
                )
            elif stats is None:
                row["status"] = "scipy_unavailable"
                warnings_list.append(
                    _warning(
                        "correlation",
                        pair_name,
                        f"Skipped correlation for {pair_name}; scipy is unavailable.",
                    )
                )
            else:
                pearson = stats.pearsonr(paired[x_column], paired[y_column])
                spearman = stats.spearmanr(paired[x_column], paired[y_column])
                row["pearson_r"] = float(pearson.statistic)
                row["pearson_p_value"] = float(pearson.pvalue)
                row["spearman_rho"] = float(spearman.statistic)
                row["spearman_p_value"] = float(spearman.pvalue)

            rows.append(row)

    return pd.DataFrame(rows, columns=CORRELATION_COLUMNS)


def build_regression_results(
    master: pd.DataFrame,
    warnings_list: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Run guarded OLS regressions for each available return target."""

    warnings_list = warnings_list if warnings_list is not None else []
    feature_columns = _available_columns(master, SENTIMENT_COLUMNS)
    target_columns = _available_columns(master, RETURN_COLUMNS)
    rows: list[dict[str, Any]] = []

    if not target_columns:
        warnings_list.append(
            _warning("regression", "all_targets", "No available return target columns for regression.")
        )
        return pd.DataFrame(columns=REGRESSION_COLUMNS)

    for target_column in target_columns:
        columns = [target_column] + feature_columns
        numeric = _to_numeric_frame(master, columns).dropna()
        n_observations = int(len(numeric))
        row: dict[str, Any] = {
            "target_column": target_column,
            "feature_columns": "|".join(feature_columns),
            "n_observations": n_observations,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "coefficients": "{}",
            "p_values": "{}",
            "status": "ok",
        }

        if not feature_columns:
            row["status"] = "missing_features"
            warnings_list.append(
                _warning(
                    "regression",
                    target_column,
                    f"Skipped regression for {target_column}; no sentiment feature columns are available.",
                )
            )
        elif sm is None:
            row["status"] = "statsmodels_unavailable"
            warnings_list.append(
                _warning(
                    "regression",
                    target_column,
                    f"Skipped regression for {target_column}; statsmodels is unavailable.",
                )
            )
        elif n_observations < len(feature_columns) + 2:
            row["status"] = "insufficient_sample_size"
            warnings_list.append(
                _warning(
                    "regression",
                    target_column,
                    "Skipped regression for "
                    f"{target_column}; n={n_observations} < features+2 ({len(feature_columns) + 2}).",
                )
            )
        else:
            y = numeric[target_column]
            x = sm.add_constant(numeric[feature_columns], has_constant="add")
            result = sm.OLS(y, x, missing="drop").fit()
            row["r_squared"] = float(result.rsquared)
            row["adj_r_squared"] = float(result.rsquared_adj)
            row["coefficients"] = _jsonify_mapping(result.params.to_dict())
            row["p_values"] = _jsonify_mapping(result.pvalues.to_dict())

        rows.append(row)

    return pd.DataFrame(rows, columns=REGRESSION_COLUMNS)


def add_t_test_warnings(
    master: pd.DataFrame,
    warnings_list: list[dict[str, str]],
) -> None:
    """Record whether positive-vs-negative sentiment return t-tests can run."""

    available_returns = _available_columns(master, RETURN_COLUMNS)
    if not available_returns:
        warnings_list.append(
            _warning("t_test", "all_targets", "Skipped t-tests; no return columns are available.")
        )
        return

    direction_column = None
    for candidate in ["finbert_direction", "lm_direction"]:
        if candidate in master.columns:
            direction_column = candidate
            break

    if direction_column is None:
        warnings_list.append(
            _warning("t_test", "sentiment_groups", "Skipped t-tests; no sentiment direction column is available.")
        )
        return

    directions = master[direction_column].astype("string").str.lower()
    for target_column in available_returns:
        returns = pd.to_numeric(master[target_column], errors="coerce")
        positive = returns[directions == "positive"].dropna()
        negative = returns[directions == "negative"].dropna()
        if len(positive) < 2 or len(negative) < 2:
            warnings_list.append(
                _warning(
                    "t_test",
                    target_column,
                    "Skipped t-test for "
                    f"{target_column}; positive_n={len(positive)} and negative_n={len(negative)} "
                    "must both be at least 2.",
                )
            )


def run_statistical_analysis(
    master_parquet_path: Path = MASTER_PARQUET_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
    summary_path: Path = STATISTICAL_SUMMARY_PATH,
    correlation_path: Path = CORRELATION_ANALYSIS_PATH,
    regression_path: Path = REGRESSION_RESULTS_PATH,
    warnings_path: Path = STATISTICAL_WARNINGS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    """Load the master dataset, run analyses, and export all Day 12 tables."""

    warnings_list: list[dict[str, str]] = []
    master = load_master_dataset(
        master_parquet_path=master_parquet_path,
        master_csv_path=master_csv_path,
    )
    summary = build_statistical_summary(master, warnings_list)
    correlation = build_correlation_analysis(master, warnings_list)
    regression = build_regression_results(master, warnings_list)
    add_t_test_warnings(master, warnings_list)
    warnings_df = pd.DataFrame(warnings_list, columns=WARNING_COLUMNS)

    output_paths = [summary_path, correlation_path, regression_path, warnings_path]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)
    correlation.to_csv(correlation_path, index=False)
    regression.to_csv(regression_path, index=False)
    warnings_df.to_csv(warnings_path, index=False)

    return summary, correlation, regression, warnings_df, output_paths


def main() -> None:
    summary, correlation, regression, warnings_df, output_paths = run_statistical_analysis()
    row_count = summary.loc[summary["metric"] == "row_count", "value"].iloc[0]

    print(f"Ran statistical analysis for {row_count} master dataset rows.")
    print("Summary:")
    print(summary.to_string(index=False))
    print("Correlation statuses:")
    print(correlation["status"].value_counts(dropna=False).to_string())
    print("Regression statuses:")
    print(regression["status"].value_counts(dropna=False).to_string())
    print(f"Warnings: {len(warnings_df)}")
    if not warnings_df.empty:
        print(warnings_df.to_string(index=False))
    print("Wrote outputs:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
