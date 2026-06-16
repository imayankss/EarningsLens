"""Build the ML-ready master dataset for downstream modeling."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSCRIPTS_CLEANED_PATH = PROJECT_ROOT / "data/interim/transcripts/transcripts_cleaned.parquet"
SENTIMENT_FEATURES_PATH = PROJECT_ROOT / "data/processed/sentiment/sentiment_features.parquet"
LM_SCORES_PATH = PROJECT_ROOT / "data/processed/sentiment/lm_scores.parquet"
EVENT_STUDY_PATH = PROJECT_ROOT / "data/processed/event_study/event_study.parquet"
MARKET_DATA_PATH = PROJECT_ROOT / "data/processed/market_data.parquet"
COMPARISON_PATH = PROJECT_ROOT / "data/processed/analysis/sentiment_model_comparison.parquet"

MASTER_PARQUET_PATH = PROJECT_ROOT / "data/processed/master_dataset.parquet"
MASTER_CSV_PATH = PROJECT_ROOT / "data/processed/master_dataset.csv"
SUMMARY_CSV_PATH = PROJECT_ROOT / "reports/tables/master_dataset_summary.csv"


BASE_COLUMNS = [
    "transcript_id",
    "ticker",
    "earnings_date",
    "fiscal_quarter",
    "year",
    "company_name",
    "word_count_clean",
]

SENTIMENT_COLUMNS = [
    "transcript_id",
    "finbert_score",
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "chunk_count",
    "mean_confidence",
]

LM_COLUMNS = [
    "transcript_id",
    "lm_tone_score",
    "lm_tone",
    "lm_positive_count",
    "lm_negative_count",
    "lm_word_count",
    "lm_label",
]

EVENT_COLUMNS = [
    "transcript_id",
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

COMPARISON_COLUMNS = [
    "transcript_id",
    "finbert_direction",
    "lm_direction",
    "directional_agreement",
    "score_difference",
    "absolute_difference",
    "divergence_flag",
]

MARKET_AGG_COLUMNS = [
    "transcript_id",
    "market_row_count",
    "market_window_start",
    "market_window_end",
    "mean_return",
    "std_return",
    "mean_abnormal_return",
    "mean_volume",
    "price_min",
    "price_max",
]

CANONICAL_COLUMNS = (
    BASE_COLUMNS
    + SENTIMENT_COLUMNS[1:]
    + LM_COLUMNS[1:]
    + COMPARISON_COLUMNS[1:]
    + EVENT_COLUMNS[1:]
)

IMPORTANT_COLUMNS = CANONICAL_COLUMNS + MARKET_AGG_COLUMNS[1:]


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_parquet(path)


def _warn_missing_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        warnings.warn(
            f"{name} is missing important columns: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )


def _require_transcript_id(df: pd.DataFrame, name: str) -> None:
    if "transcript_id" not in df.columns:
        raise ValueError(f"{name} is missing required column: transcript_id")


def _select_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


def _dedupe_by_transcript_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_transcript_id(df, name)
    if df["transcript_id"].duplicated().any():
        warnings.warn(
            f"{name} contains duplicate transcript_id rows; keeping the first row per transcript_id.",
            RuntimeWarning,
            stacklevel=2,
        )
        return df.drop_duplicates(subset=["transcript_id"], keep="first").copy()
    return df.copy()


def _safe_left_merge(base: pd.DataFrame, incoming: pd.DataFrame, name: str) -> pd.DataFrame:
    """Left-join incoming columns without creating pandas _x/_y suffix columns."""

    _require_transcript_id(incoming, name)
    incoming = _dedupe_by_transcript_id(incoming, name)
    duplicate_columns = [
        column
        for column in incoming.columns
        if column != "transcript_id" and column in base.columns
    ]
    if duplicate_columns:
        incoming = incoming.drop(columns=duplicate_columns)
    return base.merge(incoming, on="transcript_id", how="left", validate="one_to_one")


def _direction_from_score(score: object) -> str | pd.NA:
    if pd.isna(score):
        return pd.NA
    value = float(score)
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _normalise_direction(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    label = str(value).strip().lower()
    if label in {"pos", "positive", "bullish"}:
        return "positive"
    if label in {"neg", "negative", "bearish"}:
        return "negative"
    if label in {"neu", "neutral", "mixed"}:
        return "neutral"
    try:
        return _direction_from_score(float(label))
    except ValueError:
        return label or pd.NA


def _prepare_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    _require_transcript_id(comparison, "sentiment_model_comparison")
    comparison = comparison.copy()

    if "finbert_direction" not in comparison.columns:
        if "finbert_label" in comparison.columns:
            comparison["finbert_direction"] = comparison["finbert_label"].map(_normalise_direction)
        elif "finbert_score" in comparison.columns:
            comparison["finbert_direction"] = comparison["finbert_score"].map(_direction_from_score)

    if "lm_direction" not in comparison.columns:
        if "lm_label_normalized" in comparison.columns:
            comparison["lm_direction"] = comparison["lm_label_normalized"].map(_normalise_direction)
        elif "lm_label" in comparison.columns:
            comparison["lm_direction"] = comparison["lm_label"].map(_normalise_direction)
        elif "lm_tone" in comparison.columns:
            comparison["lm_direction"] = comparison["lm_tone"].map(_normalise_direction)
        elif "lm_tone_score" in comparison.columns:
            comparison["lm_direction"] = comparison["lm_tone_score"].map(_direction_from_score)

    if "directional_agreement" not in comparison.columns:
        if "sentiment_label_agree" in comparison.columns:
            comparison["directional_agreement"] = comparison["sentiment_label_agree"]
        elif {"finbert_direction", "lm_direction"}.issubset(comparison.columns):
            comparison["directional_agreement"] = (
                comparison["finbert_direction"] == comparison["lm_direction"]
            )

    if "score_difference" not in comparison.columns:
        if "sentiment_score_gap" in comparison.columns:
            comparison["score_difference"] = comparison["sentiment_score_gap"]
        elif {"finbert_score", "lm_tone_score"}.issubset(comparison.columns):
            comparison["score_difference"] = (
                comparison["finbert_score"] - comparison["lm_tone_score"]
            )

    if "absolute_difference" not in comparison.columns and "score_difference" in comparison.columns:
        comparison["absolute_difference"] = comparison["score_difference"].abs()

    if "divergence_flag" not in comparison.columns:
        if "directional_agreement" in comparison.columns:
            comparison["divergence_flag"] = ~comparison["directional_agreement"].astype("boolean")
        elif "absolute_difference" in comparison.columns:
            comparison["divergence_flag"] = comparison["absolute_difference"] > 0.5

    return _select_existing(comparison, COMPARISON_COLUMNS)


def aggregate_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate potentially many market rows into one feature row per transcript."""

    _require_transcript_id(market_data, "market_data")
    if market_data.empty:
        return pd.DataFrame(columns=MARKET_AGG_COLUMNS)

    grouped = market_data.groupby("transcript_id", dropna=False)
    aggregated = grouped.size().rename("market_row_count").to_frame()

    if "date" in market_data.columns:
        aggregated["market_window_start"] = grouped["date"].min()
        aggregated["market_window_end"] = grouped["date"].max()

    if "daily_return" in market_data.columns:
        aggregated["mean_return"] = grouped["daily_return"].mean()
        aggregated["std_return"] = grouped["daily_return"].std()
    elif "return_1d" in market_data.columns:
        aggregated["mean_return"] = grouped["return_1d"].mean()
        aggregated["std_return"] = grouped["return_1d"].std()

    if "abnormal_return" in market_data.columns:
        aggregated["mean_abnormal_return"] = grouped["abnormal_return"].mean()
    elif "abnormal_return_1d" in market_data.columns:
        aggregated["mean_abnormal_return"] = grouped["abnormal_return_1d"].mean()

    if "volume" in market_data.columns:
        aggregated["mean_volume"] = grouped["volume"].mean()

    if "low" in market_data.columns:
        aggregated["price_min"] = grouped["low"].min()
    elif "close" in market_data.columns:
        aggregated["price_min"] = grouped["close"].min()

    if "high" in market_data.columns:
        aggregated["price_max"] = grouped["high"].max()
    elif "close" in market_data.columns:
        aggregated["price_max"] = grouped["close"].max()

    return aggregated.reset_index()


def load_master_inputs(
    transcripts_cleaned_path: Path = TRANSCRIPTS_CLEANED_PATH,
    sentiment_features_path: Path = SENTIMENT_FEATURES_PATH,
    lm_scores_path: Path = LM_SCORES_PATH,
    event_study_path: Path = EVENT_STUDY_PATH,
    market_data_path: Path = MARKET_DATA_PATH,
    comparison_path: Path = COMPARISON_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all Day 11 inputs from parquet files."""

    return (
        _read_parquet(transcripts_cleaned_path),
        _read_parquet(sentiment_features_path),
        _read_parquet(lm_scores_path),
        _read_parquet(event_study_path),
        _read_parquet(market_data_path),
        _read_parquet(comparison_path),
    )


def build_master_dataset(
    transcripts_cleaned: pd.DataFrame,
    sentiment_features: pd.DataFrame,
    lm_scores: pd.DataFrame,
    event_study: pd.DataFrame,
    market_data: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build one ML-ready master dataset using transcripts as the base table."""

    _require_transcript_id(transcripts_cleaned, "transcripts_cleaned")
    if transcripts_cleaned["transcript_id"].duplicated().any():
        raise ValueError("transcripts_cleaned contains duplicate transcript_id rows.")

    _warn_missing_columns(transcripts_cleaned, BASE_COLUMNS, "transcripts_cleaned")
    _warn_missing_columns(sentiment_features, SENTIMENT_COLUMNS, "sentiment_features")
    _warn_missing_columns(lm_scores, LM_COLUMNS, "lm_scores")
    _warn_missing_columns(event_study, EVENT_COLUMNS, "event_study")
    _warn_missing_columns(comparison, COMPARISON_COLUMNS, "sentiment_model_comparison")

    master = _select_existing(transcripts_cleaned, BASE_COLUMNS)
    master = _safe_left_merge(
        master,
        _select_existing(sentiment_features, SENTIMENT_COLUMNS),
        "sentiment_features",
    )
    master = _safe_left_merge(master, _select_existing(lm_scores, LM_COLUMNS), "lm_scores")
    master = _safe_left_merge(master, _select_existing(event_study, EVENT_COLUMNS), "event_study")
    master = _safe_left_merge(master, aggregate_market_data(market_data), "market_data")
    master = _safe_left_merge(master, _prepare_comparison(comparison), "sentiment_model_comparison")

    for column in CANONICAL_COLUMNS + MARKET_AGG_COLUMNS[1:]:
        if column not in master.columns:
            master[column] = pd.NA

    ordered_columns = CANONICAL_COLUMNS + [
        column for column in MARKET_AGG_COLUMNS[1:] if column in master.columns
    ]
    remaining_columns = [column for column in master.columns if column not in ordered_columns]
    master = master[ordered_columns + remaining_columns]

    validate_master_dataset(master)
    return master


def validate_master_dataset(master: pd.DataFrame) -> None:
    """Validate the final master dataset and emit warnings for soft issues."""

    _require_transcript_id(master, "master_dataset")
    if master["transcript_id"].duplicated().any():
        duplicates = master.loc[master["transcript_id"].duplicated(), "transcript_id"].tolist()
        raise ValueError(f"master_dataset contains duplicate transcript_id rows: {duplicates}")

    if len(master) == 1:
        warnings.warn(
            "master_dataset row_count is only 1; downstream ML validation will be limited.",
            RuntimeWarning,
            stacklevel=2,
        )

    _warn_missing_columns(master, IMPORTANT_COLUMNS, "master_dataset")


def summarize_master_dataset(master: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table for reports/tables."""

    return pd.DataFrame(
        [
            {"metric": "row_count", "value": float(len(master))},
            {"metric": "column_count", "value": float(len(master.columns))},
            {"metric": "unique_transcript_id_count", "value": float(master["transcript_id"].nunique())},
            {"metric": "duplicate_transcript_id_count", "value": float(master["transcript_id"].duplicated().sum())},
            {"metric": "ticker_count", "value": float(master["ticker"].nunique()) if "ticker" in master.columns else pd.NA},
            {
                "metric": "directional_agreement_rate",
                "value": master["directional_agreement"].mean()
                if "directional_agreement" in master.columns
                else pd.NA,
            },
            {
                "metric": "divergence_rate",
                "value": master["divergence_flag"].mean()
                if "divergence_flag" in master.columns
                else pd.NA,
            },
        ]
    )


def export_outputs(
    master: pd.DataFrame,
    summary: pd.DataFrame,
    master_parquet_path: Path = MASTER_PARQUET_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
    summary_csv_path: Path = SUMMARY_CSV_PATH,
) -> list[Path]:
    """Write the master dataset and report summary."""

    output_paths = [master_parquet_path, master_csv_path, summary_csv_path]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    master.to_parquet(master_parquet_path, index=False)
    master.to_csv(master_csv_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    return output_paths


def run_master_dataset_builder(
    transcripts_cleaned_path: Path = TRANSCRIPTS_CLEANED_PATH,
    sentiment_features_path: Path = SENTIMENT_FEATURES_PATH,
    lm_scores_path: Path = LM_SCORES_PATH,
    event_study_path: Path = EVENT_STUDY_PATH,
    market_data_path: Path = MARKET_DATA_PATH,
    comparison_path: Path = COMPARISON_PATH,
    master_parquet_path: Path = MASTER_PARQUET_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
    summary_csv_path: Path = SUMMARY_CSV_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Load inputs, build the master dataset, summarize it, and export outputs."""

    inputs = load_master_inputs(
        transcripts_cleaned_path=transcripts_cleaned_path,
        sentiment_features_path=sentiment_features_path,
        lm_scores_path=lm_scores_path,
        event_study_path=event_study_path,
        market_data_path=market_data_path,
        comparison_path=comparison_path,
    )
    master = build_master_dataset(*inputs)
    summary = summarize_master_dataset(master)
    output_paths = export_outputs(
        master=master,
        summary=summary,
        master_parquet_path=master_parquet_path,
        master_csv_path=master_csv_path,
        summary_csv_path=summary_csv_path,
    )
    return master, summary, output_paths


def main() -> None:
    master, summary, output_paths = run_master_dataset_builder()

    print(f"Built master dataset with {len(master)} rows and {len(master.columns)} columns.")
    print("Columns:")
    print(", ".join(master.columns))
    print("Summary:")
    print(summary.to_string(index=False))
    print("Wrote outputs:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
