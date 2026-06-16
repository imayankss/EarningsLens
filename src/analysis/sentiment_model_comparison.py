"""Compare FinBERT sentiment outputs with Loughran-McDonald scores."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import classification_report
except ImportError:  # pragma: no cover - depends on the local environment
    classification_report = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENT_FEATURES_PATH = PROJECT_ROOT / "data/processed/sentiment/sentiment_features.parquet"
LM_SCORES_PATH = PROJECT_ROOT / "data/processed/sentiment/lm_scores.parquet"
EVENT_STUDY_PATH = PROJECT_ROOT / "data/processed/event_study/event_study.parquet"

ANALYSIS_DIR = PROJECT_ROOT / "data/processed/analysis"
REPORT_TABLES_DIR = PROJECT_ROOT / "reports/tables"

ANALYSIS_PARQUET_PATH = ANALYSIS_DIR / "sentiment_model_comparison.parquet"
ANALYSIS_CSV_PATH = ANALYSIS_DIR / "sentiment_model_comparison.csv"
REPORT_COMPARISON_CSV_PATH = REPORT_TABLES_DIR / "sentiment_model_comparison.csv"
REPORT_SUMMARY_CSV_PATH = REPORT_TABLES_DIR / "sentiment_model_comparison_summary.csv"


SENTIMENT_FEATURE_COLUMNS = [
    "transcript_id",
    "ticker",
    "earnings_date",
    "fiscal_quarter",
    "finbert_score",
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "chunk_count",
    "mean_confidence",
]

LM_SCORE_COLUMNS = [
    "transcript_id",
    "ticker",
    "earnings_date",
    "lm_positive_count",
    "lm_negative_count",
    "lm_tone_score",
    "lm_tone",
    "lm_word_count",
    "lm_total_words",
    "lm_total_tokens",
    "lm_label",
]

EVENT_REQUIRED_COLUMNS = ["transcript_id", "ticker", "event_date"]
EVENT_DUPLICATE_SENTIMENT_COLUMNS = {"finbert_score", "lm_tone", "lm_tone_score"}


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_parquet(path)


def _finbert_label(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def _normalise_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    label = str(value).strip().lower()
    if label in {"pos", "positive", "bullish"}:
        return "positive"
    if label in {"neg", "negative", "bearish"}:
        return "negative"
    if label in {"neu", "neutral", "mixed"}:
        return "neutral"
    return label or "unknown"


def _safe_correlation(df: pd.DataFrame, left: str, right: str) -> float:
    paired = df[[left, right]].dropna()
    if len(paired) < 2:
        warnings.warn(
            f"Correlation between {left} and {right} requires at least 2 observations; returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan

    if paired[left].nunique() < 2 or paired[right].nunique() < 2:
        warnings.warn(
            f"Correlation between {left} and {right} is undefined because one series is constant; returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan

    return float(paired[left].corr(paired[right]))


def load_model_inputs(
    sentiment_features_path: Path = SENTIMENT_FEATURES_PATH,
    lm_scores_path: Path = LM_SCORES_PATH,
    event_study_path: Path = EVENT_STUDY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three Day 10 inputs from processed parquet files."""

    sentiment_features = _read_parquet(sentiment_features_path)
    lm_scores = _read_parquet(lm_scores_path)
    event_study = _read_parquet(event_study_path)

    _require_columns(sentiment_features, SENTIMENT_FEATURE_COLUMNS, "sentiment_features")
    _require_columns(lm_scores, LM_SCORE_COLUMNS, "lm_scores")
    _require_columns(event_study, EVENT_REQUIRED_COLUMNS, "event_study")

    return sentiment_features, lm_scores, event_study


def build_sentiment_model_comparison(
    sentiment_features: pd.DataFrame,
    lm_scores: pd.DataFrame,
    event_study: pd.DataFrame,
) -> pd.DataFrame:
    """Merge FinBERT, LM, and event-study data on transcript_id."""

    _require_columns(sentiment_features, SENTIMENT_FEATURE_COLUMNS, "sentiment_features")
    _require_columns(lm_scores, LM_SCORE_COLUMNS, "lm_scores")
    _require_columns(event_study, EVENT_REQUIRED_COLUMNS, "event_study")

    sentiment_features = sentiment_features[SENTIMENT_FEATURE_COLUMNS].copy()

    lm_columns = [
        column
        for column in LM_SCORE_COLUMNS
        if column not in {"ticker", "earnings_date"}
    ]
    lm_scores = lm_scores[lm_columns].copy()

    event_columns = [
        column
        for column in event_study.columns
        if column == "transcript_id" or column not in EVENT_DUPLICATE_SENTIMENT_COLUMNS | {"ticker"}
    ]
    event_study = event_study[event_columns].copy()

    comparison = sentiment_features.merge(lm_scores, on="transcript_id", how="inner")
    comparison = comparison.merge(event_study, on="transcript_id", how="inner")

    comparison["finbert_label"] = comparison["finbert_score"].apply(_finbert_label)
    comparison["lm_label_normalized"] = comparison["lm_label"].apply(_normalise_label)
    comparison["lm_tone_normalized"] = comparison["lm_tone"].apply(_normalise_label)
    comparison["sentiment_label_agree"] = (
        comparison["finbert_label"] == comparison["lm_label_normalized"]
    )
    comparison["sentiment_score_gap"] = comparison["finbert_score"] - comparison["lm_tone_score"]

    return comparison


def summarize_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table for reports/tables."""

    _require_columns(
        comparison,
        ["finbert_score", "lm_tone_score", "sentiment_label_agree"],
        "comparison",
    )

    correlation = _safe_correlation(comparison, "finbert_score", "lm_tone_score")
    summary = pd.DataFrame(
        [
            {"metric": "observation_count", "value": float(len(comparison))},
            {"metric": "finbert_mean", "value": comparison["finbert_score"].mean()},
            {"metric": "lm_tone_mean", "value": comparison["lm_tone_score"].mean()},
            {"metric": "finbert_lm_correlation", "value": correlation},
            {"metric": "label_agreement_rate", "value": comparison["sentiment_label_agree"].mean()},
        ]
    )
    return summary


def export_outputs(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    analysis_parquet_path: Path = ANALYSIS_PARQUET_PATH,
    analysis_csv_path: Path = ANALYSIS_CSV_PATH,
    report_comparison_csv_path: Path = REPORT_COMPARISON_CSV_PATH,
    report_summary_csv_path: Path = REPORT_SUMMARY_CSV_PATH,
) -> list[Path]:
    """Write analysis and report outputs."""

    output_paths = [
        analysis_parquet_path,
        analysis_csv_path,
        report_comparison_csv_path,
        report_summary_csv_path,
    ]

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_parquet(analysis_parquet_path, index=False)
    comparison.to_csv(analysis_csv_path, index=False)
    comparison.to_csv(report_comparison_csv_path, index=False)
    summary.to_csv(report_summary_csv_path, index=False)

    return output_paths


def print_classification_summary(comparison: pd.DataFrame) -> None:
    """Print a label comparison report when scikit-learn is installed."""

    if classification_report is None:
        print("scikit-learn is not installed; skipping classification_report.")
        return

    report = classification_report(
        comparison["finbert_label"],
        comparison["lm_label_normalized"],
        zero_division=0,
    )
    print("FinBERT vs LM classification report")
    print(report)


def run_comparison(
    sentiment_features_path: Path = SENTIMENT_FEATURES_PATH,
    lm_scores_path: Path = LM_SCORES_PATH,
    event_study_path: Path = EVENT_STUDY_PATH,
    analysis_parquet_path: Path = ANALYSIS_PARQUET_PATH,
    analysis_csv_path: Path = ANALYSIS_CSV_PATH,
    report_comparison_csv_path: Path = REPORT_COMPARISON_CSV_PATH,
    report_summary_csv_path: Path = REPORT_SUMMARY_CSV_PATH,
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Load inputs, build the comparison table, and export all Day 10 outputs."""

    sentiment_features, lm_scores, event_study = load_model_inputs(
        sentiment_features_path=sentiment_features_path,
        lm_scores_path=lm_scores_path,
        event_study_path=event_study_path,
    )
    comparison = build_sentiment_model_comparison(sentiment_features, lm_scores, event_study)
    summary = summarize_comparison(comparison)
    output_paths = export_outputs(
        comparison=comparison,
        summary=summary,
        analysis_parquet_path=analysis_parquet_path,
        analysis_csv_path=analysis_csv_path,
        report_comparison_csv_path=report_comparison_csv_path,
        report_summary_csv_path=report_summary_csv_path,
    )

    if print_report:
        print_classification_summary(comparison)

    return comparison, summary, output_paths


def main() -> None:
    comparison, summary, output_paths = run_comparison()

    print(f"Built sentiment model comparison with {len(comparison)} rows.")
    print("Summary:")
    print(summary.to_string(index=False))
    print("Wrote outputs:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
