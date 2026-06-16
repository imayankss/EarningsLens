"""Generate Day 12 portfolio-ready sentiment and event-study figures."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.analysis.sentiment_model_comparison import build_sentiment_model_comparison
except ImportError:  # pragma: no cover - supports direct script execution
    build_sentiment_model_comparison = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPARISON_PATH = PROJECT_ROOT / "data/processed/analysis/sentiment_model_comparison.parquet"
EVENT_STUDY_PATH = PROJECT_ROOT / "data/processed/event_study/event_study.parquet"
LM_SCORES_PATH = PROJECT_ROOT / "data/processed/sentiment/lm_scores.parquet"
SENTIMENT_FEATURES_PATH = PROJECT_ROOT / "data/processed/sentiment/sentiment_features.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports/figures"

FIGURE_FILENAMES = {
    "score_comparison": "day12_finbert_vs_lm_score_comparison.png",
    "sentiment_return": "day12_sentiment_vs_return_snapshot.png",
    "return_horizon": "day12_event_study_return_horizon.png",
    "agreement": "day12_model_agreement_summary.png",
}


def _read_parquet(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} input: {path}")
    return pd.read_parquet(path)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _add_small_n_note(ax: plt.Axes, n: int) -> None:
    if n >= 2:
        return

    ax.text(
        0.01,
        0.99,
        "n < 2: descriptive snapshot only; no correlation or trend inferred",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#8a4b00",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff4d6", "edgecolor": "#d8a526"},
    )


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _short_label(row: pd.Series) -> str:
    ticker = str(row.get("ticker", "")).strip()
    transcript_id = str(row.get("transcript_id", "")).strip()
    if ticker and ticker.lower() != "nan":
        return ticker
    return transcript_id or "event"


def _return_snapshot_column(df: pd.DataFrame) -> str:
    priority = ["return_0", "ar_0", "car_0_1", "event_return", "abnormal_return", "return"]
    for column in priority:
        if column in df.columns:
            return column

    candidates = [
        column
        for column in df.columns
        if re.search(r"(^|_)(return|ar|car)(_|$)", column, flags=re.IGNORECASE)
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not candidates:
        raise ValueError("comparison has no numeric return/ar/car column for the return snapshot")
    return candidates[0]


def _horizon_columns(event_study: pd.DataFrame) -> list[tuple[int, str]]:
    preferred_columns = [
        ("ar_1d", 1),
        ("ar_2d", 2),
        ("ar_3d", 3),
        ("ar_5d", 5),
        ("ar_10d", 10),
        ("return_1d", 1),
        ("return_2d", 2),
        ("return_3d", 3),
        ("return_5d", 5),
        ("return_10d", 10),
    ]
    matches = [
        (horizon, column)
        for column, horizon in preferred_columns
        if column in event_study.columns and pd.api.types.is_numeric_dtype(event_study[column])
    ]
    if matches:
        return matches

    preferred_prefixes = ["ar", "return", "car"]
    for prefix in preferred_prefixes:
        matches: list[tuple[int, str]] = []
        for column in event_study.columns:
            if not pd.api.types.is_numeric_dtype(event_study[column]):
                continue

            match = re.fullmatch(rf"{prefix}_?(-?\d+)(?:_(-?\d+))?", column, flags=re.IGNORECASE)
            if not match:
                continue

            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            horizon = end if prefix.lower() == "car" else start
            matches.append((horizon, column))

        if matches:
            return sorted(matches, key=lambda item: item[0])

    return []


def _load_inputs(
    comparison_path: Path,
    event_study_path: Path,
    lm_scores_path: Path,
    sentiment_features_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_study = _read_parquet(event_study_path, "event_study")

    if comparison_path.exists():
        comparison = pd.read_parquet(comparison_path)
    else:
        if build_sentiment_model_comparison is None:
            raise FileNotFoundError(f"Missing comparison input: {comparison_path}")
        sentiment_features = _read_parquet(sentiment_features_path, "sentiment_features")
        lm_scores = _read_parquet(lm_scores_path, "lm_scores")
        comparison = build_sentiment_model_comparison(sentiment_features, lm_scores, event_study)

    return comparison, event_study


def plot_score_comparison(comparison: pd.DataFrame, output_path: Path) -> Path:
    _require_columns(comparison, ["finbert_score", "lm_tone_score"], "comparison")
    n = len(comparison)
    labels = [_short_label(row) for _, row in comparison.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_positions = range(n)
    width = 0.36
    ax.bar(
        [x - width / 2 for x in x_positions],
        comparison["finbert_score"],
        width=width,
        label="FinBERT score",
        color="#246b6f",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        comparison["lm_tone_score"],
        width=width,
        label="LM tone score",
        color="#c05a2b",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("FinBERT vs LM Sentiment Scores")
    ax.set_ylabel("Sentiment score")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=0 if n <= 3 else 30, ha="center" if n <= 3 else "right")
    ax.legend(frameon=False)
    _add_small_n_note(ax, n)
    return _save_figure(fig, output_path)


def plot_sentiment_return_snapshot(comparison: pd.DataFrame, output_path: Path) -> Path:
    _require_columns(comparison, ["finbert_score", "lm_tone_score"], "comparison")
    return_column = _return_snapshot_column(comparison)
    n = len(comparison)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        comparison["finbert_score"],
        comparison[return_column],
        s=95,
        color="#246b6f",
        label="FinBERT",
        alpha=0.9,
    )
    ax.scatter(
        comparison["lm_tone_score"],
        comparison[return_column],
        s=95,
        color="#c05a2b",
        label="LM",
        marker="s",
        alpha=0.9,
    )

    for _, row in comparison.iterrows():
        ax.annotate(
            _short_label(row),
            (row["finbert_score"], row[return_column]),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )

    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_title("Sentiment vs Return Snapshot")
    ax.set_xlabel("Sentiment score")
    ax.set_ylabel(return_column)
    ax.legend(frameon=False)
    _add_small_n_note(ax, n)
    return _save_figure(fig, output_path)


def plot_event_study_return_horizon(event_study: pd.DataFrame, output_path: Path) -> Path:
    horizon_columns = _horizon_columns(event_study)
    n = len(event_study)

    if not horizon_columns:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No event-study horizon return columns available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="#333333",
        )
        ax.set_title("Event-Study Return Horizon")
        _add_small_n_note(ax, n)
        return _save_figure(fig, output_path)

    horizons = [horizon for horizon, _ in horizon_columns]
    values = [event_study[column].mean() for _, column in horizon_columns]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if len(horizons) == 1:
        ax.bar([str(horizons[0])], values, color="#4f6d7a")
        ax.set_xlabel("Event horizon")
    else:
        ax.plot(horizons, values, marker="o", linewidth=2, color="#4f6d7a")
        ax.set_xlabel("Event horizon")

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Event-Study Return Horizon")
    ax.set_ylabel("Mean observed return")
    _add_small_n_note(ax, n)
    return _save_figure(fig, output_path)


def plot_model_agreement_summary(comparison: pd.DataFrame, output_path: Path) -> Path:
    if "sentiment_label_agree" not in comparison.columns:
        _require_columns(comparison, ["finbert_label", "lm_label_normalized"], "comparison")
        comparison = comparison.copy()
        comparison["sentiment_label_agree"] = (
            comparison["finbert_label"] == comparison["lm_label_normalized"]
        )

    n = len(comparison)
    agreement = int(comparison["sentiment_label_agree"].fillna(False).sum())
    disagreement = int(n - agreement)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.bar(["Agree", "Disagree"], [agreement, disagreement], color=["#2f7d59", "#b84a45"])
    ax.set_title("Model Agreement Summary")
    ax.set_ylabel("Transcript count")
    ax.set_ylim(0, max(1, agreement, disagreement) + 0.5)

    for index, value in enumerate([agreement, disagreement]):
        ax.text(index, value + 0.04, str(value), ha="center", va="bottom", fontsize=10)

    _add_small_n_note(ax, n)
    return _save_figure(fig, output_path)


def generate_day12_figures(
    comparison_path: Path = COMPARISON_PATH,
    event_study_path: Path = EVENT_STUDY_PATH,
    lm_scores_path: Path = LM_SCORES_PATH,
    sentiment_features_path: Path = SENTIMENT_FEATURES_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate all Day 12 figures under reports/figures."""

    comparison, event_study = _load_inputs(
        comparison_path=comparison_path,
        event_study_path=event_study_path,
        lm_scores_path=lm_scores_path,
        sentiment_features_path=sentiment_features_path,
    )
    if len(comparison) < 2:
        print("Warning: n < 2; figures are descriptive snapshots only.")

    output_paths = [
        plot_score_comparison(comparison, figures_dir / FIGURE_FILENAMES["score_comparison"]),
        plot_sentiment_return_snapshot(comparison, figures_dir / FIGURE_FILENAMES["sentiment_return"]),
        plot_event_study_return_horizon(event_study, figures_dir / FIGURE_FILENAMES["return_horizon"]),
        plot_model_agreement_summary(comparison, figures_dir / FIGURE_FILENAMES["agreement"]),
    ]
    return output_paths


def main() -> None:
    output_paths = generate_day12_figures()
    print("Wrote Day 12 figures:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
