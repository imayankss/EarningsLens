"""Generate professional Day 13 charts from the master analysis artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_PARQUET_PATH = PROJECT_ROOT / "data/processed/master_dataset.parquet"
MASTER_CSV_PATH = PROJECT_ROOT / "data/processed/master_dataset.csv"
COMPARISON_PATH = PROJECT_ROOT / "data/processed/analysis/sentiment_model_comparison.parquet"
CORRELATION_PATH = PROJECT_ROOT / "reports/tables/correlation_analysis.csv"
STATISTICAL_SUMMARY_PATH = PROJECT_ROOT / "reports/tables/statistical_summary.csv"
FIGURES_DIR = PROJECT_ROOT / "reports/figures"

SMALL_N_NOTE = "Descriptive snapshot only; insufficient sample size for inference."

FIGURE_FILENAMES = {
    "sentiment_distribution": "sentiment_distribution.png",
    "finbert_lm_comparison": "finbert_lm_comparison.png",
    "returns_vs_sentiment": "returns_vs_sentiment.png",
    "event_study_return_horizon": "event_study_return_horizon.png",
    "correlation_heatmap": "correlation_heatmap.png",
    "model_agreement_summary": "model_agreement_summary.png",
}

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


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e6e8eb", linewidth=0.8)
    ax.set_axisbelow(True)


def _add_small_n_note(ax: plt.Axes, n: int) -> None:
    if n >= 2:
        return
    ax.text(
        0.02,
        0.98,
        SMALL_N_NOTE,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#7a4a00",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#fff4d6",
            "edgecolor": "#c99721",
            "linewidth": 0.8,
        },
    )


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _placeholder_figure(
    output_path: Path,
    title: str,
    message: str,
    warnings_list: list[str],
) -> Path:
    warnings_list.append(f"{title}: {message}")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=14)
    ax.text(
        0.5,
        0.52,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#333333",
        wrap=True,
    )
    return _save_figure(fig, output_path)


def _available_numeric(df: pd.DataFrame, columns: list[str]) -> list[str]:
    available = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            available.append(column)
    return available


def _short_label(row: pd.Series) -> str:
    ticker = str(row.get("ticker", "")).strip()
    transcript_id = str(row.get("transcript_id", "")).strip()
    if ticker and ticker.lower() != "nan":
        return ticker
    return transcript_id if transcript_id and transcript_id.lower() != "nan" else "Transcript"


def load_master_dataset(
    master_parquet_path: Path = MASTER_PARQUET_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
) -> pd.DataFrame:
    """Load master_dataset from parquet, falling back to CSV."""

    if master_parquet_path.exists():
        return pd.read_parquet(master_parquet_path)
    if master_csv_path.exists():
        return pd.read_csv(master_csv_path)
    raise FileNotFoundError(
        f"Missing master dataset. Tried parquet={master_parquet_path} and csv={master_csv_path}"
    )


def _read_optional_parquet(path: Path, warnings_list: list[str], name: str) -> pd.DataFrame:
    if not path.exists():
        warnings_list.append(f"{name}: missing optional input {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_optional_csv(path: Path, warnings_list: list[str], name: str) -> pd.DataFrame:
    if not path.exists():
        warnings_list.append(f"{name}: missing optional input {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_sentiment_distribution(
    master: pd.DataFrame,
    output_path: Path,
    warnings_list: list[str],
) -> Path:
    columns = _available_numeric(master, SENTIMENT_COLUMNS)
    if not columns:
        return _placeholder_figure(
            output_path,
            "Sentiment Score Distribution",
            f"Missing numeric sentiment columns: {', '.join(SENTIMENT_COLUMNS)}",
            warnings_list,
        )

    fig, ax = plt.subplots(figsize=(8.5, 5))
    data = [pd.to_numeric(master[column], errors="coerce").dropna() for column in columns]
    labels = [column.replace("_", " ") for column in columns]

    if len(master) < 2:
        means = [series.iloc[0] if len(series) else np.nan for series in data]
        ax.bar(labels, means, color=["#2f6f73", "#c65f34", "#6f5aa7", "#5b7f3b"][: len(labels)])
        ax.set_ylabel("Score value")
    else:
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_ylabel("Score distribution")

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Sentiment Score Distribution")
    ax.set_xlabel("Sentiment metric")
    ax.tick_params(axis="x", rotation=20)
    _style_axes(ax)
    _add_small_n_note(ax, len(master))
    return _save_figure(fig, output_path)


def plot_finbert_lm_comparison(
    master: pd.DataFrame,
    output_path: Path,
    warnings_list: list[str],
) -> Path:
    required = ["finbert_score", "lm_tone_score"]
    missing = [column for column in required if column not in _available_numeric(master, required)]
    if missing:
        return _placeholder_figure(
            output_path,
            "FinBERT vs LM Comparison",
            f"Missing numeric columns: {', '.join(missing)}",
            warnings_list,
        )

    labels = [_short_label(row) for _, row in master.iterrows()]
    x = np.arange(len(master))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(
        x - width / 2,
        pd.to_numeric(master["finbert_score"], errors="coerce"),
        width=width,
        label="FinBERT score",
        color="#2f6f73",
    )
    ax.bar(
        x + width / 2,
        pd.to_numeric(master["lm_tone_score"], errors="coerce"),
        width=width,
        label="LM tone score",
        color="#c65f34",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("FinBERT vs Loughran-McDonald Sentiment")
    ax.set_xlabel("Transcript")
    ax.set_ylabel("Sentiment score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if len(labels) <= 3 else 30, ha="center" if len(labels) <= 3 else "right")
    ax.legend(frameon=False)
    _style_axes(ax)
    _add_small_n_note(ax, len(master))
    return _save_figure(fig, output_path)


def plot_returns_vs_sentiment(
    master: pd.DataFrame,
    output_path: Path,
    warnings_list: list[str],
) -> Path:
    sentiment_candidates = _available_numeric(master, ["finbert_score", "lm_tone_score"])
    return_candidates = _available_numeric(master, RETURN_COLUMNS)
    if not sentiment_candidates or not return_candidates:
        missing = []
        if not sentiment_candidates:
            missing.append("numeric sentiment score column")
        if not return_candidates:
            missing.append("numeric return/ar column")
        return _placeholder_figure(
            output_path,
            "Returns vs Sentiment",
            f"Missing required data: {', '.join(missing)}",
            warnings_list,
        )

    x_column = "finbert_score" if "finbert_score" in sentiment_candidates else sentiment_candidates[0]
    y_column = "return_1d" if "return_1d" in return_candidates else return_candidates[0]
    x = pd.to_numeric(master[x_column], errors="coerce")
    y = pd.to_numeric(master[y_column], errors="coerce")
    labels = [_short_label(row) for _, row in master.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.scatter(x, y, s=90, color="#2f6f73", edgecolor="white", linewidth=0.8, label=y_column)
    for label, x_value, y_value in zip(labels, x, y):
        if pd.notna(x_value) and pd.notna(y_value):
            ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(7, 6), fontsize=9)

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Returns vs Sentiment")
    ax.set_xlabel(x_column.replace("_", " "))
    ax.set_ylabel(y_column.replace("_", " "))
    ax.legend(frameon=False)
    _style_axes(ax)
    _add_small_n_note(ax, len(master))
    return _save_figure(fig, output_path)


def _horizon_columns(master: pd.DataFrame) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for column in RETURN_COLUMNS:
        if column not in master.columns:
            continue
        values = pd.to_numeric(master[column], errors="coerce")
        if not values.notna().any():
            continue
        horizon = int(column.split("_")[1].replace("d", ""))
        matches.append((horizon, column))
    return matches


def plot_event_study_return_horizon(
    master: pd.DataFrame,
    output_path: Path,
    warnings_list: list[str],
) -> Path:
    horizon_columns = _horizon_columns(master)
    if not horizon_columns:
        return _placeholder_figure(
            output_path,
            "Event-Study Return Horizon",
            f"Missing numeric horizon return columns: {', '.join(RETURN_COLUMNS)}",
            warnings_list,
        )

    horizons = [horizon for horizon, _ in horizon_columns]
    values = [pd.to_numeric(master[column], errors="coerce").mean() for _, column in horizon_columns]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(horizons, values, marker="o", linewidth=2.2, color="#516f8f", label="Mean return/AR")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Event-Study Return Horizon")
    ax.set_xlabel("Event horizon in trading days")
    ax.set_ylabel("Mean return")
    ax.legend(frameon=False)
    _style_axes(ax)
    _add_small_n_note(ax, len(master))
    return _save_figure(fig, output_path)


def plot_correlation_heatmap(
    correlation: pd.DataFrame,
    output_path: Path,
    warnings_list: list[str],
) -> Path:
    required = {"x_column", "y_column", "pearson_r", "status"}
    if correlation.empty or not required.issubset(correlation.columns):
        return _placeholder_figure(
            output_path,
            "Correlation Heatmap",
            "Missing correlation analysis columns needed for a heatmap.",
            warnings_list,
        )

    usable = correlation[correlation["status"].eq("ok")].copy()
    usable["pearson_r"] = pd.to_numeric(usable["pearson_r"], errors="coerce")
    usable = usable.dropna(subset=["pearson_r"])
    if usable.empty:
        return _placeholder_figure(
            output_path,
            "Correlation Heatmap",
            "No valid correlations available; insufficient sample size or missing numeric pairs.",
            warnings_list,
        )

    matrix = usable.pivot_table(
        index="x_column",
        columns="y_column",
        values="pearson_r",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(matrix.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title("Sentiment/Return Correlation Heatmap")
    ax.set_xlabel("Return target")
    ax.set_ylabel("Sentiment feature")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(image, ax=ax, label="Pearson r")
    return _save_figure(fig, output_path)


def _agreement_counts(master: pd.DataFrame, comparison: pd.DataFrame) -> tuple[int, int] | None:
    if "directional_agreement" in master.columns:
        values = master["directional_agreement"].astype("boolean")
        agreement = int(values.fillna(False).sum())
        return agreement, int(len(values) - agreement)

    if "sentiment_label_agree" in comparison.columns:
        values = comparison["sentiment_label_agree"].astype("boolean")
        agreement = int(values.fillna(False).sum())
        return agreement, int(len(values) - agreement)

    if {"finbert_direction", "lm_direction"}.issubset(master.columns):
        values = master["finbert_direction"].astype("string") == master["lm_direction"].astype("string")
        agreement = int(values.fillna(False).sum())
        return agreement, int(len(values) - agreement)

    return None


def plot_model_agreement_summary(
    master: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
    warnings_list: list[str],
) -> Path:
    counts = _agreement_counts(master, comparison)
    if counts is None:
        return _placeholder_figure(
            output_path,
            "Model Agreement Summary",
            "Missing agreement columns: directional_agreement or sentiment_label_agree.",
            warnings_list,
        )

    agreement, disagreement = counts
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Agreement", "Divergence"], [agreement, disagreement], color=["#2f7d59", "#b84a45"])
    ax.set_title("Model Agreement Summary")
    ax.set_xlabel("Model direction comparison")
    ax.set_ylabel("Transcript count")
    ax.set_ylim(0, max(1, agreement, disagreement) + 0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.04, f"{int(height)}", ha="center", va="bottom")
    _style_axes(ax)
    _add_small_n_note(ax, len(master))
    return _save_figure(fig, output_path)


def generate_professional_charts(
    master_parquet_path: Path = MASTER_PARQUET_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
    comparison_path: Path = COMPARISON_PATH,
    correlation_path: Path = CORRELATION_PATH,
    statistical_summary_path: Path = STATISTICAL_SUMMARY_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[list[Path], list[str]]:
    """Generate all Day 13 figures under reports/figures."""

    warnings_list: list[str] = []
    master = load_master_dataset(master_parquet_path, master_csv_path)
    comparison = _read_optional_parquet(comparison_path, warnings_list, "sentiment_model_comparison")
    correlation = _read_optional_csv(correlation_path, warnings_list, "correlation_analysis")
    _read_optional_csv(statistical_summary_path, warnings_list, "statistical_summary")

    if len(master) < 2:
        warnings_list.append(SMALL_N_NOTE)

    output_paths = [
        plot_sentiment_distribution(
            master,
            figures_dir / FIGURE_FILENAMES["sentiment_distribution"],
            warnings_list,
        ),
        plot_finbert_lm_comparison(
            master,
            figures_dir / FIGURE_FILENAMES["finbert_lm_comparison"],
            warnings_list,
        ),
        plot_returns_vs_sentiment(
            master,
            figures_dir / FIGURE_FILENAMES["returns_vs_sentiment"],
            warnings_list,
        ),
        plot_event_study_return_horizon(
            master,
            figures_dir / FIGURE_FILENAMES["event_study_return_horizon"],
            warnings_list,
        ),
        plot_correlation_heatmap(
            correlation,
            figures_dir / FIGURE_FILENAMES["correlation_heatmap"],
            warnings_list,
        ),
        plot_model_agreement_summary(
            master,
            comparison,
            figures_dir / FIGURE_FILENAMES["model_agreement_summary"],
            warnings_list,
        ),
    ]
    return output_paths, warnings_list


def main() -> None:
    output_paths, warnings_list = generate_professional_charts()
    print("Wrote Day 13 professional figures:")
    for path in output_paths:
        print(f"- {path}")
    if warnings_list:
        print("Warnings:")
        for warning in warnings_list:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
