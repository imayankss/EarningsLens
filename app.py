"""Recruiter-ready Streamlit dashboard for the earnings-call sentiment project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.utils.io_utils import file_exists as _shared_file_exists
from src.utils.io_utils import format_file_size


PROJECT_ROOT = Path(__file__).resolve().parent

MASTER_PARQUET = PROJECT_ROOT / "data/processed/master_dataset.parquet"
MASTER_CSV = PROJECT_ROOT / "data/processed/master_dataset.csv"
COMPARISON_PARQUET = PROJECT_ROOT / "data/processed/analysis/sentiment_model_comparison.parquet"
COMPARISON_CSV = PROJECT_ROOT / "data/processed/analysis/sentiment_model_comparison.csv"
NLP_PARQUET = PROJECT_ROOT / "data/processed/nlp/advanced_nlp_features.parquet"
NLP_CSV = PROJECT_ROOT / "data/processed/nlp/advanced_nlp_features.csv"
TABLES_DIR = PROJECT_ROOT / "reports/tables"
FIGURES_DIR = PROJECT_ROOT / "reports/figures"
MODEL_METADATA = PROJECT_ROOT / "models/predictive_model_metadata.json"

NAVIGATION = [
    "Overview",
    "Sentiment Analysis",
    "Event Study",
    "Advanced NLP",
    "Statistical Analysis",
    "Prediction Modeling",
    "Figures Gallery",
    "Downloads",
]

DAY13_FIGURES = [
    "sentiment_distribution.png",
    "finbert_lm_comparison.png",
    "returns_vs_sentiment.png",
    "event_study_return_horizon.png",
    "correlation_heatmap.png",
    "model_agreement_summary.png",
]

DAY12_FIGURES = [
    "day12_finbert_vs_lm_score_comparison.png",
    "day12_sentiment_vs_return_snapshot.png",
    "day12_event_study_return_horizon.png",
    "day12_model_agreement_summary.png",
]

FIGURE_FILES = DAY13_FIGURES + DAY12_FIGURES

DOWNLOAD_GROUPS = {
    "Core datasets": [
        PROJECT_ROOT / "data/processed/master_dataset.csv",
    ],
    "Sentiment reports": [
        TABLES_DIR / "sentiment_model_comparison.csv",
        TABLES_DIR / "sentiment_model_comparison_summary.csv",
    ],
    "Statistical reports": [
        TABLES_DIR / "statistical_summary.csv",
        TABLES_DIR / "correlation_analysis.csv",
        TABLES_DIR / "regression_results.csv",
        TABLES_DIR / "statistical_warnings.csv",
    ],
    "NLP reports": [
        TABLES_DIR / "keyword_summary.csv",
        TABLES_DIR / "topic_frequency.csv",
        TABLES_DIR / "bigram_summary.csv",
        TABLES_DIR / "uncertainty_summary.csv",
        TABLES_DIR / "speaker_sentiment_summary.csv",
    ],
    "Prediction reports": [
        TABLES_DIR / "prediction_model_summary.csv",
        TABLES_DIR / "prediction_model_metrics.csv",
        TABLES_DIR / "prediction_model_warnings.csv",
        TABLES_DIR / "prediction_feature_importance.csv",
    ],
}

PIPELINE_CHECKS = {
    "FinBERT sentiment complete": PROJECT_ROOT / "data/processed/sentiment/sentiment_features.parquet",
    "LM baseline complete": PROJECT_ROOT / "data/processed/sentiment/lm_scores.parquet",
    "Event-study complete": PROJECT_ROOT / "data/processed/event_study/event_study.parquet",
    "Master dataset complete": MASTER_PARQUET,
    "Statistical analysis complete": TABLES_DIR / "statistical_summary.csv",
    "Prediction modeling complete": TABLES_DIR / "prediction_model_metrics.csv",
}

SMALL_N_MESSAGE = (
    "Current dataset contains one event, so charts and statistics are descriptive snapshots, "
    "not inferential conclusions."
)


def file_exists(path: Path | str) -> bool:
    return _shared_file_exists(path)


def get_file_size(path: Path | str) -> str:
    return format_file_size(path)


def count_available_files(paths: list[Path]) -> int:
    return sum(1 for path in paths if path.exists())


def format_filename_caption(path: Path | str) -> str:
    return Path(path).stem.replace("_", " ").replace("-", " ").title()


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


@st.cache_data(show_spinner=False)
def safe_load_table(path: str, fallback_path: str | None = None) -> pd.DataFrame:
    """Load a CSV or parquet table safely, returning an empty DataFrame if missing."""

    candidates = [Path(path)]
    if fallback_path is not None:
        candidates.append(Path(fallback_path))

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".parquet":
                return pd.read_parquet(candidate)
            if candidate.suffix.lower() == ".csv":
                return pd.read_csv(candidate)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def safe_load_json(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_figures(figures_dir: Path = FIGURES_DIR, figure_files: list[str] | None = None) -> list[Path]:
    names = figure_files if figure_files is not None else FIGURE_FILES
    return [figures_dir / name for name in names if (figures_dir / name).exists()]


def count_report_tables(tables_dir: Path = TABLES_DIR) -> int:
    return len(list(tables_dir.glob("*.csv"))) if tables_dir.exists() else 0


def extract_overview_metrics(
    master: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    tables_dir: Path = TABLES_DIR,
) -> dict[str, Any]:
    """Extract compact dashboard metrics from a master dataset."""

    if master.empty:
        return {
            "transcripts": 0,
            "tickers": 0,
            "current_ticker": "-",
            "finbert_score": None,
            "lm_tone_score": None,
            "directional_agreement": "-",
            "figures": len(discover_figures(figures_dir)),
            "reports": count_report_tables(tables_dir),
        }

    first = master.iloc[0]
    directional_agreement = first.get("directional_agreement", "-")
    if hasattr(directional_agreement, "item"):
        directional_agreement = directional_agreement.item()
    return {
        "transcripts": int(master["transcript_id"].nunique()) if "transcript_id" in master.columns else len(master),
        "tickers": int(master["ticker"].nunique()) if "ticker" in master.columns else 0,
        "current_ticker": first.get("ticker", "-"),
        "finbert_score": first.get("finbert_score"),
        "lm_tone_score": first.get("lm_tone_score"),
        "directional_agreement": directional_agreement,
        "figures": len(discover_figures(figures_dir)),
        "reports": count_report_tables(tables_dir),
    }


def _status_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "status" not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df["status"].value_counts(dropna=False).items()}


def _filter_status(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty or "status" not in df.columns or label == "All":
        return df
    return df[df["status"].astype(str) == label]


def _load_all() -> dict[str, Any]:
    return {
        "master": safe_load_table(str(MASTER_PARQUET), str(MASTER_CSV)),
        "comparison": safe_load_table(str(COMPARISON_PARQUET), str(COMPARISON_CSV)),
        "nlp_features": safe_load_table(str(NLP_PARQUET), str(NLP_CSV)),
        "master_summary": safe_load_table(str(TABLES_DIR / "master_dataset_summary.csv")),
        "comparison_table": safe_load_table(str(TABLES_DIR / "sentiment_model_comparison.csv")),
        "comparison_summary": safe_load_table(str(TABLES_DIR / "sentiment_model_comparison_summary.csv")),
        "statistical_summary": safe_load_table(str(TABLES_DIR / "statistical_summary.csv")),
        "correlation": safe_load_table(str(TABLES_DIR / "correlation_analysis.csv")),
        "regression": safe_load_table(str(TABLES_DIR / "regression_results.csv")),
        "statistical_warnings": safe_load_table(str(TABLES_DIR / "statistical_warnings.csv")),
        "keywords": safe_load_table(str(TABLES_DIR / "keyword_summary.csv")),
        "topics": safe_load_table(str(TABLES_DIR / "topic_frequency.csv")),
        "bigrams": safe_load_table(str(TABLES_DIR / "bigram_summary.csv")),
        "uncertainty": safe_load_table(str(TABLES_DIR / "uncertainty_summary.csv")),
        "speaker_sentiment": safe_load_table(str(TABLES_DIR / "speaker_sentiment_summary.csv")),
        "prediction_summary": safe_load_table(str(TABLES_DIR / "prediction_model_summary.csv")),
        "prediction_metrics": safe_load_table(str(TABLES_DIR / "prediction_model_metrics.csv")),
        "prediction_warnings": safe_load_table(str(TABLES_DIR / "prediction_model_warnings.csv")),
        "prediction_importance": safe_load_table(str(TABLES_DIR / "prediction_feature_importance.csv")),
        "metadata": safe_load_json(str(MODEL_METADATA)),
    }


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-panel: rgba(15, 23, 42, 0.46);
            --bg-panel-soft: rgba(15, 23, 42, 0.28);
            --border: rgba(148, 163, 184, 0.22);
            --border-strong: rgba(148, 163, 184, 0.34);
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #5eead4;
            --warning: #fde68a;
        }
        .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--border);
        }
        .app-header {
            padding: 1.35rem 1.5rem;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.40));
            margin-bottom: 1.1rem;
        }
        .app-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .app-title {
            color: var(--text);
            font-size: 2.05rem;
            font-weight: 780;
            letter-spacing: 0;
            margin: 0;
        }
        .app-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin-top: 0.45rem;
            max-width: 780px;
        }
        .badge-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.85rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border: 1px solid rgba(94, 234, 212, 0.28);
            color: #99f6e4;
            background: rgba(20, 184, 166, 0.10);
            border-radius: 999px;
            font-size: 0.82rem;
        }
        .metric-card {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--bg-panel);
            padding: 0.9rem 1rem 0.95rem 1rem;
            min-height: 100px;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.10);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            font-weight: 650;
        }
        .metric-value {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 760;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }
        .metric-help {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }
        .section-card, .executive-card {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--bg-panel-soft);
            padding: 1rem 1.05rem;
            margin: 0.65rem 0 1rem 0;
        }
        .executive-card {
            border-color: var(--border-strong);
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.55), rgba(15, 23, 42, 0.28));
        }
        .warning-panel {
            border: 1px solid rgba(251, 191, 36, 0.34);
            border-radius: 13px;
            background: rgba(251, 191, 36, 0.10);
            color: var(--warning);
            padding: 0.9rem 1rem;
            margin: 0.65rem 0 1rem 0;
        }
        .success-panel {
            border: 1px solid rgba(94, 234, 212, 0.25);
            border-radius: 13px;
            background: rgba(20, 184, 166, 0.08);
            color: #ccfbf1;
            padding: 0.85rem 1rem;
            margin: 0.65rem 0 1rem 0;
        }
        .missing-panel {
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.22);
            color: var(--muted);
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
        }
        .sidebar-card {
            border: 1px solid var(--border);
            border-radius: 13px;
            background: rgba(15, 23, 42, 0.35);
            padding: 0.85rem 0.9rem;
            margin: 0.75rem 0;
            font-size: 0.9rem;
        }
        .check-row {
            padding: 0.22rem 0;
            color: var(--text);
        }
        .caption-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: -0.2rem;
            margin-bottom: 0.6rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 11px;
            overflow: hidden;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(metrics: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-kicker">Financial NLP Research Prototype</div>
            <div class="app-title">Earnings Call Sentiment Analyzer</div>
            <div class="app-subtitle">
                Financial NLP, explainable sentiment, event-study analysis, and prediction modeling,
                presented as a compact production-style research dashboard.
            </div>
            <div class="badge-row">
                <span class="status-badge">Research prototype</span>
                <span class="status-badge">{metrics["transcripts"]} transcript currently loaded</span>
                <span class="status-badge">Descriptive snapshot only</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    help_markup = f'<div class="metric-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{_format_value(value)}</div>
            {help_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning_panel(message: str) -> None:
    st.markdown(f'<div class="warning-panel">{message}</div>', unsafe_allow_html=True)


def _render_success_panel(message: str) -> None:
    st.markdown(f'<div class="success-panel">{message}</div>', unsafe_allow_html=True)


def _render_missing_panel(label: str, path: Path | None = None) -> None:
    detail = f"Missing optional file: {path}" if path else "Required data is not available."
    st.markdown(f'<div class="missing-panel"><strong>{label}</strong><br>{detail}</div>', unsafe_allow_html=True)


def _section(title: str, body: str | None = None) -> None:
    st.subheader(title)
    if body:
        st.markdown(f'<div class="section-card">{body}</div>', unsafe_allow_html=True)


def render_table_section(
    title: str,
    df: pd.DataFrame,
    description: str | None = None,
    max_rows: int = 20,
    status_filter: bool = False,
) -> None:
    _section(title, description)
    if df.empty:
        _render_missing_panel(title)
        return
    display_df = df
    if status_filter and "status" in df.columns:
        options = ["All"] + sorted(df["status"].astype(str).dropna().unique().tolist())
        selected = st.selectbox(f"Filter {title} by status", options, key=f"{title}_status_filter")
        display_df = _filter_status(df, selected)
    st.dataframe(display_df.head(max_rows), width="stretch", hide_index=True)


def _show_figure(path: Path, caption: str | None = None) -> None:
    label = caption or format_filename_caption(path)
    if path.exists():
        st.image(str(path), caption=label, width="stretch")
    else:
        _render_missing_panel(label, path)


def render_download_button(path: Path) -> None:
    if not path.exists():
        _render_missing_panel(path.name, path)
        return
    st.download_button(
        label=f"{path.name} · {get_file_size(path)}",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
        width="stretch",
    )


def _render_sidebar(metrics: dict[str, Any]) -> str:
    st.sidebar.markdown("### Earnings Call Sentiment")
    st.sidebar.caption("Recruiter-ready research dashboard")
    page = st.sidebar.radio("Navigate", NAVIGATION)
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <strong>Dataset status</strong><br>
            Transcripts: {_format_value(metrics["transcripts"])}<br>
            Tickers: {_format_value(metrics["tickers"])}<br>
            Current ticker: {_format_value(metrics["current_ticker"])}
        </div>
        <div class="sidebar-card">
            <strong>Prototype dataset: n=1</strong><br>
            Statistical inference and ML validation are intentionally skipped until more events are loaded.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def _metric_grid(items: list[tuple[str, Any, str | None]], columns: int = 4) -> None:
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (label, value, help_text) in zip(cols, items[start : start + columns]):
            with col:
                render_metric_card(label, value, help_text)


def _first_row_value(df: pd.DataFrame, column: str) -> Any:
    if df.empty or column not in df.columns:
        return None
    return df[column].iloc[0]


def _return_summary(master: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    rows = []
    for column in columns:
        if column in master.columns:
            rows.append(
                {
                    "metric": column,
                    "type": "abnormal return" if column.startswith("ar_") else "raw return",
                    "value": pd.to_numeric(master[column], errors="coerce").iloc[0] if not master.empty else None,
                }
            )
    return pd.DataFrame(rows)


def page_overview(data: dict[str, Any], metrics: dict[str, Any]) -> None:
    render_warning_panel(SMALL_N_MESSAGE)
    st.markdown(
        """
        <div class="executive-card">
            <strong>Executive summary.</strong> This dashboard demonstrates an end-to-end financial NLP workflow:
            transcript cleaning, FinBERT sentiment, Loughran-McDonald baseline comparison, event-study returns,
            advanced NLP summaries, guarded statistical analysis, and prediction-model scaffolding. The current
            dataset has one event, so the interface is intentionally framed as pipeline-ready rather than
            performance-validated.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _metric_grid(
        [
            ("Transcripts", metrics["transcripts"], "Loaded events"),
            ("Tickers", metrics["tickers"], "Unique symbols"),
            ("Figures", metrics["figures"], "Available PNGs"),
            ("Reports", metrics["reports"], "CSV tables"),
            ("FinBERT score", metrics["finbert_score"], "Contextual tone"),
            ("LM score", metrics["lm_tone_score"], "Dictionary baseline"),
            ("Agreement", metrics["directional_agreement"], "Direction match"),
        ],
        columns=4,
    )

    _section("Pipeline Status")
    for label, path in PIPELINE_CHECKS.items():
        marker = "Complete" if path.exists() else "Missing"
        st.markdown(f'<div class="check-row"><strong>{marker}</strong> · {label}</div>', unsafe_allow_html=True)

    _section(
        "What This Dashboard Demonstrates",
        "A clean artifact-first workflow for financial NLP: each page reads generated data products, "
        "surfaces limitations explicitly, and avoids fabricating statistics or model performance.",
    )
    render_table_section("Master Dataset Preview", data["master"], max_rows=10)


def page_sentiment(data: dict[str, Any]) -> None:
    master = data["master"]
    comparison = data["comparison_table"] if not data["comparison_table"].empty else data["comparison"]

    _section(
        "Sentiment Analysis",
        "FinBERT captures contextual tone; LM provides an explainable dictionary baseline.",
    )
    _metric_grid(
        [
            ("FinBERT score", _first_row_value(master, "finbert_score"), "Positive minus negative probability"),
            ("LM tone score", _first_row_value(master, "lm_tone_score"), "Dictionary tone"),
            ("Score difference", _first_row_value(master, "score_difference"), "FinBERT minus LM"),
            ("Divergence flag", _first_row_value(master, "divergence_flag"), "Directional disagreement"),
        ],
        columns=4,
    )

    agreement = _first_row_value(master, "directional_agreement")
    if agreement is not None:
        panel = "FinBERT and LM agree on sentiment direction." if bool(agreement) else "FinBERT and LM diverge on sentiment direction."
        _render_success_panel(panel)

    columns = [
        "transcript_id",
        "ticker",
        "finbert_score",
        "lm_tone_score",
        "score_difference",
        "absolute_difference",
        "directional_agreement",
        "divergence_flag",
    ]
    compact = master[[column for column in columns if column in master.columns]] if not master.empty else pd.DataFrame()
    render_table_section("Compact Sentiment Metrics", compact, max_rows=20)
    render_table_section("FinBERT vs LM Comparison Table", comparison, max_rows=20)

    col1, col2 = st.columns(2)
    with col1:
        _show_figure(FIGURES_DIR / "finbert_lm_comparison.png", "FinBERT and LM score comparison")
    with col2:
        _show_figure(FIGURES_DIR / "sentiment_distribution.png", "Sentiment score distribution")


def page_event_study(data: dict[str, Any]) -> None:
    master = data["master"]
    render_warning_panel("Event-study conclusions require more events; this page shows one event-level snapshot.")
    returns = _return_summary(master)
    _metric_grid(
        [
            ("AR 1D", _first_row_value(master, "ar_1d"), "Abnormal return"),
            ("Return 1D", _first_row_value(master, "return_1d"), "Raw return"),
            ("AR 3D", _first_row_value(master, "ar_3d"), "Abnormal return"),
            ("Return 5D", _first_row_value(master, "return_5d"), "Raw return"),
        ],
        columns=4,
    )
    render_table_section(
        "Return Horizon Table",
        returns,
        "AR columns are abnormal returns; return columns are raw post-event returns.",
        max_rows=20,
    )
    col1, col2 = st.columns(2)
    with col1:
        _show_figure(FIGURES_DIR / "event_study_return_horizon.png", "Event-study return horizon")
    with col2:
        _show_figure(FIGURES_DIR / "returns_vs_sentiment.png", "Returns vs sentiment snapshot")


def page_advanced_nlp(data: dict[str, Any]) -> None:
    _section("Advanced NLP", "Lightweight explainability features extracted from transcript text and chunk-level scores.")
    tabs = st.tabs(["Keywords", "Topics", "Bigrams", "Uncertainty", "Speaker Sentiment"])
    tab_config = [
        (tabs[0], "Keywords", data["keywords"], "Most frequent content words in the transcript.", TABLES_DIR / "keyword_summary.csv"),
        (tabs[1], "Topics", data["topics"], "Financial-topic bucket counts from a small curated lexicon.", TABLES_DIR / "topic_frequency.csv"),
        (tabs[2], "Bigrams", data["bigrams"], "Common two-word phrases after stopword removal.", TABLES_DIR / "bigram_summary.csv"),
        (tabs[3], "Uncertainty", data["uncertainty"], "Hedging and uncertainty language counts.", TABLES_DIR / "uncertainty_summary.csv"),
        (tabs[4], "Speaker Sentiment", data["speaker_sentiment"], "Chunk sentiment grouped by speaker or section when available.", TABLES_DIR / "speaker_sentiment_summary.csv"),
    ]
    for tab, title, df, description, path in tab_config:
        with tab:
            render_table_section(title, df, description, max_rows=20)
            render_download_button(path)


def page_statistical_analysis(data: dict[str, Any]) -> None:
    correlation = data["correlation"]
    regression = data["regression"]
    corr_counts = _status_counts(correlation)
    reg_counts = _status_counts(regression)
    if corr_counts == {"insufficient_sample_size": len(correlation)} and reg_counts == {"insufficient_sample_size": len(regression)}:
        render_warning_panel("Statistical tests were skipped because n=1. The module is ready for larger datasets.")

    _metric_grid(
        [
            ("Correlation rows", len(correlation), "Sentiment/return pairs"),
            ("Regression rows", len(regression), "Target models"),
            ("Skipped correlations", corr_counts.get("insufficient_sample_size", 0), "Guarded results"),
            ("Skipped regressions", reg_counts.get("insufficient_sample_size", 0), "Guarded results"),
        ],
        columns=4,
    )
    _section("Why Tests Are Skipped", "Pearson/Spearman correlations need at least two paired observations, and regressions require more observations than features. With n=1, the correct result is a transparent skip.")
    render_table_section("Statistical Summary", data["statistical_summary"], max_rows=20)
    render_table_section("Correlation Analysis", correlation, max_rows=40, status_filter=True)
    render_table_section("Regression Results", regression, max_rows=30, status_filter=True)
    render_table_section("Statistical Warnings", data["statistical_warnings"], max_rows=40)


def page_prediction_modeling(data: dict[str, Any]) -> None:
    metadata = data["metadata"]
    metrics = data["prediction_metrics"]
    summary = data["prediction_summary"]
    status_counts = _status_counts(metrics)
    attempted = len(metadata.get("models_attempted", [])) if metadata else len(metrics)
    trained = len(metadata.get("models_trained", [])) if metadata else status_counts.get("trained", 0)
    skipped = attempted - trained
    feature_columns = metadata.get("feature_columns", []) if metadata else []
    target_columns = metadata.get("target_columns", []) if metadata else []

    if metrics.empty or status_counts.get("insufficient_sample_size", 0) == len(metrics):
        render_warning_panel("Model training was correctly skipped because the dataset has fewer than 10 rows.")
    _section("Prediction Modeling", "Pipeline-ready, not performance-validated yet. No model performance is claimed for the current n=1 dataset.")
    _metric_grid(
        [
            ("Attempted models", attempted, "Target/model combinations"),
            ("Trained models", trained, "Expected 0 for n=1"),
            ("Skipped models", skipped, "Guarded skips"),
            ("Feature count", len(feature_columns), "Numeric features"),
            ("Target count", len(target_columns), "Binary targets"),
        ],
        columns=5,
    )
    with st.expander("Feature columns", expanded=False):
        st.write(feature_columns if feature_columns else "No feature metadata available.")
    render_table_section("Prediction Summary", summary, max_rows=20)
    render_table_section("Prediction Metrics", metrics, max_rows=40, status_filter=True)
    render_table_section("Prediction Warnings", data["prediction_warnings"], max_rows=40)
    render_table_section("Feature Importance", data["prediction_importance"], max_rows=30)
    _section("Model Metadata")
    st.json(metadata if metadata else {"status": "metadata unavailable"})


def page_figures_gallery() -> None:
    _section("Figures Gallery", "Generated PNG artifacts are shown directly; missing figures appear as quiet placeholders.")
    groups = {
        "Day 13 Professional Figures": DAY13_FIGURES,
        "Day 12 Legacy Figures": DAY12_FIGURES,
    }
    for group_name, names in groups.items():
        with st.expander(group_name, expanded=group_name.startswith("Day 13")):
            paths = [FIGURES_DIR / name for name in names]
            for start in range(0, len(paths), 2):
                cols = st.columns(2)
                for col, path in zip(cols, paths[start : start + 2]):
                    with col:
                        _show_figure(path, format_filename_caption(path))


def page_downloads() -> None:
    _section("Downloads", "Download generated CSV artifacts grouped by analysis stage.")
    for group, paths in DOWNLOAD_GROUPS.items():
        with st.expander(f"{group} · {count_available_files(paths)}/{len(paths)} available", expanded=True):
            for path in paths:
                render_download_button(path)


def main() -> None:
    st.set_page_config(
        page_title="Earnings Call Sentiment Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    data = _load_all()
    metrics = extract_overview_metrics(data["master"])
    _render_header(metrics)
    page = _render_sidebar(metrics)

    if page == "Overview":
        page_overview(data, metrics)
    elif page == "Sentiment Analysis":
        page_sentiment(data)
    elif page == "Event Study":
        page_event_study(data)
    elif page == "Advanced NLP":
        page_advanced_nlp(data)
    elif page == "Statistical Analysis":
        page_statistical_analysis(data)
    elif page == "Prediction Modeling":
        page_prediction_modeling(data)
    elif page == "Figures Gallery":
        page_figures_gallery()
    elif page == "Downloads":
        page_downloads()


if __name__ == "__main__":
    main()
