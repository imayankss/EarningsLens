"""
app/streamlit_app.py
=====================
Earnings Call Sentiment Analyzer — Streamlit Dashboard

Run from project root:
    streamlit run app/streamlit_app.py

Pages:
    Overview        — project summary and pipeline status
    Sentiment       — FinBERT and LM score explorer
    Event Study     — abnormal return analysis
    Methodology     — technical documentation
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Make src importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config, get_all_tickers
from src.visualization.charts import SentimentCharts

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Call Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_SENTIMENT_DIR   = Path("data/processed/sentiment")
_EVENT_STUDY_DIR = Path("data/processed/event_study")
_PROCESSED_MARKET = Path("data/processed/market")


# ── Helpers ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_sentiment_data() -> pd.DataFrame | None:
    files = list(_SENTIMENT_DIR.glob("*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@st.cache_data(ttl=300)
def load_event_data() -> pd.DataFrame | None:
    files = list(_EVENT_STUDY_DIR.glob("*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _pipeline_status() -> dict[str, bool]:
    return {
        "Transcripts downloaded" : any(Path("data/raw/transcripts").glob("*.parquet")),
        "Transcripts cleaned"    : any(Path("data/interim/transcripts").glob("*.parquet")),
        "Chunks generated"       : any(Path("data/interim/chunks").glob("*.parquet")),
        "FinBERT scores computed": any(_SENTIMENT_DIR.glob("*.parquet")),
        "Event study computed"   : any(_EVENT_STUDY_DIR.glob("*.parquet")),
    }


# ── Pages ────────────────────────────────────────────────────────
def page_overview() -> None:
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers",      len(get_all_tickers()))
    col2.metric("NLP Model",    "FinBERT")
    col3.metric("Baseline",     "LM Dict")
    col4.metric("Event Windows","1/2/3/5/10D")

    st.divider()
    st.subheader("Pipeline Status")
    status = _pipeline_status()
    for step, done in status.items():
        icon = "✅" if done else "⬜"
        st.write(f"{icon}  {step}")

    if not any(status.values()):
        st.info(
            "Pipeline has not been run yet.  \n"
            "Execute: `python scripts/run_pipeline.py` to populate data."
        )

    st.divider()
    st.subheader("Architecture")
    st.code("""
Earnings Transcripts (HuggingFace)
        │
        ▼
  TranscriptLoader  →  data/raw/transcripts/
        │
        ▼
  TranscriptCleaner + TranscriptChunker  →  data/interim/
        │
   ┌────┴────┐
   ▼         ▼
FinBERT    LM Baseline
Pipeline   (Loughran-McDonald)
   │         │
   └────┬────┘
        ▼
  SentimentAggregator  →  data/processed/sentiment/
        │
        ▼
  MarketDataLoader (yfinance)  →  data/raw/market/
        │
        ▼
  EventStudyEngine  →  data/processed/event_study/
        │
        ▼
  Streamlit Dashboard  ←  you are here
    """, language="text")


def page_sentiment() -> None:
    st.header("Sentiment Analysis")

    df = load_sentiment_data()
    if df is None:
        st.warning("No sentiment data found. Run the pipeline first.")
        return

    # ── Filters ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    tickers = sorted(df["ticker"].unique().tolist()) if "ticker" in df.columns else []
    selected = col1.multiselect("Filter tickers", tickers, default=tickers[:5])
    score_col = col2.selectbox("Score column", ["finbert_score", "lm_tone"])

    if selected:
        df = df[df["ticker"].isin(selected)]

    # ── Summary metrics ──────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Transcripts",    f"{len(df):,}")
    m2.metric("Mean Score",     f"{df[score_col].mean():.4f}" if score_col in df.columns else "—")
    m3.metric("% Positive",
              f"{(df[score_col] > 0).mean()*100:.1f}%" if score_col in df.columns else "—")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Distribution")
        if score_col in df.columns:
            fig = SentimentCharts.sentiment_distribution(df, score_col=score_col)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "finbert_score" in df.columns and "lm_tone" in df.columns:
            st.subheader("FinBERT vs LM Tone")
            fig = SentimentCharts.finbert_vs_lm(df)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Data Table")
    st.dataframe(df.head(200), use_container_width=True)


def page_event_study() -> None:
    st.header("Event Study Results")

    event_df = load_event_data()
    if event_df is None:
        st.warning("No event study data found. Run the pipeline first.")
        return

    # ── Summary ──────────────────────────────────────────────────
    windows = [1, 2, 3, 5, 10]
    cols = st.columns(len(windows))
    for col, w in zip(cols, windows):
        rc = f"ar_{w}d"
        if rc in event_df.columns:
            col.metric(f"{w}D Mean AR", f"{event_df[rc].mean():.3%}")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        return_col = st.selectbox("Return window", [f"ar_{w}d" for w in windows])
        sent_col   = st.selectbox("Sentiment", ["finbert_score", "lm_tone"])
        if all(c in event_df.columns for c in [sent_col, return_col]):
            fig = SentimentCharts.sentiment_vs_returns(
                event_df, sentiment_col=sent_col, return_col=return_col
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        tickers = sorted(event_df["ticker"].unique().tolist()) if "ticker" in event_df.columns else []
        ticker  = st.selectbox("Ticker for CAR timeline", tickers)
        if ticker:
            fig = SentimentCharts.car_timeline(event_df, ticker=ticker)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Correlation Matrix")
    try:
        from src.event_study.engine import EventStudyEngine
        engine   = EventStudyEngine()
        corr_df  = engine.sentiment_return_correlation(event_df)
        if not corr_df.empty:
            fig = SentimentCharts.correlation_heatmap(corr_df)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(corr_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not compute correlations: {e}")

    st.divider()
    st.subheader("Raw Events Table")
    st.dataframe(event_df.head(300), use_container_width=True)


def page_methodology() -> None:
    st.header("Methodology")
    st.markdown("""
### NLP Models

**FinBERT** (`ProsusAI/finbert`)
- BERT fine-tuned on financial news (Reuters, Bloomberg)
- Outputs: `positive_prob`, `negative_prob`, `neutral_prob`
- Chunk score: `sentiment_score = positive_prob − negative_prob ∈ [−1, +1]`
- Transcript score: confidence-weighted average across chunks

**Loughran-McDonald (2011) Lexicon**
- Gold-standard financial sentiment wordlist
- `LM_Tone = (Positive − Negative) / (Positive + Negative)`
- Used as interpretable academic baseline

---

### Preprocessing

| Step | Detail |
|---|---|
| Cleaning | Remove boilerplate, legal disclaimers, operator cues |
| Segmentation | Split prepared remarks from analyst Q&A |
| Chunking | 400-token sliding window, 50-token sentence-aware overlap |
| Tokeniser | FinBERT tokenizer (WordPiece, finance vocab) |

---

### Event Study Design

| Parameter | Value |
|---|---|
| Benchmark | S&P 500 (^GSPC) |
| Abnormal Return | AR = R_stock − R_benchmark |
| CAR | Σ AR over window |
| Windows | 1D, 2D, 3D, 5D, 10D |
| Timing | After-market → T+1 event date |
| Estimation Window | 120 trading days pre-event |

---

### Dataset Sources

| Component | Source |
|---|---|
| Transcripts | [Bose345/sp500_earnings_transcripts](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts) |
| Sentiment Labels | [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) |
| Market Data | yfinance (adjusted close) |
| LM Dictionary | [SRAF Notre Dame](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) |
| SEC Filings | SEC EDGAR APIs |
    """)


# ── Main ────────────────────────────────────────────────────────
def main() -> None:
    st.title("📊 Earnings Call Sentiment Analyzer")
    st.caption("Financial NLP · FinBERT · Loughran-McDonald · Event Study")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Sentiment Analysis", "Event Study", "Methodology"],
    )

    if   page == "Overview"           : page_overview()
    elif page == "Sentiment Analysis" : page_sentiment()
    elif page == "Event Study"        : page_event_study()
    elif page == "Methodology"        : page_methodology()


if __name__ == "__main__":
    main()
