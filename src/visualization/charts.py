"""
src/visualization/charts.py
=============================
Production Plotly charts for the Streamlit dashboard and PDF reports.

Design rule: every function returns a plotly Figure object.
Nothing is rendered here — the caller decides where to display it.
This keeps every chart independently testable.

Charts available:
  sentiment_distribution()    — histogram of FinBERT scores
  sentiment_vs_returns()      — scatter: sentiment vs abnormal return
  car_timeline()              — CAR over time for one ticker
  correlation_heatmap()       — sentiment × return window matrix
  finbert_vs_lm()             — FinBERT score vs LM tone scatter
  sector_sentiment_bar()      — mean sentiment grouped by sector
  return_distribution()       — histogram of abnormal returns
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_TEMPLATE = "plotly_white"
_PRIMARY  = "#2563EB"   # blue
_RED      = "#DC2626"
_GREEN    = "#16A34A"


class SentimentCharts:
    """Static chart factory — all methods return plotly Figure objects."""

    # ── Sentiment ───────────────────────────────────────────────

    @staticmethod
    def sentiment_distribution(
        df        : pd.DataFrame,
        score_col : str = "finbert_score",
        title     : str = "FinBERT Sentiment Score Distribution",
    ) -> go.Figure:
        """Histogram of transcript-level sentiment scores."""
        fig = px.histogram(
            df,
            x=score_col,
            nbins=50,
            title=title,
            labels={score_col: "Sentiment Score (positive − negative)"},
            color_discrete_sequence=[_PRIMARY],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.6)
        fig.update_layout(
            xaxis_title="Sentiment Score",
            yaxis_title="Number of Transcripts",
            template=_TEMPLATE,
            showlegend=False,
        )
        return fig

    @staticmethod
    def sentiment_vs_returns(
        event_df     : pd.DataFrame,
        sentiment_col: str = "finbert_score",
        return_col   : str = "ar_5d",
    ) -> go.Figure:
        """Scatter plot: sentiment score vs. N-day abnormal return."""
        clean = event_df.dropna(subset=[sentiment_col, return_col])
        fig = px.scatter(
            clean,
            x=sentiment_col,
            y=return_col,
            color="ticker",
            hover_data=["ticker", "event_date"],
            title=f"Sentiment Score vs {return_col.upper()} Abnormal Return",
            labels={
                sentiment_col: "FinBERT Sentiment Score",
                return_col   : "Abnormal Return",
            },
            trendline="ols",
            template=_TEMPLATE,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        return fig

    @staticmethod
    def car_timeline(
        event_df: pd.DataFrame,
        ticker  : str,
    ) -> go.Figure:
        """Line chart of CAR across windows for a single ticker."""
        data    = event_df[event_df["ticker"] == ticker].sort_values("event_date")
        windows = [1, 2, 3, 5, 10]
        colors  = [_PRIMARY, "#7C3AED", "#059669", "#D97706", _RED]

        fig = go.Figure()
        for w, color in zip(windows, colors):
            col = f"ar_{w}d"
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data["event_date"],
                    y=data[col],
                    name=f"{w}D AR",
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=f"Abnormal Returns — {ticker}",
            xaxis_title="Earnings Date",
            yaxis_title="Abnormal Return",
            template=_TEMPLATE,
            hovermode="x unified",
        )
        return fig

    @staticmethod
    def correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
        """Heatmap of Pearson correlations: sentiment measure × return window."""
        pivot = corr_df.pivot(
            index="sentiment_measure",
            columns="return_window",
            values="correlation",
        )
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-0.3,
            zmax=0.3,
            text=pivot.values.round(3),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(
            title="Sentiment–Return Correlation Matrix",
            xaxis_title="Return Window",
            yaxis_title="Sentiment Measure",
            template=_TEMPLATE,
        )
        return fig

    @staticmethod
    def finbert_vs_lm(df: pd.DataFrame) -> go.Figure:
        """Scatter: FinBERT score vs LM tone — methodology comparison."""
        clean = df.dropna(subset=["finbert_score", "lm_tone"])
        fig = px.scatter(
            clean,
            x="lm_tone",
            y="finbert_score",
            color="ticker",
            title="FinBERT Score vs Loughran-McDonald Tone",
            labels={
                "lm_tone"      : "LM Tone Score",
                "finbert_score": "FinBERT Sentiment Score",
            },
            trendline="ols",
            template=_TEMPLATE,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        return fig

    @staticmethod
    def sector_sentiment_bar(
        df         : pd.DataFrame,
        sector_map : dict[str, str],
        score_col  : str = "finbert_score",
    ) -> go.Figure:
        """
        Horizontal bar chart of mean sentiment grouped by sector.

        Args:
            df        : Transcript-level sentiment DataFrame
            sector_map: {ticker: sector} mapping dict
            score_col : Column name for sentiment score
        """
        df = df.copy()
        df["sector"] = df["ticker"].map(sector_map).fillna("Other")
        grouped = (
            df.groupby("sector")[score_col]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean")
        )
        grouped.columns = ["sector", "mean_score", "n"]
        fig = px.bar(
            grouped,
            x="mean_score",
            y="sector",
            orientation="h",
            title="Mean FinBERT Sentiment by Sector",
            labels={"mean_score": "Mean Sentiment Score", "sector": "Sector"},
            color="mean_score",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            template=_TEMPLATE,
        )
        return fig

    @staticmethod
    def return_distribution(
        event_df  : pd.DataFrame,
        return_col: str = "ar_5d",
    ) -> go.Figure:
        """Histogram of abnormal returns across all events."""
        clean = event_df[return_col].dropna()
        fig = px.histogram(
            clean,
            nbins=60,
            title=f"Distribution of {return_col.upper()} Abnormal Returns",
            labels={"value": "Abnormal Return", "count": "Events"},
            color_discrete_sequence=[_PRIMARY],
            template=_TEMPLATE,
        )
        fig.add_vline(
            x=float(clean.mean()),
            line_dash="dash",
            line_color=_RED,
            annotation_text=f"Mean: {clean.mean():.3f}",
        )
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
        return fig
