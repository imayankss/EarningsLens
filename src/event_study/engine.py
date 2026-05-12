"""
src/event_study/engine.py
==========================
Event study engine — measures stock price reactions around earnings events.

Methodology:
  1. For each earnings event, compute forward returns over N windows
  2. Subtract benchmark (S&P 500) return → Abnormal Return (AR)
  3. Compound across window → Cumulative Abnormal Return (CAR)
  4. Regress CAR on FinBERT/LM sentiment → the core research finding

Event date alignment (critical for correctness):
  after_market → event_date = T+1 business day  (most common)
  pre_market   → event_date = T+0               (before open)

Event windows: [1, 2, 3, 5, 10] trading days

Usage:
    engine   = EventStudyEngine()
    event_df = engine.build_event_dataset(sentiment_df, returns_df)
    corr_df  = engine.sentiment_return_correlation(event_df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class EventStudyEngine:
    """Computes abnormal returns and CARs around earnings events."""

    def __init__(self) -> None:
        cfg = load_config()["event_study"]
        self.windows    = cfg["windows"]      # [1, 2, 3, 5, 10]
        self.benchmark  = load_config()["market"]["benchmark_ticker"]

    # ── Public API ──────────────────────────────────────────────

    def build_event_dataset(
        self,
        sentiment_df: pd.DataFrame,
        returns_df  : pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the event-study dataset by joining sentiment scores with
        post-event returns for every earnings event.

        Args:
            sentiment_df: Transcript-level sentiment features
                          Must contain: ticker, earnings_date, finbert_score
            returns_df  : Daily returns DataFrame
                          DatetimeIndex rows, tickers as columns

        Returns:
            Event-study DataFrame with one row per earnings event:
                ticker, event_date, finbert_score, lm_tone,
                return_Nd, ar_Nd, car_Nd  (for each window N)
        """
        rows: list[dict] = []

        for _, row in sentiment_df.iterrows():
            ticker     = row.get("ticker", "")
            event_date = pd.Timestamp(row["earnings_date"])

            if ticker not in returns_df.columns:
                continue

            metrics = self._compute_event_metrics(
                ticker      = ticker,
                event_date  = event_date,
                returns_df  = returns_df,
                sentiment_row = row,
            )
            if metrics:
                rows.append(metrics)

        result = pd.DataFrame(rows)
        log.info(f"Event dataset: {len(result):,} events × {result.shape[1]} cols")
        return result

    def sentiment_return_correlation(
        self,
        event_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute Pearson correlations between sentiment measures and
        forward abnormal returns. This is the primary research output.

        Args:
            event_df: Output of build_event_dataset()

        Returns:
            Correlation matrix DataFrame:
                sentiment_measure | return_window | correlation | n_observations
        """
        sent_cols   = [c for c in ["finbert_score", "lm_tone"] if c in event_df.columns]
        return_cols = [f"ar_{w}d" for w in self.windows if f"ar_{w}d" in event_df.columns]

        records: list[dict] = []
        for sc in sent_cols:
            for rc in return_cols:
                valid = event_df[[sc, rc]].dropna()
                corr  = valid[sc].corr(valid[rc]) if len(valid) > 1 else np.nan
                records.append({
                    "sentiment_measure": sc,
                    "return_window"    : rc,
                    "correlation"      : round(corr, 4),
                    "n_observations"   : len(valid),
                })

        return pd.DataFrame(records)

    def summary_stats(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Descriptive statistics for event study results.
        Returns mean, std, t-stat for each return window.
        """
        from scipy import stats  # type: ignore

        records = []
        for w in self.windows:
            col = f"ar_{w}d"
            if col not in event_df.columns:
                continue
            data = event_df[col].dropna()
            t_stat, p_val = stats.ttest_1samp(data, 0)
            records.append({
                "window": f"{w}D",
                "mean_ar": data.mean(),
                "std_ar" : data.std(),
                "t_stat" : t_stat,
                "p_value": p_val,
                "n"      : len(data),
            })
        return pd.DataFrame(records)

    # ── Private helpers ─────────────────────────────────────────

    def _compute_event_metrics(
        self,
        ticker      : str,
        event_date  : pd.Timestamp,
        returns_df  : pd.DataFrame,
        sentiment_row: pd.Series,
    ) -> dict | None:
        """Compute all return metrics for a single earnings event."""
        trading_dates = returns_df.index
        event_loc     = trading_dates.searchsorted(event_date)

        if event_loc >= len(trading_dates):
            return None

        row: dict = {
            "ticker"       : ticker,
            "event_date"   : event_date,
            "transcript_id": sentiment_row.get("transcript_id"),
            "finbert_score": sentiment_row.get("finbert_score"),
            "lm_tone"      : sentiment_row.get("lm_tone"),
        }

        for w in self.windows:
            end_loc = event_loc + w
            if end_loc > len(trading_dates):
                row[f"return_{w}d"] = np.nan
                row[f"ar_{w}d"]     = np.nan
                continue

            window_dates   = trading_dates[event_loc:end_loc]
            stock_ret      = returns_df.loc[window_dates, ticker]
            market_ret     = (
                returns_df.loc[window_dates, self.benchmark]
                if self.benchmark in returns_df.columns
                else pd.Series(0.0, index=window_dates)
            )

            cum_stock  = float((1 + stock_ret.fillna(0)).prod() - 1)
            cum_market = float((1 + market_ret.fillna(0)).prod() - 1)

            row[f"return_{w}d"] = cum_stock
            row[f"ar_{w}d"]     = cum_stock - cum_market

        return row
