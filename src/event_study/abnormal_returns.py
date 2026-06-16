"""
abnormal_returns.py
===================
DAY 6 — Abnormal Return Computation Engine
Earnings Call Sentiment Analyzer — Event Study Pipeline

Computes benchmark-adjusted abnormal returns (AR) for each earnings event
across configurable forward-return horizons.  This module contains all
financial formulas for abnormal-return calculation and is the exclusive
owner of that logic within the project.

Financial methodology
---------------------
Single-day abnormal return (market-adjusted model):

    AR_t = R_stock,t − R_benchmark,t

Multi-day compounded abnormal return:

    R_stock,[1..n]    = ∏(1 + r_i) − 1   for i in {1…n}
    R_benchmark,[1..n] = ∏(1 + b_i) − 1  for i in {1…n}
    AR[1..n]          = R_stock,[1..n] − R_benchmark,[1..n]

Naive summation (AR = Σ r_i − Σ b_i) is intentionally NOT used because
compounding is the standard in event-study research and prevents cumulative
drift error over multi-day windows.

Module boundaries
-----------------
This module does NOT:
  - compute CAR (→ car_calculator.py)
  - generate event windows (→ event_window_generator.py)
  - download benchmark data (→ benchmark_loader.py)
  - implement trading-calendar logic (→ trading_calendar.py)
  - orchestrate the pipeline (→ event_study.py)

Architecture position
---------------------
    event_study.py (orchestrator)
          │
          ▼
    abnormal_returns.py   ← THIS FILE
          │
          ▼
    car_calculator.py

Python : 3.11+
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Final, Optional, Sequence

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 3, 5)

# A daily return magnitude beyond this threshold is flagged as extreme.
_EXTREME_RETURN_THRESHOLD: Final[float] = 0.50  # 50 %

# Minimum number of valid (non-NaN) benchmark observations required per event
# window before we attempt to compute an abnormal return.
_MIN_BENCHMARK_OBS: Final[int] = 1

# Columns required in the market DataFrame handed in by the orchestrator.
_REQUIRED_MARKET_COLS: Final[frozenset[str]] = frozenset(
    {
        "transcript_id",
        "ticker",
        "date",
        "aligned_event_date",
        "daily_return",
        "benchmark_return",
    }
)

# Columns required in the event-level returns frame (output of Stage 4).
_REQUIRED_EVENT_COLS: Final[frozenset[str]] = frozenset(
    {
        "transcript_id",
        "ticker",
        "aligned_event_date",
    }
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbnormalReturnConfig:
    """
    Configuration for the AbnormalReturnCalculator.

    Parameters
    ----------
    return_horizons:
        Forward-return horizons (in trading days) for which abnormal
        returns are computed.  Must be a non-empty, ascending tuple of
        positive integers.
    extreme_return_threshold:
        Absolute daily-return magnitude above which a data point is
        flagged as extreme (default 0.50 = 50 %).  Extreme observations
        are reported but NOT dropped — the caller decides.
    min_benchmark_obs:
        Minimum number of valid benchmark observations required inside an
        event window to compute an abnormal return.  Windows with fewer
        observations produce NaN for that horizon.
    fill_missing_benchmark:
        When True, a missing benchmark return on a specific date is filled
        with 0.0 (neutral — market moved nothing).  When False, the
        corresponding abnormal return becomes NaN.
    warn_on_nan:
        Emit a warning when NaN abnormal returns are produced.
    """

    return_horizons: tuple[int, ...] = _DEFAULT_HORIZONS
    extreme_return_threshold: float = _EXTREME_RETURN_THRESHOLD
    min_benchmark_obs: int = _MIN_BENCHMARK_OBS
    fill_missing_benchmark: bool = False
    warn_on_nan: bool = True

    def __post_init__(self) -> None:
        if not self.return_horizons:
            raise ValueError("return_horizons must be a non-empty tuple.")
        if sorted(self.return_horizons) != list(self.return_horizons):
            raise ValueError("return_horizons must be in ascending order.")
        if any(h <= 0 for h in self.return_horizons):
            raise ValueError("All return_horizons must be positive integers.")
        if not (0.0 < self.extreme_return_threshold <= 10.0):
            raise ValueError("extreme_return_threshold must be in (0, 10].")
        if self.min_benchmark_obs < 1:
            raise ValueError("min_benchmark_obs must be >= 1.")


# ---------------------------------------------------------------------------


@dataclass
class BenchmarkCoverageReport:
    """
    Summary of benchmark-data availability for the event population.

    Attributes
    ----------
    total_event_dates:
        Number of unique (ticker, event_date) pairs checked.
    fully_covered:
        Events for which every required horizon date had benchmark data.
    partially_covered:
        Events where at least one horizon date lacked benchmark data.
    uncovered:
        Events where NO horizon date had benchmark data.
    missing_rows:
        List of (transcript_id, date) tuples that lacked benchmark data.
    coverage_rate:
        Fraction of event-dates that were fully covered (0–1).
    """

    total_event_dates: int = 0
    fully_covered: int = 0
    partially_covered: int = 0
    uncovered: int = 0
    missing_rows: list[tuple[str, pd.Timestamp]] = field(default_factory=list)

    @property
    def coverage_rate(self) -> float:
        if self.total_event_dates == 0:
            return 0.0
        return self.fully_covered / self.total_event_dates

    def log_summary(self) -> None:
        logger.info(
            "BenchmarkCoverage — total=%d  fully_covered=%d  partial=%d  "
            "uncovered=%d  rate=%.1f%%",
            self.total_event_dates,
            self.fully_covered,
            self.partially_covered,
            self.uncovered,
            self.coverage_rate * 100,
        )
        if self.missing_rows:
            sample = self.missing_rows[:5]
            logger.warning(
                "Missing benchmark rows (first %d of %d): %s",
                len(sample),
                len(self.missing_rows),
                sample,
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "total_event_dates": self.total_event_dates,
            "fully_covered": self.fully_covered,
            "partially_covered": self.partially_covered,
            "uncovered": self.uncovered,
            "missing_row_count": len(self.missing_rows),
            "coverage_rate_pct": round(self.coverage_rate * 100, 2),
        }


# ---------------------------------------------------------------------------


@dataclass
class AbnormalReturnResult:
    """
    Output of a completed AbnormalReturnCalculator.compute() call.

    Attributes
    ----------
    ar_df:
        Event-level DataFrame enriched with abnormal_return_{h}d columns
        for every configured horizon.
    coverage_report:
        Benchmark-data coverage statistics.
    horizons_computed:
        Horizons for which abnormal return columns were produced.
    extreme_events:
        transcript_ids of events where at least one horizon return
        exceeded the configured extreme_return_threshold.
    nan_counts:
        Mapping of column → number of NaN values produced.
    total_events:
        Number of unique transcript_ids in ar_df.
    """

    ar_df: pd.DataFrame
    coverage_report: BenchmarkCoverageReport
    horizons_computed: tuple[int, ...]
    extreme_events: list[str] = field(default_factory=list)
    nan_counts: dict[str, int] = field(default_factory=dict)
    total_events: int = 0

    def summary_dict(self) -> dict[str, object]:
        return {
            "total_events": self.total_events,
            "horizons_computed": self.horizons_computed,
            "extreme_events_count": len(self.extreme_events),
            "nan_counts": self.nan_counts,
            **self.coverage_report.as_dict(),
        }

    def log_summary(self) -> None:
        logger.info(
            "AbnormalReturnResult — events=%d  horizons=%s  extreme=%d",
            self.total_events,
            self.horizons_computed,
            len(self.extreme_events),
        )
        for col, n in self.nan_counts.items():
            if n:
                logger.warning("  NaN count in %-30s: %d", col, n)
        if self.extreme_events:
            logger.warning(
                "  Extreme-return events (%d): %s",
                len(self.extreme_events),
                self.extreme_events[:10],
            )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _compound_return(returns: Sequence[float]) -> float:
    """
    Compute compounded total return from a sequence of period returns.

        R_compound = ∏(1 + r_i) − 1

    NaN values in ``returns`` are treated as 0.0 (neutral period).

    Parameters
    ----------
    returns:
        Iterable of per-period fractional returns (e.g. 0.01 = +1 %).

    Returns
    -------
    float
        Compounded total return.  Returns NaN when the sequence is empty.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return float("nan")
    arr = np.where(np.isnan(arr), 0.0, arr)
    return float(np.prod(1.0 + arr) - 1.0)


def _compound_return_series(series: pd.Series) -> float:
    """
    Wrapper around :func:`_compound_return` that accepts a pandas Series.
    Preserves NaN propagation: if the series is *entirely* NaN, return NaN.
    """
    if series.isna().all():
        return float("nan")
    return _compound_return(series.dropna().to_numpy())


def _validate_columns(df: pd.DataFrame, required: frozenset[str], label: str) -> None:
    """Raise ValueError if any required column is absent from *df*."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame '{label}' is missing required columns: {sorted(missing)}"
        )


def _flag_extreme_returns(
    df: pd.DataFrame,
    return_cols: list[str],
    threshold: float,
) -> list[str]:
    """
    Return a list of transcript_ids where any return column exceeds *threshold*
    in absolute value.
    """
    mask = pd.Series(False, index=df.index)
    for col in return_cols:
        if col in df.columns:
            mask |= df[col].abs() > threshold
    if "transcript_id" not in df.columns:
        return []
    return df.loc[mask, "transcript_id"].unique().tolist()


# ---------------------------------------------------------------------------
# AbnormalReturnCalculator
# ---------------------------------------------------------------------------


class AbnormalReturnCalculator:
    """
    Compute benchmark-adjusted abnormal returns for earnings-call events.

    This class is the single owner of abnormal-return financial logic in
    the project.  It is called by ``EventStudyEngine`` (event_study.py)
    after event windows and forward returns have been attached, and before
    CAR computation (car_calculator.py).

    The market-adjusted model is used:

        AR_t = R_stock,t − R_market,t

    For multi-day windows both the stock and benchmark returns are
    compounded before subtraction, in line with event-study research
    standards.

    Parameters
    ----------
    config:
        AbnormalReturnConfig controlling horizons, thresholds, and
        NaN-handling behaviour.  Defaults to AbnormalReturnConfig().

    Examples
    --------
    >>> calc = AbnormalReturnCalculator()
    >>> result = calc.compute(
    ...     event_returns_df=event_df,
    ...     market_df=market_df,
    ...     horizons=(1, 3, 5),
    ... )
    >>> result.ar_df.columns.tolist()
    [..., 'abnormal_return_1d', 'abnormal_return_3d', 'abnormal_return_5d']
    """

    def __init__(
        self,
        config: Optional[AbnormalReturnConfig] = None,
        # Accept an EventStudyConfig-shaped object so the orchestrator can
        # pass its own config directly without manual conversion.
        **kwargs: object,
    ) -> None:
        if config is None:
            # Pull compatible fields from a passed-in EventStudyConfig if present.
            horizons = kwargs.get("return_horizons", _DEFAULT_HORIZONS)
            if hasattr(horizons, "__iter__"):
                horizons = tuple(horizons)
            config = AbnormalReturnConfig(return_horizons=horizons)
        self.config: AbnormalReturnConfig = config
        logger.debug("AbnormalReturnCalculator initialised — config=%s", config)

    # ------------------------------------------------------------------
    # Public interface (called by EventStudyEngine)
    # ------------------------------------------------------------------

    def compute(
        self,
        event_returns_df: pd.DataFrame,
        market_df: pd.DataFrame,
        horizons: Optional[tuple[int, ...]] = None,
    ) -> pd.DataFrame:
        """
        Primary entry point called by EventStudyEngine.

        Orchestrates:
          1. Input validation
          2. Benchmark alignment check
          3. Abnormal return computation for each horizon
          4. Extreme-return detection
          5. Return enriched event-level DataFrame

        Parameters
        ----------
        event_returns_df:
            Event-level DataFrame (one row per earnings event) produced by
            EventWindowGenerator.attach_forward_returns().  Must contain at
            minimum: transcript_id, ticker, aligned_event_date, and
            return_{h}d columns for each horizon.
        market_df:
            Full daily market DataFrame (one row per ticker × date) from
            DAY 5.  Must contain: transcript_id, ticker, date,
            aligned_event_date, daily_return, benchmark_return.
        horizons:
            Override the horizons from config.  Defaults to
            self.config.return_horizons.

        Returns
        -------
        pd.DataFrame
            ``event_returns_df`` enriched with:
            - abnormal_return_{h}d  for each horizon h
        """
        horizons = horizons or self.config.return_horizons
        logger.info(
            "AbnormalReturnCalculator.compute() — %d events, horizons=%s",
            len(event_returns_df),
            horizons,
        )

        # ── Validate inputs ───────────────────────────────────────────
        self._validate_event_returns_df(event_returns_df, horizons)
        _validate_columns(market_df, _REQUIRED_MARKET_COLS, "market_df")

        # ── Benchmark alignment ───────────────────────────────────────
        coverage = self.validate_benchmark_alignment(event_returns_df, market_df, horizons)
        coverage.log_summary()

        # ── Compute abnormal returns ──────────────────────────────────
        result_df = self.build_abnormal_return_frame(
            event_returns_df=event_returns_df,
            market_df=market_df,
            horizons=horizons,
        )

        logger.info(
            "compute() complete — %d event rows, AR columns: %s",
            len(result_df),
            [f"abnormal_return_{h}d" for h in horizons],
        )
        return result_df

    # ------------------------------------------------------------------

    def build_abnormal_return_frame(
        self,
        event_returns_df: pd.DataFrame,
        market_df: pd.DataFrame,
        horizons: Optional[tuple[int, ...]] = None,
    ) -> pd.DataFrame:
        """
        Construct the abnormal-return enriched DataFrame.

        For each horizon *h*:
          1. Extract benchmark compound return over [event_date+1 .. event_date+h].
          2. Compute: AR_{h}d = compound_stock_return_{h}d − compound_bench_{h}d

        The stock compound return for horizon h (return_{h}d) is assumed to
        already exist in ``event_returns_df`` as produced by
        EventWindowGenerator.  The benchmark compound return is computed
        here from the raw daily benchmark series in ``market_df``.

        Parameters
        ----------
        event_returns_df:
            Event-level frame with return_{h}d columns.
        market_df:
            Full daily frame with daily_return and benchmark_return columns.
        horizons:
            Horizons to process.

        Returns
        -------
        pd.DataFrame
            Copy of ``event_returns_df`` with abnormal_return_{h}d columns appended.
        """
        horizons = horizons or self.config.return_horizons
        df = event_returns_df.copy()

        # Build a per-event benchmark compound return lookup keyed on
        # (transcript_id, horizon).
        bench_returns = self._compute_benchmark_compound_returns(
            event_returns_df=df,
            market_df=market_df,
            horizons=horizons,
        )

        for h in horizons:
            stock_col = f"return_{h}d"
            ar_col = f"abnormal_return_{h}d"
            bench_col = f"_bench_return_{h}d"

            if stock_col not in df.columns:
                logger.warning(
                    "Column '%s' not found in event_returns_df — "
                    "'%s' will be NaN.",
                    stock_col,
                    ar_col,
                )
                df[ar_col] = float("nan")
                continue

            # Merge benchmark compound returns for this horizon
            if bench_col in bench_returns.columns:
                df = df.merge(
                    bench_returns[["transcript_id", bench_col]],
                    on="transcript_id",
                    how="left",
                )
            else:
                df[bench_col] = float("nan")

            df[ar_col] = self.compute_abnormal_returns(
                stock_returns=df[stock_col],
                benchmark_returns=df[bench_col],
            )

            # Drop the temporary benchmark column
            df = df.drop(columns=[bench_col], errors="ignore")

            nan_count = df[ar_col].isna().sum()
            logger.debug(
                "abnormal_return_%dd — NaN: %d / %d", h, nan_count, len(df)
            )
            if nan_count and self.config.warn_on_nan:
                warnings.warn(
                    f"abnormal_return_{h}d contains {nan_count} NaN values. "
                    "Check benchmark coverage.",
                    stacklevel=2,
                )

        return df

    # ------------------------------------------------------------------

    def compute_abnormal_returns(
        self,
        stock_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.Series:
        """
        Compute element-wise abnormal returns.

            AR = R_stock − R_benchmark

        Both series are expected to be compound returns over the same
        horizon.  NaN-safe: if either input is NaN the result is NaN.

        Parameters
        ----------
        stock_returns:
            Series of compound stock returns (one value per event).
        benchmark_returns:
            Series of compound benchmark returns aligned to the same events.

        Returns
        -------
        pd.Series
            Abnormal returns.  Index matches the input series.
        """
        ar = stock_returns.subtract(benchmark_returns)
        logger.debug(
            "compute_abnormal_returns — mean=%.5f  std=%.5f  nan=%d",
            ar.mean() if not ar.isna().all() else float("nan"),
            ar.std() if not ar.isna().all() else float("nan"),
            ar.isna().sum(),
        )
        return ar

    # ------------------------------------------------------------------

    def compute_forward_abnormal_returns(
        self,
        market_df: pd.DataFrame,
        event_df: pd.DataFrame,
        horizons: Optional[tuple[int, ...]] = None,
    ) -> pd.DataFrame:
        """
        Alternative entry point that computes stock AND benchmark compound
        returns internally, without requiring pre-attached return_{h}d columns.

        Useful for recalculation or testing when event_returns_df has not
        yet had forward returns attached.

        Parameters
        ----------
        market_df:
            Daily market frame (ticker × date).
        event_df:
            Event-level frame with transcript_id, ticker, aligned_event_date.
        horizons:
            Horizons to process.

        Returns
        -------
        pd.DataFrame
            ``event_df`` enriched with return_{h}d AND abnormal_return_{h}d
            columns for every horizon.
        """
        horizons = horizons or self.config.return_horizons
        logger.info(
            "compute_forward_abnormal_returns — %d events, horizons=%s",
            len(event_df),
            horizons,
        )
        _validate_columns(market_df, _REQUIRED_MARKET_COLS, "market_df")
        _validate_columns(event_df, _REQUIRED_EVENT_COLS, "event_df")

        df = event_df.copy()
        market_sorted = market_df.sort_values(["ticker", "date"]).copy()

        for h in horizons:
            stock_col = f"return_{h}d"
            bench_col = f"_bench_{h}d"
            ar_col = f"abnormal_return_{h}d"

            stock_series: list[float] = []
            bench_series: list[float] = []

            for _, row in df.iterrows():
                tid: str = row["transcript_id"]
                ticker: str = row["ticker"]
                event_date: pd.Timestamp = pd.Timestamp(row["aligned_event_date"])

                ticker_data = market_sorted[market_sorted["ticker"] == ticker]
                window = self._extract_forward_window(ticker_data, event_date, h)

                stock_series.append(
                    _compound_return_series(window["daily_return"])
                )
                bench_series.append(
                    _compound_return_series(window["benchmark_return"])
                )

            df[stock_col] = stock_series
            df[bench_col] = bench_series
            df[ar_col] = self.compute_abnormal_returns(
                stock_returns=pd.Series(stock_series, index=df.index),
                benchmark_returns=pd.Series(bench_series, index=df.index),
            )
            df = df.drop(columns=[bench_col], errors="ignore")

        return df

    # ------------------------------------------------------------------

    def validate_benchmark_alignment(
        self,
        event_returns_df: pd.DataFrame,
        market_df: pd.DataFrame,
        horizons: Optional[tuple[int, ...]] = None,
    ) -> BenchmarkCoverageReport:
        """
        Verify that benchmark return data exists for every event-window date.

        For each event, checks that the market_df contains benchmark_return
        observations for the max(horizons) trading days following the
        aligned_event_date.

        Parameters
        ----------
        event_returns_df:
            Event-level DataFrame with transcript_id, ticker, aligned_event_date.
        market_df:
            Daily market DataFrame with benchmark_return.
        horizons:
            Horizons to validate against.

        Returns
        -------
        BenchmarkCoverageReport
            Detailed coverage statistics.
        """
        horizons = horizons or self.config.return_horizons
        max_horizon = max(horizons)
        report = BenchmarkCoverageReport()

        market_sorted = market_df.sort_values(["ticker", "date"])
        bench_dates: dict[str, set[pd.Timestamp]] = {}
        for ticker, grp in market_sorted.groupby("ticker"):
            bench_dates[str(ticker)] = set(
                pd.to_datetime(grp.loc[grp["benchmark_return"].notna(), "date"])
            )

        for _, row in event_returns_df.iterrows():
            tid = str(row["transcript_id"])
            ticker = str(row["ticker"])
            event_date = pd.Timestamp(row["aligned_event_date"])
            report.total_event_dates += 1

            available = bench_dates.get(ticker, set())
            # Generate max_horizon forward business days as a proxy
            fwd_dates = pd.bdate_range(
                start=event_date + pd.tseries.offsets.BDay(1),
                periods=max_horizon,
            )
            missing = [d for d in fwd_dates if d not in available]

            if not missing:
                report.fully_covered += 1
            elif len(missing) == len(fwd_dates):
                report.uncovered += 1
                for d in missing:
                    report.missing_rows.append((tid, d))
            else:
                report.partially_covered += 1
                for d in missing:
                    report.missing_rows.append((tid, d))

        return report

    # ------------------------------------------------------------------

    def summarize_abnormal_returns(
        self,
        ar_df: pd.DataFrame,
        horizons: Optional[tuple[int, ...]] = None,
    ) -> pd.DataFrame:
        """
        Produce a descriptive statistics summary of computed abnormal returns.

        Returns a DataFrame with rows = metrics, columns = AR horizon columns.
        Metrics include: count, mean, median, std, min, max, skew,
        nan_count, extreme_count.

        Parameters
        ----------
        ar_df:
            DataFrame containing abnormal_return_{h}d columns.
        horizons:
            Horizons to summarise.

        Returns
        -------
        pd.DataFrame
            Descriptive statistics table.
        """
        horizons = horizons or self.config.return_horizons
        ar_cols = [f"abnormal_return_{h}d" for h in horizons]
        present = [c for c in ar_cols if c in ar_df.columns]

        if not present:
            logger.warning("summarize_abnormal_returns — no AR columns found.")
            return pd.DataFrame()

        records: dict[str, dict[str, float]] = {}
        for col in present:
            s = ar_df[col].dropna()
            records[col] = {
                "count": float(len(s)),
                "nan_count": float(ar_df[col].isna().sum()),
                "mean": float(s.mean()) if len(s) else float("nan"),
                "median": float(s.median()) if len(s) else float("nan"),
                "std": float(s.std()) if len(s) else float("nan"),
                "min": float(s.min()) if len(s) else float("nan"),
                "max": float(s.max()) if len(s) else float("nan"),
                "skew": float(s.skew()) if len(s) > 2 else float("nan"),
                "extreme_count": float(
                    (s.abs() > self.config.extreme_return_threshold).sum()
                ),
            }

        summary = pd.DataFrame(records)
        logger.info("summarize_abnormal_returns:\n%s", summary.to_string())
        return summary

    # ------------------------------------------------------------------
    # Internal helpers (private)
    # ------------------------------------------------------------------

    def _compute_benchmark_compound_returns(
        self,
        event_returns_df: pd.DataFrame,
        market_df: pd.DataFrame,
        horizons: tuple[int, ...],
    ) -> pd.DataFrame:
        """
        For each event, compute the compounded benchmark return over each
        horizon window.

        Returns a DataFrame with columns:
            transcript_id, _bench_return_{h}d  for each h in horizons.
        """
        market_sorted = market_df.sort_values(["ticker", "date"])
        out_rows: list[dict[str, object]] = []

        for _, row in event_returns_df.iterrows():
            tid = str(row["transcript_id"])
            ticker = str(row["ticker"])
            event_date = pd.Timestamp(row["aligned_event_date"])

            ticker_data = market_sorted[market_sorted["ticker"] == ticker]
            record: dict[str, object] = {"transcript_id": tid}

            for h in horizons:
                window = self._extract_forward_window(ticker_data, event_date, h)
                bench_series = window["benchmark_return"]

                if self.config.fill_missing_benchmark:
                    bench_series = bench_series.fillna(0.0)

                if bench_series.notna().sum() < self.config.min_benchmark_obs:
                    record[f"_bench_return_{h}d"] = float("nan")
                else:
                    record[f"_bench_return_{h}d"] = _compound_return_series(
                        bench_series
                    )

            out_rows.append(record)

        return pd.DataFrame(out_rows)

    @staticmethod
    def _extract_forward_window(
        ticker_data: pd.DataFrame,
        event_date: pd.Timestamp,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Extract the *horizon* trading-day rows immediately following *event_date*.

        Uses positional slicing on sorted data so that non-trading days
        (weekends / holidays) are naturally excluded — the forward window
        consists of at most *horizon* rows from the sorted daily frame.

        Parameters
        ----------
        ticker_data:
            Daily rows for a single ticker, sorted by date ascending.
        event_date:
            The aligned event date (t=0).
        horizon:
            Number of forward trading days to include.

        Returns
        -------
        pd.DataFrame
            Subset of ticker_data representing the forward window.
            May have fewer than *horizon* rows if data is unavailable.
        """
        dates = pd.to_datetime(ticker_data["date"])
        post_mask = dates > event_date
        post_data = ticker_data.loc[post_mask].head(horizon)
        return post_data

    def _validate_event_returns_df(
        self,
        df: pd.DataFrame,
        horizons: tuple[int, ...],
    ) -> None:
        """
        Validate event_returns_df has required base columns and, optionally,
        pre-attached return columns.  Missing return columns produce warnings
        (not errors) because build_abnormal_return_frame handles them gracefully.
        """
        _validate_columns(df, _REQUIRED_EVENT_COLS, "event_returns_df")

        if len(df) == 0:
            raise ValueError("event_returns_df is empty — nothing to compute.")

        missing_return_cols = [
            f"return_{h}d" for h in horizons if f"return_{h}d" not in df.columns
        ]
        if missing_return_cols:
            logger.warning(
                "event_returns_df is missing forward-return columns %s — "
                "those abnormal returns will be NaN.  "
                "Ensure EventWindowGenerator.attach_forward_returns() ran first.",
                missing_return_cols,
            )

    def _detect_extreme_returns(
        self,
        df: pd.DataFrame,
        horizons: tuple[int, ...],
    ) -> list[str]:
        """
        Identify transcript_ids where any abnormal-return column exceeds the
        configured extreme_return_threshold in absolute value.
        """
        ar_cols = [f"abnormal_return_{h}d" for h in horizons if f"abnormal_return_{h}d" in df.columns]
        return _flag_extreme_returns(df, ar_cols, self.config.extreme_return_threshold)


# ---------------------------------------------------------------------------
# Self-test / demo block
# ---------------------------------------------------------------------------


def _make_demo_market_df(
    tickers: list[str],
    event_date: pd.Timestamp,
    n_days_before: int = 20,
    n_days_after: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic daily market DataFrame for the demo."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    all_dates = pd.bdate_range(
        start=event_date - pd.tseries.offsets.BDay(n_days_before),
        end=event_date + pd.tseries.offsets.BDay(n_days_after),
    )
    for ticker in tickers:
        t_id = f"{ticker}_Q1_2025"
        for dt in all_dates:
            rows.append(
                {
                    "transcript_id": t_id,
                    "ticker": ticker,
                    "date": dt,
                    "aligned_event_date": event_date,
                    "daily_return": float(rng.normal(0.001, 0.015)),
                    "benchmark_return": float(rng.normal(0.0005, 0.010)),
                }
            )
    return pd.DataFrame(rows)


def _make_demo_event_df(
    tickers: list[str],
    event_date: pd.Timestamp,
    horizons: tuple[int, ...],
    seed: int = 99,
) -> pd.DataFrame:
    """Build a synthetic event-level DataFrame with pre-attached return columns."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        row: dict[str, object] = {
            "transcript_id": f"{ticker}_Q1_2025",
            "ticker": ticker,
            "aligned_event_date": event_date,
        }
        for h in horizons:
            row[f"return_{h}d"] = float(rng.normal(0.005, 0.02))
        rows.append(row)
    return pd.DataFrame(rows)


def _run_demo() -> None:
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger.info("=" * 60)
    logger.info("SELF-TEST / DEMO — abnormal_returns.py")
    logger.info("=" * 60)

    tickers = ["AAPL", "MSFT", "NVDA"]
    event_date = pd.Timestamp("2025-01-30")
    horizons: tuple[int, ...] = (1, 3, 5)

    market_df = _make_demo_market_df(tickers, event_date)
    event_df = _make_demo_event_df(tickers, event_date, horizons)

    config = AbnormalReturnConfig(
        return_horizons=horizons,
        fill_missing_benchmark=False,
        warn_on_nan=True,
    )
    calculator = AbnormalReturnCalculator(config=config)

    # ── Test 1: full compute() via orchestrator interface ──────────────
    logger.info("\n--- Test 1: compute() ---")
    result_df = calculator.compute(
        event_returns_df=event_df,
        market_df=market_df,
        horizons=horizons,
    )

    ar_cols = [f"abnormal_return_{h}d" for h in horizons]
    assert all(c in result_df.columns for c in ar_cols), "Missing AR columns!"
    assert result_df[ar_cols].isna().sum().sum() == 0, "Unexpected NaNs!"
    print("\n[Test 1] result_df head:")
    print(result_df[["transcript_id"] + ar_cols].to_string())

    # ── Test 2: benchmark alignment validation ─────────────────────────
    logger.info("\n--- Test 2: validate_benchmark_alignment() ---")
    coverage = calculator.validate_benchmark_alignment(event_df, market_df, horizons)
    coverage.log_summary()
    assert coverage.total_event_dates == len(tickers)

    # ── Test 3: compute_forward_abnormal_returns() ─────────────────────
    logger.info("\n--- Test 3: compute_forward_abnormal_returns() ---")
    fwd_df = calculator.compute_forward_abnormal_returns(
        market_df=market_df,
        event_df=event_df[["transcript_id", "ticker", "aligned_event_date"]],
        horizons=horizons,
    )
    assert all(f"abnormal_return_{h}d" in fwd_df.columns for h in horizons)
    print("\n[Test 3] forward AR columns present:", [c for c in fwd_df.columns if "return" in c])

    # ── Test 4: summarize_abnormal_returns() ──────────────────────────
    logger.info("\n--- Test 4: summarize_abnormal_returns() ---")
    summary = calculator.summarize_abnormal_returns(result_df, horizons)
    assert not summary.empty, "Summary should not be empty!"
    print("\n[Test 4] Summary:\n", summary.to_string())

    # ── Test 5: compound_return utility ───────────────────────────────
    logger.info("\n--- Test 5: _compound_return utility ---")
    daily = [0.01, -0.005, 0.02]
    expected = (1.01 * 0.995 * 1.02) - 1.0
    got = _compound_return(daily)
    assert abs(got - expected) < 1e-12, f"Compound return mismatch: {got} != {expected}"
    nan_result = _compound_return([])
    assert np.isnan(nan_result), "Empty sequence should return NaN"
    print(f"[Test 5] compound_return([0.01, -0.005, 0.02]) = {got:.8f}  ✓")
    print(f"[Test 5] compound_return([])                   = {nan_result}  ✓")

    # ── Test 6: AbnormalReturnConfig validation ────────────────────────
    logger.info("\n--- Test 6: AbnormalReturnConfig validation ---")
    try:
        AbnormalReturnConfig(return_horizons=())
        assert False, "Should have raised"
    except ValueError:
        pass
    try:
        AbnormalReturnConfig(return_horizons=(5, 1, 3))
        assert False, "Should have raised"
    except ValueError:
        pass
    print("[Test 6] Config validation ✓")

    print("\n" + "=" * 60)
    print("  SELF-TEST PASSED ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _run_demo()
