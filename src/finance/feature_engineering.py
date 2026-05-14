"""
src/finance/feature_engineering.py
=====================================
Financial feature engineering engine for the Earnings Call Sentiment
Analyzer pipeline.

Pipeline position:
    market_data_loader.py   → raw OHLCV parquet
    benchmark_loader.py     → benchmark_return column merged in
    trading_calendar.py     → event dates aligned to valid sessions
        → [THIS MODULE]  FinancialFeatureEngineer
        → analytics-ready DataFrame (all features appended)
        → event_alignment.py / market_dataset_builder.py (next stages)

Responsibilities (strictly scoped):
    ✓  Daily returns from adjusted close
    ✓  Forward returns (1d, 2d, 3d, 5d)
    ✓  Benchmark return passthrough / alignment
    ✓  Abnormal returns (AR) vs benchmark
    ✓  Cumulative abnormal returns (CAR)
    ✓  Rolling annualised volatility
    ✓  Simple & exponential moving averages
    ✓  Momentum indicators
    ✓  Volume features (relative volume, volume change)
    ✓  Validation of engineered features
    ✓  Feature coverage diagnostics

Out of scope:
    ✗  Market data downloads (→ market_data_loader.py)
    ✗  Trading calendar alignment (→ trading_calendar.py)
    ✗  Pipeline orchestration (→ market_dataset_builder.py)

Author : Earnings Call Sentiment Analyzer — Day 5
Python : 3.11+
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    }
)

# benchmark_return is needed only for AR / CAR computations;
# all other features work without it.
BENCHMARK_COLUMN: str = "benchmark_return"

ANNUALISATION_FACTOR: int = 252  # NYSE trading days per year


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureEngineeringConfig:
    """
    Configuration for FinancialFeatureEngineer.

    Attributes
    ----------
    return_horizons : list[int]
        Forward-return horizons in trading days.
        Default: [1, 2, 3, 5]
    rolling_vol_window : int
        Lookback window for rolling volatility (trading days).
        Default: 20
    moving_average_windows : list[int]
        Windows for simple moving averages (trading days).
        Default: [20, 50]
    ema_span : int
        Span parameter for exponential moving average.
        Default: 20
    momentum_windows : list[int]
        Lookback windows for momentum calculation (trading days).
        Default: [5, 20]
    annualisation_factor : int
        Trading days per year used to annualise volatility.
        Default: 252
    min_periods_vol : int
        Minimum observations required for rolling volatility.
        Default: 10
    min_periods_ma : int
        Minimum observations required for moving averages.
        Default: 1
    relative_volume_window : int
        Lookback window for average volume used in relative-volume.
        Default: 20
    clip_extreme_returns : bool
        If True, daily returns outside [-0.99, 0.99] are clipped
        (protects against data artefacts).
        Default: True
    """

    return_horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 5])
    rolling_vol_window: int = 20
    moving_average_windows: list[int] = field(default_factory=lambda: [20, 50])
    ema_span: int = 20
    momentum_windows: list[int] = field(default_factory=lambda: [5, 20])
    annualisation_factor: int = ANNUALISATION_FACTOR
    min_periods_vol: int = 10
    min_periods_ma: int = 1
    relative_volume_window: int = 20
    clip_extreme_returns: bool = True

    def __post_init__(self) -> None:
        if self.rolling_vol_window < 2:
            raise ValueError("rolling_vol_window must be >= 2.")
        if self.annualisation_factor <= 0:
            raise ValueError("annualisation_factor must be > 0.")
        if not self.return_horizons:
            raise ValueError("return_horizons must not be empty.")
        if not self.moving_average_windows:
            raise ValueError("moving_average_windows must not be empty.")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureEngineeringResult:
    """
    Carries the output of a feature engineering run.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame with all engineered feature columns appended.
    ticker : str
        Ticker symbol (or 'BATCH' for multi-ticker DataFrames).
    features_added : list[str]
        Names of columns that were successfully computed and appended.
    features_failed : list[str]
        Names of features that could not be computed (missing inputs, etc.).
    validation_errors : list[str]
        Errors raised by the post-computation validation pass.
    validation_warnings : list[str]
        Non-fatal issues raised by the validation pass.
    row_count : int
        Number of rows in the output DataFrame.
    """

    df: pd.DataFrame
    ticker: str = "UNKNOWN"
    features_added: list[str] = field(default_factory=list)
    features_failed: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    row_count: int = 0

    @property
    def is_valid(self) -> bool:
        """True when no validation errors were raised."""
        return len(self.validation_errors) == 0

    def summary(self) -> str:
        """Return a human-readable run summary."""
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines = [
            f"[{status}] {self.ticker} | {self.row_count} rows",
            f"  Features added   : {len(self.features_added)}",
            f"  Features failed  : {len(self.features_failed)}",
        ]
        if self.features_failed:
            lines.append(f"  Failed           : {self.features_failed}")
        for e in self.validation_errors:
            lines.append(f"  ERROR   : {e}")
        for w in self.validation_warnings:
            lines.append(f"  WARNING : {w}")
        return "\n".join(lines)

    def coverage_report(self) -> pd.DataFrame:
        """
        Return a DataFrame showing NaN coverage per feature column.

        Useful for diagnosing how much of the event window is covered
        after rolling / forward-looking calculations produce leading
        or trailing NaNs.
        """
        all_features = self.features_added + self.features_failed
        if not all_features or self.df.empty:
            return pd.DataFrame()

        present = [c for c in all_features if c in self.df.columns]
        total = len(self.df)
        records = []
        for col in present:
            nan_n = int(self.df[col].isna().sum())
            records.append(
                {
                    "feature": col,
                    "total_rows": total,
                    "non_null": total - nan_n,
                    "null_count": nan_n,
                    "coverage_pct": round(100 * (total - nan_n) / total, 1),
                }
            )
        return pd.DataFrame(records).set_index("feature")


# ---------------------------------------------------------------------------
# Core feature engineering class
# ---------------------------------------------------------------------------


class FinancialFeatureEngineer:
    """
    Compute financial features on a normalised OHLCV DataFrame.

    All methods accept a ``pd.DataFrame`` with the schema produced by
    ``MarketDataLoader.standardize_columns()`` (plus a ``benchmark_return``
    column merged in by ``BenchmarkLoader``) and return a **new** DataFrame
    with feature columns appended — the input is never mutated.

    Multi-ticker DataFrames are handled by grouping on ``ticker`` before
    applying each computation, ensuring no cross-contamination between
    stocks (rolling windows never bleed across ticker boundaries).

    Parameters
    ----------
    config : FeatureEngineeringConfig, optional
        Configuration object.  Uses library defaults when omitted.

    Examples
    --------
    >>> eng = FinancialFeatureEngineer()
    >>> result = eng.build_feature_set(df)
    >>> print(result.summary())
    >>> result.df.head()
    """

    def __init__(
        self, config: Optional[FeatureEngineeringConfig] = None
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        logger.info(
            "FinancialFeatureEngineer initialised | vol_window=%d | "
            "horizons=%s | ma_windows=%s",
            self.config.rolling_vol_window,
            self.config.return_horizons,
            self.config.moving_average_windows,
        )

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def build_feature_set(
        self, df: pd.DataFrame, ticker: str = "BATCH"
    ) -> FeatureEngineeringResult:
        """
        Run the full feature-engineering pipeline on a market DataFrame.

        Executes every feature computation in the correct dependency order:

        1. Daily returns           (basis for everything else)
        2. Forward returns         (requires daily prices)
        3. Benchmark returns       (passthrough / alignment check)
        4. Abnormal returns        (requires daily + benchmark returns)
        5. Cumulative AR           (requires abnormal returns)
        6. Rolling volatility      (requires daily returns)
        7. Moving averages         (requires adj_close)
        8. Momentum features       (requires adj_close + daily returns)
        9. Relative volume         (requires volume)

        Then runs ``validate_features()``.

        Parameters
        ----------
        df : pd.DataFrame
            Normalised OHLCV DataFrame (DatetimeIndex named ``date``).
            Must contain the columns in ``REQUIRED_INPUT_COLUMNS``.
        ticker : str
            Label used in logging and the result object.
            Pass the actual ticker for single-ticker DataFrames.

        Returns
        -------
        FeatureEngineeringResult
            Result object containing the enriched DataFrame, list of
            features added/failed, and validation outcome.
        """
        _check_required_columns(df, REQUIRED_INPUT_COLUMNS, context="build_feature_set")
        df = _prepare_dataframe(df)

        features_added: list[str] = []
        features_failed: list[str] = []

        def _run(method_name: str, *args, **kwargs) -> pd.DataFrame:
            """Run a feature method and track success/failure."""
            nonlocal df
            try:
                new_df = getattr(self, method_name)(df, *args, **kwargs)
                new_cols = [
                    c for c in new_df.columns if c not in df.columns
                ]
                features_added.extend(new_cols)
                logger.debug(
                    "%s → added columns: %s", method_name, new_cols
                )
                return new_df
            except Exception as exc:
                logger.error("%s failed: %s", method_name, exc, exc_info=True)
                features_failed.append(method_name)
                return df  # return unchanged

        df = _run("compute_daily_returns")
        df = _run("compute_forward_returns")
        df = _run("compute_benchmark_returns")
        df = _run("compute_abnormal_returns")
        df = _run("compute_cumulative_abnormal_returns")
        df = _run("compute_rolling_volatility")
        df = _run("compute_moving_averages")
        df = _run("compute_momentum_features")
        df = _run("compute_relative_volume")

        val_errors, val_warnings = self.validate_features(df)

        result = FeatureEngineeringResult(
            df=df,
            ticker=ticker,
            features_added=features_added,
            features_failed=features_failed,
            validation_errors=val_errors,
            validation_warnings=val_warnings,
            row_count=len(df),
        )

        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Feature computation methods
    # ------------------------------------------------------------------

    def compute_daily_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily log-equivalent percentage returns from adjusted close.

        Formula
        -------
        ::

            return_t = (adj_close_t / adj_close_{t-1}) - 1

        Uses ``pct_change()`` which is equivalent to the above for
        non-zero prices and avoids log-return approximation error on
        large daily moves.

        Grouped by ``ticker`` to prevent cross-contamination in
        multi-ticker DataFrames.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``adj_close`` and ``ticker``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``daily_return`` appended.
        """
        df = df.copy()

        def _returns(group: pd.DataFrame) -> pd.Series:
            r = group["adj_close"].pct_change()
            if self.config.clip_extreme_returns:
                r = r.clip(lower=-0.99, upper=0.99)
            return r

        df["daily_return"] = df.groupby("ticker", group_keys=False)[
            "adj_close"
        ].transform(lambda s: s.pct_change())

        if self.config.clip_extreme_returns:
            df["daily_return"] = df["daily_return"].clip(-0.99, 0.99)

        logger.debug("compute_daily_returns: done (%d rows).", len(df))
        return df

    def compute_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forward (future) returns for each configured horizon.

        Formula
        -------
        ::

            return_Nd = (adj_close_{t+N} / adj_close_t) - 1

        Implemented as ``adj_close.shift(-N) / adj_close - 1`` so each row
        contains the future N-day return from that date.
        Grouped by ``ticker``.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``adj_close`` and ``ticker``.
            ``daily_return`` should already be present (computed first)
            but is not strictly required here.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``return_Nd`` columns appended for each
            horizon N in ``config.return_horizons``.

        Notes
        -----
        Forward-return columns will be NaN for the last N rows of each
        ticker group — this is expected and correct.
        """
        df = df.copy()

        for n in self.config.return_horizons:
            col = f"return_{n}d"
            df[col] = df.groupby("ticker", group_keys=False)[
                "adj_close"
            ].transform(lambda s, _n=n: (s.shift(-_n) / s) - 1)

            if self.config.clip_extreme_returns:
                df[col] = df[col].clip(-0.99, 0.99)

        logger.debug(
            "compute_forward_returns: horizons=%s.", self.config.return_horizons
        )
        return df

    def compute_benchmark_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalise the benchmark return column.

        The ``benchmark_return`` column is merged in by ``BenchmarkLoader``
        before this module is called.  This method:

        1. Confirms the column is present (logs a warning if absent).
        2. Forward-fills up to 1 day to handle sporadic missing sessions
           (e.g. early-close days where the benchmark has no record).
        3. Clips extreme values matching the stock-return clip policy.

        Parameters
        ----------
        df : pd.DataFrame
            Should contain ``benchmark_return``.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``benchmark_return`` normalised in-place.
            If the column is absent, a NaN column is added so downstream
            methods degrade gracefully rather than raising.
        """
        df = df.copy()

        if BENCHMARK_COLUMN not in df.columns:
            logger.warning(
                "Column '%s' not found — abnormal returns will be NaN. "
                "Ensure BenchmarkLoader has merged benchmark data before "
                "calling FinancialFeatureEngineer.",
                BENCHMARK_COLUMN,
            )
            df[BENCHMARK_COLUMN] = np.nan
            return df

        # Forward-fill gaps up to 1 day (e.g. early-close sessions)
        df[BENCHMARK_COLUMN] = df[BENCHMARK_COLUMN].ffill(limit=1)

        if self.config.clip_extreme_returns:
            df[BENCHMARK_COLUMN] = df[BENCHMARK_COLUMN].clip(-0.99, 0.99)

        logger.debug("compute_benchmark_returns: normalised.")
        return df

    def compute_abnormal_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute abnormal returns (AR) for each forward-return horizon.

        Formula
        -------
        ::

            AR_Nd = return_Nd - benchmark_return_Nd

        Where ``benchmark_return_Nd`` is the N-day cumulative return of the
        benchmark over the same forward window, computed as:

        ::

            benchmark_return_Nd = (1 + benchmark_return)^N  −  1

        This compound formulation is more accurate than simply multiplying
        ``benchmark_return × N`` for multi-day windows.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``benchmark_return`` and ``return_Nd`` columns
            for each configured horizon.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``abnormal_return_Nd`` columns appended.

        Notes
        -----
        Also computes ``abnormal_return`` (1-day) as a convenience alias
        equal to ``daily_return - benchmark_return`` for use in event-study
        CAR calculations.
        """
        df = df.copy()

        if BENCHMARK_COLUMN not in df.columns or df[BENCHMARK_COLUMN].isna().all():
            logger.warning(
                "benchmark_return unavailable — skipping abnormal return computation."
            )
            for n in self.config.return_horizons:
                df[f"abnormal_return_{n}d"] = np.nan
            df["abnormal_return"] = np.nan
            return df

        # 1-day abnormal return (convenience alias)
        if "daily_return" in df.columns:
            df["abnormal_return"] = (
                df["daily_return"] - df[BENCHMARK_COLUMN]
            )

        # N-day abnormal returns using compound benchmark
        for n in self.config.return_horizons:
            fwd_col = f"return_{n}d"
            ar_col = f"abnormal_return_{n}d"

            if fwd_col not in df.columns:
                logger.warning(
                    "Forward return column '%s' missing — skipping %s.",
                    fwd_col,
                    ar_col,
                )
                df[ar_col] = np.nan
                continue

            # Compound benchmark return over N days
            compound_bench = (1 + df[BENCHMARK_COLUMN]) ** n - 1
            df[ar_col] = df[fwd_col] - compound_bench

        logger.debug(
            "compute_abnormal_returns: horizons=%s.", self.config.return_horizons
        )
        return df

    def compute_cumulative_abnormal_returns(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute Cumulative Abnormal Returns (CAR) over event windows.

        Formula
        -------
        ::

            CAR(0, N) = Σ_{t=0}^{N} AR_t

        where AR_t is the daily abnormal return on trading day t relative
        to the earnings event.

        Implementation note
        -------------------
        Rather than summing per-ticker historical AR (which is a
        time-series cumsum), this method provides event-window CARs by
        summing ``abnormal_return_Nd`` values directly.  This is the
        standard definition used in finance event-study literature
        (MacKinlay, 1997):

        ::

            car_3d = abnormal_return_1d + abnormal_return_2d + abnormal_return_3d
            car_5d = Σ abnormal_return_{1..5}d

        If any constituent AR day is NaN the CAR is NaN (correct behaviour:
        incomplete windows should not produce partial sums).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``abnormal_return_Nd`` columns for each horizon.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``car_3d`` and ``car_5d`` appended.
        """
        df = df.copy()

        def _car(horizons: list[int], col_name: str) -> None:
            cols = [
                f"abnormal_return_{n}d"
                for n in horizons
                if f"abnormal_return_{n}d" in df.columns
            ]
            if not cols:
                logger.warning(
                    "Cannot compute %s — no AR columns found for horizons %s.",
                    col_name,
                    horizons,
                )
                df[col_name] = np.nan
                return
            df[col_name] = df[cols].sum(axis=1, skipna=False)

        # CAR over 3 and 5 days (standard event-study windows)
        _car([1, 2, 3], "car_3d")
        _car([1, 2, 3, 4, 5], "car_5d")

        logger.debug("compute_cumulative_abnormal_returns: car_3d, car_5d computed.")
        return df

    def compute_rolling_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute annualised rolling historical volatility.

        Formula
        -------
        ::

            rolling_volatility_20d_t =
                sqrt(annualisation_factor) × std(daily_return_{t-W+1 : t})

        where W = ``config.rolling_vol_window``.

        The standard deviation uses the default ``ddof=1`` (sample std),
        which is standard in finance.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``daily_return`` and ``ticker``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``rolling_volatility_{W}d`` appended.
        """
        df = df.copy()
        w = self.config.rolling_vol_window
        col = f"rolling_volatility_{w}d"
        sqrt_annual = np.sqrt(self.config.annualisation_factor)

        df[col] = df.groupby("ticker", group_keys=False)[
            "daily_return"
        ].transform(
            lambda s: s.rolling(
                window=w, min_periods=self.config.min_periods_vol
            ).std()
            * sqrt_annual
        )

        # Volatility must be non-negative; clip floating-point noise
        df[col] = df[col].clip(lower=0.0)

        logger.debug(
            "compute_rolling_volatility: window=%d, annualised by sqrt(%d).",
            w,
            self.config.annualisation_factor,
        )
        return df

    def compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple moving averages (SMA) and an exponential MA (EMA).

        SMAs use ``adj_close`` with a rolling window.
        EMA uses ``adj_close`` with ``ewm(span=config.ema_span)``.

        All calculations are grouped by ``ticker`` to prevent bleeding
        across stocks in multi-ticker DataFrames.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``adj_close`` and ``ticker``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with the following columns appended:

            - ``sma_{W}`` for each W in ``config.moving_average_windows``
            - ``ema_{config.ema_span}``
        """
        df = df.copy()

        for w in self.config.moving_average_windows:
            col = f"sma_{w}"
            df[col] = df.groupby("ticker", group_keys=False)[
                "adj_close"
            ].transform(
                lambda s, _w=w: s.rolling(
                    window=_w, min_periods=self.config.min_periods_ma
                ).mean()
            )
            # Prices are non-negative; clip floating-point noise
            df[col] = df[col].clip(lower=0.0)

        ema_col = f"ema_{self.config.ema_span}"
        df[ema_col] = df.groupby("ticker", group_keys=False)[
            "adj_close"
        ].transform(
            lambda s: s.ewm(
                span=self.config.ema_span, adjust=False, min_periods=1
            ).mean()
        )
        df[ema_col] = df[ema_col].clip(lower=0.0)

        logger.debug(
            "compute_moving_averages: sma_windows=%s, ema_span=%d.",
            self.config.moving_average_windows,
            self.config.ema_span,
        )
        return df

    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price momentum and intraday range features.

        Features produced
        -----------------
        momentum_Nd
            Percentage change of adjusted close over the last N trading
            days.  Positive → price trended up; negative → trended down.

            ::

                momentum_Nd = (adj_close_t / adj_close_{t-N}) - 1

        price_gap
            Overnight gap as a fraction of the prior close:

            ::

                price_gap_t = (open_t / adj_close_{t-1}) - 1

            Captures after-hours / pre-market price movement.

        intraday_range
            Normalised intraday high–low spread:

            ::

                intraday_range_t = (high_t - low_t) / adj_close_t

            Proxy for intraday volatility / uncertainty.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``adj_close``, ``open``, ``high``, ``low``,
            and ``ticker``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with momentum and range features appended.
        """
        df = df.copy()

        # --- Momentum -----------------------------------------------
        for n in self.config.momentum_windows:
            col = f"momentum_{n}d"
            df[col] = df.groupby("ticker", group_keys=False)[
                "adj_close"
            ].transform(lambda s, _n=n: s.pct_change(_n))

            if self.config.clip_extreme_returns:
                df[col] = df[col].clip(-0.99, 0.99)

        # --- Price gap (overnight) ----------------------------------
        prior_close = df.groupby("ticker", group_keys=False)[
            "adj_close"
        ].transform(lambda s: s.shift(1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["price_gap"] = (df["open"] / prior_close) - 1

        if self.config.clip_extreme_returns:
            df["price_gap"] = df["price_gap"].clip(-0.99, 0.99)

        # --- Intraday range ------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["intraday_range"] = (
                (df["high"] - df["low"]) / df["adj_close"]
            ).clip(lower=0.0)

        logger.debug(
            "compute_momentum_features: windows=%s.", self.config.momentum_windows
        )
        return df

    def compute_relative_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume-based features.

        Features produced
        -----------------
        volume_change
            Day-over-day percentage change in trading volume:

            ::

                volume_change_t = (volume_t / volume_{t-1}) - 1

        relative_volume
            Today's volume relative to the rolling mean volume:

            ::

                relative_volume_t = volume_t /
                    rolling_mean(volume, window=config.relative_volume_window)

            Values > 1 indicate above-average activity; values < 1
            indicate below-average activity.  Useful for detecting
            abnormal trading around earnings events.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``volume`` and ``ticker``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``volume_change`` and ``relative_volume``
            appended.
        """
        df = df.copy()
        w = self.config.relative_volume_window

        # Volume change
        df["volume_change"] = df.groupby("ticker", group_keys=False)[
            "volume"
        ].transform(lambda s: s.pct_change())
        df["volume_change"] = df["volume_change"].clip(-10.0, 100.0)

        # Rolling average volume (shifted by 1 to avoid look-ahead)
        rolling_avg = df.groupby("ticker", group_keys=False)[
            "volume"
        ].transform(
            lambda s: s.shift(1)
            .rolling(window=w, min_periods=max(1, self.config.min_periods_ma))
            .mean()
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["relative_volume"] = df["volume"] / rolling_avg

        # Cap at reasonable upper bound (>50× average is almost always
        # a data artefact or a corporate event)
        df["relative_volume"] = df["relative_volume"].clip(lower=0.0, upper=50.0)

        logger.debug(
            "compute_relative_volume: rolling_window=%d.", w
        )
        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_features(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """
        Run post-computation integrity checks on the enriched DataFrame.

        Checks
        ------
        1. No duplicate (ticker, date) combinations.
        2. ``daily_return`` absolute values < 1 (after clipping).
        3. ``rolling_volatility_*`` values are non-negative.
        4. SMA / EMA columns are non-negative.
        5. No infinite values in any numeric column.
        6. Required input columns still present.
        7. (Warning) NaN fraction > 50 % in any feature column.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched DataFrame output of ``build_feature_set()``.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(errors, warnings)`` — both are lists of strings.
        """
        errors: list[str] = []
        warnings_: list[str] = []

        # 1. Duplicate (ticker, date) check. Date may live either as a
        # DatetimeIndex or as a materialized column depending on the caller.
        if "ticker" in df.columns:
            if "date" in df.columns:
                dup_count = df.duplicated(subset=["ticker", "date"]).sum()
            elif df.index.name is not None:
                dup_count = (
                    df.reset_index()
                    .duplicated(subset=["ticker", df.index.name])
                    .sum()
                )
            else:
                dup_count = df.duplicated(subset=["ticker"]).sum()
        else:
            dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            errors.append(
                f"Duplicate (ticker, date) combinations: {dup_count}."
            )

        # 2. Return sanity — daily_return
        if "daily_return" in df.columns:
            max_abs = df["daily_return"].abs().max()
            if max_abs >= 1.0:
                errors.append(
                    f"daily_return has |value| >= 1.0 (max={max_abs:.4f}). "
                    "Check for data artefacts."
                )

        # 3. Volatility non-negative
        vol_cols = [c for c in df.columns if c.startswith("rolling_volatility")]
        for col in vol_cols:
            min_vol = df[col].min()
            if pd.notna(min_vol) and min_vol < -1e-9:
                errors.append(f"{col} has negative values (min={min_vol:.6f}).")

        # 4. Moving averages non-negative
        ma_cols = [
            c for c in df.columns if c.startswith("sma_") or c.startswith("ema_")
        ]
        for col in ma_cols:
            min_ma = df[col].min()
            if pd.notna(min_ma) and min_ma < -1e-9:
                errors.append(f"{col} has negative values (min={min_ma:.4f}).")

        # 5. Infinite values
        numeric = df.select_dtypes(include=[np.number])
        inf_cols = numeric.columns[np.isinf(numeric).any()].tolist()
        if inf_cols:
            errors.append(f"Infinite values found in columns: {inf_cols}.")

        # 6. Required input columns still present
        missing_input = REQUIRED_INPUT_COLUMNS - set(df.columns)
        if missing_input:
            errors.append(
                f"Required input columns dropped: {sorted(missing_input)}."
            )

        # 7. (Warning) High NaN fraction in feature columns
        feature_cols = [
            c
            for c in df.columns
            if c not in REQUIRED_INPUT_COLUMNS and c != BENCHMARK_COLUMN
        ]
        total = len(df)
        if total > 0:
            for col in feature_cols:
                nan_frac = df[col].isna().mean()
                if nan_frac > 0.5:
                    warnings_.append(
                        f"'{col}' is {nan_frac:.0%} NaN — "
                        "consider a longer download window."
                    )

        return errors, warnings_

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def feature_summary_statistics(
        df: pd.DataFrame, feature_cols: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Return descriptive statistics for engineered feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched DataFrame.
        feature_cols : list[str], optional
            Columns to describe.  Defaults to all numeric columns not
            in ``REQUIRED_INPUT_COLUMNS``.

        Returns
        -------
        pd.DataFrame
            Standard ``describe()`` output (transposed for readability).
        """
        if feature_cols is None:
            feature_cols = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c not in REQUIRED_INPUT_COLUMNS
            ]
        if not feature_cols:
            return pd.DataFrame()
        return df[feature_cols].describe().T

    @staticmethod
    def get_event_window_slice(
        df: pd.DataFrame,
        event_date: str | pd.Timestamp,
        pre_days: int = 5,
        post_days: int = 5,
    ) -> pd.DataFrame:
        """
        Slice the DataFrame to an event window around a given date.

        Useful for inspecting features in the N days before/after an
        earnings event.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched DataFrame with DatetimeIndex.
        event_date : str | pd.Timestamp
            Centre of the event window.
        pre_days : int
            Trading days before the event to include.
        post_days : int
            Trading days after the event to include.

        Returns
        -------
        pd.DataFrame
            Sliced DataFrame covering the event window.
        """
        ts = pd.Timestamp(event_date)
        dates = df.index.sort_values()
        pos = dates.searchsorted(ts)

        start_pos = max(0, pos - pre_days)
        end_pos = min(len(dates) - 1, pos + post_days)

        start_date = dates[start_pos]
        end_date = dates[end_pos]

        return df.loc[start_date:end_date]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _check_required_columns(
    df: pd.DataFrame,
    required: frozenset[str],
    context: str = "",
) -> None:
    """Raise ValueError if any required column is absent."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns.tolist())}"
        )


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive preparation before feature computation.

    1. Makes a copy (never mutates input).
    2. Ensures DatetimeIndex is sorted ascending per ticker.
    3. Removes duplicate index entries (keeps last).
    4. Resets to a clean DatetimeIndex named 'date'.
    """
    df = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.index.name = "date"

    # Sort ascending
    df = df.sort_index()

    # Remove duplicates (keep last, consistent with MarketDataLoader)
    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    removed = before - len(df)
    if removed:
        logger.warning(
            "_prepare_dataframe: removed %d duplicate date row(s).", removed
        )

    return df


# ---------------------------------------------------------------------------
# Self-tests (lightweight, no external dependencies)
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    """
    Lightweight formula verification tests.

    Verifies:
    - Daily return formula
    - Forward return direction
    - Abnormal return formula (AR = stock_return - benchmark_return)
    - CAR additive property
    - Rolling volatility non-negativity
    """
    import math

    print("Running self-tests…")

    # Build a minimal 30-row synthetic DataFrame
    dates = pd.date_range("2024-01-02", periods=30, freq="B")  # business days
    np.random.seed(42)

    prices = 150.0 * np.cumprod(1 + np.random.normal(0.001, 0.015, 30))
    benchmark_r = np.random.normal(0.0005, 0.010, 30)
    benchmark_r[0] = np.nan  # first day has no return

    df = pd.DataFrame(
        {
            "ticker": "TEST",
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "adj_close": prices,
            "volume": np.random.randint(1_000_000, 5_000_000, 30).astype(float),
            "benchmark_return": benchmark_r,
        },
        index=dates,
    )
    df.index.name = "date"

    eng = FinancialFeatureEngineer()
    result = eng.build_feature_set(df, ticker="TEST")
    out = result.df

    # --- Test 1: daily return formula --------------------------------
    expected_r2 = (prices[1] / prices[0]) - 1
    actual_r2 = out["daily_return"].iloc[1]
    assert math.isclose(actual_r2, expected_r2, rel_tol=1e-9), (
        f"daily_return mismatch: expected {expected_r2:.8f}, got {actual_r2:.8f}"
    )
    print("  ✓ daily_return formula correct.")

    # --- Test 2: forward return direction ----------------------------
    # return_1d at row 0 should equal (price[1] / price[0]) - 1
    expected_fwd1 = (prices[1] / prices[0]) - 1
    actual_fwd1 = out["return_1d"].iloc[0]
    assert math.isclose(actual_fwd1, expected_fwd1, rel_tol=1e-9), (
        f"return_1d mismatch: expected {expected_fwd1:.8f}, got {actual_fwd1:.8f}"
    )
    print("  ✓ return_1d (forward) formula correct.")

    # --- Test 3: abnormal return formula -----------------------------
    # AR_1d at row 1 = daily_return[1] - benchmark_return[1]
    # (benchmark_return[0] is NaN so use row 1)
    br = benchmark_r[1]
    expected_ar = actual_r2 - br
    actual_ar = out["abnormal_return"].iloc[1]
    assert math.isclose(actual_ar, expected_ar, rel_tol=1e-9), (
        f"abnormal_return mismatch: expected {expected_ar:.8f}, got {actual_ar:.8f}"
    )
    print("  ✓ abnormal_return (AR) formula correct.")

    # --- Test 4: CAR additive property -------------------------------
    # car_3d = abnormal_return_1d + abnormal_return_2d + abnormal_return_3d
    row = out.iloc[5]  # pick a row with enough history for forward windows
    expected_car3 = (
        row.get("abnormal_return_1d", np.nan)
        + row.get("abnormal_return_2d", np.nan)
        + row.get("abnormal_return_3d", np.nan)
    )
    actual_car3 = row.get("car_3d", np.nan)
    if not (math.isnan(expected_car3) and math.isnan(actual_car3)):
        assert math.isclose(actual_car3, expected_car3, rel_tol=1e-9), (
            f"car_3d mismatch: expected {expected_car3:.8f}, got {actual_car3:.8f}"
        )
    print("  ✓ car_3d additive property correct.")

    # --- Test 5: rolling volatility non-negativity -------------------
    vol_col = f"rolling_volatility_{eng.config.rolling_vol_window}d"
    assert vol_col in out.columns, f"Column {vol_col} not found."
    min_vol = out[vol_col].dropna().min()
    assert min_vol >= 0.0, f"Negative volatility: {min_vol}"
    print("  ✓ rolling_volatility non-negative.")

    # --- Test 6: validation passes -----------------------------------
    assert result.is_valid, f"Validation failed: {result.validation_errors}"
    print("  ✓ validate_features passes on clean synthetic data.")

    print("\nAll self-tests passed.")


# ---------------------------------------------------------------------------
# CLI / demo entry point
# ---------------------------------------------------------------------------


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    """
    Demo: build feature set on synthetic OHLCV data.

    Run with:
        python src/finance/feature_engineering.py

    In production, ``df`` comes from MarketDataLoader + BenchmarkLoader:
        df = loader.load_parquet("AAPL")
        df = benchmark_loader.merge_benchmark_returns(df)
        result = FinancialFeatureEngineer().build_feature_set(df, "AAPL")
    """
    _setup_logging()

    # ----------------------------------------------------------------
    # Run self-tests first
    # ----------------------------------------------------------------
    _run_self_tests()

    # ----------------------------------------------------------------
    # Build a realistic synthetic dataset
    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DEMO — FinancialFeatureEngineer on synthetic AAPL-like data")
    print("=" * 65)

    np.random.seed(0)
    N = 120  # ~6 months of trading days

    dates = pd.date_range("2024-06-01", periods=N, freq="B")
    prices = 185.0 * np.cumprod(1 + np.random.normal(0.0008, 0.016, N))
    volumes = np.random.randint(40_000_000, 120_000_000, N).astype(float)
    bench_r = np.concatenate([[np.nan], np.random.normal(0.0004, 0.010, N - 1)])

    df = pd.DataFrame(
        {
            "ticker": "AAPL",
            "open": prices * (1 + np.random.uniform(-0.003, 0.003, N)),
            "high": prices * (1 + np.random.uniform(0.000, 0.012, N)),
            "low": prices * (1 - np.random.uniform(0.000, 0.012, N)),
            "close": prices,
            "adj_close": prices,
            "volume": volumes,
            "benchmark_return": bench_r,
        },
        index=dates,
    )
    df.index.name = "date"

    # ----------------------------------------------------------------
    # Custom config
    # ----------------------------------------------------------------
    config = FeatureEngineeringConfig(
        return_horizons=[1, 2, 3, 5],
        rolling_vol_window=20,
        moving_average_windows=[20, 50],
        ema_span=20,
        momentum_windows=[5, 20],
        annualisation_factor=252,
        clip_extreme_returns=True,
    )

    eng = FinancialFeatureEngineer(config=config)

    # ----------------------------------------------------------------
    # Build full feature set
    # ----------------------------------------------------------------
    result = eng.build_feature_set(df, ticker="AAPL")

    print("\n--- Result Summary ---")
    print(result.summary())

    print("\n--- Last 5 rows (selected features) ---")
    display_cols = [
        "adj_close",
        "daily_return",
        "return_1d",
        "return_5d",
        "abnormal_return",
        "car_3d",
        "car_5d",
        f"rolling_volatility_{config.rolling_vol_window}d",
        "sma_20",
        "ema_20",
        "momentum_5d",
        "relative_volume",
    ]
    present = [c for c in display_cols if c in result.df.columns]
    pd.set_option("display.float_format", "{:.5f}".format)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 140)
    print(result.df[present].tail(5).to_string())

    print("\n--- Feature Summary Statistics ---")
    stats = FinancialFeatureEngineer.feature_summary_statistics(result.df)
    print(stats[["mean", "std", "min", "max"]].round(5).to_string())

    print("\n--- Coverage Report ---")
    print(result.coverage_report().to_string())

    # ----------------------------------------------------------------
    # Event-window slice demo
    # ----------------------------------------------------------------
    print("\n--- Event Window Slice (earnings_date=2024-10-30, ±5 days) ---")
    window = FinancialFeatureEngineer.get_event_window_slice(
        result.df,
        event_date="2024-10-30",
        pre_days=5,
        post_days=5,
    )
    print(window[present].to_string())

    print("\nDemo complete.")
