"""
src/event_study/car_calculator.py
===================================
Cumulative Abnormal Return (CAR) computation engine for the DAY 6
event-study pipeline.

Responsibilities:
    - Compute cumulative abnormal returns over configurable horizons
    - Aggregate abnormal returns across post-event trading sessions
    - Generate finance-research-ready CAR metrics (car_3d, car_5d, ...)
    - Validate CAR series completeness and magnitude
    - Produce export-ready DataFrames (parquet-friendly)

Scope:
    - CAR computation ONLY
    - No abnormal return computation (sourced from abnormal_returns.py)
    - No benchmark logic
    - No trading-calendar logic
    - No sentiment merging
    - No pipeline orchestration

Formula:
    CAR(0, N) = Σ AR_t  for t in [1, N]   (post-event sessions only)

    Where:
        AR_t = abnormal return on trading day t relative to event
        t=0  = event day (aligned_event_date)
        t=1..N = post-event trading sessions

    Note: day-0 (event day itself) is excluded from CAR by default because
    after-hours earnings releases mean the event-day return is ambiguous.
    This is configurable via CARConfig.include_event_day_in_car.

Author: Earnings Call Sentiment Analyzer Pipeline
Python: 3.11+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CAR_HORIZONS: Final[tuple[int, ...]] = (1, 3, 5)
_MAX_SANE_CAR: Final[float] = 1.0          # 100% CAR — flag anything beyond
_MIN_SANE_CAR: Final[float] = -1.0         # -100%
_ABNORMAL_RETURN_COL: Final[str] = "abnormal_return"
_RELATIVE_DAY_COL: Final[str] = "relative_day"
_TRANSCRIPT_ID_COL: Final[str] = "transcript_id"
_TICKER_COL: Final[str] = "ticker"
_EVENT_DATE_COL: Final[str] = "aligned_event_date"

_REQUIRED_INPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        _TRANSCRIPT_ID_COL,
        _TICKER_COL,
        _EVENT_DATE_COL,
        _RELATIVE_DAY_COL,
        _ABNORMAL_RETURN_COL,
    }
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CARConfig:
    """
    Immutable configuration for CAR computation.

    Parameters
    ----------
    horizons:
        Post-event trading-day horizons for which CAR is computed.
        Each value N produces a column ``car_{N}d`` representing
        CAR over relative days [1, N] (or [0, N] if
        ``include_event_day_in_car`` is True).
        Default: (1, 3, 5).
    include_event_day_in_car:
        If True, include the event day (relative_day == 0) in the
        cumulative sum.  If False (default), CAR starts at t=1.
    nan_policy:
        How to handle NaN abnormal returns within a window.
        ``"raise"``  — raise ValueError on any NaN.
        ``"skip"``   — skip NaN days (CAR = sum of available days).
        ``"zero"``   — treat NaN as 0.0 before summing.
        Default: ``"skip"``.
    extreme_car_threshold:
        Absolute CAR values exceeding this threshold are flagged as
        potentially erroneous.  Default: 1.0 (100%).
    min_post_days_required:
        Minimum number of non-NaN post-event abnormal return observations
        required to compute CAR for a horizon.  Events below this threshold
        are assigned NaN for that horizon.  Default: 1.
    """

    horizons: tuple[int, ...] = _DEFAULT_CAR_HORIZONS
    include_event_day_in_car: bool = False
    nan_policy: str = "skip"
    extreme_car_threshold: float = _MAX_SANE_CAR
    min_post_days_required: int = 1

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("horizons must contain at least one value.")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(
                f"All horizon values must be > 0, got {self.horizons}."
            )
        valid_policies = {"raise", "skip", "zero"}
        if self.nan_policy not in valid_policies:
            raise ValueError(
                f"nan_policy must be one of {valid_policies}, "
                f"got '{self.nan_policy}'."
            )
        if self.extreme_car_threshold <= 0:
            raise ValueError(
                "extreme_car_threshold must be > 0."
            )
        if self.min_post_days_required < 1:
            raise ValueError("min_post_days_required must be >= 1.")

    @property
    def car_column_names(self) -> list[str]:
        return [f"car_{h}d" for h in self.horizons]

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)


@dataclass(slots=True)
class CARResult:
    """
    CAR result for a single event.

    Attributes
    ----------
    transcript_id:
        Primary key.
    ticker:
        Stock ticker.
    event_date:
        Aligned trading session (t=0).
    car_values:
        Mapping from horizon N → computed CAR value (or NaN if insufficient
        data).
    post_day_count:
        Number of post-event relative days actually present in the input.
    nan_day_count:
        Number of post-event days with NaN abnormal returns.
    is_complete:
        True iff every requested horizon had sufficient non-NaN data.
    extreme_flags:
        Set of horizons whose |CAR| exceeded the configured threshold.
    """

    transcript_id: str
    ticker: str
    event_date: pd.Timestamp
    car_values: dict[int, float]         # horizon → CAR
    post_day_count: int
    nan_day_count: int
    is_complete: bool
    extreme_flags: set[int] = field(default_factory=set)

    def to_series(self, config: CARConfig) -> pd.Series:
        """Flatten to a named Series for easy DataFrame assembly."""
        data: dict = {
            _TRANSCRIPT_ID_COL: self.transcript_id,
            _TICKER_COL: self.ticker,
            _EVENT_DATE_COL: self.event_date,
            "post_day_count": self.post_day_count,
            "nan_day_count": self.nan_day_count,
            "car_complete": self.is_complete,
        }
        for h in config.horizons:
            data[f"car_{h}d"] = self.car_values.get(h, np.nan)
        return pd.Series(data)


@dataclass(slots=True)
class CARSummary:
    """
    Aggregate summary statistics across all events in a CAR computation run.

    Attributes
    ----------
    total_events:
        Total number of events processed.
    complete_events:
        Events with sufficient data for all requested horizons.
    incomplete_events:
        Events missing data for at least one horizon.
    extreme_events:
        Events where at least one horizon's |CAR| exceeded the threshold.
    car_stats:
        Per-horizon descriptive statistics dict:
        ``{horizon: {mean, std, min, max, count_nan}}``.
    config_snapshot:
        Copy of the CARConfig used for this run.
    validation_errors:
        List of validation error strings.
    """

    total_events: int = 0
    complete_events: int = 0
    incomplete_events: int = 0
    extreme_events: int = 0
    car_stats: dict[int, dict] = field(default_factory=dict)
    config_snapshot: dict = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        return round(self.complete_events / self.total_events, 4) if self.total_events else 0.0

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class CARCalculator:
    """
    Computes Cumulative Abnormal Returns (CAR) for a set of earnings events.

    The calculator consumes a long-format abnormal-return DataFrame produced
    by ``abnormal_returns.py`` — one row per (event, trading session) — and
    outputs an event-level CAR DataFrame.

    Parameters
    ----------
    config:
        :class:`CARConfig` controlling horizons, NaN policy, and thresholds.

    Usage
    -----
    ::

        from src.event_study.car_calculator import CARCalculator, CARConfig

        config = CARConfig(horizons=(1, 3, 5))
        calc   = CARCalculator(config)

        car_df = calc.compute_car(abnormal_returns_df)
        car_df.to_parquet("data/interim/car_metrics.parquet", index=False)
    """

    def __init__(self, config: CARConfig) -> None:
        self._config = config
        logger.info(
            "CARCalculator initialised | horizons=%s nan_policy='%s' "
            "include_event_day=%s threshold=%.2f",
            config.horizons,
            config.nan_policy,
            config.include_event_day_in_car,
            config.extreme_car_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_car(self, abnormal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute CAR for all events in *abnormal_df*.

        Parameters
        ----------
        abnormal_df:
            Long-format DataFrame with one row per (event, trading session).
            Required columns:
                ``transcript_id``, ``ticker``, ``aligned_event_date``,
                ``relative_day``, ``abnormal_return``.

        Returns
        -------
        pd.DataFrame
            Event-level CAR DataFrame with columns:
                ``transcript_id``, ``ticker``, ``aligned_event_date``,
                ``car_1d``, ``car_3d``, ``car_5d`` (per configured horizons),
                ``post_day_count``, ``nan_day_count``, ``car_complete``.
            One row per event, sorted by ``transcript_id``.

        Raises
        ------
        ValueError
            If required columns are absent or the DataFrame is empty.
        """
        self._validate_input_schema(abnormal_df)

        df = abnormal_df.copy()
        df[_EVENT_DATE_COL] = pd.to_datetime(df[_EVENT_DATE_COL])
        df[_RELATIVE_DAY_COL] = df[_RELATIVE_DAY_COL].astype(int)

        n_events = df[_TRANSCRIPT_ID_COL].nunique()
        logger.info(
            "Computing CAR for %d events | horizons=%s",
            n_events,
            self._config.horizons,
        )

        results: list[CARResult] = []
        for tid, group in df.groupby(_TRANSCRIPT_ID_COL, sort=True):
            result = self.compute_event_car(group)
            results.append(result)

        car_df = self.build_car_frame(results)

        errors = self.validate_car_series(car_df)
        if errors:
            for e in errors:
                logger.error("CAR validation error: %s", e)
        else:
            logger.info("CAR validation passed — no errors detected.")

        summary = self.summarize_car_metrics(results, car_df, errors)
        self._log_summary(summary)

        return car_df

    def compute_event_car(self, event_df: pd.DataFrame) -> CARResult:
        """
        Compute CAR for a single event from its abnormal-return series.

        Parameters
        ----------
        event_df:
            Subset of the long-format DataFrame for one ``transcript_id``.
            Must contain ``relative_day`` and ``abnormal_return`` columns.

        Returns
        -------
        CARResult
        """
        transcript_id = str(event_df[_TRANSCRIPT_ID_COL].iloc[0])
        ticker = str(event_df[_TICKER_COL].iloc[0])
        event_date = pd.Timestamp(event_df[_EVENT_DATE_COL].iloc[0])

        # Sort by relative_day — deterministic
        event_df = event_df.sort_values(_RELATIVE_DAY_COL).reset_index(drop=True)

        # Select the post-event slice (and optionally event-day)
        start_rel = 0 if self._config.include_event_day_in_car else 1
        post_mask = event_df[_RELATIVE_DAY_COL] >= start_rel
        post_df = event_df.loc[post_mask].copy()

        post_day_count = int(post_mask.sum())
        nan_day_count = int(post_df[_ABNORMAL_RETURN_COL].isna().sum())

        # Apply NaN policy
        ar_series = self._apply_nan_policy(
            post_df[_ABNORMAL_RETURN_COL].values,
            transcript_id,
        )

        relative_days = post_df[_RELATIVE_DAY_COL].values

        car_values: dict[int, float] = {}
        extreme_flags: set[int] = set()
        is_complete = True

        for horizon in self._config.horizons:
            # Select days in [start_rel, horizon]
            horizon_mask = relative_days <= horizon
            ar_horizon = ar_series[horizon_mask]
            days_horizon = relative_days[horizon_mask]

            valid_count = int(np.sum(~np.isnan(ar_horizon)))
            has_horizon_coverage = bool(np.any(days_horizon == horizon))

            if valid_count < self._config.min_post_days_required:
                logger.debug(
                    "Insufficient data for CAR_%dd | id=%s valid_days=%d required=%d",
                    horizon,
                    transcript_id,
                    valid_count,
                    self._config.min_post_days_required,
                )
                car_values[horizon] = np.nan
                is_complete = False
                continue

            if not has_horizon_coverage:
                is_complete = False

            # Summation (NaNs already handled by policy)
            car = float(np.nansum(ar_horizon))
            car_values[horizon] = car

            # Extreme-value flag
            if abs(car) > self._config.extreme_car_threshold:
                extreme_flags.add(horizon)
                logger.warning(
                    "Extreme CAR detected | id=%s horizon=%dd car=%.4f threshold=%.2f",
                    transcript_id,
                    horizon,
                    car,
                    self._config.extreme_car_threshold,
                )

        logger.debug(
            "CAR computed | id=%s ticker=%s event=%s post_days=%d nan=%d "
            "complete=%s values=%s",
            transcript_id,
            ticker,
            event_date.date(),
            post_day_count,
            nan_day_count,
            is_complete,
            {f"{h}d": round(v, 6) for h, v in car_values.items() if not np.isnan(v)},
        )

        return CARResult(
            transcript_id=transcript_id,
            ticker=ticker,
            event_date=event_date,
            car_values=car_values,
            post_day_count=post_day_count,
            nan_day_count=nan_day_count,
            is_complete=is_complete,
            extreme_flags=extreme_flags,
        )

    def validate_car_series(self, car_df: pd.DataFrame) -> list[str]:
        """
        Validate the event-level CAR DataFrame for research integrity.

        Checks:
        1. No duplicate ``transcript_id`` values.
        2. All CAR columns present.
        3. ``aligned_event_date`` has no nulls.
        4. Fraction of NaN CAR values per horizon (logged as warning, not error).
        5. CAR magnitudes within [-threshold, +threshold].
        6. ``car_complete`` column present and boolean.

        Parameters
        ----------
        car_df:
            Output of :meth:`build_car_frame`.

        Returns
        -------
        list[str]
            Empty on success; one message per violated check.
        """
        errors: list[str] = []

        if car_df.empty:
            errors.append("CAR DataFrame is empty — no events were processed.")
            return errors

        # 1. Duplicate transcript_ids
        dup = car_df[_TRANSCRIPT_ID_COL].duplicated().sum()
        if dup:
            errors.append(
                f"Duplicate transcript_id values in CAR output: {int(dup)} duplicates."
            )

        # 2. CAR columns present
        for col in self._config.car_column_names:
            if col not in car_df.columns:
                errors.append(f"Expected CAR column '{col}' is missing.")

        # 3. Null event dates
        null_dates = car_df[_EVENT_DATE_COL].isnull().sum()
        if null_dates:
            errors.append(
                f"{int(null_dates)} rows have null aligned_event_date."
            )

        # 4. NaN CAR fraction (warning only — not an error)
        for h in self._config.horizons:
            col = f"car_{h}d"
            if col in car_df.columns:
                nan_frac = car_df[col].isna().mean()
                if nan_frac > 0.1:
                    logger.warning(
                        "CAR column '%s' has %.1f%% NaN values — "
                        "check for insufficient post-event data.",
                        col,
                        nan_frac * 100,
                    )

        # 5. Extreme CAR magnitudes
        for h in self._config.horizons:
            col = f"car_{h}d"
            if col not in car_df.columns:
                continue
            extreme_mask = car_df[col].abs() > self._config.extreme_car_threshold
            n_extreme = int(extreme_mask.sum())
            if n_extreme:
                errors.append(
                    f"{n_extreme} events have |{col}| > "
                    f"{self._config.extreme_car_threshold:.2f} (extreme threshold)."
                )

        # 6. car_complete column
        if "car_complete" not in car_df.columns:
            errors.append("Column 'car_complete' is missing from CAR output.")
        elif car_df["car_complete"].dtype != bool:
            errors.append(
                f"Column 'car_complete' has unexpected dtype "
                f"'{car_df['car_complete'].dtype}'; expected bool."
            )

        return errors

    def summarize_car_metrics(
        self,
        results: Sequence[CARResult],
        car_df: pd.DataFrame,
        validation_errors: list[str] | None = None,
    ) -> CARSummary:
        """
        Generate aggregate summary statistics across all CARResult objects.

        Parameters
        ----------
        results:
            Sequence of per-event :class:`CARResult` instances.
        car_df:
            Event-level CAR DataFrame from :meth:`build_car_frame`.
        validation_errors:
            Optional list of validation error strings.

        Returns
        -------
        CARSummary
        """
        total = len(results)
        complete = sum(r.is_complete for r in results)
        extreme = sum(len(r.extreme_flags) > 0 for r in results)

        car_stats: dict[int, dict] = {}
        for h in self._config.horizons:
            col = f"car_{h}d"
            if col in car_df.columns:
                series = car_df[col].dropna()
                car_stats[h] = {
                    "mean": round(float(series.mean()), 6) if len(series) else np.nan,
                    "median": round(float(series.median()), 6) if len(series) else np.nan,
                    "std": round(float(series.std()), 6) if len(series) else np.nan,
                    "min": round(float(series.min()), 6) if len(series) else np.nan,
                    "max": round(float(series.max()), 6) if len(series) else np.nan,
                    "count_valid": int(len(series)),
                    "count_nan": int(car_df[col].isna().sum()),
                    "pct_positive": round(
                        float((series > 0).mean()) * 100, 2
                    ) if len(series) else np.nan,
                }

        summary = CARSummary(
            total_events=total,
            complete_events=complete,
            incomplete_events=total - complete,
            extreme_events=extreme,
            car_stats=car_stats,
            config_snapshot={
                "horizons": list(self._config.horizons),
                "include_event_day_in_car": self._config.include_event_day_in_car,
                "nan_policy": self._config.nan_policy,
                "extreme_car_threshold": self._config.extreme_car_threshold,
                "min_post_days_required": self._config.min_post_days_required,
            },
            validation_errors=validation_errors or [],
        )
        return summary

    def build_car_frame(self, results: Sequence[CARResult]) -> pd.DataFrame:
        """
        Assemble a clean event-level CAR DataFrame from per-event results.

        Output columns (deterministic order):
            ``transcript_id``, ``ticker``, ``aligned_event_date``,
            ``car_1d``, ``car_3d``, ``car_5d`` (per horizons),
            ``post_day_count``, ``nan_day_count``, ``car_complete``.

        Parameters
        ----------
        results:
            Sequence of :class:`CARResult` objects from :meth:`compute_event_car`.

        Returns
        -------
        pd.DataFrame
            One row per event, sorted by ``transcript_id``.
        """
        if not results:
            logger.warning("build_car_frame called with empty results list.")
            cols = (
                [_TRANSCRIPT_ID_COL, _TICKER_COL, _EVENT_DATE_COL]
                + self._config.car_column_names
                + ["post_day_count", "nan_day_count", "car_complete"]
            )
            return pd.DataFrame(columns=cols)

        rows = [r.to_series(self._config) for r in results]
        df = pd.DataFrame(rows)

        # Enforce dtypes
        df[_EVENT_DATE_COL] = pd.to_datetime(df[_EVENT_DATE_COL])
        df["post_day_count"] = df["post_day_count"].astype(np.int32)
        df["nan_day_count"] = df["nan_day_count"].astype(np.int32)
        df["car_complete"] = df["car_complete"].astype(bool)

        for col in self._config.car_column_names:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        # Deterministic sort
        df.sort_values(_TRANSCRIPT_ID_COL, ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Canonical column order
        ordered_cols = (
            [_TRANSCRIPT_ID_COL, _TICKER_COL, _EVENT_DATE_COL]
            + self._config.car_column_names
            + ["post_day_count", "nan_day_count", "car_complete"]
        )
        extra = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + extra]

        logger.debug(
            "Built CAR frame | rows=%d cols=%s",
            len(df),
            list(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_nan_policy(
        self,
        ar_values: np.ndarray,
        transcript_id: str,
    ) -> np.ndarray:
        """
        Apply the configured NaN policy to a raw abnormal-return array.

        Returns a float64 array with NaNs handled per policy.
        """
        ar = ar_values.astype(np.float64)
        nan_mask = np.isnan(ar)

        if not nan_mask.any():
            return ar

        n_nan = int(nan_mask.sum())

        if self._config.nan_policy == "raise":
            raise ValueError(
                f"abnormal_return has {n_nan} NaN value(s) for "
                f"transcript_id='{transcript_id}' and nan_policy='raise'."
            )
        elif self._config.nan_policy == "zero":
            logger.debug(
                "Replacing %d NaN abnormal returns with 0.0 | id=%s",
                n_nan,
                transcript_id,
            )
            ar = np.where(nan_mask, 0.0, ar)
        else:
            # "skip" — leave NaNs in place; nansum will ignore them
            logger.debug(
                "Skipping %d NaN abnormal returns | id=%s",
                n_nan,
                transcript_id,
            )

        return ar

    def _log_summary(self, summary: CARSummary) -> None:
        logger.info(
            "CAR summary | events=%d complete=%d (%.1f%%) incomplete=%d extreme=%d",
            summary.total_events,
            summary.complete_events,
            summary.completion_rate * 100,
            summary.incomplete_events,
            summary.extreme_events,
        )
        for h, stats in summary.car_stats.items():
            logger.info(
                "  CAR_%dd | mean=%+.4f median=%+.4f std=%.4f "
                "min=%+.4f max=%+.4f valid=%d nan=%d pct_pos=%.1f%%",
                h,
                stats.get("mean", np.nan),
                stats.get("median", np.nan),
                stats.get("std", np.nan),
                stats.get("min", np.nan),
                stats.get("max", np.nan),
                stats.get("count_valid", 0),
                stats.get("count_nan", 0),
                stats.get("pct_positive", np.nan),
            )
        if not summary.is_valid:
            logger.error(
                "%d validation error(s) found in CAR output.",
                len(summary.validation_errors),
            )

    @staticmethod
    def _validate_input_schema(df: pd.DataFrame) -> None:
        missing = _REQUIRED_INPUT_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"abnormal_df is missing required columns: {sorted(missing)}. "
                f"Expected: {sorted(_REQUIRED_INPUT_COLUMNS)}"
            )
        if df.empty:
            raise ValueError(
                "abnormal_df is empty — no abnormal returns to process."
            )
        null_ids = df[_TRANSCRIPT_ID_COL].isnull().sum()
        if null_ids:
            raise ValueError(
                f"{null_ids} rows have null transcript_id in abnormal_df."
            )
        null_rel = df[_RELATIVE_DAY_COL].isnull().sum()
        if null_rel:
            raise ValueError(
                f"{null_rel} rows have null relative_day in abnormal_df."
            )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_car_calculator(
    horizons: tuple[int, ...] = _DEFAULT_CAR_HORIZONS,
    include_event_day_in_car: bool = False,
    nan_policy: str = "skip",
    extreme_car_threshold: float = _MAX_SANE_CAR,
    min_post_days_required: int = 1,
) -> CARCalculator:
    """
    Convenience factory for constructing a :class:`CARCalculator`.

    Parameters
    ----------
    horizons:
        Post-event trading-day horizons.  Default: (1, 3, 5).
    include_event_day_in_car:
        Whether to include the event day in the cumulative sum.
    nan_policy:
        ``"skip"`` | ``"zero"`` | ``"raise"``.
    extreme_car_threshold:
        Absolute CAR values beyond this are flagged.  Default: 1.0.
    min_post_days_required:
        Minimum valid post-event days to produce a non-NaN CAR.

    Returns
    -------
    CARCalculator
    """
    config = CARConfig(
        horizons=horizons,
        include_event_day_in_car=include_event_day_in_car,
        nan_policy=nan_policy,
        extreme_car_threshold=extreme_car_threshold,
        min_post_days_required=min_post_days_required,
    )
    return CARCalculator(config)
