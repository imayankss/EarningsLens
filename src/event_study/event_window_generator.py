"""
src/event_study/event_window_generator.py
==========================================
Trading-session event window generation engine for the DAY 6 event-study pipeline.

Responsibilities:
    - Generate trading-session windows around earnings events
    - Attach relative trading days (t-N ... t=0 ... t+N)
    - Support configurable pre/post windows
    - Validate window completeness and integrity
    - Preserve deterministic ordering
    - Produce event-study-ready window datasets

Scope:
    - Window generation ONLY
    - No abnormal return computation
    - No CAR computation
    - No benchmark download logic
    - No trading-calendar engine (injected via dependency)

Dependencies:
    - pandas >= 2.0
    - numpy >= 1.24
    - pandas_market_calendars >= 4.0  (injected; not imported here)
    - trading_calendar.py  (provides valid session lists)

Author: Earnings Call Sentiment Analyzer Pipeline
Python: 3.11+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Final, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PRE_WINDOW: Final[int] = 10
_DEFAULT_POST_WINDOW: Final[int] = 5
_REQUIRED_INPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    {"transcript_id", "ticker", "aligned_event_date"}
)
_OUTPUT_COLUMN_ORDER: Final[list[str]] = [
    "transcript_id",
    "ticker",
    "aligned_event_date",
    "window_start",
    "window_end",
    "date",
    "relative_day",
    "event_day_flag",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventWindowConfig:
    """
    Immutable configuration for event-window generation.

    Parameters
    ----------
    pre_event_days:
        Number of trading sessions to include *before* the event (t=0).
        Must be >= 0.  Default: 10.
    post_event_days:
        Number of trading sessions to include *after* the event (t=0).
        Must be >= 0.  Default: 5.
    include_event_day:
        Whether to include the event session itself (relative_day == 0).
        Default: True.
    min_required_post_days:
        Minimum post-event trading sessions a window must contain to be
        considered complete.  Windows with fewer sessions are flagged.
        Default: equal to post_event_days (strict).
    min_required_pre_days:
        Minimum pre-event trading sessions a window must contain to be
        considered complete.  Default: equal to pre_event_days (strict).
    """

    pre_event_days: int = _DEFAULT_PRE_WINDOW
    post_event_days: int = _DEFAULT_POST_WINDOW
    include_event_day: bool = True
    min_required_post_days: int | None = None
    min_required_pre_days: int | None = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def effective_min_post(self) -> int:
        return (
            self.post_event_days
            if self.min_required_post_days is None
            else self.min_required_post_days
        )

    @property
    def effective_min_pre(self) -> int:
        return (
            self.pre_event_days
            if self.min_required_pre_days is None
            else self.min_required_pre_days
        )

    @property
    def total_window_size(self) -> int:
        event_contrib = 1 if self.include_event_day else 0
        return self.pre_event_days + event_contrib + self.post_event_days

    def __post_init__(self) -> None:
        if self.pre_event_days < 0:
            raise ValueError(
                f"pre_event_days must be >= 0, got {self.pre_event_days}"
            )
        if self.post_event_days < 0:
            raise ValueError(
                f"post_event_days must be >= 0, got {self.post_event_days}"
            )


@dataclass(slots=True)
class EventWindow:
    """
    Represents the fully-resolved trading-session window for a single event.

    Attributes
    ----------
    transcript_id:
        Primary key linking back to the transcript metadata.
    ticker:
        Stock ticker symbol.
    aligned_event_date:
        The NYSE-aligned trading session used as t=0.
    window_start:
        Earliest session in the window (t = -pre_event_days).
    window_end:
        Latest session in the window (t = +post_event_days).
    sessions:
        Ordered list of valid NYSE trading dates within the window.
    relative_days:
        Integer offsets corresponding to each session in ``sessions``.
    is_complete:
        True iff all required pre- and post-event sessions are present.
    missing_pre_count:
        Number of pre-event sessions that could not be sourced.
    missing_post_count:
        Number of post-event sessions that could not be sourced.
    """

    transcript_id: str
    ticker: str
    aligned_event_date: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    sessions: list[pd.Timestamp]
    relative_days: list[int]
    is_complete: bool
    missing_pre_count: int = 0
    missing_post_count: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Expand this window into a tidy long-format DataFrame row per session."""
        n = len(self.sessions)
        return pd.DataFrame(
            {
                "transcript_id": [self.transcript_id] * n,
                "ticker": [self.ticker] * n,
                "aligned_event_date": [self.aligned_event_date] * n,
                "window_start": [self.window_start] * n,
                "window_end": [self.window_end] * n,
                "date": self.sessions,
                "relative_day": self.relative_days,
                "event_day_flag": [rd == 0 for rd in self.relative_days],
            }
        )


@dataclass(slots=True)
class WindowGenerationResult:
    """
    Container for the full output of a window-generation run.

    Attributes
    ----------
    windows:
        One :class:`EventWindow` per input event, in input order.
    window_df:
        Long-format DataFrame (one row per event×session) ready for
        downstream pipeline stages.
    summary:
        Human-readable summary statistics as a dict.
    incomplete_event_ids:
        ``transcript_id`` values whose windows did not meet completeness
        thresholds.
    validation_errors:
        List of validation error messages.  Empty iff validation passed.
    """

    windows: list[EventWindow] = field(default_factory=list)
    window_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)
    incomplete_event_ids: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0

    @property
    def total_events(self) -> int:
        return len(self.windows)

    @property
    def complete_events(self) -> int:
        return sum(w.is_complete for w in self.windows)


# ---------------------------------------------------------------------------
# Type alias for the trading-session provider
# ---------------------------------------------------------------------------

# Callers inject a function that, given a start date and end date,
# returns a sorted list of valid NYSE trading-session Timestamps.
TradingSessionProvider = Callable[
    [pd.Timestamp, pd.Timestamp], list[pd.Timestamp]
]


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class EventWindowGenerator:
    """
    Generates trading-session event windows for a set of earnings events.

    The generator does **not** own or implement the trading-calendar logic.
    Instead it accepts a ``session_provider`` callable that returns valid
    NYSE sessions for any date range.  This keeps the calendar logic in
    ``trading_calendar.py`` and satisfies the no-duplicated-calendar-logic
    requirement.

    Parameters
    ----------
    config:
        Window-generation configuration.
    session_provider:
        Callable ``(start: Timestamp, end: Timestamp) -> list[Timestamp]``
        that returns sorted valid trading sessions.  Typically sourced from
        ``trading_calendar.TradingCalendar.get_sessions()``.

    Usage
    -----
    ::

        from src.finance.trading_calendar import TradingCalendar
        from src.event_study.event_window_generator import (
            EventWindowConfig,
            EventWindowGenerator,
        )

        calendar = TradingCalendar()
        config   = EventWindowConfig(pre_event_days=10, post_event_days=5)
        gen      = EventWindowGenerator(config, calendar.get_sessions)

        result = gen.generate_event_windows(events_df)
        result.window_df.to_parquet("data/interim/aligned_event_windows.parquet")
    """

    def __init__(
        self,
        config: EventWindowConfig,
        session_provider: TradingSessionProvider,
    ) -> None:
        self._config = config
        self._session_provider = session_provider
        logger.info(
            "EventWindowGenerator initialised | pre=%d post=%d include_event_day=%s",
            config.pre_event_days,
            config.post_event_days,
            config.include_event_day,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_event_windows(
        self,
        events_df: pd.DataFrame,
    ) -> WindowGenerationResult:
        """
        Main entry point: generate event windows for every row in *events_df*.

        Parameters
        ----------
        events_df:
            DataFrame with at minimum the columns:
            ``transcript_id``, ``ticker``, ``aligned_event_date``.
            Additional columns are silently preserved in the result via
            ``build_event_dataset()``.

        Returns
        -------
        WindowGenerationResult
            Contains the per-event :class:`EventWindow` objects, the merged
            long-format DataFrame, summary statistics, and validation output.

        Raises
        ------
        ValueError
            If required columns are absent.
        """
        self._validate_input_schema(events_df)

        events_df = events_df.copy()
        events_df["aligned_event_date"] = pd.to_datetime(
            events_df["aligned_event_date"]
        )

        logger.info(
            "Generating event windows for %d events across %d tickers.",
            len(events_df),
            events_df["ticker"].nunique(),
        )

        windows: list[EventWindow] = []
        for _, row in events_df.iterrows():
            win = self._build_single_window(
                transcript_id=str(row["transcript_id"]),
                ticker=str(row["ticker"]),
                event_date=pd.Timestamp(row["aligned_event_date"]),
            )
            windows.append(win)

        result = WindowGenerationResult(windows=windows)
        result.window_df = self.build_event_dataset(windows)
        result.incomplete_event_ids = [
            w.transcript_id for w in windows if not w.is_complete
        ]
        result.validation_errors = self.validate_windows(result.window_df)
        result.summary = self.summarize_windows(result)

        if result.incomplete_event_ids:
            logger.warning(
                "%d events have incomplete windows: %s",
                len(result.incomplete_event_ids),
                result.incomplete_event_ids[:10],
            )

        if result.validation_errors:
            for err in result.validation_errors:
                logger.error("Validation error: %s", err)
        else:
            logger.info("All validation checks passed.")

        logger.info(
            "Window generation complete | total=%d complete=%d incomplete=%d rows=%d",
            result.total_events,
            result.complete_events,
            len(result.incomplete_event_ids),
            len(result.window_df),
        )

        return result

    def build_event_dataset(
        self,
        windows: Sequence[EventWindow],
    ) -> pd.DataFrame:
        """
        Concatenate all per-event windows into a single long-format DataFrame.

        The resulting DataFrame has one row per (event, trading session),
        sorted deterministically by ``(transcript_id, relative_day)``.

        Columns returned follow ``_OUTPUT_COLUMN_ORDER``.

        Parameters
        ----------
        windows:
            Iterable of :class:`EventWindow` instances.

        Returns
        -------
        pd.DataFrame
        """
        if not windows:
            logger.warning("build_event_dataset called with empty window list.")
            return pd.DataFrame(columns=_OUTPUT_COLUMN_ORDER)

        frames = [w.to_dataframe() for w in windows]
        df = pd.concat(frames, ignore_index=True)

        # Enforce deterministic sort: (transcript_id, relative_day)
        df.sort_values(
            ["transcript_id", "relative_day"],
            ascending=[True, True],
            inplace=True,
            ignore_index=True,
        )

        # Canonical column order
        extra_cols = [c for c in df.columns if c not in _OUTPUT_COLUMN_ORDER]
        df = df[_OUTPUT_COLUMN_ORDER + extra_cols]

        # Ensure correct dtypes for parquet compatibility
        df["date"] = pd.to_datetime(df["date"])
        df["aligned_event_date"] = pd.to_datetime(df["aligned_event_date"])
        df["window_start"] = pd.to_datetime(df["window_start"])
        df["window_end"] = pd.to_datetime(df["window_end"])
        df["relative_day"] = df["relative_day"].astype(np.int16)
        df["event_day_flag"] = df["event_day_flag"].astype(bool)

        logger.debug(
            "Built event dataset: %d rows for %d events.",
            len(df),
            df["transcript_id"].nunique(),
        )
        return df

    def attach_relative_days(
        self,
        df: pd.DataFrame,
        sessions_col: str = "date",
        event_date_col: str = "aligned_event_date",
        ticker_col: str = "ticker",
        output_col: str = "relative_day",
    ) -> pd.DataFrame:
        """
        (Re-)compute ``relative_day`` for an existing long-format DataFrame.

        Relative days are computed as the integer offset from the event
        session within *valid trading sessions only* — not calendar days.

        This method is useful when a caller already has a merged market
        DataFrame and wants to attach or recompute relative-day offsets
        without regenerating the full window.

        Parameters
        ----------
        df:
            Long-format DataFrame with at minimum ``sessions_col``,
            ``event_date_col``, and ``ticker_col``.
        sessions_col:
            Column containing the trading-session date.
        event_date_col:
            Column containing the aligned event date (t=0).
        ticker_col:
            Column containing the ticker (used for groupby).
        output_col:
            Name of the relative-day output column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``output_col`` added / replaced.
        """
        logger.debug("Attaching relative trading days to DataFrame.")

        df = df.copy()
        df[sessions_col] = pd.to_datetime(df[sessions_col])
        df[event_date_col] = pd.to_datetime(df[event_date_col])

        group_cols = ["transcript_id"] if "transcript_id" in df.columns else [ticker_col, event_date_col]
        df[output_col] = np.nan

        def _relative_days_for_group(group: pd.DataFrame) -> pd.Series:
            event_date = group[event_date_col].iloc[0]
            sorted_sessions = sorted(group[sessions_col].unique())

            # Build offset map: session → relative_day integer
            try:
                event_idx = sorted_sessions.index(event_date)
            except ValueError:
                # event date not in the group — offset relative to nearest
                logger.warning(
                    "Event date %s not found in sessions for ticker %s; "
                    "computing offset relative to closest session.",
                    event_date.date(),
                    group[ticker_col].iloc[0],
                )
                diffs = [abs((s - event_date).days) for s in sorted_sessions]
                event_idx = int(np.argmin(diffs))

            offset_map: dict[pd.Timestamp, int] = {
                s: i - event_idx for i, s in enumerate(sorted_sessions)
            }
            return group[sessions_col].map(offset_map)

        for _, group in df.groupby(group_cols, sort=False):
            df.loc[group.index, output_col] = _relative_days_for_group(group).to_numpy()

        df[output_col] = df[output_col].astype(np.int16)
        return df

    def validate_windows(self, window_df: pd.DataFrame) -> list[str]:
        """
        Run all integrity checks on the long-format window DataFrame.

        Checks performed:

        1. Required columns present.
        2. No duplicate (transcript_id, date) rows.
        3. Relative-day sequences are monotonically increasing per event.
        4. No missing values in key columns.
        5. Each event has the expected session count.
        6. ``event_day_flag`` is True exactly once per event (when
           ``include_event_day`` is True).
        7. ``window_start`` <= ``date`` <= ``window_end`` for every row.

        Parameters
        ----------
        window_df:
            Output of :meth:`build_event_dataset`.

        Returns
        -------
        list[str]
            Empty list on success; one error message per failed check.
        """
        errors: list[str] = []

        if window_df.empty:
            errors.append("window_df is empty — no windows generated.")
            return errors

        # 1. Required columns
        missing_cols = set(_OUTPUT_COLUMN_ORDER) - set(window_df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {sorted(missing_cols)}")
            return errors  # cannot proceed safely

        # 2. Duplicate (transcript_id, date) pairs
        dup_mask = window_df.duplicated(["transcript_id", "date"])
        if dup_mask.any():
            dup_count = int(dup_mask.sum())
            errors.append(
                f"Found {dup_count} duplicate (transcript_id, date) rows."
            )

        # 3. Monotonic relative days per event
        def _is_monotone(grp: pd.DataFrame) -> bool:
            sorted_rd = grp["relative_day"].sort_values()
            return bool((sorted_rd.diff().dropna() >= 1).all())

        non_monotone = (
            window_df.groupby("transcript_id")
            .apply(_is_monotone)
            .pipe(lambda s: s[~s])
        )
        if not non_monotone.empty:
            bad_ids = non_monotone.index.tolist()
            errors.append(
                f"Non-monotonic relative_day sequences for "
                f"{len(bad_ids)} events: {bad_ids[:5]}"
            )

        # 4. Nulls in key columns
        for col in ["transcript_id", "ticker", "date", "relative_day",
                     "aligned_event_date", "window_start", "window_end"]:
            null_count = int(window_df[col].isnull().sum())
            if null_count:
                errors.append(
                    f"Column '{col}' has {null_count} null value(s)."
                )

        # 5. Session count per event
        expected_min = self._config.effective_min_pre + self._config.effective_min_post
        if self._config.include_event_day:
            expected_min += 1

        session_counts = window_df.groupby("transcript_id")["date"].count()
        under_min = session_counts[session_counts < expected_min]
        if not under_min.empty:
            errors.append(
                f"{len(under_min)} events have fewer than {expected_min} "
                f"sessions (min required). First few: "
                f"{under_min.head(3).to_dict()}"
            )

        # 6. Event-day flag count
        if self._config.include_event_day:
            event_flag_counts = (
                window_df[window_df["event_day_flag"]]
                .groupby("transcript_id")["event_day_flag"]
                .sum()
            )
            all_events = set(window_df["transcript_id"].unique())
            flagged_events = set(event_flag_counts.index)
            missing_flag = all_events - flagged_events
            if missing_flag:
                errors.append(
                    f"{len(missing_flag)} events have no event_day_flag=True row: "
                    f"{list(missing_flag)[:5]}"
                )
            multi_flag = event_flag_counts[event_flag_counts > 1]
            if not multi_flag.empty:
                errors.append(
                    f"{len(multi_flag)} events have multiple event_day_flag=True rows."
                )

        # 7. Date bounds: window_start <= date <= window_end
        out_of_bounds = window_df[
            (window_df["date"] < window_df["window_start"])
            | (window_df["date"] > window_df["window_end"])
        ]
        if not out_of_bounds.empty:
            errors.append(
                f"{len(out_of_bounds)} rows have 'date' outside "
                f"[window_start, window_end]."
            )

        return errors

    def summarize_windows(
        self, result: WindowGenerationResult
    ) -> dict:
        """
        Produce a structured summary dict for the generation run.

        Parameters
        ----------
        result:
            The :class:`WindowGenerationResult` whose ``windows`` and
            ``window_df`` are already populated.

        Returns
        -------
        dict
            Keys include ``total_events``, ``complete_events``,
            ``incomplete_events``, ``completion_rate``,
            ``total_rows``, ``tickers``, ``session_count_stats``,
            ``relative_day_range``, ``config``.
        """
        windows = result.windows
        df = result.window_df

        total = len(windows)
        complete = sum(w.is_complete for w in windows)
        incomplete = total - complete

        session_counts = (
            df.groupby("transcript_id")["date"].count()
            if not df.empty
            else pd.Series(dtype=int)
        )

        rd_min = int(df["relative_day"].min()) if not df.empty else 0
        rd_max = int(df["relative_day"].max()) if not df.empty else 0

        missing_pre_total = sum(w.missing_pre_count for w in windows)
        missing_post_total = sum(w.missing_post_count for w in windows)

        summary = {
            "total_events": total,
            "complete_events": complete,
            "incomplete_events": incomplete,
            "completion_rate": round(complete / total, 4) if total else 0.0,
            "total_rows": len(df),
            "tickers": sorted(df["ticker"].unique().tolist()) if not df.empty else [],
            "session_count_stats": {
                "mean": round(float(session_counts.mean()), 2) if len(session_counts) else 0,
                "min": int(session_counts.min()) if len(session_counts) else 0,
                "max": int(session_counts.max()) if len(session_counts) else 0,
            },
            "relative_day_range": {"min": rd_min, "max": rd_max},
            "missing_pre_sessions_total": missing_pre_total,
            "missing_post_sessions_total": missing_post_total,
            "validation_passed": result.is_valid,
            "validation_error_count": len(result.validation_errors),
            "config": {
                "pre_event_days": self._config.pre_event_days,
                "post_event_days": self._config.post_event_days,
                "include_event_day": self._config.include_event_day,
                "min_required_pre": self._config.effective_min_pre,
                "min_required_post": self._config.effective_min_post,
                "total_window_size": self._config.total_window_size,
            },
        }

        logger.info(
            "Summary | events=%d complete=%d (%.1f%%) rows=%d rd=[%d,%d]",
            total,
            complete,
            summary["completion_rate"] * 100,
            len(df),
            rd_min,
            rd_max,
        )
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_single_window(
        self,
        transcript_id: str,
        ticker: str,
        event_date: pd.Timestamp,
    ) -> EventWindow:
        """
        Resolve the session list and relative-day map for one event.

        The session provider is asked for a *generous* date range
        (calendar days, not trading days) to ensure we always fetch
        enough trading sessions even over long holiday stretches.
        """
        # Use a 3× calendar-day buffer so that weekends + holidays never
        # starve the window of requested trading days.
        buffer_multiplier = 3
        start_cal = event_date - timedelta(
            days=self._config.pre_event_days * buffer_multiplier
        )
        end_cal = event_date + timedelta(
            days=self._config.post_event_days * buffer_multiplier
        )

        all_sessions: list[pd.Timestamp] = self._session_provider(
            start_cal, end_cal
        )

        if not all_sessions:
            logger.error(
                "Session provider returned no sessions for ticker=%s event=%s "
                "range=[%s, %s]",
                ticker,
                event_date.date(),
                start_cal.date(),
                end_cal.date(),
            )
            return self._empty_window(transcript_id, ticker, event_date)

        all_sessions_sorted = sorted(set(all_sessions))

        # Locate event date in session list
        try:
            event_idx = all_sessions_sorted.index(event_date)
        except ValueError:
            # event_date is not a trading session — find nearest future session
            future = [s for s in all_sessions_sorted if s >= event_date]
            if not future:
                logger.error(
                    "No trading session on or after event_date=%s for ticker=%s",
                    event_date.date(),
                    ticker,
                )
                return self._empty_window(transcript_id, ticker, event_date)
            nearest = future[0]
            logger.warning(
                "event_date=%s is not a trading session for ticker=%s; "
                "using nearest future session=%s",
                event_date.date(),
                ticker,
                nearest.date(),
            )
            event_date = nearest
            event_idx = all_sessions_sorted.index(event_date)

        # Slice pre / post from session list
        pre_start_idx = event_idx - self._config.pre_event_days
        post_end_idx = event_idx + self._config.post_event_days + 1  # inclusive slice

        available_pre = all_sessions_sorted[:event_idx]
        available_post = all_sessions_sorted[event_idx + 1:]

        pre_sessions: list[pd.Timestamp] = (
            available_pre[-self._config.pre_event_days:]
            if self._config.pre_event_days > 0
            else []
        )
        post_sessions: list[pd.Timestamp] = (
            available_post[: self._config.post_event_days]
            if self._config.post_event_days > 0
            else []
        )

        # Assemble session sequence
        window_sessions: list[pd.Timestamp] = list(pre_sessions)
        if self._config.include_event_day:
            window_sessions.append(event_date)
        window_sessions.extend(post_sessions)
        window_sessions.sort()  # deterministic

        if not window_sessions:
            return self._empty_window(transcript_id, ticker, event_date)

        # Assign relative days: each session gets its offset from event_date
        # as counted in *trading sessions* (not calendar days)
        relative_days: list[int] = self._compute_relative_days(
            window_sessions, event_date
        )

        # Completeness assessment
        missing_pre = self._config.pre_event_days - len(pre_sessions)
        missing_post = self._config.post_event_days - len(post_sessions)
        is_complete = (
            missing_pre <= (self._config.pre_event_days - self._config.effective_min_pre)
            and missing_post <= (self._config.post_event_days - self._config.effective_min_post)
        )

        window_start = window_sessions[0]
        window_end = window_sessions[-1]

        logger.debug(
            "Built window | id=%s ticker=%s event=%s sessions=%d "
            "missing_pre=%d missing_post=%d complete=%s",
            transcript_id,
            ticker,
            event_date.date(),
            len(window_sessions),
            missing_pre,
            missing_post,
            is_complete,
        )

        return EventWindow(
            transcript_id=transcript_id,
            ticker=ticker,
            aligned_event_date=event_date,
            window_start=window_start,
            window_end=window_end,
            sessions=window_sessions,
            relative_days=relative_days,
            is_complete=is_complete,
            missing_pre_count=max(missing_pre, 0),
            missing_post_count=max(missing_post, 0),
        )

    @staticmethod
    def _compute_relative_days(
        sessions: list[pd.Timestamp],
        event_date: pd.Timestamp,
    ) -> list[int]:
        """
        Compute integer trading-session offsets relative to *event_date*.

        Offsets are based on *position* in the ordered session sequence,
        not calendar-day differences, so weekends and holidays are ignored.

        Returns
        -------
        list[int]
            One integer per session in *sessions*.
        """
        if event_date not in sessions:
            # Fallback: compute closest-index offset
            dates_arr = sorted(sessions)
            diffs = [abs((s - event_date).days) for s in dates_arr]
            event_pos = int(np.argmin(diffs))
        else:
            event_pos = sessions.index(event_date)

        return [i - event_pos for i in range(len(sessions))]

    @staticmethod
    def _empty_window(
        transcript_id: str,
        ticker: str,
        event_date: pd.Timestamp,
    ) -> EventWindow:
        """Return a degenerate empty window for events that could not be resolved."""
        return EventWindow(
            transcript_id=transcript_id,
            ticker=ticker,
            aligned_event_date=event_date,
            window_start=event_date,
            window_end=event_date,
            sessions=[],
            relative_days=[],
            is_complete=False,
            missing_pre_count=-1,
            missing_post_count=-1,
        )

    @staticmethod
    def _validate_input_schema(df: pd.DataFrame) -> None:
        missing = _REQUIRED_INPUT_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {sorted(missing)}. "
                f"Expected at minimum: {sorted(_REQUIRED_INPUT_COLUMNS)}"
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty — no events to process.")
        null_events = df["aligned_event_date"].isnull().sum()
        if null_events:
            raise ValueError(
                f"{null_events} rows have null 'aligned_event_date'. "
                "All events must have a valid aligned trading session."
            )
        null_ids = df["transcript_id"].isnull().sum()
        if null_ids:
            raise ValueError(
                f"{null_ids} rows have null 'transcript_id'. "
                "Every event must have a unique transcript_id."
            )
        dup_ids = df["transcript_id"].duplicated().sum()
        if dup_ids:
            raise ValueError(
                f"{dup_ids} duplicate 'transcript_id' values found in input. "
                "transcript_id must be unique per event."
            )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_event_window_generator(
    pre_event_days: int = _DEFAULT_PRE_WINDOW,
    post_event_days: int = _DEFAULT_POST_WINDOW,
    include_event_day: bool = True,
    session_provider: TradingSessionProvider | None = None,
    min_required_pre_days: int | None = None,
    min_required_post_days: int | None = None,
) -> EventWindowGenerator:
    """
    Convenience factory for constructing an :class:`EventWindowGenerator`.

    Parameters
    ----------
    pre_event_days:
        Trading sessions before the event.
    post_event_days:
        Trading sessions after the event.
    include_event_day:
        Whether to include t=0.
    session_provider:
        Callable returning valid sessions for a date range.  If ``None``,
        a naive weekday-only provider is used (no holiday handling).
        **Production usage should always pass a real NYSE calendar provider.**
    min_required_pre_days:
        Minimum pre sessions for completeness.  Defaults to ``pre_event_days``.
    min_required_post_days:
        Minimum post sessions for completeness.  Defaults to ``post_event_days``.

    Returns
    -------
    EventWindowGenerator
    """
    config = EventWindowConfig(
        pre_event_days=pre_event_days,
        post_event_days=post_event_days,
        include_event_day=include_event_day,
        min_required_pre_days=min_required_pre_days,
        min_required_post_days=min_required_post_days,
    )

    if session_provider is None:
        logger.warning(
            "No session_provider supplied — using naive weekday provider. "
            "This ignores market holidays. Pass a TradingCalendar provider "
            "for production-grade correctness."
        )
        session_provider = _naive_weekday_provider

    return EventWindowGenerator(config=config, session_provider=session_provider)


def _naive_weekday_provider(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    """
    Fallback session provider: returns Mon–Fri dates only (no holiday awareness).
    Suitable only for testing / development.  Replace with a real NYSE calendar.
    """
    return [
        pd.Timestamp(d)
        for d in pd.bdate_range(start=start, end=end, freq="B")
    ]
