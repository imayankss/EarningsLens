"""
src/finance/trading_calendar.py
================================
NYSE trading-calendar engine and earnings-event alignment system for the
Earnings Call Sentiment Analyzer pipeline.

This is the **most critical** file in the Day 5 market pipeline because every
downstream component — market_data_loader, benchmark_loader, event_alignment,
and market_dataset_builder — depends on the dates it produces being correct.

Responsibilities
----------------
- Validate NYSE trading sessions (holiday + weekend awareness)
- Find the previous / next valid NYSE trading day from any date
- Normalize arbitrary timestamps to a canonical timezone-safe form
- Align an earnings-release timestamp to the correct event study date
  following the before/after-market-close rule:

      ┌────────────────────────────────────────────────────────┐
      │  release BEFORE market open → same trading day         │
      │  release AFTER  market close → NEXT trading day        │
      │  release on weekend / holiday → NEXT valid NYSE session │
      └────────────────────────────────────────────────────────┘

- Produce deterministic, reproducible aligned dates
- Validate a batch of aligned event dates for pipeline quality gates

What this module does NOT do
-----------------------------
- Download prices                   (→ market_data_loader.py)
- Compute returns or volatility      (→ feature_engineering.py)
- Compute abnormal returns          (→ abnormal_returns.py)
- Merge stock / benchmark datasets  (→ event_alignment.py)
- Orchestrate the full pipeline     (→ market_dataset_builder.py)

Design decisions
----------------
* All internal comparisons are done in **UTC** to avoid DST surprises.
* The NYSE session schedule is **cached** on first access; subsequent calls
  to the same engine instance do zero network / disk I/O.
* ``EventAlignmentResult`` is an immutable dataclass — results are safe to
  hash, log, and store without defensive copying.
* Every public method is fully type-hinted and carries a production docstring.

Usage
-----
    engine = TradingCalendarEngine()
    result = engine.align_earnings_event("2024-01-26 16:30", "America/New_York")
    print(result.aligned_event_date)   # 2024-01-29  (next trading day)

    # Or use the convenience one-liner
    aligned_date = engine.align_earnings_event(ts, tz).aligned_event_date
"""

from __future__ import annotations

import logging
import zoneinfo
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Optional, Sequence, Union

import pandas as pd
import pandas_market_calendars as mcal

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DateLike = Union[str, date, datetime, pd.Timestamp]

# ---------------------------------------------------------------------------
# Market-hours constants (NYSE, America/New_York)
# ---------------------------------------------------------------------------
NYSE_TZ: str = "America/New_York"
NYSE_CALENDAR: str = "NYSE"

#: Regular market open — 09:30 ET
MARKET_OPEN_TIME: time = time(9, 30, 0)

#: Regular market close — 16:00 ET
MARKET_CLOSE_TIME: time = time(16, 0, 0)

#: Earliest date the engine will accept for calendar lookups.
CALENDAR_START: str = "2000-01-01"

#: Latest date the engine will pre-load in its session cache.
CALENDAR_END: str = "2035-12-31"

# ---------------------------------------------------------------------------
# Alignment-reason labels (use these constants everywhere for consistency)
# ---------------------------------------------------------------------------
REASON_BEFORE_OPEN: str = "before_market_open"
REASON_AFTER_CLOSE: str = "after_market_close"
REASON_WEEKEND: str = "weekend_adjustment"
REASON_HOLIDAY: str = "holiday_adjustment"
REASON_VALID: str = "already_valid"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class TradingCalendarConfig:
    """
    Tunable parameters for :class:`TradingCalendarEngine`.

    Attributes
    ----------
    calendar_name : str
        Exchange calendar identifier understood by ``pandas_market_calendars``.
        Default ``"NYSE"``.
    timezone : str
        IANA timezone string used when a naive timestamp is supplied to the
        engine and no explicit timezone override is given.
        Default ``"America/New_York"``.
    market_open : datetime.time
        Regular session open in *local exchange time*.  Default 09:30.
    market_close : datetime.time
        Regular session close in *local exchange time*.  Default 16:00.
    calendar_start : str
        Inclusive start of the pre-loaded NYSE session window (``YYYY-MM-DD``).
    calendar_end : str
        Inclusive end of the pre-loaded NYSE session window (``YYYY-MM-DD``).
    log_level : int
        Python logging level used by engine-internal log calls.
        Default ``logging.DEBUG``.
    """

    calendar_name: str = NYSE_CALENDAR
    timezone: str = NYSE_TZ
    market_open: time = field(default_factory=lambda: MARKET_OPEN_TIME)
    market_close: time = field(default_factory=lambda: MARKET_CLOSE_TIME)
    calendar_start: str = CALENDAR_START
    calendar_end: str = CALENDAR_END
    log_level: int = logging.DEBUG


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EventAlignmentResult:
    """
    Immutable record describing how one earnings-event timestamp was aligned
    to a valid NYSE trading session.

    Attributes
    ----------
    original_event_datetime : pd.Timestamp
        The raw earnings timestamp supplied by the caller, converted to UTC.
    aligned_event_date : pd.Timestamp
        The aligned trading date (tz-naive, midnight, represents a calendar
        date) that should be used as the event anchor in downstream analysis.
    alignment_reason : str
        One of the ``REASON_*`` module constants explaining *why* the date
        was adjusted (or not).
    was_adjusted : bool
        ``True`` when the original date differed from the aligned date.
    market_session : str
        Human-readable description of the session category, e.g.
        ``"regular"`` or ``"extended_afterhours"``.
    timezone : str
        The IANA timezone string in which the original timestamp was
        interpreted.
    """

    original_event_datetime: pd.Timestamp
    aligned_event_date: pd.Timestamp
    alignment_reason: str
    was_adjusted: bool
    market_session: str
    timezone: str

    def __str__(self) -> str:  # pragma: no cover
        adj_flag = "ADJUSTED" if self.was_adjusted else "unchanged"
        return (
            f"EventAlignmentResult("
            f"original={self.original_event_datetime.isoformat()}, "
            f"aligned={self.aligned_event_date.date()}, "
            f"reason='{self.alignment_reason}', "
            f"session='{self.market_session}', "
            f"flag={adj_flag})"
        )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class TradingCalendarEngine:
    """
    NYSE trading-calendar validator and earnings-event alignment engine.

    The engine pre-loads the NYSE session schedule on first access (lazy
    initialisation) and caches it for the lifetime of the instance, making
    repeated alignment calls cheap.

    Parameters
    ----------
    config : TradingCalendarConfig, optional
        If omitted a default config is used (NYSE, America/New_York).

    Examples
    --------
    >>> engine = TradingCalendarEngine()
    >>> result = engine.align_earnings_event(
    ...     "2024-07-25 16:30:00", "America/New_York"
    ... )
    >>> result.aligned_event_date
    Timestamp('2024-07-26 00:00:00')  # next trading day
    >>> result.alignment_reason
    'after_market_close'
    """

    def __init__(self, config: Optional[TradingCalendarConfig] = None) -> None:
        self.config: TradingCalendarConfig = config or TradingCalendarConfig()
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(self.config.log_level)

        # Lazy-loaded caches — populated by _ensure_calendar_loaded()
        self._calendar: Optional[mcal.MarketCalendar] = None
        self._schedule: Optional[pd.DataFrame] = None         # market_open/close cols in UTC
        self._trading_dates: Optional[pd.DatetimeIndex] = None  # tz-naive date index

        self._log.info(
            "TradingCalendarEngine created | exchange=%s | tz=%s | window=%s → %s",
            self.config.calendar_name,
            self.config.timezone,
            self.config.calendar_start,
            self.config.calendar_end,
        )

    # ------------------------------------------------------------------
    # Session cache
    # ------------------------------------------------------------------

    def _ensure_calendar_loaded(self) -> None:
        """
        Lazily load and cache the NYSE session schedule.

        After the first call this is a no-op (O(1)).  The schedule covers
        ``config.calendar_start`` → ``config.calendar_end``.
        """
        if self._schedule is not None:
            return

        self._log.debug(
            "Loading %s calendar (%s → %s) …",
            self.config.calendar_name,
            self.config.calendar_start,
            self.config.calendar_end,
        )
        self._calendar = mcal.get_calendar(self.config.calendar_name)
        self._schedule = self._calendar.schedule(
            start_date=self.config.calendar_start,
            end_date=self.config.calendar_end,
        )
        # The schedule index is a tz-naive DatetimeIndex of valid trading days.
        self._trading_dates = pd.DatetimeIndex(self._schedule.index)
        self._log.info(
            "Calendar loaded | valid trading days=%d | first=%s | last=%s",
            len(self._trading_dates),
            self._trading_dates[0].date(),
            self._trading_dates[-1].date(),
        )

    @property
    def trading_dates(self) -> pd.DatetimeIndex:
        """Cached tz-naive DatetimeIndex of all valid NYSE trading days."""
        self._ensure_calendar_loaded()
        return self._trading_dates  # type: ignore[return-value]

    @property
    def schedule(self) -> pd.DataFrame:
        """
        Cached NYSE session schedule.

        Columns
        -------
        market_open : datetime64[us, UTC]
        market_close : datetime64[us, UTC]
        Index : tz-naive date of each trading session.
        """
        self._ensure_calendar_loaded()
        return self._schedule  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API — trading day queries
    # ------------------------------------------------------------------

    def is_trading_day(self, dt: DateLike) -> bool:
        """
        Return ``True`` when *dt* falls on a valid NYSE trading session.

        Weekends, federal holidays, and NYSE-observed market closures all
        return ``False``.

        Parameters
        ----------
        dt : DateLike
            Any date-like object (str ``"YYYY-MM-DD"``, ``datetime``,
            ``date``, or ``pd.Timestamp``).  Time-of-day and timezone
            information are ignored; only the calendar date is tested.

        Returns
        -------
        bool

        Examples
        --------
        >>> engine.is_trading_day("2024-01-01")   # New Year's Day
        False
        >>> engine.is_trading_day("2024-01-02")   # First trading day 2024
        True
        >>> engine.is_trading_day("2024-01-06")   # Saturday
        False
        """
        ts = self._to_naive_date_ts(dt)
        result: bool = ts in self.trading_dates
        self._log.debug("is_trading_day(%s) → %s", ts.date(), result)
        return result

    def get_next_trading_day(self, dt: DateLike) -> pd.Timestamp:
        """
        Return the **first** NYSE trading session that is strictly *after* *dt*.

        Parameters
        ----------
        dt : DateLike
            Reference date.  If *dt* is itself a trading day it is still
            skipped — use :meth:`get_nearest_trading_day` if you want to
            stay on the same day when valid.

        Returns
        -------
        pd.Timestamp
            Tz-naive midnight timestamp of the next trading day.

        Raises
        ------
        ValueError
            When no future trading day exists within the loaded calendar
            window (i.e. *dt* is beyond ``config.calendar_end``).

        Examples
        --------
        >>> engine.get_next_trading_day("2024-01-05")   # Friday
        Timestamp('2024-01-08 00:00:00')                # Monday
        >>> engine.get_next_trading_day("2024-01-06")   # Saturday
        Timestamp('2024-01-08 00:00:00')                # Monday
        """
        ts = self._to_naive_date_ts(dt)
        future = self.trading_dates[self.trading_dates > ts]
        if future.empty:
            raise ValueError(
                f"No trading day found after {ts.date()} within calendar "
                f"window (end={self.config.calendar_end})."
            )
        result = future[0]
        self._log.debug("get_next_trading_day(%s) → %s", ts.date(), result.date())
        return result

    def get_previous_trading_day(self, dt: DateLike) -> pd.Timestamp:
        """
        Return the **last** NYSE trading session strictly *before* *dt*.

        Parameters
        ----------
        dt : DateLike
            Reference date.  The date itself is excluded from the search.

        Returns
        -------
        pd.Timestamp
            Tz-naive midnight timestamp of the previous trading day.

        Raises
        ------
        ValueError
            When no past trading day exists within the loaded calendar
            window.

        Examples
        --------
        >>> engine.get_previous_trading_day("2024-01-08")   # Monday
        Timestamp('2024-01-05 00:00:00')                    # Friday
        """
        ts = self._to_naive_date_ts(dt)
        past = self.trading_dates[self.trading_dates < ts]
        if past.empty:
            raise ValueError(
                f"No trading day found before {ts.date()} within calendar "
                f"window (start={self.config.calendar_start})."
            )
        result = past[-1]
        self._log.debug(
            "get_previous_trading_day(%s) → %s", ts.date(), result.date()
        )
        return result

    def get_nearest_trading_day(self, dt: DateLike) -> pd.Timestamp:
        """
        Return the nearest *upcoming* NYSE trading session on or after *dt*.

        If *dt* is already a trading day it is returned unchanged.  If it
        falls on a weekend or holiday the next valid session is returned.

        Parameters
        ----------
        dt : DateLike
            Reference date.

        Returns
        -------
        pd.Timestamp
            Tz-naive midnight timestamp.

        Examples
        --------
        >>> engine.get_nearest_trading_day("2024-01-01")   # New Year's Day
        Timestamp('2024-01-02 00:00:00')
        >>> engine.get_nearest_trading_day("2024-01-02")   # Trading day
        Timestamp('2024-01-02 00:00:00')
        """
        ts = self._to_naive_date_ts(dt)
        if self.is_trading_day(ts):
            self._log.debug("get_nearest_trading_day(%s) → same day", ts.date())
            return ts
        result = self.get_next_trading_day(ts)
        self._log.debug(
            "get_nearest_trading_day(%s) → %s (skipped non-trading)",
            ts.date(),
            result.date(),
        )
        return result

    # ------------------------------------------------------------------
    # Public API — session schedule
    # ------------------------------------------------------------------

    def get_trading_sessions(
        self,
        start: DateLike,
        end: DateLike,
    ) -> pd.DataFrame:
        """
        Return a subset of the NYSE session schedule for a date window.

        Parameters
        ----------
        start : DateLike
            Inclusive start date.
        end : DateLike
            Inclusive end date.

        Returns
        -------
        pd.DataFrame
            Rows indexed by tz-naive trading date; columns are
            ``market_open`` and ``market_close`` (both ``datetime64[us, UTC]``).

        Examples
        --------
        >>> sessions = engine.get_trading_sessions("2024-01-02", "2024-01-05")
        >>> sessions.columns.tolist()
        ['market_open', 'market_close']
        """
        start_ts = self._to_naive_date_ts(start)
        end_ts = self._to_naive_date_ts(end)
        mask = (self.trading_dates >= start_ts) & (self.trading_dates <= end_ts)
        subset = self.schedule.loc[self.trading_dates[mask]]
        self._log.debug(
            "get_trading_sessions(%s → %s) → %d sessions",
            start_ts.date(),
            end_ts.date(),
            len(subset),
        )
        return subset

    # ------------------------------------------------------------------
    # Public API — event alignment (core business logic)
    # ------------------------------------------------------------------

    def align_earnings_event(
        self,
        event_datetime: DateLike,
        event_timezone: Optional[str] = None,
    ) -> EventAlignmentResult:
        """
        Align an earnings-release datetime to the correct NYSE trading date.

        Alignment rules
        ---------------
        1. **Weekend or holiday**: move to the *next* valid NYSE session.
        2. **Before market open** (< 09:30 ET on a trading day): use the
           *same* trading day.
        3. **After market close** (≥ 16:00 ET on a trading day): move to the
           *next* trading day.
        4. **During trading hours** (09:30–16:00 ET): use the same trading
           day (identical to rule 2 outcome; treated as "before_market_open"
           variant for simplicity).

        The method handles:

        * Tz-aware timestamps (any IANA zone, including UTC)
        * Tz-naive timestamps (interpreted as ``event_timezone`` or
          ``config.timezone`` if neither is supplied)
        * String timestamps in common formats (ISO-8601, US-style)
        * DST transitions — all comparisons happen in UTC

        Parameters
        ----------
        event_datetime : DateLike
            The raw earnings-release timestamp.  Can be date-only (in which
            case no time-of-day is known and the event is assumed to have
            occurred *after* market close — conservative assumption).
        event_timezone : str, optional
            IANA timezone name to apply when *event_datetime* is tz-naive.
            Falls back to ``config.timezone`` when omitted.

        Returns
        -------
        EventAlignmentResult
            Frozen dataclass with ``aligned_event_date`` and supporting
            diagnostic fields.

        Examples
        --------
        >>> engine.align_earnings_event("2024-01-25 16:30", "America/New_York")
        # after_market_close → aligned to 2024-01-26

        >>> engine.align_earnings_event("2024-01-26 08:00", "America/New_York")
        # before_market_open → aligned to 2024-01-26 (same day)

        >>> engine.align_earnings_event("2024-01-27")   # Saturday
        # weekend_adjustment → aligned to 2024-01-29 (Monday)
        """
        tz_str = event_timezone or self.config.timezone
        utc_ts, is_date_only = self.normalize_timestamp(event_datetime, tz_str)

        self._log.debug(
            "align_earnings_event | input=%s | tz=%s | utc=%s | date_only=%s",
            event_datetime,
            tz_str,
            utc_ts.isoformat(),
            is_date_only,
        )

        # ── Resolve the calendar-date of the event in ET ──────────────
        et_ts = utc_ts.tz_convert(NYSE_TZ)
        event_date_naive = self._to_naive_date_ts(et_ts)

        # ── Case 1: weekend / holiday ──────────────────────────────────
        if not self.is_trading_day(event_date_naive):
            day_type = "weekend" if et_ts.dayofweek >= 5 else "holiday"
            aligned = self.get_next_trading_day(event_date_naive)
            reason = REASON_WEEKEND if day_type == "weekend" else REASON_HOLIDAY
            self._log.info(
                "Event on %s (%s) → aligned to %s (%s)",
                event_date_naive.date(),
                day_type,
                aligned.date(),
                reason,
            )
            return EventAlignmentResult(
                original_event_datetime=utc_ts,
                aligned_event_date=aligned,
                alignment_reason=reason,
                was_adjusted=(aligned != event_date_naive),
                market_session="non_trading",
                timezone=tz_str,
            )

        # ── Trading day: compare time against session boundaries ───────
        # When only a date was provided (no time component) we conservatively
        # assume the release happened after the close (e.g. after-hours press
        # release on the day — most common real-world pattern).
        if is_date_only:
            aligned = self.get_next_trading_day(event_date_naive)
            self._log.info(
                "Date-only event on %s → conservatively aligned to %s "
                "(after_market_close assumption)",
                event_date_naive.date(),
                aligned.date(),
            )
            return EventAlignmentResult(
                original_event_datetime=utc_ts,
                aligned_event_date=aligned,
                alignment_reason=REASON_AFTER_CLOSE,
                was_adjusted=True,
                market_session="unknown_time",
                timezone=tz_str,
            )

        # Fetch the session's UTC open and close from the cached schedule
        session_open_utc, session_close_utc = self._get_session_boundaries_utc(
            event_date_naive
        )

        # ── Case 2: before market open (pre-market / overnight) ───────
        if utc_ts <= session_open_utc:
            aligned = event_date_naive
            self._log.info(
                "Event at %s ET (pre-market) → aligned to same day %s",
                et_ts.strftime("%H:%M"),
                aligned.date(),
            )
            return EventAlignmentResult(
                original_event_datetime=utc_ts,
                aligned_event_date=aligned,
                alignment_reason=REASON_BEFORE_OPEN,
                was_adjusted=False,
                market_session="pre_market",
                timezone=tz_str,
            )

        # ── Case 3: during regular hours (same day) ────────────────────
        if utc_ts <= session_close_utc:
            aligned = event_date_naive
            self._log.info(
                "Event at %s ET (regular hours) → aligned to same day %s",
                et_ts.strftime("%H:%M"),
                aligned.date(),
            )
            return EventAlignmentResult(
                original_event_datetime=utc_ts,
                aligned_event_date=aligned,
                alignment_reason=REASON_BEFORE_OPEN,   # same rule bucket
                was_adjusted=False,
                market_session="regular",
                timezone=tz_str,
            )

        # ── Case 4: after market close (most common earnings pattern) ──
        aligned = self.get_next_trading_day(event_date_naive)
        self._log.info(
            "Event at %s ET (after-hours) → aligned to next day %s",
            et_ts.strftime("%H:%M"),
            aligned.date(),
        )
        return EventAlignmentResult(
            original_event_datetime=utc_ts,
            aligned_event_date=aligned,
            alignment_reason=REASON_AFTER_CLOSE,
            was_adjusted=True,
            market_session="after_hours",
            timezone=tz_str,
        )

    def validate_event_dates(
        self,
        event_series: pd.Series,
        id_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Validate a batch of already-aligned event dates for pipeline quality.

        Checks performed
        ----------------
        1. No null / NaT event dates.
        2. Every date is a valid NYSE trading session.
        3. No duplicate aligned dates within the same event ID (when
           *id_series* is supplied).

        Parameters
        ----------
        event_series : pd.Series
            Series of aligned event dates (``DateLike`` values).
        id_series : pd.Series, optional
            Parallel series of event identifiers (e.g. ``transcript_id``).
            When supplied, duplicate ``(id, date)`` pairs are flagged.

        Returns
        -------
        pd.DataFrame
            Validation report with columns:

            - ``index``   — original positional index
            - ``date``    — the aligned event date
            - ``is_valid_trading_day`` — bool
            - ``is_null`` — bool
            - ``is_duplicate`` — bool (always ``False`` when no id_series)
            - ``error_summary`` — human-readable issue string or ``""``

        Raises
        ------
        ValueError
            When any critical validation fails (null dates or non-trading
            days). The report DataFrame is attached to the exception as
            ``exception.report``.

        Examples
        --------
        >>> dates = pd.Series(["2024-01-02", "2024-01-01"])   # 2nd valid, 1st not
        >>> report = engine.validate_event_dates(dates)
        ValueError: Validation failed: 1 non-trading day(s) found.
        """
        self._log.info(
            "validate_event_dates | n_events=%d", len(event_series)
        )

        records: list[dict] = []
        for idx, raw_date in event_series.items():
            is_null = pd.isnull(raw_date)
            is_valid = False
            is_dup = False
            errors: list[str] = []

            if is_null:
                errors.append("null/NaT date")
            else:
                ts = self._to_naive_date_ts(raw_date)
                is_valid = self.is_trading_day(ts)
                if not is_valid:
                    errors.append(f"non-trading day: {ts.date()}")

                if id_series is not None:
                    event_id = id_series.iloc[idx] if hasattr(idx, "__index__") else id_series[idx]
                    dup_mask = (
                        (event_series == raw_date) &
                        (id_series == event_id)
                    )
                    if dup_mask.sum() > 1:
                        is_dup = True
                        errors.append(f"duplicate (id={event_id}, date={ts.date()})")

            records.append(
                {
                    "index": idx,
                    "date": raw_date,
                    "is_valid_trading_day": is_valid,
                    "is_null": is_null,
                    "is_duplicate": is_dup,
                    "error_summary": "; ".join(errors) if errors else "",
                }
            )

        report = pd.DataFrame(records)

        null_count = report["is_null"].sum()
        bad_count = (~report["is_valid_trading_day"] & ~report["is_null"]).sum()
        dup_count = report["is_duplicate"].sum()

        issues: list[str] = []
        if null_count:
            issues.append(f"{null_count} null date(s)")
        if bad_count:
            issues.append(f"{bad_count} non-trading day(s) found")
        if dup_count:
            issues.append(f"{dup_count} duplicate event(s) found")

        if issues:
            msg = "Validation failed: " + ", ".join(issues) + "."
            self._log.error(msg)
            exc = ValueError(msg)
            exc.report = report  # type: ignore[attr-defined]
            raise exc

        self._log.info("validate_event_dates passed — all %d dates OK.", len(report))
        return report

    # ------------------------------------------------------------------
    # Public API — timestamp normalisation
    # ------------------------------------------------------------------

    def normalize_timestamp(
        self,
        ts: DateLike,
        assume_timezone: Optional[str] = None,
    ) -> tuple[pd.Timestamp, bool]:
        """
        Convert any date-like input to a tz-aware UTC ``pd.Timestamp``.

        Handles
        -------
        * Tz-aware ``pd.Timestamp`` / ``datetime`` — converted to UTC as-is.
        * Tz-naive ``pd.Timestamp`` / ``datetime`` / ``str`` — localized to
          *assume_timezone* (or ``config.timezone``), then converted to UTC.
        * ``date`` objects (no time component) — treated as midnight local
          time; the second return value is ``True`` to signal that no
          time-of-day was available.
        * UTC timestamps — passed through unchanged.
        * DST-ambiguous local times — resolved using ``ambiguous="NaT"``
          guard then fallback to ``"infer"``.

        Parameters
        ----------
        ts : DateLike
            Raw timestamp or date string.
        assume_timezone : str, optional
            IANA timezone to apply to naive inputs.  Defaults to
            ``config.timezone``.

        Returns
        -------
        (utc_timestamp, is_date_only) : tuple[pd.Timestamp, bool]
            *utc_timestamp* — timezone-aware UTC ``pd.Timestamp``.
            *is_date_only*  — ``True`` when the input had no time component.

        Examples
        --------
        >>> engine.normalize_timestamp("2024-01-25 16:30", "America/New_York")
        (Timestamp('2024-01-25 21:30:00+0000', tz='UTC'), False)

        >>> engine.normalize_timestamp("2024-01-25")
        (Timestamp('2024-01-25 05:00:00+0000', tz='UTC'), True)
        """
        tz_str = assume_timezone or self.config.timezone

        # Detect date-only inputs before converting to Timestamp
        is_date_only = isinstance(ts, date) and not isinstance(ts, datetime)

        # Coerce to pd.Timestamp (preserves tz info if already present)
        try:
            pts = pd.Timestamp(ts)
        except Exception as exc:
            raise ValueError(
                f"Cannot parse '{ts}' as a valid timestamp: {exc}"
            ) from exc

        # String "2024-01-25" parses to midnight with no tz — treat as date-only
        if not is_date_only and isinstance(ts, str):
            stripped = ts.strip()
            # Purely date-formatted string: YYYY-MM-DD or MM/DD/YYYY (no time part)
            is_date_only = (
                len(stripped) == 10
                and ":" not in stripped
            )

        # Apply timezone if naive
        if pts.tzinfo is None:
            try:
                pts = pts.tz_localize(tz_str, ambiguous="raise")
            except Exception:
                # DST ambiguity fallback
                try:
                    pts = pts.tz_localize(tz_str, ambiguous="infer")
                except Exception:
                    pts = pts.tz_localize("UTC")   # last resort

        # Convert to UTC
        utc = pts.tz_convert("UTC")
        return utc, is_date_only

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_naive_date_ts(dt: DateLike) -> pd.Timestamp:
        """
        Coerce *dt* to a tz-naive midnight ``pd.Timestamp``.

        This is the canonical internal date representation used for all
        calendar index comparisons (the schedule index is tz-naive).
        """
        if isinstance(dt, pd.Timestamp):
            pts = dt
        elif isinstance(dt, datetime):
            pts = pd.Timestamp(dt)
        elif isinstance(dt, date):
            pts = pd.Timestamp(dt)
        else:
            pts = pd.Timestamp(dt)

        # Strip tz and normalize to midnight
        if pts.tzinfo is not None:
            pts = pts.tz_localize(None)
        return pts.normalize()

    def _get_session_boundaries_utc(
        self, trading_date_naive: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Return the UTC market-open and market-close for *trading_date_naive*.

        Falls back to computing canonical open/close from
        ``config.market_open`` / ``config.market_close`` when the date is
        not in the schedule (should never happen if called after
        ``is_trading_day`` returns ``True``).

        Returns
        -------
        (session_open_utc, session_close_utc) : tuple[pd.Timestamp, pd.Timestamp]
        """
        try:
            row = self.schedule.loc[trading_date_naive]
            return (
                pd.Timestamp(row["market_open"]),
                pd.Timestamp(row["market_close"]),
            )
        except KeyError:
            # Defensive fallback: construct from config times
            self._log.warning(
                "Session for %s not in schedule — using config defaults.",
                trading_date_naive.date(),
            )
            local_open = pd.Timestamp(
                datetime.combine(trading_date_naive.date(), self.config.market_open),
                tz=self.config.timezone,
            )
            local_close = pd.Timestamp(
                datetime.combine(trading_date_naive.date(), self.config.market_close),
                tz=self.config.timezone,
            )
            return local_open.tz_convert("UTC"), local_close.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Convenience module-level helpers
# ---------------------------------------------------------------------------

def quick_align(
    event_datetime: DateLike,
    event_timezone: str = NYSE_TZ,
) -> pd.Timestamp:
    """
    One-line convenience wrapper: align a single earnings timestamp.

    Creates a default-config engine, aligns, and returns the date.
    For repeated use prefer creating a :class:`TradingCalendarEngine` once
    and calling :meth:`~TradingCalendarEngine.align_earnings_event` directly.

    Parameters
    ----------
    event_datetime : DateLike
        The earnings-release timestamp.
    event_timezone : str
        IANA timezone for naive inputs.

    Returns
    -------
    pd.Timestamp
        Tz-naive midnight aligned trading date.
    """
    engine = TradingCalendarEngine()
    return engine.align_earnings_event(event_datetime, event_timezone).aligned_event_date


# ---------------------------------------------------------------------------
# Self-tests (lightweight, run without pytest)
# ---------------------------------------------------------------------------

def _run_self_tests(engine: TradingCalendarEngine) -> None:
    """
    Deterministic correctness assertions.

    These verify the four canonical edge cases from the Day 5 spec.
    Raise ``AssertionError`` on any failure.
    """
    print("\n" + "═" * 65)
    print("  Self-tests")
    print("═" * 65)

    failures: list[str] = []

    def check(label: str, got: pd.Timestamp, expected_date: str) -> None:
        exp = pd.Timestamp(expected_date)
        ok = got.normalize() == exp
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {label}")
        print(f"          got={got.date()}  expected={exp.date()}")
        if not ok:
            failures.append(f"{label}: got {got.date()}, expected {exp.date()}")

    # ── 1. Friday after market close ──────────────────────────────────
    # 2024-01-05 is a Friday; 16:30 ET is after close → next trading day = Mon Jan 8
    r1 = engine.align_earnings_event("2024-01-05 16:30:00", "America/New_York")
    assert r1.alignment_reason == REASON_AFTER_CLOSE, r1.alignment_reason
    assert r1.was_adjusted is True
    check("Friday 16:30 ET (after close) → Monday", r1.aligned_event_date, "2024-01-08")

    # ── 2. Saturday earnings announcement ─────────────────────────────
    # 2024-01-06 is Saturday → next trading day = Monday Jan 8
    r2 = engine.align_earnings_event("2024-01-06 09:00:00", "America/New_York")
    assert r2.alignment_reason == REASON_WEEKEND, r2.alignment_reason
    assert r2.was_adjusted is True
    check("Saturday 09:00 ET (weekend) → Monday", r2.aligned_event_date, "2024-01-08")

    # ── 3. Monday pre-market ──────────────────────────────────────────
    # 2024-01-08 is Monday; 07:00 ET is before open → same day
    r3 = engine.align_earnings_event("2024-01-08 07:00:00", "America/New_York")
    assert r3.alignment_reason == REASON_BEFORE_OPEN, r3.alignment_reason
    assert r3.was_adjusted is False
    check("Monday 07:00 ET (pre-market) → same day", r3.aligned_event_date, "2024-01-08")

    # ── 4. Holiday adjustment — Christmas 2024 ────────────────────────
    # 2024-12-25 is a market holiday → next trading day = Dec 26
    r4 = engine.align_earnings_event("2024-12-25 18:00:00", "America/New_York")
    assert r4.alignment_reason == REASON_HOLIDAY, r4.alignment_reason
    assert r4.was_adjusted is True
    check("Christmas 2024 (holiday) → Dec 26", r4.aligned_event_date, "2024-12-26")

    # ── 5. New Year's Day 2024 (holiday, Sun → observed Mon) ──────────
    # Jan 1 2024 is observed holiday; Jan 2 is first trading day
    r5 = engine.align_earnings_event("2024-01-01 10:00:00", "America/New_York")
    assert r5.was_adjusted is True
    check("New Year's Day 2024 → Jan 2", r5.aligned_event_date, "2024-01-02")

    # ── 6. UTC timestamp input ────────────────────────────────────────
    # 2024-07-25 20:30 UTC = 16:30 ET (after close) → next day Jul 26
    r6 = engine.align_earnings_event("2024-07-25 20:30:00+00:00", "UTC")
    assert r6.alignment_reason == REASON_AFTER_CLOSE, r6.alignment_reason
    check("UTC 20:30 (= ET 16:30, after close) → Jul 26", r6.aligned_event_date, "2024-07-26")

    # ── 7. DST boundary — Friday before spring-forward ────────────────
    # 2024-03-08 is Friday before the spring-forward weekend (Mar 10).
    # An after-hours release at 16:30 ET on a winter-offset day must still
    # correctly convert to UTC (21:30 UTC, EST = UTC-5) and align forward.
    # The next trading day is Monday 2024-03-11 (first day of summer time).
    r7 = engine.align_earnings_event("2024-03-08 16:30:00", "America/New_York")
    assert r7.alignment_reason == REASON_AFTER_CLOSE, r7.alignment_reason
    check("DST boundary: Fri Mar 8 16:30 ET after close → Mon Mar 11", r7.aligned_event_date, "2024-03-11")

    print("═" * 65)
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
        raise AssertionError(f"{len(failures)} self-test(s) failed.")
    print(f"  All {7} self-tests passed.")
    print("═" * 65)


# ---------------------------------------------------------------------------
# __main__ — demo + self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("═" * 65)
    print("  TradingCalendarEngine — demo run")
    print("═" * 65)

    engine = TradingCalendarEngine()

    # ── Demo 1: after-hours earnings (most common pattern) ────────────
    print("\n[Demo 1] After-hours earnings release")
    print("  Apple typically reports ~4:30 PM ET on a Thursday")
    r = engine.align_earnings_event("2024-08-01 16:30:00", "America/New_York")
    print(f"  {r}")

    # ── Demo 2: pre-market earnings release ───────────────────────────
    print("\n[Demo 2] Pre-market earnings release")
    print("  Some companies report before market open (~7 AM ET)")
    r = engine.align_earnings_event("2024-07-23 07:00:00", "America/New_York")
    print(f"  {r}")

    # ── Demo 3: weekend / Saturday announcement ───────────────────────
    print("\n[Demo 3] Weekend announcement")
    r = engine.align_earnings_event("2024-08-03 10:00:00", "America/New_York")
    print(f"  {r}")

    # ── Demo 4: holiday (Christmas) ───────────────────────────────────
    print("\n[Demo 4] Holiday release (Christmas 2024)")
    r = engine.align_earnings_event("2024-12-25 09:00:00", "America/New_York")
    print(f"  {r}")

    # ── Demo 5: DST boundary — Friday before spring-forward ───────────
    print("\n[Demo 5] DST boundary — Fri Mar 8 2024 after close (EST → UTC-5)")
    print("  After spring-forward (Mar 10), the next session is Mon Mar 11.")
    r = engine.align_earnings_event("2024-03-08 16:30:00", "America/New_York")
    print(f"  {r}")

    # ── Demo 6: is_trading_day checks ────────────────────────────────
    print("\n[Demo 6] is_trading_day checks")
    for d in ["2024-01-01", "2024-01-02", "2024-01-06", "2024-07-04", "2024-11-28"]:
        flag = engine.is_trading_day(d)
        print(f"  {d}: {'trading day' if flag else 'NOT a trading day'}")

    # ── Demo 7: next / prev trading day ──────────────────────────────
    print("\n[Demo 7] get_next / get_previous trading day")
    for d in ["2024-01-05", "2024-01-06", "2024-12-31"]:
        nxt = engine.get_next_trading_day(d)
        prv = engine.get_previous_trading_day(d)
        print(f"  {d}  →  prev={prv.date()}  next={nxt.date()}")

    # ── Demo 8: nearest trading day ───────────────────────────────────
    print("\n[Demo 8] get_nearest_trading_day")
    for d in ["2024-01-01", "2024-01-02", "2024-01-07"]:
        nearest = engine.get_nearest_trading_day(d)
        print(f"  {d} → nearest={nearest.date()}")

    # ── Demo 9: session window ────────────────────────────────────────
    print("\n[Demo 9] get_trading_sessions (Jan 2–5 2024)")
    sessions = engine.get_trading_sessions("2024-01-02", "2024-01-05")
    print(sessions.to_string())

    # ── Demo 10: batch validation ─────────────────────────────────────
    print("\n[Demo 10] validate_event_dates (all valid)")
    good_dates = pd.Series(["2024-01-02", "2024-01-03", "2024-01-04"])
    report = engine.validate_event_dates(good_dates)
    print(report[["date", "is_valid_trading_day", "error_summary"]].to_string(index=False))

    # ── Self-tests ────────────────────────────────────────────────────
    _run_self_tests(engine)

    print("\n✅  Demo complete.")
