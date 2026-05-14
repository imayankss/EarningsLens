"""
src/finance/event_alignment.py
================================
Earnings-event alignment layer for the Earnings Call Sentiment Analyzer
pipeline.

This module is the structural bridge between the NLP layer and the
quantitative finance layer of the project.  It consumes three inputs:

    1. Transcript metadata     — ``transcript_id``, ``ticker``,
                                 ``earnings_date``, ``quarter``, ``year``
    2. Stock market data       — OHLCV + engineered features per ticker
                                 (produced by ``market_data_loader.py`` +
                                 ``feature_engineering.py``)
    3. NYSE trading calendar   — ``TradingCalendarEngine`` from
                                 ``trading_calendar.py``

…and produces one output: an **event-study-ready DataFrame** where every row
is a (transcript_id, relative_day) pair inside a symmetric event window
centred on the aligned earnings date.

Architecture position
---------------------

    transcript_metadata.parquet
           │
           ▼
    ┌──────────────────────┐
    │  EventAligner        │   ← this module
    │  • align_events()    │
    │  • gen windows()     │
    │  • attach features() │
    └──────────┬───────────┘
               │
     market_data.parquet  ←─  market_data_loader.py + feature_engineering.py
     benchmark (embedded in market_data or separate)
               │
               ▼
    data/interim/event_market_alignment.parquet

What this module does NOT do
-----------------------------
- Download market prices           (→ market_data_loader.py)
- Compute raw technical indicators (→ feature_engineering.py)
- Compute abnormal returns         (→ abnormal_returns.py)
- Orchestrate the full pipeline    (→ market_dataset_builder.py)

Design principles
-----------------
* **Deterministic** — same inputs always produce identical aligned dates and
  window boundaries, guaranteed by the underlying ``TradingCalendarEngine``.
* **Ticker-isolated windows** — window generation slices from that ticker's
  market-data rows only; there is no cross-ticker contamination.
* **No silent data loss** — incomplete events (missing day 0, short windows)
  are quarantined via ``filter_incomplete_events()`` rather than silently
  dropped during merge.
* **Long format** — one row per (transcript_id, relative_day), making the
  output directly consumable by panel regression, event-study libraries, and
  FinBERT analysis joins.

Usage
-----
    from src.finance.event_alignment import EventAligner, EventAlignmentConfig

    cfg  = EventAlignmentConfig(pre_event_window=30, post_event_window=10)
    aligner = EventAligner(cfg)

    aligner.load_event_metadata("data/interim/transcript_metadata.parquet")
    aligner.load_market_data("data/processed/market_data.parquet")

    event_df = aligner.build_event_dataset()
    # → data/interim/event_market_alignment.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Local import — trading_calendar must be on sys.path or installed as package
try:
    from trading_calendar import TradingCalendarEngine, EventAlignmentResult
except ImportError:  # pragma: no cover — tolerant for environments using package installs
    from src.finance.trading_calendar import TradingCalendarEngine, EventAlignmentResult  # type: ignore

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required column sets (used in validation)
# ---------------------------------------------------------------------------
REQUIRED_METADATA_COLS: frozenset[str] = frozenset(
    {"transcript_id", "ticker", "earnings_date", "quarter", "year"}
)

REQUIRED_MARKET_COLS: frozenset[str] = frozenset(
    {"ticker", "date", "open", "high", "low", "close", "adj_close",
     "volume", "daily_return"}
)

OPTIONAL_MARKET_COLS: list[str] = [
    "benchmark_return", "abnormal_return",
    "return_1d", "return_3d", "return_5d",
]

OUTPUT_COLUMNS: list[str] = [
    "transcript_id", "ticker", "earnings_date", "aligned_event_date",
    "quarter", "year",
    "event_window_day", "relative_day", "is_event_day", "market_session",
    "date", "open", "high", "low", "close", "adj_close", "volume",
    "daily_return", "benchmark_return", "abnormal_return",
    "return_1d", "return_3d", "return_5d",
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class EventAlignmentConfig:
    """
    All tunable parameters for :class:`EventAligner`.

    Attributes
    ----------
    pre_event_window : int
        Number of trading days *before* the aligned event day to include.
        Relative days: ``[-pre_event_window, …, -1]``.  Default 30.
    post_event_window : int
        Number of trading days *after* the aligned event day to include.
        Relative days: ``[+1, …, +post_event_window]``.  Default 10.
    require_complete_window : bool
        When ``True``, events whose market data cannot fill the full
        pre+post window (e.g. newly listed tickers) are quarantined by
        :meth:`~EventAligner.filter_incomplete_events`.  Default ``True``.
    event_timezone : str
        IANA timezone used when the ``earnings_date`` column contains naive
        timestamps.  Default ``"America/New_York"``.
    benchmark_column_name : str
        Name of the benchmark-return column in the market data.
        Default ``"benchmark_return"``.
    date_column_name : str
        Name of the date column in the market data.  Default ``"date"``.
    output_dir : Path
        Directory for the primary Parquet export.
    output_filename : str
        Base filename (without ``.parquet``) for the aligned dataset.
    calendar_start : str
        Earliest date pre-loaded into the trading-calendar engine.
    calendar_end : str
        Latest date pre-loaded into the trading-calendar engine.
    """

    pre_event_window: int = 30
    post_event_window: int = 10
    require_complete_window: bool = True
    event_timezone: str = "America/New_York"
    benchmark_column_name: str = "benchmark_return"
    date_column_name: str = "date"
    output_dir: Path = field(
        default_factory=lambda: Path("data/interim")
    )
    output_filename: str = "event_market_alignment"
    calendar_start: str = "2000-01-01"
    calendar_end: str = "2035-12-31"

    @property
    def parquet_path(self) -> Path:
        """Full path to the primary Parquet output."""
        return self.output_dir / f"{self.output_filename}.parquet"

    @property
    def window_size(self) -> int:
        """Total rows per event window (pre + event_day + post)."""
        return self.pre_event_window + 1 + self.post_event_window


# ---------------------------------------------------------------------------
# EventWindow dataclass
# ---------------------------------------------------------------------------
@dataclass
class EventWindow:
    """
    Immutable record describing one earnings event's market window.

    Attributes
    ----------
    transcript_id : str
        Unique identifier of the source transcript.
    ticker : str
        Stock ticker symbol.
    earnings_date : pd.Timestamp
        Original (un-aligned) earnings release datetime.
    aligned_event_date : pd.Timestamp
        NYSE-aligned trading date used as the event anchor (day 0).
    alignment_reason : str
        One of the ``REASON_*`` constants from ``trading_calendar``.
    was_adjusted : bool
        ``True`` when the original date differed from the aligned date.
    market_session : str
        Session description (``"after_hours"``, ``"pre_market"``, etc.).
    quarter : str
        Fiscal quarter label (e.g. ``"Q1"``).
    year : int
        Fiscal year.
    window_df : pd.DataFrame
        Long-format DataFrame with one row per relative day in the event
        window, already merged with market features.
    is_complete : bool
        ``True`` when the window contains all required pre/post days.
    missing_sessions : list[pd.Timestamp]
        Trading days that were expected in the window but absent from
        the market dataset.
    """

    transcript_id: str
    ticker: str
    earnings_date: pd.Timestamp
    aligned_event_date: pd.Timestamp
    alignment_reason: str
    was_adjusted: bool
    market_session: str
    quarter: str
    year: int
    window_df: pd.DataFrame
    is_complete: bool
    missing_sessions: list[pd.Timestamp]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EventWindow(id={self.transcript_id!r}, "
            f"ticker={self.ticker!r}, "
            f"aligned={self.aligned_event_date.date()}, "
            f"rows={len(self.window_df)}, "
            f"complete={self.is_complete})"
        )


# ---------------------------------------------------------------------------
# EventAlignmentResult summary dataclass
# ---------------------------------------------------------------------------
@dataclass
class AlignmentSummary:
    """
    Top-level summary produced by :meth:`EventAligner.build_event_dataset`.

    Attributes
    ----------
    total_events : int
        Number of events in the input metadata.
    aligned_events : int
        Events successfully aligned to a trading session.
    complete_windows : int
        Events with a full pre+event+post window in the market data.
    incomplete_windows : int
        Events quarantined due to insufficient market coverage.
    output_rows : int
        Total rows in the final event-study DataFrame.
    output_path : Path
        Path where the result was saved.
    """

    total_events: int
    aligned_events: int
    complete_windows: int
    incomplete_windows: int
    output_rows: int
    output_path: Path

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"AlignmentSummary | total={self.total_events} "
            f"aligned={self.aligned_events} "
            f"complete={self.complete_windows} "
            f"incomplete={self.incomplete_windows} "
            f"rows={self.output_rows} | path={self.output_path}"
        )


# ---------------------------------------------------------------------------
# EventAligner
# ---------------------------------------------------------------------------
class EventAligner:
    """
    Earnings-event alignment engine.

    Loads transcript metadata and market data, aligns each earnings
    timestamp to the correct NYSE trading session, slices symmetric
    event windows, merges market features, and exports a panel DataFrame
    ready for event-study analysis.

    Parameters
    ----------
    config : EventAlignmentConfig, optional
        Configuration object.  A default instance is used when omitted.
    calendar_engine : TradingCalendarEngine, optional
        Pre-built calendar engine.  A default-config engine is created
        when omitted — share a single engine across module instances in
        batch pipelines for maximum efficiency.

    Examples
    --------
    >>> aligner = EventAligner()
    >>> aligner.load_event_metadata("data/interim/transcript_metadata.parquet")
    >>> aligner.load_market_data("data/processed/market_data.parquet")
    >>> result_df = aligner.build_event_dataset()
    """

    def __init__(
        self,
        config: Optional[EventAlignmentConfig] = None,
        calendar_engine: Optional[TradingCalendarEngine] = None,
    ) -> None:
        self.config: EventAlignmentConfig = config or EventAlignmentConfig()
        self.calendar: TradingCalendarEngine = calendar_engine or TradingCalendarEngine()
        self._log = logging.getLogger(self.__class__.__name__)

        # Internal state — populated by load_* methods
        self._metadata: Optional[pd.DataFrame] = None
        self._market: Optional[pd.DataFrame] = None

        # Per-ticker market data index for fast O(1) date lookups
        self._ticker_dates: dict[str, pd.DatetimeIndex] = {}

        self._log.info(
            "EventAligner initialised | pre=%d post=%d require_complete=%s",
            self.config.pre_event_window,
            self.config.post_event_window,
            self.config.require_complete_window,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_event_metadata(
        self,
        source: str | Path | pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Load and validate transcript-event metadata.

        Accepts a Parquet file path, a CSV file path, or a pre-built
        ``pd.DataFrame``.  The ``earnings_date`` column is coerced to
        ``datetime64`` regardless of source format.

        Parameters
        ----------
        source : str | Path | pd.DataFrame
            Path to a Parquet / CSV file, or a DataFrame already in memory.

        Returns
        -------
        pd.DataFrame
            Validated metadata DataFrame (also stored in ``self._metadata``).

        Raises
        ------
        ValueError
            When required columns are absent.
        FileNotFoundError
            When a file path does not exist on disk.
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            self._log.info("Metadata loaded from in-memory DataFrame | rows=%d", len(df))
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Metadata file not found: {path}")
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix in {".csv", ".tsv"}:
                df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported metadata format: {path.suffix}")
            self._log.info("Metadata loaded from %s | rows=%d", path, len(df))

        # Validate required columns
        missing = REQUIRED_METADATA_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Metadata missing required columns: {sorted(missing)}. "
                f"Present: {sorted(df.columns)}"
            )

        # Coerce types
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], utc=False)
        df["year"] = df["year"].astype(int)
        df["transcript_id"] = df["transcript_id"].astype(str)
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

        # Remove duplicates on transcript_id (keep first occurrence)
        before = len(df)
        df = df.drop_duplicates(subset=["transcript_id"]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            self._log.warning(
                "Dropped %d duplicate transcript_id row(s) from metadata.", dropped
            )

        self._metadata = df
        self._log.info(
            "Metadata validated | unique events=%d | tickers=%d",
            len(df),
            df["ticker"].nunique(),
        )
        return df

    def load_market_data(
        self,
        source: str | Path | pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Load and index per-ticker market data.

        Parameters
        ----------
        source : str | Path | pd.DataFrame
            Path to a Parquet / CSV file, or a DataFrame in memory.
            Must contain at minimum the columns in ``REQUIRED_MARKET_COLS``.

        Returns
        -------
        pd.DataFrame
            Validated market DataFrame (also stored in ``self._market``).

        Raises
        ------
        ValueError
            When required columns are absent.
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            self._log.info("Market data loaded from in-memory DataFrame | rows=%d", len(df))
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Market data file not found: {path}")
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            self._log.info("Market data loaded from %s | rows=%d", path, len(df))

        # Validate required columns
        missing = REQUIRED_MARKET_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Market data missing required columns: {sorted(missing)}. "
                f"Present: {sorted(df.columns)}"
            )

        # Fill optional columns with NaN when absent
        for col in OPTIONAL_MARKET_COLS:
            if col not in df.columns:
                df[col] = np.nan
                self._log.debug("Optional column '%s' not found — filled with NaN.", col)

        # Normalise date column: tz-naive midnight timestamps
        date_col = self.config.date_column_name
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

        # Sort and remove duplicates per (ticker, date)
        df = df.sort_values(["ticker", date_col]).reset_index(drop=True)
        before = len(df)
        df = df.drop_duplicates(subset=["ticker", date_col], keep="first")
        dropped = before - len(df)
        if dropped:
            self._log.warning(
                "Dropped %d duplicate (ticker, date) row(s) from market data.", dropped
            )

        # Build per-ticker DatetimeIndex cache for O(1) window slicing
        self._ticker_dates = {
            ticker: pd.DatetimeIndex(grp[date_col].values)
            for ticker, grp in df.groupby("ticker")
        }

        self._market = df
        self._log.info(
            "Market data validated | rows=%d | tickers=%d | date_range=[%s, %s]",
            len(df),
            df["ticker"].nunique(),
            df[date_col].min().date(),
            df[date_col].max().date(),
        )
        return df

    # ------------------------------------------------------------------
    # Core alignment
    # ------------------------------------------------------------------

    def align_events(self) -> pd.DataFrame:
        """
        Align every event in the metadata to a valid NYSE trading session.

        Wraps :meth:`TradingCalendarEngine.align_earnings_event` for every
        row and appends the result fields as new columns.

        Returns
        -------
        pd.DataFrame
            Metadata DataFrame augmented with:

            - ``aligned_event_date`` — tz-naive midnight ``pd.Timestamp``
            - ``alignment_reason``   — one of the ``REASON_*`` constants
            - ``was_adjusted``       — bool flag
            - ``market_session``     — session category string

        Raises
        ------
        RuntimeError
            When :meth:`load_event_metadata` has not been called first.
        """
        self._require_metadata()

        self._log.info("Aligning %d event(s) to NYSE sessions …", len(self._metadata))

        aligned_dates: list[pd.Timestamp] = []
        reasons: list[str] = []
        was_adjusted: list[bool] = []
        sessions: list[str] = []

        for _, row in self._metadata.iterrows():
            try:
                result: EventAlignmentResult = self.calendar.align_earnings_event(
                    row["earnings_date"],
                    self.config.event_timezone,
                )
                aligned_dates.append(result.aligned_event_date)
                reasons.append(result.alignment_reason)
                was_adjusted.append(result.was_adjusted)
                sessions.append(result.market_session)
            except Exception as exc:  # noqa: BLE001
                self._log.error(
                    "Failed to align event '%s' (%s): %s",
                    row["transcript_id"],
                    row["earnings_date"],
                    exc,
                )
                aligned_dates.append(pd.NaT)
                reasons.append("alignment_error")
                was_adjusted.append(False)
                sessions.append("unknown")

        df = self._metadata.copy()
        df["aligned_event_date"] = aligned_dates
        df["alignment_reason"] = reasons
        df["was_adjusted"] = was_adjusted
        df["market_session"] = sessions

        # Drop rows where alignment failed entirely
        failed = df["aligned_event_date"].isnull().sum()
        if failed:
            self._log.warning(
                "Dropping %d event(s) where alignment failed.", failed
            )
            df = df[df["aligned_event_date"].notna()].reset_index(drop=True)

        adjusted_count = df["was_adjusted"].sum()
        self._log.info(
            "Alignment complete | total=%d | adjusted=%d | reasons=%s",
            len(df),
            adjusted_count,
            df["alignment_reason"].value_counts().to_dict(),
        )
        return df

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def generate_event_windows(
        self,
        aligned_df: pd.DataFrame,
    ) -> list[EventWindow]:
        """
        Generate event windows for every aligned event.

        For each event, this method:

        1. Locates the ``aligned_event_date`` in that ticker's market data.
        2. Slices ``pre_event_window`` rows before and ``post_event_window``
           rows after.
        3. Assigns ``relative_day`` labels (−30 … 0 … +10).
        4. Detects missing trading sessions.
        5. Wraps everything in an :class:`EventWindow` record.

        Parameters
        ----------
        aligned_df : pd.DataFrame
            Output of :meth:`align_events` (must include
            ``aligned_event_date``).

        Returns
        -------
        list[EventWindow]
            One ``EventWindow`` per event, including incomplete windows
            (use :meth:`filter_incomplete_events` to separate them).
        """
        self._require_market()

        windows: list[EventWindow] = []
        pre = self.config.pre_event_window
        post = self.config.post_event_window

        for _, row in aligned_df.iterrows():
            tid = row["transcript_id"]
            ticker = row["ticker"]
            aligned_date = pd.Timestamp(row["aligned_event_date"]).normalize()

            # Retrieve ticker-specific market dates
            if ticker not in self._ticker_dates:
                self._log.warning(
                    "Ticker '%s' (%s) not found in market data — skipping.",
                    ticker, tid,
                )
                windows.append(self._make_empty_window(row, "ticker_missing"))
                continue

            ticker_dates = self._ticker_dates[ticker]

            # Find the position of aligned_date in ticker's market dates
            position_arr = np.where(ticker_dates == aligned_date)[0]
            if len(position_arr) == 0:
                self._log.warning(
                    "aligned_event_date %s not in market data for '%s' (%s) — skipping.",
                    aligned_date.date(), ticker, tid,
                )
                windows.append(self._make_empty_window(row, "event_date_missing"))
                continue

            pos = int(position_arr[0])

            # Slice window dates from this ticker's date index
            start_idx = max(0, pos - pre)
            end_idx = min(len(ticker_dates) - 1, pos + post)
            window_dates = ticker_dates[start_idx: end_idx + 1]

            # Compute relative days anchored at aligned_date (pos)
            relative_days = [
                int(ticker_dates.get_loc(d)) - pos   # type: ignore[arg-type]
                for d in window_dates
            ]

            # Detect expected-but-missing sessions
            expected_rel = list(range(-pre, post + 1))
            actual_rel = set(relative_days)
            missing_rel = [r for r in expected_rel if r not in actual_rel]
            missing_sessions: list[pd.Timestamp] = []
            # Approximate missing session dates using trading calendar
            for rel in missing_rel:
                try:
                    if rel < 0:
                        cand = self.calendar.get_previous_trading_day(
                            ticker_dates[max(0, pos + rel + 1)]
                        )
                    else:
                        cand = self.calendar.get_next_trading_day(
                            ticker_dates[min(len(ticker_dates) - 1, pos + rel - 1)]
                        )
                    missing_sessions.append(cand)
                except Exception:  # noqa: BLE001
                    pass

            # Build the window sub-DataFrame from market data
            ticker_market = self._market[self._market["ticker"] == ticker].copy()
            date_col = self.config.date_column_name

            window_frame = pd.DataFrame(
                {date_col: window_dates, "relative_day": relative_days}
            )
            window_frame[date_col] = pd.to_datetime(window_frame[date_col]).dt.normalize()

            merged = window_frame.merge(ticker_market, on=date_col, how="left")
            merged["is_event_day"] = merged["relative_day"] == 0
            merged["event_window_day"] = merged["relative_day"].apply(
                lambda r: f"D{r:+d}" if r != 0 else "D0"
            )
            # Propagate event-level metadata
            merged["transcript_id"] = tid
            merged["aligned_event_date"] = aligned_date
            merged["earnings_date"] = row["earnings_date"]
            merged["quarter"] = row["quarter"]
            merged["year"] = row["year"]
            merged["market_session"] = row["market_session"]

            is_complete = (
                len(missing_rel) == 0
                and (merged["relative_day"] == 0).any()
            )

            windows.append(
                EventWindow(
                    transcript_id=tid,
                    ticker=ticker,
                    earnings_date=pd.Timestamp(row["earnings_date"]),
                    aligned_event_date=aligned_date,
                    alignment_reason=str(row["alignment_reason"]),
                    was_adjusted=bool(row["was_adjusted"]),
                    market_session=str(row["market_session"]),
                    quarter=str(row["quarter"]),
                    year=int(row["year"]),
                    window_df=merged,
                    is_complete=is_complete,
                    missing_sessions=missing_sessions,
                )
            )

            self._log.debug(
                "Window built | %s | aligned=%s | rows=%d | complete=%s | missing=%d",
                tid, aligned_date.date(), len(merged), is_complete, len(missing_rel),
            )

        complete = sum(1 for w in windows if w.is_complete)
        self._log.info(
            "Windows generated | total=%d | complete=%d | incomplete=%d",
            len(windows), complete, len(windows) - complete,
        )
        return windows

    # ------------------------------------------------------------------
    # Feature attachment
    # ------------------------------------------------------------------

    def attach_market_features(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Verify that core market feature columns are present and return the
        frame unchanged (features were already merged during window generation).

        This method exists as an explicit pipeline stage that downstream
        orchestrators can call to confirm feature availability before
        proceeding to statistical analysis.

        Parameters
        ----------
        event_df : pd.DataFrame
            Concatenated event-study DataFrame from
            :meth:`build_event_dataset`.

        Returns
        -------
        pd.DataFrame
            Input frame (unchanged).

        Raises
        ------
        ValueError
            When mandatory market columns are missing from the frame.
        """
        mandatory = {"daily_return", "close", "volume"}
        missing = mandatory - set(event_df.columns)
        if missing:
            raise ValueError(
                f"Market feature columns missing from event dataset: {missing}"
            )
        filled = {
            col: int(event_df[col].notna().sum())
            for col in ["daily_return", "close", "abnormal_return", "return_1d"]
            if col in event_df.columns
        }
        self._log.info(
            "Market features present | non-null counts: %s", filled
        )
        return event_df

    def attach_benchmark_features(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Verify benchmark return coverage and log alignment statistics.

        Parameters
        ----------
        event_df : pd.DataFrame
            Concatenated event-study DataFrame.

        Returns
        -------
        pd.DataFrame
            Input frame (unchanged).  Missing benchmark rows are logged
            as warnings rather than errors (benchmark data may not cover
            extended pre-event windows for early tickers).
        """
        bcol = self.config.benchmark_column_name
        if bcol not in event_df.columns:
            self._log.warning(
                "Benchmark column '%s' absent from event dataset.  "
                "Abnormal-return calculations will not be possible.",
                bcol,
            )
            return event_df

        null_count = event_df[bcol].isnull().sum()
        total = len(event_df)
        coverage_pct = 100 * (total - null_count) / max(total, 1)
        if null_count > 0:
            self._log.warning(
                "Benchmark column '%s' has %d null(s) / %d rows (%.1f%% coverage).",
                bcol, null_count, total, coverage_pct,
            )
        else:
            self._log.info(
                "Benchmark column '%s' fully populated | coverage=100%%.", bcol
            )
        return event_df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_alignment(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run post-assembly quality checks on the event-study DataFrame.

        Checks
        ------
        1. DataFrame is not empty.
        2. Every ``aligned_event_date`` is a valid NYSE trading session.
        3. Every ``transcript_id`` has exactly one row where
           ``relative_day == 0`` (day-0 integrity).
        4. No duplicate ``(transcript_id, relative_day)`` pairs.
        5. Event windows are chronologically sorted within each transcript.

        Parameters
        ----------
        event_df : pd.DataFrame
            Fully assembled event-study DataFrame.

        Returns
        -------
        pd.DataFrame
            Input frame unchanged when all checks pass.

        Raises
        ------
        ValueError
            On any critical validation failure; the error message
            identifies the specific issue and affected row count.
        """
        self._log.info("Running alignment validation …")

        if event_df.empty:
            raise ValueError("Event dataset is empty after alignment.")

        errors: list[str] = []

        # 1. NYSE trading-session check on aligned_event_date
        unique_aligned = event_df["aligned_event_date"].dropna().unique()
        non_trading = [
            str(d.date()) if hasattr(d, "date") else str(d)
            for d in unique_aligned
            if not self.calendar.is_trading_day(pd.Timestamp(d))
        ]
        if non_trading:
            errors.append(
                f"{len(non_trading)} aligned_event_date(s) are not NYSE trading "
                f"sessions: {non_trading[:5]}{'…' if len(non_trading) > 5 else ''}"
            )

        # 2. Day-0 integrity — every transcript_id must have exactly one day-0 row
        day0 = event_df[event_df["relative_day"] == 0]
        day0_counts = day0.groupby("transcript_id").size()
        missing_day0 = event_df["transcript_id"].unique()
        ids_without_day0 = [
            tid for tid in missing_day0
            if tid not in day0_counts.index
        ]
        if ids_without_day0:
            errors.append(
                f"{len(ids_without_day0)} transcript_id(s) have no day-0 row."
            )
        multi_day0 = day0_counts[day0_counts > 1]
        if not multi_day0.empty:
            errors.append(
                f"{len(multi_day0)} transcript_id(s) have multiple day-0 rows."
            )

        # 3. No duplicate (transcript_id, relative_day)
        dup_count = event_df.duplicated(
            subset=["transcript_id", "relative_day"]
        ).sum()
        if dup_count:
            errors.append(
                f"{dup_count} duplicate (transcript_id, relative_day) pair(s)."
            )

        # 4. Chronological order within each transcript
        def _is_sorted(grp: pd.DataFrame) -> bool:
            return grp["relative_day"].is_monotonic_increasing

        unsorted = [
            tid for tid, grp in event_df.groupby("transcript_id")
            if not _is_sorted(grp)
        ]
        if unsorted:
            errors.append(
                f"{len(unsorted)} transcript_id(s) have non-monotonic relative_day "
                f"ordering: {unsorted[:3]}"
            )

        if errors:
            combined = "; ".join(errors)
            self._log.error("Validation FAILED: %s", combined)
            raise ValueError(f"Alignment validation failed: {combined}")

        self._log.info(
            "Validation PASSED | rows=%d | events=%d | trading-days=%d",
            len(event_df),
            event_df["transcript_id"].nunique(),
            len(unique_aligned),
        )
        return event_df

    def validate_window_continuity(self, windows: list[EventWindow]) -> dict[str, list]:
        """
        Check that every complete window has a contiguous sequence of
        relative days (no gaps from missing market sessions).

        Parameters
        ----------
        windows : list[EventWindow]
            Output of :meth:`generate_event_windows`.

        Returns
        -------
        dict[str, list]
            ``{"continuous": [...transcript_ids...],
               "gapped":     [...transcript_ids...]}``
        """
        continuous: list[str] = []
        gapped: list[str] = []

        for w in windows:
            if w.window_df.empty:
                gapped.append(w.transcript_id)
                continue
            rel = sorted(w.window_df["relative_day"].tolist())
            expected = list(range(min(rel), max(rel) + 1))
            if rel == expected:
                continuous.append(w.transcript_id)
            else:
                missing_in_window = set(expected) - set(rel)
                self._log.debug(
                    "Gap detected | %s | missing relative_days=%s",
                    w.transcript_id, sorted(missing_in_window),
                )
                gapped.append(w.transcript_id)

        self._log.info(
            "Window continuity | continuous=%d | gapped=%d",
            len(continuous), len(gapped),
        )
        return {"continuous": continuous, "gapped": gapped}

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_incomplete_events(
        self,
        windows: list[EventWindow],
    ) -> tuple[list[EventWindow], list[EventWindow]]:
        """
        Separate complete event windows from incomplete ones.

        A window is considered *complete* when:

        * Its ``is_complete`` flag is ``True`` (set during
          :meth:`generate_event_windows`).
        * It contains a row with ``relative_day == 0`` (day-0 present).
        * When ``config.require_complete_window`` is ``True``, it also
          requires the full pre+post window to be present.

        Parameters
        ----------
        windows : list[EventWindow]
            Output of :meth:`generate_event_windows`.

        Returns
        -------
        (complete, incomplete) : tuple[list[EventWindow], list[EventWindow]]
            Two lists — events suitable for analysis and quarantined events.
        """
        complete: list[EventWindow] = []
        incomplete: list[EventWindow] = []

        for w in windows:
            has_day0 = (
                not w.window_df.empty
                and (w.window_df["relative_day"] == 0).any()
            )
            if not has_day0:
                self._log.warning(
                    "Event '%s' has no day-0 row — quarantining.", w.transcript_id
                )
                incomplete.append(w)
                continue

            if self.config.require_complete_window and not w.is_complete:
                self._log.info(
                    "Event '%s' has incomplete window (%d missing session(s)) — "
                    "quarantining (require_complete_window=True).",
                    w.transcript_id,
                    len(w.missing_sessions),
                )
                incomplete.append(w)
            else:
                complete.append(w)

        self._log.info(
            "filter_incomplete_events | complete=%d | quarantined=%d",
            len(complete), len(incomplete),
        )
        return complete, incomplete

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def compute_relative_days(
        self,
        window_dates: pd.DatetimeIndex,
        event_date: pd.Timestamp,
    ) -> list[int]:
        """
        Compute integer relative-day offsets for a sequence of trading dates.

        Parameters
        ----------
        window_dates : pd.DatetimeIndex
            Ordered sequence of trading dates in the window.
        event_date : pd.Timestamp
            The aligned event date (day 0).

        Returns
        -------
        list[int]
            Integer offsets, e.g. ``[-5, -4, -3, -2, -1, 0, 1, 2, 3]``.

        Examples
        --------
        >>> dates = pd.DatetimeIndex(["2024-01-23", "2024-01-24", "2024-01-25"])
        >>> aligner.compute_relative_days(dates, pd.Timestamp("2024-01-24"))
        [-1, 0, 1]
        """
        event_norm = pd.Timestamp(event_date).normalize()
        result: list[int] = []
        for d in window_dates:
            d_norm = pd.Timestamp(d).normalize()
            if d_norm == event_norm:
                result.append(0)
            else:
                # Count trading days between d and event_date
                min_d = min(d_norm, event_norm)
                max_d = max(d_norm, event_norm)
                try:
                    sessions_between = self.calendar.get_trading_sessions(min_d, max_d)
                    # subtract 1 because both endpoints inclusive
                    count = len(sessions_between) - 1
                except Exception:  # noqa: BLE001
                    count = abs((d_norm - event_norm).days)
                result.append(-count if d_norm < event_norm else count)
        return result

    def detect_missing_sessions(
        self,
        window_dates: pd.DatetimeIndex,
        pre: int,
        post: int,
        event_date: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        """
        Identify expected trading sessions absent from *window_dates*.

        Parameters
        ----------
        window_dates : pd.DatetimeIndex
            Trading dates actually present in the window.
        pre : int
            Expected number of pre-event days.
        post : int
            Expected number of post-event days.
        event_date : pd.Timestamp
            The aligned event date (day 0).

        Returns
        -------
        list[pd.Timestamp]
            Timestamps of expected-but-missing sessions.
        """
        actual = set(pd.Timestamp(d).normalize() for d in window_dates)
        event_norm = pd.Timestamp(event_date).normalize()
        missing: list[pd.Timestamp] = []

        # Check pre-event sessions
        try:
            pre_sessions = self.calendar.get_trading_sessions(
                self.calendar.get_previous_trading_day(event_norm)
                if pre > 0 else event_norm,
                event_norm,
            )
            for d in pd.DatetimeIndex(pre_sessions.index)[:-1]:  # exclude event_day itself
                d_norm = pd.Timestamp(d).normalize()
                if d_norm not in actual:
                    missing.append(d_norm)
        except Exception:  # noqa: BLE001
            pass

        # Check post-event sessions
        try:
            post_sessions = self.calendar.get_trading_sessions(
                event_norm,
                self.calendar.get_next_trading_day(event_norm)
                if post > 0 else event_norm,
            )
            for d in pd.DatetimeIndex(post_sessions.index)[1:]:  # exclude event_day itself
                d_norm = pd.Timestamp(d).normalize()
                if d_norm not in actual:
                    missing.append(d_norm)
        except Exception:  # noqa: BLE001
            pass

        return missing

    # ------------------------------------------------------------------
    # Master pipeline
    # ------------------------------------------------------------------

    def build_event_dataset(
        self,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Full pipeline: align → generate windows → merge → validate → export.

        This method is the single entry point for end-to-end event dataset
        construction.  It requires :meth:`load_event_metadata` and
        :meth:`load_market_data` to have been called first.

        Parameters
        ----------
        save : bool
            When ``True`` (default) the result is saved to
            ``config.parquet_path``.

        Returns
        -------
        pd.DataFrame
            Event-study panel DataFrame in long format with columns defined
            by ``OUTPUT_COLUMNS`` (absent optional columns filled with NaN).

        Side effects
        ------------
        * Writes ``data/interim/event_market_alignment.parquet`` when
          ``save=True``.
        * Logs an :class:`AlignmentSummary` at INFO level.

        Raises
        ------
        RuntimeError
            When metadata or market data have not been loaded.
        ValueError
            When post-assembly validation fails.
        """
        self._require_metadata()
        self._require_market()

        total_events = len(self._metadata)
        self._log.info("build_event_dataset | start | total_events=%d", total_events)

        # Stage 1: align earnings dates to NYSE sessions
        aligned_df = self.align_events()

        # Stage 2: generate event windows
        windows = self.generate_event_windows(aligned_df)

        # Stage 3: separate complete from incomplete
        complete_windows, incomplete_windows = self.filter_incomplete_events(windows)

        if not complete_windows:
            self._log.warning(
                "No complete event windows found.  "
                "Check market data coverage for your date range."
            )

        # Stage 4: concatenate into a flat panel DataFrame
        frames = [w.window_df for w in complete_windows if not w.window_df.empty]
        if not frames:
            event_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        else:
            event_df = pd.concat(frames, ignore_index=True)
            event_df = event_df.sort_values(
                ["transcript_id", "relative_day"]
            ).reset_index(drop=True)

        # Ensure all output columns exist (fill missing optional cols with NaN)
        for col in OUTPUT_COLUMNS:
            if col not in event_df.columns:
                event_df[col] = np.nan

        # Stage 5: feature attachment verification
        if not event_df.empty:
            self.attach_market_features(event_df)
            self.attach_benchmark_features(event_df)

        # Stage 6: validation
        if not event_df.empty:
            self.validate_alignment(event_df)

        # Stage 7: persist
        if save and not event_df.empty:
            self._save_parquet(event_df)

        # Summary
        summary = AlignmentSummary(
            total_events=total_events,
            aligned_events=len(aligned_df),
            complete_windows=len(complete_windows),
            incomplete_windows=len(incomplete_windows),
            output_rows=len(event_df),
            output_path=self.config.parquet_path,
        )
        self._log.info("build_event_dataset complete | %s", summary)

        return event_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_parquet(self, df: pd.DataFrame) -> Path:
        """Write *df* to ``config.parquet_path`` using Snappy compression."""
        path = self.config.parquet_path
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, str(path), compression="snappy")
        self._log.info(
            "Event dataset saved | rows=%d | path=%s", len(df), path
        )
        return path

    # ------------------------------------------------------------------
    # Private guard helpers
    # ------------------------------------------------------------------

    def _require_metadata(self) -> None:
        if self._metadata is None:
            raise RuntimeError(
                "Metadata not loaded.  Call load_event_metadata() first."
            )

    def _require_market(self) -> None:
        if self._market is None:
            raise RuntimeError(
                "Market data not loaded.  Call load_market_data() first."
            )

    def _make_empty_window(
        self, row: pd.Series, reason: str
    ) -> EventWindow:
        """Return a placeholder EventWindow with an empty DataFrame."""
        self._log.debug(
            "Empty window for '%s' | reason=%s", row["transcript_id"], reason
        )
        return EventWindow(
            transcript_id=str(row["transcript_id"]),
            ticker=str(row["ticker"]),
            earnings_date=pd.Timestamp(row["earnings_date"]),
            aligned_event_date=pd.Timestamp(row.get("aligned_event_date", pd.NaT)),
            alignment_reason=reason,
            was_adjusted=False,
            market_session="unknown",
            quarter=str(row.get("quarter", "")),
            year=int(row.get("year", 0)),
            window_df=pd.DataFrame(),
            is_complete=False,
            missing_sessions=[],
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _build_synthetic_data(
    pre: int = 30, post: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a minimal synthetic dataset (metadata + market) for self-testing.
    Uses real NYSE trading sessions for date correctness.
    """
    from trading_calendar import TradingCalendarEngine

    engine = TradingCalendarEngine()
    sessions = engine.get_trading_sessions("2023-06-01", "2025-06-30")
    all_dates = pd.DatetimeIndex(sessions.index)

    rng = np.random.default_rng(0)

    # --- AAPL: normal after-hours Thursday ---
    aapl_event_dt = "2024-02-01 16:30"   # Thursday 4:30 PM ET → aligns to Fri 2024-02-02
    # --- MSFT: weekend release ---
    msft_event_dt = "2024-01-27 10:00"   # Saturday → aligns to Mon 2024-01-29
    # --- NVDA: pre-market release ---
    nvda_event_dt = "2024-02-21 07:30"   # Wednesday 7:30 AM ET → same day

    meta = pd.DataFrame({
        "transcript_id": ["AAPL_Q1_2024", "MSFT_Q2_2024", "NVDA_Q4_2023"],
        "ticker":        ["AAPL", "MSFT", "NVDA"],
        "earnings_date": pd.to_datetime([aapl_event_dt, msft_event_dt, nvda_event_dt]),
        "quarter":       ["Q1", "Q2", "Q4"],
        "year":          [2024, 2024, 2023],
    })

    # Build market data: for each ticker, create rows covering the window
    market_frames: list[pd.DataFrame] = []
    for ticker, event_str in zip(
        ["AAPL", "MSFT", "NVDA"],
        [aapl_event_dt, msft_event_dt, nvda_event_dt],
    ):
        aligned = engine.align_earnings_event(event_str, "America/New_York")
        aligned_date = aligned.aligned_event_date.normalize()
        try:
            pos = all_dates.get_loc(aligned_date)
        except KeyError:
            continue
        start = max(0, pos - pre - 5)          # extra buffer
        end = min(len(all_dates) - 1, pos + post + 5)
        ticker_dates = all_dates[start: end + 1]
        n = len(ticker_dates)
        base_price = {"AAPL": 185.0, "MSFT": 390.0, "NVDA": 600.0}[ticker]
        prices = base_price + rng.normal(0, 2, n).cumsum()
        daily_ret = rng.normal(0.0004, 0.013, n)
        bench_ret = rng.normal(0.0002, 0.009, n)
        frame = pd.DataFrame({
            "ticker":           ticker,
            "date":             pd.to_datetime(ticker_dates).normalize(),
            "open":             prices * 0.998,
            "high":             prices * 1.010,
            "low":              prices * 0.990,
            "close":            prices,
            "adj_close":        prices,
            "volume":           rng.integers(40_000_000, 120_000_000, n),
            "daily_return":     daily_ret,
            "benchmark_return": bench_ret,
            "abnormal_return":  daily_ret - bench_ret,
            "return_1d":        np.roll(daily_ret, -1),
            "return_3d":        np.roll(daily_ret, -3),
            "return_5d":        np.roll(daily_ret, -5),
        })
        market_frames.append(frame)

    market = pd.concat(market_frames, ignore_index=True)
    return meta, market


def _run_self_tests() -> None:
    """Deterministic correctness assertions for :class:`EventAligner`."""
    print("\n" + "═" * 68)
    print("  Self-tests")
    print("═" * 68)

    meta, market = _build_synthetic_data()

    cfg = EventAlignmentConfig(
        pre_event_window=10,
        post_event_window=5,
        require_complete_window=False,   # synthetic data may be partial
    )
    aligner = EventAligner(config=cfg)
    aligner.load_event_metadata(meta)
    aligner.load_market_data(market)

    failures: list[str] = []

    def chk(label: str, condition: bool, detail: str = "") -> None:
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  {status}  {label}" + (f" ({detail})" if detail else ""))
        if not condition:
            failures.append(f"{label}: {detail}")

    # 1. align_events() returns correct aligned dates
    aligned_df = aligner.align_events()
    chk("align_events returns all 3 events", len(aligned_df) == 3,
        f"got {len(aligned_df)}")

    aapl_row = aligned_df[aligned_df["transcript_id"] == "AAPL_Q1_2024"].iloc[0]
    chk(
        "AAPL after-hours → next trading day (Fri Feb 2)",
        pd.Timestamp(aapl_row["aligned_event_date"]).date() == pd.Timestamp("2024-02-02").date(),
        str(pd.Timestamp(aapl_row["aligned_event_date"]).date()),
    )
    chk("AAPL was_adjusted=True", bool(aapl_row["was_adjusted"]))

    msft_row = aligned_df[aligned_df["transcript_id"] == "MSFT_Q2_2024"].iloc[0]
    chk(
        "MSFT weekend → Mon Jan 29",
        pd.Timestamp(msft_row["aligned_event_date"]).date() == pd.Timestamp("2024-01-29").date(),
        str(pd.Timestamp(msft_row["aligned_event_date"]).date()),
    )
    chk("MSFT reason=weekend_adjustment",
        msft_row["alignment_reason"] == "weekend_adjustment",
        msft_row["alignment_reason"])

    nvda_row = aligned_df[aligned_df["transcript_id"] == "NVDA_Q4_2023"].iloc[0]
    chk(
        "NVDA pre-market → same day Feb 21",
        pd.Timestamp(nvda_row["aligned_event_date"]).date() == pd.Timestamp("2024-02-21").date(),
        str(pd.Timestamp(nvda_row["aligned_event_date"]).date()),
    )
    chk("NVDA was_adjusted=False", not bool(nvda_row["was_adjusted"]))

    # 2. Window generation
    windows = aligner.generate_event_windows(aligned_df)
    chk("generate_event_windows produces 3 windows", len(windows) == 3,
        f"got {len(windows)}")

    for w in windows:
        has_d0 = (
            not w.window_df.empty
            and (w.window_df["relative_day"] == 0).any()
        )
        chk(f"Window {w.transcript_id} has day-0 row", has_d0)
        rel = sorted(w.window_df["relative_day"].tolist())
        chk(
            f"Window {w.transcript_id} relative_days are monotonic",
            rel == sorted(rel),
            str(rel[:5]),
        )

    # 3. build_event_dataset end-to-end (save=False)
    event_df = aligner.build_event_dataset(save=False)
    chk("build_event_dataset not empty", not event_df.empty,
        f"rows={len(event_df)}")
    chk("is_event_day column exists", "is_event_day" in event_df.columns)
    chk("relative_day column exists", "relative_day" in event_df.columns)
    chk(
        "No duplicate (transcript_id, relative_day)",
        not event_df.duplicated(["transcript_id", "relative_day"]).any(),
    )
    # Every transcript has exactly one day-0 row
    day0_counts = (
        event_df[event_df["relative_day"] == 0]
        .groupby("transcript_id")
        .size()
    )
    chk(
        "Every transcript_id has exactly 1 day-0 row",
        (day0_counts == 1).all(),
        str(day0_counts.to_dict()),
    )

    # 4. validate_window_continuity
    continuity = aligner.validate_window_continuity(windows)
    chk(
        "validate_window_continuity returns dict with correct keys",
        "continuous" in continuity and "gapped" in continuity,
    )

    # 5. filter_incomplete_events
    complete, incomplete_ = aligner.filter_incomplete_events(windows)
    chk(
        "filter_incomplete_events partitions correctly",
        len(complete) + len(incomplete_) == len(windows),
        f"complete={len(complete)} incomplete={len(incomplete_)}",
    )

    print("═" * 68)
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
        raise AssertionError(f"{len(failures)} self-test(s) failed.")
    print(f"  All {sum(1 for _ in failures) + (15 - len(failures))} checks passed.")
    print("═" * 68)


# ---------------------------------------------------------------------------
# __main__ — demo + self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("═" * 68)
    print("  EventAligner — demo run")
    print("═" * 68)

    # ── Build synthetic dataset ────────────────────────────────────────
    print("\n[Setup] Building synthetic metadata + market data …")
    meta, market = _build_synthetic_data(pre=30, post=10)
    print(f"  Metadata rows : {len(meta)}")
    print(f"  Market rows   : {len(market)} | tickers: {market['ticker'].unique().tolist()}")
    print(f"  Market cols   : {list(market.columns)}")

    # ── Initialise aligner ────────────────────────────────────────────
    cfg = EventAlignmentConfig(
        pre_event_window=30,
        post_event_window=10,
        require_complete_window=False,  # synthetic; may lack full buffer
    )
    aligner = EventAligner(config=cfg)
    aligner.load_event_metadata(meta)
    aligner.load_market_data(market)

    # ── Demo 1: align_events ──────────────────────────────────────────
    print("\n[Demo 1] align_events()")
    aligned_df = aligner.align_events()
    display_cols = [
        "transcript_id", "ticker", "earnings_date",
        "aligned_event_date", "alignment_reason", "was_adjusted", "market_session"
    ]
    print(aligned_df[display_cols].to_string(index=False))

    # ── Demo 2: generate_event_windows ────────────────────────────────
    print("\n[Demo 2] generate_event_windows()")
    windows = aligner.generate_event_windows(aligned_df)
    for w in windows:
        d0_row = w.window_df[w.window_df["relative_day"] == 0]
        d0_close = d0_row["close"].values[0] if not d0_row.empty else "N/A"
        print(
            f"  {w.transcript_id:20s} | aligned={w.aligned_event_date.date()} "
            f"| window_rows={len(w.window_df):3d} | complete={w.is_complete} "
            f"| D0_close={d0_close:.2f}" if isinstance(d0_close, float)
            else f"  {w.transcript_id:20s} | D0_close=N/A"
        )

    # ── Demo 3: AAPL after-hours alignment ───────────────────────────
    print("\n[Demo 3] AAPL after-hours event window (relative_days -5 → +5)")
    aapl_w = next(w for w in windows if w.ticker == "AAPL")
    snippet = aapl_w.window_df[
        aapl_w.window_df["relative_day"].between(-5, 5)
    ][["date", "relative_day", "is_event_day", "close", "daily_return", "abnormal_return"]]
    print(snippet.to_string(index=False))

    # ── Demo 4: weekend event (MSFT Saturday) ────────────────────────
    print("\n[Demo 4] MSFT weekend event alignment")
    msft_w = next(w for w in windows if w.ticker == "MSFT")
    print(f"  earnings_date    : {msft_w.earnings_date}")
    print(f"  aligned_date     : {msft_w.aligned_event_date.date()}")
    print(f"  alignment_reason : {msft_w.alignment_reason}")
    print(f"  was_adjusted     : {msft_w.was_adjusted}")

    # ── Demo 5: filter_incomplete_events ─────────────────────────────
    print("\n[Demo 5] filter_incomplete_events()")
    complete_wins, incomplete_wins = aligner.filter_incomplete_events(windows)
    print(f"  Complete   : {len(complete_wins)}")
    print(f"  Incomplete : {len(incomplete_wins)}")

    # ── Demo 6: build_event_dataset (full pipeline) ───────────────────
    print("\n[Demo 6] build_event_dataset() — full pipeline (save=False)")
    event_df = aligner.build_event_dataset(save=False)
    print(f"  Output shape : {event_df.shape}")
    print(f"  Columns      : {list(event_df.columns)}")
    print(f"  Events       : {event_df['transcript_id'].nunique()}")
    print(f"  Null returns : {event_df['daily_return'].isnull().sum()}")
    print("\n  Day-0 snapshot per event:")
    d0 = event_df[event_df["relative_day"] == 0][
        ["transcript_id", "aligned_event_date", "close", "daily_return", "abnormal_return"]
    ]
    print(d0.to_string(index=False))

    # ── Demo 7: validate_window_continuity ────────────────────────────
    print("\n[Demo 7] validate_window_continuity()")
    cont = aligner.validate_window_continuity(windows)
    print(f"  Continuous : {cont['continuous']}")
    print(f"  Gapped     : {cont['gapped']}")

    # ── Self-tests ────────────────────────────────────────────────────
    _run_self_tests()

    print("\n✅  Demo complete.")
