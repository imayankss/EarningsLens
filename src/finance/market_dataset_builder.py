"""
src/finance/market_dataset_builder.py
=======================================
DAY 5 MASTER MARKET PIPELINE — Orchestration layer for the Earnings Call
Sentiment Analyzer market data system.

Pipeline position (final stage of Day 5):

    data/interim/transcript_metadata.parquet
              │
              ▼
    ┌─────────────────────────────────────────────────┐
    │           MarketDatasetBuilder.run()            │
    │                                                 │
    │  STEP 1  load_transcript_metadata()             │
    │  STEP 2  extract unique tickers + dates         │
    │  STEP 3  download OHLCV per ticker              │  ← market_data_loader
    │  STEP 4  download benchmark data                │  ← benchmark_loader
    │  STEP 5  align earnings events                  │  ← trading_calendar
    │  STEP 6  generate financial features            │  ← feature_engineering
    │  STEP 7  build event windows (relative_day)     │  ← event_alignment
    │  STEP 8  merge_all_tickers()                    │
    │  STEP 9  validate_final_dataset()               │
    │  STEP 10 export_outputs()                       │
    └─────────────────────────────────────────────────┘
              │
              ▼
    data/processed/market_data.parquet
    data/processed/market_data.csv
    data/interim/aligned_events.parquet
    data/interim/benchmark_returns.parquet

Design principle: THIS MODULE ORCHESTRATES. It never duplicates business
logic that belongs to its dependencies. Each step delegates to the
responsible module; the builder's only job is sequencing, error isolation,
merging, validation, and export.

Module dependencies
-------------------
Implemented (Day 5):
    market_data_loader.py   — OHLCV download, parquet I/O
    feature_engineering.py  — all return / volatility / MA features

Contracts defined here (implement separately):
    benchmark_loader.py     — S&P500 OHLCV + benchmark_return column
    trading_calendar.py     — NYSE calendar, event-date alignment
    event_alignment.py      — relative_day tagging, event window merging

See ``BenchmarkLoaderProtocol``, ``TradingCalendarProtocol``, and
``EventAlignerProtocol`` for the exact interfaces each must satisfy.

Author : Earnings Call Sentiment Analyzer — Day 5
Python : 3.11+
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sibling module imports
# ---------------------------------------------------------------------------
# market_data_loader and feature_engineering are already implemented.
# The three remaining modules are imported defensively — the builder raises
# an informative error at runtime if they are missing, but the file itself
# always parses cleanly.

from src.finance.market_data_loader import (  # type: ignore[import]
    MarketDataConfig,
    MarketDataLoader,
)
from src.finance.feature_engineering import (  # type: ignore[import]
    FeatureEngineeringConfig,
    FinancialFeatureEngineer,
)

try:
    from src.finance.benchmark_loader import (  # type: ignore[import]
        BenchmarkConfig,
        BenchmarkLoader,
    )
    _BENCHMARK_LOADER_AVAILABLE = True
except ImportError:
    _BENCHMARK_LOADER_AVAILABLE = False

try:
    from src.finance.trading_calendar import TradingCalendarEngine  # type: ignore[import]
    _TRADING_CALENDAR_AVAILABLE = True
except ImportError:
    _TRADING_CALENDAR_AVAILABLE = False

try:
    from src.finance.event_alignment import EventAligner  # type: ignore[import]
    _EVENT_ALIGNER_AVAILABLE = True
except ImportError:
    _EVENT_ALIGNER_AVAILABLE = False

try:
    import yaml  # type: ignore[import]
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# Module-dependency Protocols
# ===========================================================================
# These Protocols are the formal contracts that benchmark_loader.py,
# trading_calendar.py, and event_alignment.py must satisfy.
# They serve as:
#   (a) the authoritative interface specification for implementors,
#   (b) type-checker targets so mypy catches mismatches immediately,
#   (c) documentation that lives at the point of use (here).
# ===========================================================================


@runtime_checkable
class BenchmarkLoaderProtocol(Protocol):
    """
    Contract for ``src/finance/benchmark_loader.py``.

    Responsibilities
    ----------------
    - Download S&P500 (``^GSPC``) OHLCV for a date range via yfinance.
    - Compute ``benchmark_return = adj_close.pct_change()`` per day.
    - Expose the result as a tidy DataFrame with a DatetimeIndex named
      ``date`` and columns: ``benchmark_return`` (minimum).
    - Persist intermediate output to
      ``data/interim/benchmark_returns.parquet``.

    Required methods
    ----------------
    download(start_date, end_date) -> pd.DataFrame
        Returns a DataFrame indexed by date with at least a
        ``benchmark_return`` column.
    merge_into(stock_df, benchmark_df) -> pd.DataFrame
        Left-joins ``benchmark_return`` onto ``stock_df`` by date,
        forward-filling up to 1 day for early-close sessions.
    save_parquet(df, path) -> Path
        Persist ``benchmark_df`` to parquet.
    load_parquet(path) -> pd.DataFrame
        Load a previously saved benchmark parquet.
    """

    def download(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame: ...

    def merge_into(
        self,
        stock_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
    ) -> pd.DataFrame: ...

    def save_parquet(self, df: pd.DataFrame, path: Path) -> Path: ...

    def load_parquet(self, path: Path) -> pd.DataFrame: ...


@runtime_checkable
class TradingCalendarProtocol(Protocol):
    """
    Contract for ``src/finance/trading_calendar.py``.

    Responsibilities
    ----------------
    - Wrap ``pandas_market_calendars`` NYSE calendar.
    - Determine whether a given date is a valid NYSE session.
    - Shift non-trading days (weekends, holidays) to the next valid
      session.
    - Align earnings event dates: if earnings were released *after*
      market close, the event day is the *next* trading session.

    Required methods
    ----------------
    is_trading_day(d) -> bool
        Return True if ``d`` is a valid NYSE trading session.
    next_trading_day(d) -> date
        Return the next valid NYSE session on or after ``d``.
    align_event_date(earnings_date, after_close) -> date
        Return the market-impact date:
          - after_close=False → same trading day (if valid) or next
          - after_close=True  → next trading day unconditionally
    get_trading_days(start_date, end_date) -> list[date]
        Return all valid NYSE sessions in [start_date, end_date].
    """

    def is_trading_day(self, d: date) -> bool: ...

    def next_trading_day(self, d: date) -> date: ...

    def align_event_date(
        self,
        earnings_date: date,
        after_close: bool = True,
    ) -> date: ...

    def get_trading_days(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]: ...


@runtime_checkable
class EventAlignerProtocol(Protocol):
    """
    Contract for ``src/finance/event_alignment.py``.

    Responsibilities
    ----------------
    - Tag each row of a market DataFrame with a ``relative_day``
      integer: 0 = aligned event date, -N = N days before, +N after.
    - Attach ``transcript_id`` and ``aligned_event_date`` from the
      transcript metadata to each market-data row.
    - Produce the event-window slice for each (ticker, transcript_id)
      pair over the configured pre/post day range.
    - Save ``data/interim/aligned_events.parquet``.

    Required methods
    ----------------
    tag_relative_days(market_df, event_date_col) -> pd.DataFrame
        Computes ``relative_day`` per (ticker, transcript_id) group.
    build_event_windows(metadata_df, market_dict, calendar) -> pd.DataFrame
        Merges transcript metadata + aligned market data into the final
        event-window DataFrame with ``relative_day``.
    save_parquet(df, path) -> Path
    load_parquet(path) -> pd.DataFrame
    """

    def tag_relative_days(
        self,
        market_df: pd.DataFrame,
        event_date_col: str = "aligned_event_date",
    ) -> pd.DataFrame: ...

    def build_event_windows(
        self,
        metadata_df: pd.DataFrame,
        market_dict: dict[str, pd.DataFrame],
        calendar: Any,
    ) -> pd.DataFrame: ...

    def save_parquet(self, df: pd.DataFrame, path: Path) -> Path: ...

    def load_parquet(self, path: Path) -> pd.DataFrame: ...


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class MarketPipelineConfig:
    """
    Master configuration for ``MarketDatasetBuilder``.

    All fields map 1-to-1 to keys in ``market_config.yaml``.  Pass a
    path to ``load_config()`` to populate from YAML; or construct
    directly with keyword arguments.

    Attributes
    ----------
    # ── Paths ──────────────────────────────────────────────────────────
    raw_market_dir : Path
        Where per-ticker OHLCV parquet files are stored.
    interim_dir : Path
        Where aligned_events.parquet and benchmark_returns.parquet go.
    processed_dir : Path
        Final output directory for market_data.parquet / .csv.
    transcript_metadata_path : Path
        Input: parquet produced by transcript preprocessing and used by
        sentiment pipelines.

    # ── Benchmark ──────────────────────────────────────────────────────
    benchmark_ticker : str
        yfinance symbol for the market benchmark.  Default: ``^GSPC``.

    # ── Download windows ───────────────────────────────────────────────
    pre_event_calendar_days : int
        Calendar days before earnings to download (generous buffer so
        rolling-window features are fully warm).  Default: 45.
    post_event_calendar_days : int
        Calendar days after earnings.  Default: 20.

    # ── Feature engineering ────────────────────────────────────────────
    return_horizons : list[int]
        Forward-return horizons in trading days.  Default: [1, 2, 3, 5].
    rolling_vol_window : int
        Volatility lookback window.  Default: 20.
    moving_average_windows : list[int]
        SMA windows.  Default: [20, 50].
    ema_span : int
        EMA span.  Default: 20.
    momentum_windows : list[int]
        Momentum lookback windows.  Default: [5, 20].

    # ── Download reliability ───────────────────────────────────────────
    retry_count : int
        yfinance download retries.  Default: 3.
    retry_delay : float
        Base retry delay in seconds (exponential back-off).  Default: 2.0.
    download_timeout : int
        HTTP timeout per request.  Default: 30.

    # ── Pipeline behaviour ─────────────────────────────────────────────
    use_cache : bool
        If True, skip re-downloading tickers whose parquet already
        exists in ``raw_market_dir``.  Default: True.
    export_csv : bool
        If True, also write market_data.csv alongside the parquet.
        Default: True.
    save_intermediates : bool
        Persist aligned_events.parquet and benchmark_returns.parquet.
        Default: True.
    annualisation_factor : int
        Trading days per year for volatility annualisation.  Default: 252.
    """

    # Paths
    raw_market_dir: Path = field(
        default_factory=lambda: Path("data/raw/market")
    )
    interim_dir: Path = field(
        default_factory=lambda: Path("data/interim")
    )
    processed_dir: Path = field(
        default_factory=lambda: Path("data/processed")
    )
    transcript_metadata_path: Path = field(
        default_factory=lambda: Path(
            "data/interim/transcripts/transcripts_cleaned.parquet"
        )
    )

    # Benchmark
    benchmark_ticker: str = "^GSPC"

    # Download windows
    pre_event_calendar_days: int = 45
    post_event_calendar_days: int = 20

    # Feature engineering
    return_horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 5])
    rolling_vol_window: int = 20
    moving_average_windows: list[int] = field(default_factory=lambda: [20, 50])
    ema_span: int = 20
    momentum_windows: list[int] = field(default_factory=lambda: [5, 20])

    # Download reliability
    retry_count: int = 3
    retry_delay: float = 2.0
    download_timeout: int = 30

    # Pipeline behaviour
    use_cache: bool = True
    export_csv: bool = True
    save_intermediates: bool = True
    annualisation_factor: int = 252

    def __post_init__(self) -> None:
        # Coerce to Path objects in case strings were passed
        for attr in (
            "raw_market_dir",
            "interim_dir",
            "processed_dir",
            "transcript_metadata_path",
        ):
            setattr(self, attr, Path(getattr(self, attr)))

    def as_market_data_config(self) -> "MarketDataConfig":
        """Produce the MarketDataConfig consumed by MarketDataLoader."""
        return MarketDataConfig(
            output_dir=self.raw_market_dir,
            auto_adjust=False,
            save_raw=True,
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
            trading_buffer_before=self.pre_event_calendar_days,
            trading_buffer_after=self.post_event_calendar_days,
            timeout=self.download_timeout,
        )

    def as_feature_config(self) -> "FeatureEngineeringConfig":
        """Produce the FeatureEngineeringConfig consumed by FinancialFeatureEngineer."""
        return FeatureEngineeringConfig(
            return_horizons=self.return_horizons,
            rolling_vol_window=self.rolling_vol_window,
            moving_average_windows=self.moving_average_windows,
            ema_span=self.ema_span,
            momentum_windows=self.momentum_windows,
            annualisation_factor=self.annualisation_factor,
        )


# ===========================================================================
# Per-ticker result
# ===========================================================================


@dataclass
class TickerResult:
    """
    Outcome of processing a single ticker through the pipeline.

    Attributes
    ----------
    ticker : str
        Stock symbol.
    transcript_id : str
        Associated transcript identifier.
    earnings_date : date | None
        Original earnings date from transcript metadata.
    aligned_event_date : date | None
        NYSE-aligned event date (set after trading_calendar step).
    success : bool
        True if all pipeline steps completed without error.
    error : str
        Error message if ``success`` is False.
    row_count : int
        Number of rows in the output DataFrame for this ticker.
    feature_count : int
        Number of feature columns successfully computed.
    df : pd.DataFrame
        Enriched OHLCV + features DataFrame, or empty on failure.
    elapsed_seconds : float
        Wall-clock time to process this ticker.
    """

    ticker: str
    transcript_id: str = ""
    earnings_date: Optional[date] = None
    aligned_event_date: Optional[date] = None
    success: bool = False
    error: str = ""
    row_count: int = 0
    feature_count: int = 0
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (excludes the DataFrame)."""
        return {
            "ticker": self.ticker,
            "transcript_id": self.transcript_id,
            "earnings_date": str(self.earnings_date) if self.earnings_date else None,
            "aligned_event_date": (
                str(self.aligned_event_date) if self.aligned_event_date else None
            ),
            "success": self.success,
            "error": self.error,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ===========================================================================
# Pipeline run summary
# ===========================================================================


@dataclass
class PipelineRunSummary:
    """
    Aggregate metrics for a complete ``MarketDatasetBuilder.run()`` call.

    Attributes
    ----------
    run_id : str
        ISO-format timestamp of the run start.
    total_transcripts : int
        Total rows in the input transcript_metadata table.
    total_tickers : int
        Unique tickers processed.
    successful_tickers : list[str]
        Tickers that completed all pipeline steps.
    failed_tickers : list[str]
        Tickers that encountered at least one unrecoverable error.
    total_rows_exported : int
        Row count in the final market_data.parquet.
    date_range_start : date | None
        Earliest date in the exported dataset.
    date_range_end : date | None
        Latest date in the exported dataset.
    missing_value_summary : dict[str, float]
        Mapping of column → fraction of NaN values (features only).
    validation_errors : list[str]
        Errors raised during final dataset validation.
    validation_warnings : list[str]
        Warnings raised during final dataset validation.
    elapsed_seconds : float
        Total wall-clock time for the full run.
    output_paths : dict[str, str]
        Mapping of output label → file path string.
    ticker_results : list[TickerResult]
        Per-ticker detail records.
    """

    run_id: str = ""
    total_transcripts: int = 0
    total_tickers: int = 0
    successful_tickers: list[str] = field(default_factory=list)
    failed_tickers: list[str] = field(default_factory=list)
    total_rows_exported: int = 0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    missing_value_summary: dict[str, float] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    output_paths: dict[str, str] = field(default_factory=dict)
    ticker_results: list[TickerResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of tickers that completed successfully."""
        if self.total_tickers == 0:
            return 0.0
        return len(self.successful_tickers) / self.total_tickers

    def print_report(self) -> None:
        """Print a human-readable pipeline run report to stdout."""
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  DAY 5 MARKET PIPELINE — RUN REPORT  [{self.run_id}]")
        print(sep)
        print(f"  Transcripts      : {self.total_transcripts}")
        print(f"  Tickers          : {self.total_tickers}")
        print(f"  Succeeded        : {len(self.successful_tickers)}")
        print(
            f"  Failed           : {len(self.failed_tickers)}"
            + (f"  → {self.failed_tickers}" if self.failed_tickers else "")
        )
        print(f"  Rows exported    : {self.total_rows_exported:,}")
        if self.date_range_start and self.date_range_end:
            print(
                f"  Date range       : {self.date_range_start} "
                f"→ {self.date_range_end}"
            )
        print(f"  Success rate     : {self.success_rate:.0%}")
        print(f"  Elapsed          : {self.elapsed_seconds:.1f}s")

        if self.output_paths:
            print("\n  Outputs:")
            for label, path in self.output_paths.items():
                print(f"    {label:<30} {path}")

        if self.validation_errors:
            print("\n  VALIDATION ERRORS:")
            for e in self.validation_errors:
                print(f"    ✗ {e}")

        if self.validation_warnings:
            print("\n  Validation warnings:")
            for w in self.validation_warnings:
                print(f"    ⚠ {w}")

        if self.missing_value_summary:
            high_missing = {
                k: v
                for k, v in self.missing_value_summary.items()
                if v > 0.1
            }
            if high_missing:
                print("\n  Columns with >10% missing values:")
                for col, frac in sorted(
                    high_missing.items(), key=lambda x: -x[1]
                ):
                    print(f"    {col:<35} {frac:.1%}")

        print(sep + "\n")


# ===========================================================================
# Required columns for the final exported dataset
# ===========================================================================

FINAL_REQUIRED_COLUMNS: list[str] = [
    "transcript_id",
    "ticker",
    "earnings_date",
    "aligned_event_date",
    "relative_day",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "daily_return",
    "benchmark_return",
    "abnormal_return",
    "return_1d",
    "return_3d",
    "return_5d",
    "rolling_volatility_20d",
    "sma_20",
    "sma_50",
    "momentum_5d",
    "relative_volume",
]


# ===========================================================================
# Master builder
# ===========================================================================


class MarketDatasetBuilder:
    """
    Day 5 master orchestration pipeline.

    Coordinates ``MarketDataLoader``, ``BenchmarkLoader``,
    ``TradingCalendar``, ``FinancialFeatureEngineer``, and
    ``EventAligner`` to produce a single analytics-ready parquet
    dataset from raw transcript metadata.

    The builder enforces a strict separation of concerns: every step
    delegates entirely to the responsible module; this class provides
    only sequencing, error isolation, merging, and I/O.

    Parameters
    ----------
    config : MarketPipelineConfig, optional
        Pipeline configuration.  Defaults to ``MarketPipelineConfig()``.
    benchmark_loader : BenchmarkLoaderProtocol, optional
        Injected benchmark loader.  If omitted the builder attempts to
        import ``BenchmarkLoader`` from ``src.finance.benchmark_loader``.
    trading_calendar : TradingCalendarProtocol, optional
        Injected trading calendar.  Falls back to import from
        ``src.finance.trading_calendar``.
    event_aligner : EventAlignerProtocol, optional
        Injected event aligner.  Falls back to import from
        ``src.finance.event_alignment``.

    Notes
    -----
    Dependency injection is the preferred pattern in tests — pass mock
    objects that satisfy the Protocols above to avoid live API calls.

    Examples
    --------
    >>> builder = MarketDatasetBuilder()
    >>> summary = builder.run()
    >>> summary.print_report()
    """

    def __init__(
        self,
        config: Optional[MarketPipelineConfig] = None,
        benchmark_loader: Optional[Any] = None,
        trading_calendar: Optional[Any] = None,
        event_aligner: Optional[Any] = None,
    ) -> None:
        self.config = config or MarketPipelineConfig()
        self._ensure_directories()

        # Market data loader (always available — already implemented)
        self._market_loader = MarketDataLoader(
            config=self.config.as_market_data_config()
        )

        # Feature engineer (always available — already implemented)
        self._feature_engineer = FinancialFeatureEngineer(
            config=self.config.as_feature_config()
        )

        # Injected or auto-imported dependencies
        self._benchmark_loader = benchmark_loader or self._resolve_benchmark_loader()
        self._trading_calendar = trading_calendar or self._resolve_trading_calendar()
        self._event_aligner = event_aligner or self._resolve_event_aligner()

        logger.info(
            "MarketDatasetBuilder ready | config=%s",
            self.config.processed_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_config(path: str | Path) -> MarketPipelineConfig:
        """
        Load pipeline configuration from a YAML file.

        The YAML file should mirror ``MarketPipelineConfig`` field names.
        Any key absent from the file falls back to the dataclass default.

        Parameters
        ----------
        path : str | Path
            Path to ``market_config.yaml``.

        Returns
        -------
        MarketPipelineConfig
            Populated configuration object.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ImportError
            If ``pyyaml`` is not installed.

        Example YAML
        ------------
        .. code-block:: yaml

            benchmark_ticker: "^GSPC"
            pre_event_calendar_days: 45
            post_event_calendar_days: 20
            return_horizons: [1, 2, 3, 5]
            rolling_vol_window: 20
            moving_average_windows: [20, 50]
            retry_count: 3
            use_cache: true
            export_csv: true
            raw_market_dir: "data/raw/market"
            interim_dir: "data/interim"
            processed_dir: "data/processed"
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "pyyaml is required to load YAML configs: pip install pyyaml"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        # Map both the flat unit-test style config and the nested project
        # config in configs/market_config.yaml onto dataclass fields.
        nested: dict[str, Any] = {}
        paths = raw.get("paths", {}) or {}
        benchmark = raw.get("benchmark", {}) or {}
        market_data = raw.get("market_data", {}) or {}
        event_alignment = raw.get("event_alignment", {}) or {}
        features = raw.get("feature_engineering", {}) or {}
        exports = raw.get("exports", {}) or {}

        nested.update(
            {
                "transcript_metadata_path": paths.get("transcript_metadata"),
                "raw_market_dir": paths.get("raw_market_dir"),
                "interim_dir": paths.get("interim_dir"),
                "processed_dir": paths.get("processed_dir"),
                "benchmark_ticker": benchmark.get("ticker"),
                "pre_event_calendar_days": market_data.get(
                    "trading_buffer_before",
                    event_alignment.get("pre_event_window"),
                ),
                "post_event_calendar_days": market_data.get(
                    "trading_buffer_after",
                    event_alignment.get("post_event_window"),
                ),
                "return_horizons": features.get("return_horizons"),
                "rolling_vol_window": features.get("rolling_volatility_window"),
                "moving_average_windows": features.get("moving_average_windows"),
                "momentum_windows": features.get("momentum_windows"),
                "retry_count": market_data.get(
                    "retry_count", benchmark.get("retry_count")
                ),
                "retry_delay": market_data.get(
                    "retry_delay", benchmark.get("retry_delay")
                ),
                "export_csv": exports.get("export_csv"),
            }
        )

        # Top-level keys override nested values when explicitly provided.
        nested.update(raw)
        raw = {k: v for k, v in nested.items() if v is not None}

        # Map YAML keys onto dataclass fields; ignore unknown keys.
        valid_fields = {f.name for f in MarketPipelineConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in valid_fields}

        # Coerce list fields that YAML may parse as None
        for list_field in ("return_horizons", "moving_average_windows", "momentum_windows"):
            if list_field in filtered and filtered[list_field] is None:
                filtered.pop(list_field)

        cfg = MarketPipelineConfig(**filtered)
        logger.info("Config loaded from %s: %s", path, filtered)
        return cfg

    def load_transcript_metadata(self) -> pd.DataFrame:
        """
        Load transcript metadata produced by the Day 4 pipeline.

        Expected schema
        ---------------
        Required columns:
            transcript_id  — unique event identifier (e.g. AAPL_Q1_2025)
            ticker         — stock symbol
            earnings_date  — date of the earnings call

        Optional but used when present:
            quarter, year, company

        Parameters (via config)
        -----------------------
        transcript_metadata_path : Path
            ``data/interim/transcripts/transcripts_cleaned.parquet`` by default.

        Returns
        -------
        pd.DataFrame
            Validated metadata table, one row per transcript.

        Raises
        ------
        FileNotFoundError
            If the parquet file does not exist.
        ValueError
            If required columns are missing.
        """
        path = self.config.transcript_metadata_path

        if not path.exists():
            raise FileNotFoundError(
                f"Transcript metadata not found at {path}. "
                "Run the Day 4 preprocessing pipeline first."
            )

        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(
            "Loaded transcript metadata: %d rows from %s", len(df), path
        )

        # Validate required columns
        required = {"transcript_id", "ticker", "earnings_date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Transcript metadata missing required columns: {sorted(missing)}. "
                f"Found: {sorted(df.columns.tolist())}"
            )

        # Normalise earnings_date to python date objects
        df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date

        # Normalise ticker (upper-case, stripped)
        df["ticker"] = df["ticker"].str.strip().str.upper()

        # Drop rows with null tickers or dates
        before = len(df)
        df = df.dropna(subset=["ticker", "earnings_date"])
        dropped = before - len(df)
        if dropped:
            logger.warning(
                "Dropped %d rows with null ticker / earnings_date.", dropped
            )

        return df.reset_index(drop=True)

    def process_ticker(
        self,
        ticker: str,
        earnings_date: date,
        transcript_id: str,
        benchmark_df: pd.DataFrame,
    ) -> TickerResult:
        """
        Run the full market data pipeline for a single ticker.

        Steps
        -----
        1. Cache check — if ``use_cache=True`` and parquet exists, load it.
        2. Download OHLCV event window (delegates to MarketDataLoader).
        3. Merge benchmark returns (delegates to BenchmarkLoader).
        4. Align event date to valid NYSE session (delegates to TradingCalendar).
        5. Attach metadata columns (transcript_id, earnings_date,
           aligned_event_date).
        6. Compute financial features (delegates to FinancialFeatureEngineer).
        7. Tag relative_day per trading session vs. aligned event date.

        Parameters
        ----------
        ticker : str
            Stock symbol.
        earnings_date : date
            Original earnings call date from transcript metadata.
        transcript_id : str
            Unique transcript identifier — propagated to every row.
        benchmark_df : pd.DataFrame
            Pre-downloaded benchmark returns DataFrame (shared across all
            tickers to avoid redundant downloads).

        Returns
        -------
        TickerResult
            Contains the enriched DataFrame and processing metadata.
            On failure, ``TickerResult.success`` is False and the
            ``error`` field holds the exception message; the DataFrame
            is empty.  The pipeline continues to the next ticker.
        """
        t0 = time.perf_counter()
        ticker = str(ticker).upper().strip()
        result = TickerResult(
            ticker=ticker,
            transcript_id=transcript_id,
            earnings_date=earnings_date,
        )

        try:
            # ── Step 1: Cache check ──────────────────────────────────
            raw_df = self._load_from_cache(ticker)

            if raw_df is None:
                # ── Step 2: Download OHLCV ───────────────────────────
                raw_df = self._market_loader.download_event_window(
                    ticker=ticker,
                    earnings_date=earnings_date,
                    pre_days=self.config.pre_event_calendar_days,
                    post_days=self.config.post_event_calendar_days,
                )

            if raw_df is None or raw_df.empty:
                raise ValueError(
                    f"No OHLCV data returned for {ticker}. "
                    "Ticker may be delisted or outside valid date range."
                )

            # ── Step 3: Merge benchmark returns ──────────────────────
            if not benchmark_df.empty and self._benchmark_loader is not None:
                raw_df = self._benchmark_loader.merge_into(raw_df, benchmark_df)
            elif "benchmark_return" not in raw_df.columns:
                raw_df["benchmark_return"] = np.nan
                logger.warning(
                    "%s: benchmark_return unavailable — AR features will be NaN.",
                    ticker,
                )

            # ── Step 4: Align event date ─────────────────────────────
            aligned_date = self._align_event_date(earnings_date)
            result.aligned_event_date = aligned_date

            # ── Step 5: Attach metadata columns ──────────────────────
            raw_df = raw_df.copy()
            raw_df["transcript_id"] = transcript_id
            raw_df["earnings_date"] = pd.Timestamp(earnings_date)
            raw_df["aligned_event_date"] = pd.Timestamp(aligned_date)

            # ── Step 6: Feature engineering ───────────────────────────
            fe_result = self._feature_engineer.build_feature_set(
                raw_df, ticker=ticker
            )

            if not fe_result.is_valid:
                logger.warning(
                    "%s: Feature engineering validation warnings: %s",
                    ticker,
                    fe_result.validation_errors,
                )

            enriched_df = fe_result.df

            # ── Step 7: Tag relative_day ─────────────────────────────
            enriched_df = self._tag_relative_days(enriched_df, aligned_date)

            result.df = enriched_df
            result.success = True
            result.row_count = len(enriched_df)
            result.feature_count = len(fe_result.features_added)

            logger.info(
                "✓ %s | %d rows | %d features | aligned_date=%s",
                ticker,
                result.row_count,
                result.feature_count,
                aligned_date,
            )

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.df = pd.DataFrame()
            logger.error(
                "✗ %s failed: %s\n%s",
                ticker,
                exc,
                traceback.format_exc(),
            )

        result.elapsed_seconds = time.perf_counter() - t0
        return result

    def build_market_dataset(
        self, metadata_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[TickerResult]]:
        """
        Run per-ticker processing across all events in the metadata table.

        Downloads benchmark data once, then calls ``process_ticker()``
        for each unique (ticker, transcript_id, earnings_date) triple.
        Failed tickers are isolated — their TickerResult records
        ``success=False`` but processing continues.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Output of ``load_transcript_metadata()``.

        Returns
        -------
        tuple[pd.DataFrame, list[TickerResult]]
            - Merged DataFrame of all successful ticker results.
            - List of per-ticker TickerResult objects (success and failure).
        """
        # ── Download benchmark once for the full date range ──────────
        benchmark_df = self._download_benchmark(metadata_df)
        if self.config.save_intermediates and not benchmark_df.empty:
            self._save_intermediate(
                benchmark_df, self.config.interim_dir / "benchmark_returns.parquet"
            )

        # ── Process each event row ───────────────────────────────────
        ticker_results: list[TickerResult] = []
        events = metadata_df[
            ["ticker", "earnings_date", "transcript_id"]
        ].drop_duplicates()

        total = len(events)
        logger.info("Processing %d ticker events…", total)

        for i, row in enumerate(events.itertuples(index=False), start=1):
            logger.info(
                "  [%d/%d] %s  earnings=%s  id=%s",
                i, total, row.ticker, row.earnings_date, row.transcript_id,
            )
            result = self.process_ticker(
                ticker=row.ticker,
                earnings_date=row.earnings_date,
                transcript_id=row.transcript_id,
                benchmark_df=benchmark_df,
            )
            ticker_results.append(result)

        merged_df = self.merge_all_tickers(ticker_results)
        return merged_df, ticker_results

    def merge_all_tickers(
        self, ticker_results: list[TickerResult]
    ) -> pd.DataFrame:
        """
        Concatenate all successful per-ticker DataFrames into one.

        Post-merge operations
        ---------------------
        1. Concatenate successful results.
        2. Reset index to a clean integer RangeIndex (date moves to column).
        3. Sort by (transcript_id, ticker, date, relative_day).
        4. Drop any all-NaN columns introduced by failed partial merges.
        5. Log shape and column inventory.

        Parameters
        ----------
        ticker_results : list[TickerResult]
            Output of ``build_market_dataset()``.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame, or empty DataFrame if all tickers failed.
        """
        successful = [r.df for r in ticker_results if r.success and not r.df.empty]

        if not successful:
            logger.error("No successful ticker results — merged DataFrame is empty.")
            return pd.DataFrame()

        df = pd.concat(successful, ignore_index=False)

        # Reset index — move DatetimeIndex 'date' into a regular column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()  # brings 'date' back as a column
            df.rename(columns={"index": "date"}, inplace=True, errors="ignore")
        else:
            df = df.reset_index(drop=True)

        # Normalise the date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)

        # Sort deterministically
        sort_cols = [c for c in ["transcript_id", "ticker", "date", "relative_day"]
                     if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        # Drop columns that are entirely NaN (artefacts from concat)
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            logger.debug("Dropped all-NaN columns: %s", all_nan_cols)

        logger.info(
            "Merged dataset: %d rows × %d columns from %d tickers.",
            len(df),
            df.shape[1],
            len(successful),
        )
        return df

    def validate_final_dataset(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Run final integrity checks on the merged, feature-enriched dataset.

        Validation checks
        -----------------
        1.  DataFrame is not empty.
        2.  All FINAL_REQUIRED_COLUMNS are present.
        3.  No duplicate (ticker, transcript_id, date) combinations.
        4.  No missing ``aligned_event_date`` values.
        5.  All tickers in the dataset are non-null strings.
        6.  Dates are sorted per (ticker, transcript_id) group.
        7.  ``relative_day=0`` exists for every (ticker, transcript_id) event.
        8.  Benchmark return alignment — every stock date with a non-null
            ``daily_return`` should have a non-null ``benchmark_return``.
        9.  No infinite values in any numeric column.
        10. (Warning) Missing value fraction > 20% for feature columns.
        11. (Warning) Tickers with fewer than 5 rows.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``merge_all_tickers()``.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(errors, warnings)`` — both lists of strings.
        """
        errors: list[str] = []
        warnings_: list[str] = []

        # 1. Empty check
        if df is None or df.empty:
            errors.append("Final dataset is empty.")
            return errors, warnings_

        # 2. Required columns
        missing_cols = [c for c in FINAL_REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # 3. Duplicate rows
        id_cols = [c for c in ["ticker", "transcript_id", "date"] if c in df.columns]
        if id_cols:
            dup_n = df.duplicated(subset=id_cols).sum()
            if dup_n:
                errors.append(f"Duplicate (ticker, transcript_id, date) rows: {dup_n}.")

        # 4. Missing aligned_event_date
        if "aligned_event_date" in df.columns:
            null_aed = df["aligned_event_date"].isna().sum()
            if null_aed:
                errors.append(f"Null aligned_event_date: {null_aed} rows.")

        # 5. Null tickers
        if "ticker" in df.columns:
            null_tickers = df["ticker"].isna().sum()
            if null_tickers:
                errors.append(f"Null ticker values: {null_tickers} rows.")

        # 6. Date sort order per group
        if "date" in df.columns and "ticker" in df.columns:
            group_cols = [c for c in ["ticker", "transcript_id"] if c in df.columns]
            for _, grp in df.groupby(group_cols):
                if not grp["date"].is_monotonic_increasing:
                    errors.append(
                        f"Dates not sorted ascending for group "
                        f"{grp[group_cols].iloc[0].to_dict()}."
                    )
                    break  # report once; don't flood

        # 7. relative_day=0 event anchor exists per group
        if "relative_day" in df.columns and "transcript_id" in df.columns:
            group_cols = [c for c in ["ticker", "transcript_id"] if c in df.columns]
            groups_without_anchor = (
                df.groupby(group_cols)["relative_day"]
                .apply(lambda s: 0 not in s.values)
                .sum()
            )
            if groups_without_anchor:
                warnings_.append(
                    f"{groups_without_anchor} event group(s) have no relative_day=0. "
                    "Earnings date may fall on a non-trading day."
                )

        # 8. Benchmark alignment
        if {"daily_return", "benchmark_return"}.issubset(df.columns):
            has_return = df["daily_return"].notna()
            no_bench = df.loc[has_return, "benchmark_return"].isna().sum()
            if no_bench > 0:
                warnings_.append(
                    f"{no_bench} rows have daily_return but no benchmark_return. "
                    "Check BenchmarkLoader date coverage."
                )

        # 9. Infinite values
        numeric = df.select_dtypes(include=[np.number])
        inf_cols = numeric.columns[np.isinf(numeric).any()].tolist()
        if inf_cols:
            errors.append(f"Infinite values in columns: {inf_cols}.")

        # 10. (Warning) High NaN fraction in feature columns
        feature_cols = [
            c for c in df.columns
            if c not in {"ticker", "transcript_id", "date",
                         "earnings_date", "aligned_event_date", "relative_day"}
        ]
        total = len(df)
        for col in feature_cols:
            frac = df[col].isna().mean()
            if frac > 0.20:
                warnings_.append(
                    f"'{col}' is {frac:.0%} NaN — consider a wider download window."
                )

        # 11. (Warning) Tickers with very few rows
        if "ticker" in df.columns:
            thin_tickers = (
                df.groupby("ticker").size().loc[lambda s: s < 5].index.tolist()
            )
            if thin_tickers:
                warnings_.append(
                    f"Tickers with < 5 rows: {thin_tickers}."
                )

        return errors, warnings_

    def export_outputs(
        self,
        df: pd.DataFrame,
        ticker_results: list[TickerResult],
    ) -> dict[str, str]:
        """
        Write the final and intermediate datasets to disk.

        Outputs
        -------
        Always written:
            data/processed/market_data.parquet

        Written when ``config.export_csv=True``:
            data/processed/market_data.csv

        Written when ``config.save_intermediates=True``:
            data/interim/aligned_events.parquet

        Parameters
        ----------
        df : pd.DataFrame
            Final merged and validated dataset.
        ticker_results : list[TickerResult]
            Per-ticker results (used to save aligned_events subset).

        Returns
        -------
        dict[str, str]
            Mapping of output label → absolute file path.
        """
        output_paths: dict[str, str] = {}
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)

        if df.empty:
            logger.warning("Export called with empty DataFrame — skipping.")
            return output_paths

        # ── Primary parquet output ───────────────────────────────────
        parquet_path = self.config.processed_dir / "market_data.parquet"
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
        output_paths["market_data.parquet"] = str(parquet_path.resolve())
        logger.info("Exported %d rows → %s", len(df), parquet_path)

        # ── Optional CSV output ───────────────────────────────────────
        if self.config.export_csv:
            csv_path = self.config.processed_dir / "market_data.csv"
            df.to_csv(csv_path, index=False)
            output_paths["market_data.csv"] = str(csv_path.resolve())
            logger.info("Exported CSV → %s", csv_path)

        # ── Intermediate: aligned events ──────────────────────────────
        if self.config.save_intermediates:
            aligned_cols = [
                c for c in
                ["ticker", "transcript_id", "earnings_date",
                 "aligned_event_date", "date", "relative_day"]
                if c in df.columns
            ]
            if aligned_cols:
                aligned_path = self.config.interim_dir / "aligned_events.parquet"
                df[aligned_cols].to_parquet(
                    aligned_path, engine="pyarrow", compression="snappy", index=False
                )
                output_paths["aligned_events.parquet"] = str(aligned_path.resolve())
                logger.info("Saved aligned events → %s", aligned_path)

        return output_paths

    def generate_pipeline_summary(
        self,
        metadata_df: pd.DataFrame,
        final_df: pd.DataFrame,
        ticker_results: list[TickerResult],
        validation_errors: list[str],
        validation_warnings: list[str],
        output_paths: dict[str, str],
        run_start: float,
        run_id: str,
    ) -> PipelineRunSummary:
        """
        Assemble a ``PipelineRunSummary`` from all pipeline artefacts.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Input transcript metadata.
        final_df : pd.DataFrame
            Final exported DataFrame.
        ticker_results : list[TickerResult]
            Per-ticker processing outcomes.
        validation_errors : list[str]
            From ``validate_final_dataset()``.
        validation_warnings : list[str]
            From ``validate_final_dataset()``.
        output_paths : dict[str, str]
            From ``export_outputs()``.
        run_start : float
            ``time.perf_counter()`` value at run start.
        run_id : str
            ISO timestamp string identifying this run.

        Returns
        -------
        PipelineRunSummary
        """
        successful = [r.ticker for r in ticker_results if r.success]
        failed = [r.ticker for r in ticker_results if not r.success]

        date_start = date_end = None
        if not final_df.empty and "date" in final_df.columns:
            dates = pd.to_datetime(final_df["date"]).dropna()
            if len(dates):
                date_start = dates.min().date()
                date_end = dates.max().date()

        missing_summary = self.summarize_missing_data(final_df)

        return PipelineRunSummary(
            run_id=run_id,
            total_transcripts=len(metadata_df),
            total_tickers=len(ticker_results),
            successful_tickers=successful,
            failed_tickers=failed,
            total_rows_exported=len(final_df),
            date_range_start=date_start,
            date_range_end=date_end,
            missing_value_summary=missing_summary,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            elapsed_seconds=time.perf_counter() - run_start,
            output_paths=output_paths,
            ticker_results=ticker_results,
        )

    def run(
        self,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> PipelineRunSummary:
        """
        Execute the complete Day 5 market data pipeline end-to-end.

        This is the primary entry point.  Orchestrates all ten pipeline
        steps in sequence, isolates per-ticker failures, validates the
        final output, exports to disk, and returns a full run summary.

        Parameters
        ----------
        metadata_df : pd.DataFrame, optional
            Pre-loaded transcript metadata.  If omitted, loaded from
            ``config.transcript_metadata_path``.

        Returns
        -------
        PipelineRunSummary
            Complete run metrics, per-ticker results, and output paths.
        """
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_start = time.perf_counter()

        logger.info("=" * 68)
        logger.info("DAY 5 MARKET PIPELINE  START  [run_id=%s]", run_id)
        logger.info("=" * 68)

        # ── STEP 1: Load transcript metadata ─────────────────────────
        if metadata_df is None:
            logger.info("STEP 1 — Loading transcript metadata…")
            metadata_df = self.load_transcript_metadata()

        logger.info("STEP 1 — %d transcripts loaded.", len(metadata_df))

        # ── STEPS 2–7: Per-ticker processing ─────────────────────────
        logger.info("STEPS 2–7 — Downloading + processing all tickers…")
        final_df, ticker_results = self.build_market_dataset(metadata_df)

        # ── STEP 8: Merge ─────────────────────────────────────────────
        # (performed inside build_market_dataset → merge_all_tickers)
        logger.info("STEP 8 — Merge complete: %d rows.", len(final_df))

        # ── STEP 9: Validate ──────────────────────────────────────────
        logger.info("STEP 9 — Validating final dataset…")
        val_errors, val_warnings = self.validate_final_dataset(final_df)
        for e in val_errors:
            logger.error("  VALIDATION ERROR: %s", e)
        for w in val_warnings:
            logger.warning("  VALIDATION WARNING: %s", w)

        # ── STEP 10: Export ───────────────────────────────────────────
        logger.info("STEP 10 — Exporting outputs…")
        output_paths = self.export_outputs(final_df, ticker_results)

        # ── Summary ───────────────────────────────────────────────────
        summary = self.generate_pipeline_summary(
            metadata_df=metadata_df,
            final_df=final_df,
            ticker_results=ticker_results,
            validation_errors=val_errors,
            validation_warnings=val_warnings,
            output_paths=output_paths,
            run_start=run_start,
            run_id=run_id,
        )

        logger.info(
            "DAY 5 PIPELINE COMPLETE in %.1fs | %d rows | %d/%d tickers OK.",
            summary.elapsed_seconds,
            summary.total_rows_exported,
            len(summary.successful_tickers),
            summary.total_tickers,
        )

        return summary

    # ------------------------------------------------------------------
    # Helper / diagnostic methods
    # ------------------------------------------------------------------

    def detect_failed_tickers(
        self, ticker_results: list[TickerResult]
    ) -> list[dict[str, Any]]:
        """
        Return structured failure records for all failed tickers.

        Parameters
        ----------
        ticker_results : list[TickerResult]

        Returns
        -------
        list[dict]
            One dict per failed ticker with keys:
            ticker, transcript_id, earnings_date, error.
        """
        return [
            {
                "ticker": r.ticker,
                "transcript_id": r.transcript_id,
                "earnings_date": str(r.earnings_date),
                "error": r.error,
            }
            for r in ticker_results
            if not r.success
        ]

    def summarize_missing_data(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Compute per-column NaN fraction for all numeric feature columns.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict[str, float]
            Mapping of column → fraction NaN (0.0–1.0).
            Only columns with at least one NaN are included.
        """
        if df.empty:
            return {}
        non_id_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in {"relative_day"}
        ]
        if not non_id_cols:
            return {}
        frac = df[non_id_cols].isna().mean()
        return {col: round(float(v), 4) for col, v in frac.items() if v > 0}

    def generate_dataset_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return descriptive statistics for the final dataset.

        Excludes identifier columns; summarises all numeric features.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Transposed ``describe()`` output with mean/std/min/max.
        """
        if df.empty:
            return pd.DataFrame()
        exclude = {"relative_day", "transcript_id", "ticker"}
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        if not num_cols:
            return pd.DataFrame()
        return df[num_cols].describe().T[["mean", "std", "min", "max"]].round(6)

    def check_event_window_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Report continuity statistics for each event's trading-day window.

        For each (ticker, transcript_id) group, reports:
        - pre_days_available  : trading days with relative_day < 0
        - post_days_available : trading days with relative_day > 0
        - has_event_day       : whether relative_day=0 is present
        - total_days          : total rows in the event window

        Parameters
        ----------
        df : pd.DataFrame
            Final merged dataset with ``relative_day`` column.

        Returns
        -------
        pd.DataFrame
            One row per (ticker, transcript_id) event.
        """
        if df.empty or "relative_day" not in df.columns:
            return pd.DataFrame()

        group_cols = [c for c in ["ticker", "transcript_id"] if c in df.columns]
        if not group_cols:
            return pd.DataFrame()

        records = []
        for key, grp in df.groupby(group_cols):
            rd = grp["relative_day"]
            record: dict[str, Any] = {}
            if isinstance(key, tuple):
                for col, val in zip(group_cols, key):
                    record[col] = val
            else:
                record[group_cols[0]] = key

            record["pre_days_available"] = int((rd < 0).sum())
            record["post_days_available"] = int((rd > 0).sum())
            record["has_event_day"] = bool((rd == 0).any())
            record["total_days"] = len(grp)
            records.append(record)

        return pd.DataFrame(records).set_index(group_cols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_directories(self) -> None:
        """Create all required output directories if absent."""
        for d in (
            self.config.raw_market_dir,
            self.config.interim_dir,
            self.config.processed_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return cached parquet if available and use_cache=True, else None."""
        if not self.config.use_cache:
            return None
        cached = self._market_loader.load_parquet(ticker)
        if not cached.empty:
            logger.debug("%s: loaded from cache (%d rows).", ticker, len(cached))
            return cached
        return None

    def _download_benchmark(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Download benchmark data spanning the full event date range.

        Adds generous buffers so rolling-window features are fully
        warm at the earliest event date.
        """
        if self._benchmark_loader is None:
            logger.warning(
                "BenchmarkLoader not available — benchmark_return will be NaN."
            )
            return pd.DataFrame()

        dates = pd.to_datetime(metadata_df["earnings_date"])
        global_start = (
            dates.min() - pd.Timedelta(days=self.config.pre_event_calendar_days + 10)
        ).date()
        global_end = (
            dates.max() + pd.Timedelta(days=self.config.post_event_calendar_days + 5)
        ).date()

        logger.info(
            "Downloading benchmark (%s) %s → %s…",
            self.config.benchmark_ticker,
            global_start,
            global_end,
        )

        try:
            bench_df = self._benchmark_loader.download(global_start, global_end)
            logger.info(
                "Benchmark downloaded: %d rows.", len(bench_df)
            )
            return bench_df
        except Exception as exc:
            logger.error("Benchmark download failed: %s", exc)
            return pd.DataFrame()

    def _align_event_date(self, earnings_date: date) -> date:
        """
        Delegate event-date alignment to TradingCalendar.

        Falls back to the raw earnings_date if TradingCalendar is
        unavailable, logging a warning.  The standard assumption is
        after_close=True (most earnings are released post-market).
        """
        if self._trading_calendar is None:
            logger.debug(
                "TradingCalendar not available — using earnings_date as-is: %s.",
                earnings_date,
            )
            return earnings_date

        try:
            if hasattr(self._trading_calendar, "align_event_date"):
                return self._trading_calendar.align_event_date(
                    earnings_date, after_close=True
                )
            if hasattr(self._trading_calendar, "align_earnings_event"):
                result = self._trading_calendar.align_earnings_event(
                    earnings_date,
                    "America/New_York",
                )
                return pd.Timestamp(result.aligned_event_date).date()
            raise AttributeError("No supported event-alignment method found.")
        except Exception as exc:
            logger.warning(
                "Event date alignment failed for %s: %s — using raw date.",
                earnings_date,
                exc,
            )
            return earnings_date

    def _tag_relative_days(
        self, df: pd.DataFrame, aligned_event_date: date
    ) -> pd.DataFrame:
        """
        Add a ``relative_day`` integer column to the DataFrame.

        ``relative_day`` = trading-session offset from the aligned event
        date.  Row where ``date == aligned_event_date`` gets 0; rows
        before get negative integers; rows after get positive integers.

        This is a lightweight fallback used when EventAligner is absent.
        For full accuracy (handling non-consecutive trading days) the
        proper implementation lives in ``event_alignment.py``.
        """
        if self._event_aligner is not None:
            try:
                return self._event_aligner.tag_relative_days(
                    df, event_date_col="aligned_event_date"
                )
            except Exception as exc:
                logger.warning(
                    "EventAligner.tag_relative_days failed: %s — using fallback.",
                    exc,
                )

        # Fallback: rank-based relative day using the sorted DatetimeIndex
        df = df.copy()
        event_ts = pd.Timestamp(aligned_event_date)

        # Use the index if it's a DatetimeIndex, otherwise look for 'date' column
        if isinstance(df.index, pd.DatetimeIndex):
            date_series = df.index.to_series().dt.normalize()
        elif "date" in df.columns:
            date_series = pd.to_datetime(df["date"]).dt.normalize()
        else:
            df["relative_day"] = np.nan
            return df

        sorted_dates = date_series.sort_values().unique()
        event_ts_norm = event_ts.normalize()

        if event_ts_norm not in sorted_dates:
            # Event date not in window — use calendar-day difference as proxy
            df["relative_day"] = (date_series - event_ts_norm).dt.days
        else:
            # Rank-based: count trading sessions from the event date
            event_rank = np.searchsorted(sorted_dates, event_ts_norm)
            date_to_rank = {d: int(i - event_rank) for i, d in enumerate(sorted_dates)}
            df["relative_day"] = date_series.map(date_to_rank).values

        return df

    def _save_intermediate(self, df: pd.DataFrame, path: Path) -> None:
        """Save a DataFrame to an intermediate parquet path."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, engine="pyarrow", compression="snappy", index=True)
            logger.info("Intermediate saved → %s (%d rows)", path, len(df))
        except Exception as exc:
            logger.warning("Could not save intermediate %s: %s", path, exc)

    def _resolve_benchmark_loader(self) -> Optional[Any]:
        if _BENCHMARK_LOADER_AVAILABLE:
            try:
                cfg = BenchmarkConfig(
                    benchmark_ticker=self.config.benchmark_ticker,
                    retry_count=self.config.retry_count,
                    retry_delay=self.config.retry_delay,
                    output_dir=self.config.raw_market_dir / "benchmark",
                )
                return BenchmarkLoader(cfg)
            except Exception as exc:
                logger.warning("BenchmarkLoader init failed: %s", exc)
        return None

    def _resolve_trading_calendar(self) -> Optional[Any]:
        if _TRADING_CALENDAR_AVAILABLE:
            try:
                return TradingCalendarEngine()
            except Exception as exc:
                logger.warning("TradingCalendar init failed: %s", exc)
        return None

    def _resolve_event_aligner(self) -> Optional[Any]:
        if _EVENT_ALIGNER_AVAILABLE:
            try:
                aligner = EventAligner()
                if not hasattr(aligner, "tag_relative_days"):
                    logger.info(
                        "EventAligner has no tag_relative_days adapter; "
                        "using builder fallback for relative-day tagging."
                    )
                    return None
                return aligner
            except Exception as exc:
                logger.warning("EventAligner init failed: %s", exc)
        return None


# ===========================================================================
# Standalone helpers
# ===========================================================================


def _build_synthetic_metadata(tickers: list[str]) -> pd.DataFrame:
    """
    Build a minimal transcript_metadata DataFrame for demo / test purposes.

    Parameters
    ----------
    tickers : list[str]
        Stock symbols to include.

    Returns
    -------
    pd.DataFrame
        Synthetic metadata with realistic earnings dates.
    """
    import random

    random.seed(42)
    base_dates = [
        date(2024, 10, 25),
        date(2024, 10, 30),
        date(2024, 11, 1),
        date(2024, 10, 23),
    ]
    rows = []
    for i, ticker in enumerate(tickers):
        ed = base_dates[i % len(base_dates)]
        quarter = "Q3"
        year = 2024
        rows.append(
            {
                "transcript_id": f"{ticker}_{quarter}_{year}",
                "ticker": ticker,
                "earnings_date": ed,
                "quarter": quarter,
                "year": year,
                "company": f"{ticker} Inc.",
            }
        )
    return pd.DataFrame(rows)


def _build_synthetic_ohlcv(
    ticker: str,
    n_days: int = 65,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV + benchmark_return data for a ticker.

    Produces a DataFrame that mirrors the schema output by
    ``MarketDataLoader.standardize_columns()`` merged with
    ``BenchmarkLoader.merge_into()``, enabling end-to-end testing
    of all downstream feature engineering and orchestration logic
    without live API calls.
    """
    np.random.seed(seed)
    dates = pd.date_range("2024-09-01", periods=n_days, freq="B")
    prices = 150.0 * np.cumprod(1 + np.random.normal(0.0008, 0.016, n_days))
    bench_r = np.concatenate([[np.nan], np.random.normal(0.0004, 0.010, n_days - 1)])

    df = pd.DataFrame(
        {
            "ticker": ticker,
            "open": prices * (1 + np.random.uniform(-0.003, 0.003, n_days)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.008, n_days))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.008, n_days))),
            "close": prices,
            "adj_close": prices,
            "volume": np.random.randint(20_000_000, 80_000_000, n_days).astype(float),
            "benchmark_return": bench_r,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ===========================================================================
# Lightweight self-tests
# ===========================================================================


def _run_self_tests() -> None:
    """
    Verify orchestration logic without live API calls.

    Tests
    -----
    1. Single-ticker run via process_ticker() with synthetic data.
    2. Multi-ticker run via build_market_dataset() (monkey-patched loader).
    3. Export verification — parquet written and reloadable.
    4. Validation checks — duplicate detection, missing column detection.
    """
    import tempfile

    print("Running self-tests…")

    # ── Setup ─────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = MarketPipelineConfig(
            raw_market_dir=tmp_path / "raw/market",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
            transcript_metadata_path=tmp_path / "interim/transcript_metadata.parquet",
            use_cache=False,
            export_csv=True,
            save_intermediates=True,
        )

        builder = MarketDatasetBuilder(config=cfg)

        # Monkey-patch the market loader to return synthetic data
        def _fake_download_event_window(ticker, earnings_date, pre_days, post_days):
            seed = abs(hash(ticker)) % 1000
            return _build_synthetic_ohlcv(ticker, n_days=65, seed=seed)

        builder._market_loader.download_event_window = _fake_download_event_window

        earnings_date_obj = date(2024, 10, 30)

        # ── Test 1: Single-ticker ─────────────────────────────────────
        bench_df = pd.DataFrame()  # no benchmark in unit test
        result = builder.process_ticker(
            ticker="AAPL",
            earnings_date=earnings_date_obj,
            transcript_id="AAPL_Q3_2024",
            benchmark_df=bench_df,
        )
        assert result.success, f"Single-ticker test failed: {result.error}"
        assert result.row_count > 0, "Expected rows > 0."
        assert "daily_return" in result.df.columns, "daily_return missing."
        assert "relative_day" in result.df.columns, "relative_day missing."
        print("  ✓ Single-ticker process_ticker() passed.")

        # ── Test 2: Multi-ticker merge ────────────────────────────────
        tickers = ["AAPL", "MSFT", "NVDA"]
        metadata_df = _build_synthetic_metadata(tickers)
        merged_df, ticker_results = builder.build_market_dataset(metadata_df)

        assert not merged_df.empty, "Merged DataFrame should not be empty."
        assert len(ticker_results) == 3, f"Expected 3 results, got {len(ticker_results)}."
        successful = [r for r in ticker_results if r.success]
        assert len(successful) == 3, f"Expected 3 successes, got {len(successful)}."
        print("  ✓ Multi-ticker build_market_dataset() passed.")

        # ── Test 3: Export verification ───────────────────────────────
        output_paths = builder.export_outputs(merged_df, ticker_results)
        parquet_path = Path(output_paths.get("market_data.parquet", ""))
        assert parquet_path.exists(), f"Parquet not found at {parquet_path}."
        reloaded = pd.read_parquet(parquet_path)
        assert len(reloaded) == len(merged_df), "Row count mismatch after reload."

        csv_path = Path(output_paths.get("market_data.csv", ""))
        assert csv_path.exists(), f"CSV not found at {csv_path}."
        print("  ✓ export_outputs() parquet + CSV passed.")

        # ── Test 4: Validation detects duplicate rows ─────────────────
        bad_df = pd.concat([merged_df, merged_df.iloc[:5]], ignore_index=True)
        errors, _ = builder.validate_final_dataset(bad_df)
        assert any("uplicate" in e for e in errors), (
            "Validation should catch duplicate rows."
        )
        print("  ✓ validate_final_dataset() duplicate detection passed.")

        # ── Test 5: Validation detects missing required columns ───────
        bad_df2 = merged_df.drop(
            columns=["aligned_event_date"], errors="ignore"
        )
        errors2, _ = builder.validate_final_dataset(bad_df2)
        assert any("aligned_event_date" in e for e in errors2), (
            "Validation should catch missing aligned_event_date."
        )
        print("  ✓ validate_final_dataset() missing column detection passed.")

        # ── Test 6: Event window integrity report ────────────────────
        integrity = builder.check_event_window_integrity(merged_df)
        assert not integrity.empty, "Event window integrity report empty."
        print("  ✓ check_event_window_integrity() passed.")

    print("\nAll self-tests passed.")


# ===========================================================================
# CLI / demo entry point
# ===========================================================================


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    """
    Demo: run the Day 5 market pipeline end-to-end on synthetic data.

    Run with:
        python src/finance/market_dataset_builder.py

    In production, replace synthetic metadata + monkey-patch with:
        builder = MarketDatasetBuilder(config=MarketPipelineConfig.load_config(...))
        summary = builder.run()
        summary.print_report()
    """
    _setup_logging()

    metadata_path = Path("data/interim/transcript_metadata.parquet")
    config_path = Path("configs/market_config.yaml")
    if "--demo" not in sys.argv and metadata_path.exists():
        cfg = (
            MarketDatasetBuilder.load_config(config_path)
            if config_path.exists()
            else MarketPipelineConfig()
        )
        cfg.use_cache = False
        builder = MarketDatasetBuilder(config=cfg)
        summary = builder.run()
        summary.print_report()
        raise SystemExit(0)

    # ── Run self-tests first ─────────────────────────────────────────
    _run_self_tests()

    print("\n" + "=" * 68)
    print("DEMO — MarketDatasetBuilder end-to-end on synthetic data")
    print("=" * 68)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ── Config ────────────────────────────────────────────────────
        cfg = MarketPipelineConfig(
            raw_market_dir=tmp_path / "raw/market",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
            transcript_metadata_path=tmp_path / "interim/transcript_metadata.parquet",
            benchmark_ticker="^GSPC",
            return_horizons=[1, 2, 3, 5],
            rolling_vol_window=20,
            moving_average_windows=[20, 50],
            use_cache=False,
            export_csv=True,
            save_intermediates=True,
        )

        builder = MarketDatasetBuilder(config=cfg)

        # ── Synthetic metadata ────────────────────────────────────────
        tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
        metadata_df = _build_synthetic_metadata(tickers)

        print(f"\nTranscript metadata ({len(metadata_df)} rows):")
        print(metadata_df[["transcript_id", "ticker", "earnings_date"]].to_string())

        # ── Monkey-patch the loader for offline demo ──────────────────
        def _fake_window(ticker, earnings_date, pre_days, post_days):
            seed = abs(hash(ticker)) % 999
            return _build_synthetic_ohlcv(ticker, n_days=65, seed=seed)

        builder._market_loader.download_event_window = _fake_window

        # ── Run pipeline ──────────────────────────────────────────────
        summary = builder.run(metadata_df=metadata_df)
        summary.print_report()

        # ── Dataset statistics ────────────────────────────────────────
        parquet_out = cfg.processed_dir / "market_data.parquet"
        if parquet_out.exists():
            final_df = pd.read_parquet(parquet_out)

            print("--- Dataset Statistics (numeric features) ---")
            stats = builder.generate_dataset_statistics(final_df)
            pd.set_option("display.float_format", "{:.5f}".format)
            pd.set_option("display.max_columns", 10)
            pd.set_option("display.width", 110)
            print(stats.to_string())

            print("\n--- Event Window Integrity ---")
            integrity = builder.check_event_window_integrity(final_df)
            print(integrity.to_string())

            print("\n--- Sample rows (AAPL, relative_day -2 → +2) ---")
            aapl = final_df[final_df["ticker"] == "AAPL"].copy()
            window = aapl[aapl["relative_day"].between(-2, 2)]
            show_cols = [
                c for c in [
                    "ticker", "date", "relative_day",
                    "adj_close", "daily_return", "benchmark_return",
                    "abnormal_return", "rolling_volatility_20d",
                    "sma_20", "relative_volume",
                ]
                if c in window.columns
            ]
            print(window[show_cols].to_string(index=False))

        print("\nDemo complete.")
