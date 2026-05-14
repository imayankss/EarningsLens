"""
src/finance/market_data_loader.py
==================================
Production-grade OHLCV market data ingestion module.

Responsibilities (strictly scoped):
    - Download OHLCV data via yfinance
    - Support single-ticker and batch downloads
    - Support event-window downloads given an earnings date
    - Normalize and standardize column names
    - Validate returned DataFrames
    - Retry failed downloads with exponential back-off
    - Remove duplicates and normalize datetime index
    - Save / load Parquet files to data/raw/market/

Out of scope for this module:
    - Return calculations
    - Abnormal return computation
    - Event-date alignment / trading calendar logic
    - Feature engineering

Pipeline position:
    Transcript Metadata (ticker + earnings_date)
        → [THIS MODULE]  MarketDataLoader
        → data/raw/market/{TICKER}_market.parquet
        → benchmark_loader.py  (next stage)
        → trading_calendar.py  (event alignment)
        → feature_engineering.py

Author: Earnings Call Sentiment Analyzer — Day 5
Python: 3.11+
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)

COLUMN_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

# yfinance sometimes returns MultiIndex columns when auto_adjust=True.
# These are the adjusted-only column names in that case.
ADJUSTED_COLUMN_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

DEFAULT_OUTPUT_DIR = Path("data/raw/market")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class MarketDataConfig:
    """
    Configuration for the MarketDataLoader.

    Attributes
    ----------
    output_dir : Path
        Directory where raw parquet files are saved.
        Default: data/raw/market/
    auto_adjust : bool
        If True yfinance returns split/dividend-adjusted prices.
        Adj Close is always preferred; set True for cleaner OHLCV.
        Default: False  — we keep raw + Adj Close for transparency.
    save_raw : bool
        Persist each ticker's download as a parquet file.
        Default: True
    retry_count : int
        Number of download attempts before giving up.
        Default: 3
    retry_delay : float
        Base delay in seconds between retries (exponential back-off).
        Default: 2.0
    trading_buffer_before : int
        Calendar days to add before an earnings date when computing
        an event window.  30 calendar days ≈ ~20 trading days.
        Default: 45
    trading_buffer_after : int
        Calendar days to add after an earnings date.
        Default: 20
    timeout : int
        HTTP request timeout forwarded to yfinance (seconds).
        Default: 30
    """

    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    auto_adjust: bool = False
    save_raw: bool = True
    retry_count: int = 3
    retry_delay: float = 2.0
    trading_buffer_before: int = 45
    trading_buffer_after: int = 20
    timeout: int = 30

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


# ---------------------------------------------------------------------------
# Validation result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """
    Holds the outcome of a DataFrame validation pass.

    Attributes
    ----------
    is_valid : bool
        True if all checks passed.
    ticker : str
        Ticker symbol that was validated.
    errors : list[str]
        Human-readable descriptions of each failed check.
    warnings : list[str]
        Non-fatal issues observed during validation.
    row_count : int
        Number of rows in the validated DataFrame.
    """

    is_valid: bool
    ticker: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"[{status}] {self.ticker} — {self.row_count} rows"]
        for e in self.errors:
            lines.append(f"  ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"  WARNING : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core loader class
# ---------------------------------------------------------------------------
class MarketDataLoader:
    """
    Production-grade OHLCV downloader backed by yfinance.

    Responsibilities
    ----------------
    - Download OHLCV data for one or many tickers.
    - Provide an event-window helper that auto-computes date ranges
      from an earnings date + configured buffer.
    - Normalize column names to snake_case.
    - Validate data integrity (no empty frames, required columns,
      non-negative volumes, positive close prices, duplicate-free index).
    - Retry failed downloads with exponential back-off.
    - Persist outputs to Parquet and reload them.

    Parameters
    ----------
    config : MarketDataConfig, optional
        Loader configuration. Uses defaults if not provided.

    Examples
    --------
    >>> loader = MarketDataLoader()
    >>> df = loader.download_ticker("AAPL", "2024-01-01", "2024-06-30")
    >>> loader.save_parquet(df, "AAPL")

    >>> batch = loader.download_batch(
    ...     ["AAPL", "MSFT", "NVDA"],
    ...     start_date="2024-01-01",
    ...     end_date="2024-06-30",
    ... )

    >>> window_df = loader.download_event_window(
    ...     ticker="TSLA",
    ...     earnings_date=date(2025, 1, 29),
    ... )
    """

    def __init__(self, config: Optional[MarketDataConfig] = None) -> None:
        self.config = config or MarketDataConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "MarketDataLoader initialised | output_dir=%s | auto_adjust=%s",
            self.config.output_dir,
            self.config.auto_adjust,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_ticker(
        self,
        ticker: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a single ticker.

        Applies retry logic, column normalisation, duplicate removal,
        and validation before returning. Optionally saves to Parquet.

        Parameters
        ----------
        ticker : str
            Stock symbol, e.g. "AAPL", "BRK.B", "^GSPC".
        start_date : str | date | datetime
            Inclusive start of the download window (YYYY-MM-DD).
        end_date : str | date | datetime
            Inclusive end of the download window (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV DataFrame with columns:
            ticker, date, open, high, low, close, adj_close, volume.
            Returns an empty DataFrame if download fails entirely.

        Raises
        ------
        Does NOT raise — failures are logged and an empty DataFrame
        is returned so batch operations continue uninterrupted.
        """
        ticker = ticker.strip().upper()
        start_str = _to_date_str(start_date)
        end_str = _to_date_str(end_date)

        logger.info(
            "Downloading %s | %s → %s", ticker, start_str, end_str
        )

        raw_df = self._download_with_retry(ticker, start_str, end_str)

        if raw_df is None or raw_df.empty:
            logger.warning(
                "No data returned for %s (%s → %s). "
                "Ticker may be delisted or outside valid date range.",
                ticker,
                start_str,
                end_str,
            )
            return pd.DataFrame()

        df = self.standardize_columns(raw_df, ticker)
        df = self._remove_duplicates(df)
        df = self._normalize_datetime_index(df)

        result = self.validate_market_data(df, ticker)
        logger.info(result.summary())

        if not result.is_valid:
            logger.error(
                "Validation failed for %s — returning empty DataFrame.",
                ticker,
            )
            return pd.DataFrame()

        if self.config.save_raw:
            self.save_parquet(df, ticker)

        return df

    def download_batch(
        self,
        tickers: list[str],
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> dict[str, pd.DataFrame]:
        """
        Download OHLCV data for multiple tickers.

        Each ticker is downloaded independently so that a failure for
        one ticker does not block the rest of the batch.

        Parameters
        ----------
        tickers : list[str]
            List of stock symbols.
        start_date : str | date | datetime
            Inclusive start of the download window.
        end_date : str | date | datetime
            Inclusive end of the download window.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ticker → normalised OHLCV DataFrame.
            Tickers that failed are mapped to empty DataFrames.
        """
        tickers = [t.strip().upper() for t in tickers]
        logger.info(
            "Batch download: %d tickers | %s → %s",
            len(tickers),
            _to_date_str(start_date),
            _to_date_str(end_date),
        )

        results: dict[str, pd.DataFrame] = {}
        failed: list[str] = []

        for ticker in tickers:
            df = self.download_ticker(ticker, start_date, end_date)
            results[ticker] = df
            if df.empty:
                failed.append(ticker)

        if failed:
            logger.warning(
                "Batch complete — %d/%d tickers failed: %s",
                len(failed),
                len(tickers),
                ", ".join(failed),
            )
        else:
            logger.info(
                "Batch complete — all %d tickers succeeded.", len(tickers)
            )

        return results

    def download_event_window(
        self,
        ticker: str,
        earnings_date: str | date | datetime,
        pre_days: Optional[int] = None,
        post_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Download OHLCV data centred on an earnings event date.

        Automatically computes the date window from the earnings date
        plus the configured (or overridden) buffer days. This helper
        does NOT perform trading-calendar alignment — that responsibility
        belongs to trading_calendar.py.

        Parameters
        ----------
        ticker : str
            Stock symbol.
        earnings_date : str | date | datetime
            Date of the earnings call / event.
        pre_days : int, optional
            Calendar days before earnings_date. Overrides config default.
        post_days : int, optional
            Calendar days after earnings_date. Overrides config default.

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV DataFrame for the event window.
            Returns empty DataFrame on failure.

        Notes
        -----
        Buffer defaults (45 before / 20 after calendar days) are
        intentionally generous to ensure ~30/10 *trading* days are
        available after calendar / holiday filtering downstream.
        """
        pre = pre_days if pre_days is not None else self.config.trading_buffer_before
        post = post_days if post_days is not None else self.config.trading_buffer_after

        event_dt = _to_date_obj(earnings_date)
        start_date = event_dt - timedelta(days=pre)
        end_date = event_dt + timedelta(days=post)

        logger.info(
            "Event window for %s | earnings=%s | window=%s → %s "
            "(pre=%d days, post=%d days)",
            ticker,
            event_dt,
            start_date,
            end_date,
            pre,
            post,
        )

        df = self.download_ticker(ticker, start_date, end_date)

        if not df.empty:
            # Tag the source earnings date for traceability
            df["earnings_date"] = pd.Timestamp(event_dt)

        return df

    def validate_market_data(
        self, df: pd.DataFrame, ticker: str = ""
    ) -> ValidationResult:
        """
        Run integrity checks on a normalised OHLCV DataFrame.

        Checks performed
        ----------------
        1. DataFrame is not empty.
        2. All required columns exist (open, high, low, close,
           adj_close, volume).
        3. DatetimeIndex is valid (no NaT values).
        4. No duplicate dates.
        5. close prices are strictly positive.
        6. volume values are non-negative.
        7. Warns if any NaN values are present.
        8. Warns if row count is suspiciously low (< 5).

        Parameters
        ----------
        df : pd.DataFrame
            Normalised DataFrame (output of standardize_columns).
        ticker : str, optional
            Ticker label used in log messages.

        Returns
        -------
        ValidationResult
            Dataclass carrying is_valid flag, error list, and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Empty check
        if df is None or df.empty:
            errors.append("DataFrame is empty.")
            return ValidationResult(
                is_valid=False,
                ticker=ticker,
                errors=errors,
                warnings=warnings,
                row_count=0,
            )

        row_count = len(df)

        # 2. Required columns
        required = {"open", "high", "low", "close", "adj_close", "volume"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")

        # 3. DatetimeIndex validity
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append(
                f"Index is not DatetimeIndex — got {type(df.index).__name__}."
            )
        else:
            if df.index.isnull().any():
                errors.append("DatetimeIndex contains NaT values.")

        # 4. Duplicate dates
        duplicate_count = df.index.duplicated().sum()
        if duplicate_count > 0:
            errors.append(
                f"Duplicate dates found: {duplicate_count} duplicates."
            )

        # Remaining checks only if required columns are present
        if "close" in df.columns:
            # 5. Positive close prices
            non_positive = (df["close"] <= 0).sum()
            if non_positive > 0:
                errors.append(
                    f"close prices <= 0: {non_positive} rows."
                )

        if "volume" in df.columns:
            # 6. Non-negative volume
            negative_vol = (df["volume"] < 0).sum()
            if negative_vol > 0:
                errors.append(
                    f"Negative volume values: {negative_vol} rows."
                )

        # 7. NaN warnings
        nan_counts = df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            warnings.append(
                f"NaN values detected — {nan_cols.to_dict()}"
            )

        # 8. Low row-count warning
        if row_count < 5:
            warnings.append(
                f"Very few rows ({row_count}). "
                "Check date range or ticker validity."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            ticker=ticker,
            errors=errors,
            warnings=warnings,
            row_count=row_count,
        )

    def standardize_columns(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        Normalise yfinance output to the project's standard schema.

        Handles two yfinance output shapes:
          - Standard:  MultiIndex columns (Ticker, Field) — flattened.
          - Flat:      Single-level columns like "Open", "Adj Close".

        Normalised output columns:
            ticker, open, high, low, close, adj_close, volume

        The index is named ``date``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame returned by yfinance.
        ticker : str
            Ticker symbol — added as a column for downstream joins.

        Returns
        -------
        pd.DataFrame
            DataFrame with standardised schema and DatetimeIndex
            named ``date``.
        """
        df = df.copy()

        # --- Flatten MultiIndex columns (yfinance batch mode) -------
        if isinstance(df.columns, pd.MultiIndex):
            # MultiIndex is (field, ticker); keep only this ticker's slice
            if ticker.upper() in df.columns.get_level_values(1):
                df = df.xs(ticker.upper(), axis=1, level=1)
            else:
                # Try collapsing by joining levels
                df.columns = [
                    " ".join(filter(None, map(str, col))).strip()
                    for col in df.columns
                ]

        # --- Handle auto_adjust=True (no "Adj Close" column) --------
        # When auto_adjust=True yfinance returns adjusted OHLCV without
        # a separate "Adj Close"; treat "Close" as adj_close.
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
            logger.debug(
                "%s: 'Adj Close' missing (auto_adjust mode?); "
                "using 'Close' as adj_close.",
                ticker,
            )

        # --- Rename to snake_case ------------------------------------
        rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # --- Keep only the required normalised columns ---------------
        keep = [c for c in COLUMN_MAP.values() if c in df.columns]
        df = df[keep]

        # --- Add ticker column ---------------------------------------
        df.insert(0, "ticker", ticker.upper())

        # --- Normalise index -----------------------------------------
        df.index.name = "date"
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        # Strip timezone info — store as timezone-naive UTC dates
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        # Keep date component only (no intraday times)
        df.index = df.index.normalize()

        return df

    def save_parquet(self, df: pd.DataFrame, ticker: str) -> Path:
        """
        Persist a normalised OHLCV DataFrame to Parquet.

        File path pattern:
            {output_dir}/{TICKER}_market.parquet

        Parameters
        ----------
        df : pd.DataFrame
            Normalised DataFrame to persist.
        ticker : str
            Ticker symbol used in the filename.

        Returns
        -------
        Path
            Absolute path of the saved file.

        Raises
        ------
        IOError
            If the file cannot be written (permissions, disk full, etc.).
        """
        ticker = ticker.strip().upper()
        path = self.config.output_dir / f"{ticker}_market.parquet"

        try:
            df.to_parquet(path, engine="pyarrow", compression="snappy")
            logger.info(
                "Saved %s → %s (%d rows)", ticker, path, len(df)
            )
        except Exception as exc:
            logger.error("Failed to save parquet for %s: %s", ticker, exc)
            raise

        return path

    def load_parquet(self, ticker: str) -> pd.DataFrame:
        """
        Load a previously saved OHLCV Parquet file.

        Parameters
        ----------
        ticker : str
            Ticker symbol — resolved to {output_dir}/{TICKER}_market.parquet.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame with DatetimeIndex named ``date``.
            Returns empty DataFrame if the file does not exist.
        """
        ticker = ticker.strip().upper()
        path = self.config.output_dir / f"{ticker}_market.parquet"

        if not path.exists():
            logger.warning(
                "Parquet not found for %s at %s.", ticker, path
            )
            return pd.DataFrame()

        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded %s from %s (%d rows)", ticker, path, len(df))
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_with_retry(
        self,
        ticker: str,
        start_str: str,
        end_str: str,
    ) -> pd.DataFrame | None:
        """
        Attempt to download data from yfinance with exponential back-off.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        start_str : str
            Start date as "YYYY-MM-DD".
        end_str : str
            End date as "YYYY-MM-DD".

        Returns
        -------
        pd.DataFrame | None
            Raw yfinance output, or None if all attempts fail.
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.config.retry_count + 1):
            try:
                raw = yf.download(
                    ticker,
                    start=start_str,
                    end=end_str,
                    auto_adjust=self.config.auto_adjust,
                    progress=False,
                    threads=False,
                    timeout=self.config.timeout,
                )

                if raw is not None and not raw.empty:
                    logger.debug(
                        "%s download succeeded on attempt %d (%d rows).",
                        ticker,
                        attempt,
                        len(raw),
                    )
                    return raw

                logger.warning(
                    "%s returned empty on attempt %d/%d.",
                    ticker,
                    attempt,
                    self.config.retry_count,
                )

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "%s download error on attempt %d/%d: %s",
                    ticker,
                    attempt,
                    self.config.retry_count,
                    exc,
                )

            if attempt < self.config.retry_count:
                delay = self.config.retry_delay * (2 ** (attempt - 1))
                logger.debug(
                    "Waiting %.1fs before retry %d for %s.",
                    delay,
                    attempt + 1,
                    ticker,
                )
                time.sleep(delay)

        if last_exception:
            logger.error(
                "All %d attempts failed for %s. Last error: %s",
                self.config.retry_count,
                ticker,
                last_exception,
            )

        return None

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate index entries, keeping the last occurrence.

        Duplicate dates can arise from timezone normalisation or
        malformed yfinance responses.
        """
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        removed = before - len(df)
        if removed:
            logger.warning("Removed %d duplicate date(s).", removed)
        return df

    @staticmethod
    def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the index is a tz-naive DatetimeIndex sorted ascending.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df.sort_index()
        return df


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def _to_date_str(dt: str | date | datetime) -> str:
    """Convert various date types to 'YYYY-MM-DD' string."""
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    if isinstance(dt, date):
        return dt.strftime("%Y-%m-%d")
    raise TypeError(f"Cannot convert {type(dt)} to date string.")


def _to_date_obj(dt: str | date | datetime) -> date:
    """Convert various date types to a ``datetime.date`` object."""
    if isinstance(dt, datetime):
        return dt.date()
    if isinstance(dt, date):
        return dt
    if isinstance(dt, str):
        return datetime.strptime(dt, "%Y-%m-%d").date()
    raise TypeError(f"Cannot convert {type(dt)} to date object.")


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
    Demo: download OHLCV data for a selection of tickers.

    Run with:
        python src/finance/market_data_loader.py

    Outputs saved to:
        data/raw/market/{TICKER}_market.parquet
    """
    _setup_logging()

    # -----------------------------------------------------------------
    # 1. Basic configuration
    # -----------------------------------------------------------------
    config = MarketDataConfig(
        output_dir=Path("data/raw/market"),
        auto_adjust=False,   # keep Adj Close alongside raw OHLCV
        save_raw=True,
        retry_count=3,
        retry_delay=2.0,
        trading_buffer_before=45,
        trading_buffer_after=20,
    )

    loader = MarketDataLoader(config=config)

    # -----------------------------------------------------------------
    # 2. Single ticker download
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO 1 — Single ticker download")
    print("=" * 60)

    aapl_df = loader.download_ticker(
        ticker="AAPL",
        start_date="2024-10-01",
        end_date="2025-03-31",
    )

    if not aapl_df.empty:
        print(f"\nAAPL — {len(aapl_df)} rows downloaded")
        print(aapl_df.tail(5).to_string())
    else:
        print("AAPL download returned no data.")

    # -----------------------------------------------------------------
    # 3. Batch download
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO 2 — Batch ticker download")
    print("=" * 60)

    batch_results = loader.download_batch(
        tickers=["MSFT", "NVDA", "BRK-B"],   # BRK.B → BRK-B in yfinance
        start_date="2024-10-01",
        end_date="2025-03-31",
    )

    for ticker, df in batch_results.items():
        status = f"{len(df)} rows" if not df.empty else "EMPTY / FAILED"
        print(f"  {ticker:<8} → {status}")

    # -----------------------------------------------------------------
    # 4. Event-window download (earnings-centric)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO 3 — Event-window download (earnings-centric)")
    print("=" * 60)

    tsla_event_df = loader.download_event_window(
        ticker="TSLA",
        earnings_date=date(2025, 1, 29),   # TSLA Q4 2024 earnings
        pre_days=45,
        post_days=20,
    )

    if not tsla_event_df.empty:
        print(f"\nTSLA event window — {len(tsla_event_df)} rows")
        print(f"Date range: {tsla_event_df.index.min().date()} "
              f"→ {tsla_event_df.index.max().date()}")
        print(tsla_event_df.head(3).to_string())
    else:
        print("TSLA event window returned no data.")

    # -----------------------------------------------------------------
    # 5. Load from parquet cache
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO 4 — Load from parquet cache")
    print("=" * 60)

    cached_df = loader.load_parquet("AAPL")
    if not cached_df.empty:
        print(f"\nLoaded AAPL from cache — {len(cached_df)} rows")
        print(cached_df.dtypes.to_string())
    else:
        print("No cached file found for AAPL.")

    # -----------------------------------------------------------------
    # 6. Standalone validation example
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO 5 — Standalone validation")
    print("=" * 60)

    if not aapl_df.empty:
        result = loader.validate_market_data(aapl_df, ticker="AAPL")
        print(result.summary())

    print("\nDemo complete.")
