"""
src/finance/benchmark_loader.py
================================
Benchmark market data loader for the earnings call sentiment analysis pipeline.

Responsibilities
----------------
- Download S&P 500 (or alternate benchmark) OHLCV data via yfinance
- Compute daily benchmark returns from adjusted close prices
- Validate downloaded data for quality and completeness
- Cache results to Parquet for deterministic, fast re-use
- Expose an alignment frame ready to be joined against stock-level data

What this module does NOT do
-----------------------------
- Compute abnormal returns          (→ abnormal_returns.py)
- Merge with individual stock data  (→ event_alignment.py)
- Calculate rolling volatility      (→ feature_engineering.py)
- Orchestrate the full pipeline     (→ market_dataset_builder.py)

Usage
-----
    python -m src.finance.benchmark_loader          # demo run via __main__
    loader = BenchmarkLoader(BenchmarkConfig())     # programmatic
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    """
    All tunable parameters for the BenchmarkLoader.

    Attributes
    ----------
    benchmark_ticker : str
        Primary benchmark symbol.  Default ``^GSPC`` (S&P 500 index).
        Supported alternatives: ``SPY``, ``QQQ``.
    start_date : str
        Inclusive download start date in ``YYYY-MM-DD`` format.
    end_date : str
        Inclusive download end date in ``YYYY-MM-DD`` format.
    retry_count : int
        Number of retry attempts on transient yfinance failures.
    retry_delay : float
        Seconds to wait between retries.
    auto_adjust : bool
        Whether yfinance should return split/dividend-adjusted prices.
        Must be ``True`` for correct return calculations.
    save_raw : bool
        Persist the raw (pre-validation) download alongside the processed file.
    output_dir : Path
        Root directory for saved Parquet files.
    filename : str
        Base filename (without extension) used for the processed Parquet.
    """

    benchmark_ticker: str = "^GSPC"
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    retry_count: int = 3
    retry_delay: float = 2.0
    auto_adjust: bool = True
    save_raw: bool = False
    output_dir: Path = field(
        default_factory=lambda: Path("data/raw/market/benchmark")
    )
    filename: str = "sp500_benchmark"

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def parquet_path(self) -> Path:
        """Full path to the processed Parquet file."""
        return self.output_dir / f"{self.filename}.parquet"

    @property
    def raw_parquet_path(self) -> Path:
        """Full path to the raw (unvalidated) Parquet file."""
        return self.output_dir / f"{self.filename}_raw.parquet"


# ---------------------------------------------------------------------------
# Required output columns
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: list[str] = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "benchmark_return",
]


# ---------------------------------------------------------------------------
# BenchmarkLoader
# ---------------------------------------------------------------------------
class BenchmarkLoader:
    """
    Downloads, validates, caches, and exposes S&P 500 benchmark data.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration object controlling all loader behaviour.

    Examples
    --------
    >>> cfg = BenchmarkConfig(start_date="2024-01-01", end_date="2024-12-31")
    >>> loader = BenchmarkLoader(cfg)
    >>> df = loader.download_benchmark()
    >>> df = loader.compute_returns(df)
    >>> loader.validate_benchmark_data(df)
    >>> loader.save_parquet(df)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config: BenchmarkConfig = config or BenchmarkConfig()
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.info(
            "BenchmarkLoader initialised | ticker=%s | window=%s → %s",
            self.config.benchmark_ticker,
            self.config.start_date,
            self.config.end_date,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_benchmark(self) -> pd.DataFrame:
        """
        Download OHLCV data for the configured benchmark ticker.

        Applies retry logic on failure, normalises column names to lowercase,
        removes timezone information from the index, drops duplicates, and
        sorts ascending by date.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``open``, ``high``, ``low``, ``close``,
            ``adj_close``, ``volume``.
            Index is a plain ``RangeIndex``; the trading date lives in the
            ``date`` column.

        Raises
        ------
        RuntimeError
            When all retry attempts fail or the downloaded frame is empty.
        """
        ticker = self.config.benchmark_ticker
        self._log.info("Downloading benchmark data for %s …", ticker)

        raw: Optional[pd.DataFrame] = None
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.retry_count + 1):
            try:
                raw = yf.download(
                    tickers=ticker,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=self.config.auto_adjust,
                    progress=False,
                    threads=False,
                )
                if raw is not None and not raw.empty:
                    self._log.info(
                        "Download succeeded on attempt %d | rows=%d",
                        attempt,
                        len(raw),
                    )
                    break
                self._log.warning("Attempt %d returned empty data.", attempt)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._log.warning(
                    "Attempt %d failed: %s — retrying in %.1fs …",
                    attempt,
                    exc,
                    self.config.retry_delay,
                )
                time.sleep(self.config.retry_delay)

        if raw is None or raw.empty:
            raise RuntimeError(
                f"Failed to download benchmark data for '{ticker}' "
                f"after {self.config.retry_count} attempt(s). "
                f"Last error: {last_exc}"
            )

        df = self._normalise_frame(raw)

        if self.config.save_raw:
            self._persist_parquet(df, self.config.raw_parquet_path, label="raw")

        return df

    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily benchmark returns from adjusted close prices.

        Formula
        -------
        ``benchmark_return_t = (adj_close_t / adj_close_{t-1}) - 1``

        The first row receives ``NaN`` because no prior price exists.

        Parameters
        ----------
        df : pd.DataFrame
            Frame produced by :meth:`download_benchmark`.

        Returns
        -------
        pd.DataFrame
            Input frame augmented with the ``benchmark_return`` column.

        Raises
        ------
        KeyError
            When ``adj_close`` column is absent.
        """
        if "adj_close" not in df.columns:
            raise KeyError(
                "'adj_close' column not found. "
                "Ensure download_benchmark() was called with auto_adjust=True."
            )

        df = df.copy()
        df["benchmark_return"] = df["adj_close"].pct_change()

        non_null = df["benchmark_return"].notna().sum()
        self._log.info(
            "benchmark_return computed | non-null rows=%d / %d",
            non_null,
            len(df),
        )
        return df

    def validate_benchmark_data(self, df: pd.DataFrame) -> None:
        """
        Run quality checks on the benchmark DataFrame.

        Checks performed
        ----------------
        1. DataFrame is not empty.
        2. All required columns are present.
        3. No missing (NaN) dates.
        4. No duplicate dates.
        5. All close prices are strictly positive.
        6. Absolute benchmark returns are < 1.0 (sanity bound).

        Parameters
        ----------
        df : pd.DataFrame
            Frame to validate (must include the ``benchmark_return`` column).

        Raises
        ------
        ValueError
            On any validation failure; message describes the specific issue.
        """
        self._log.info("Running benchmark validation …")

        # 1. Non-empty
        if df.empty:
            raise ValueError("Benchmark DataFrame is empty.")

        # 2. Required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Present: {list(df.columns)}"
            )

        # 3. No missing dates
        null_dates = df["date"].isnull().sum()
        if null_dates > 0:
            raise ValueError(f"Found {null_dates} missing date value(s).")

        # 4. No duplicate dates
        dup_dates = df.duplicated(subset=["date"]).sum()
        if dup_dates > 0:
            raise ValueError(
                f"Found {dup_dates} duplicate date(s) in benchmark data."
            )

        # 5. Positive close prices
        non_positive = (df["close"] <= 0).sum()
        if non_positive > 0:
            raise ValueError(
                f"Found {non_positive} row(s) with close price ≤ 0."
            )

        # 6. Return magnitude sanity check (excludes NaN first row)
        returns = df["benchmark_return"].dropna()
        extreme = (returns.abs() >= 1.0).sum()
        if extreme > 0:
            raise ValueError(
                f"Found {extreme} benchmark_return value(s) with |return| ≥ 1.0. "
                "This likely indicates a data error."
            )

        self._log.info(
            "Validation passed | rows=%d | date_range=[%s, %s]",
            len(df),
            df["date"].min(),
            df["date"].max(),
        )

    def save_parquet(self, df: pd.DataFrame) -> Path:
        """
        Persist the processed benchmark DataFrame to Parquet.

        The output directory is created automatically if it does not exist.

        Parameters
        ----------
        df : pd.DataFrame
            Validated benchmark DataFrame.

        Returns
        -------
        Path
            Absolute path of the written Parquet file.
        """
        return self._persist_parquet(df, self.config.parquet_path, label="processed")

    def load_parquet(self) -> pd.DataFrame:
        """
        Load a previously cached benchmark Parquet file from disk.

        Returns
        -------
        pd.DataFrame
            Cached benchmark DataFrame with ``date`` parsed as ``datetime64``.

        Raises
        ------
        FileNotFoundError
            When the expected Parquet file does not exist.
        """
        path = self.config.parquet_path
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark Parquet not found at '{path}'. "
                "Run download_benchmark() + save_parquet() first."
            )

        self._log.info("Loading benchmark data from cache: %s", path)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        self._log.info("Loaded %d rows from cache.", len(df))
        return df

    def prepare_alignment_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a minimal alignment frame suitable for left-joining onto
        stock-level event data.

        Columns retained
        ----------------
        - ``date``             — trading date (join key)
        - ``benchmark_return`` — daily S&P 500 return

        The frame is sorted ascending and the index is reset so merges
        behave predictably.

        Parameters
        ----------
        df : pd.DataFrame
            Full benchmark DataFrame (post-validation).

        Returns
        -------
        pd.DataFrame
            Alignment frame with columns ``date`` and ``benchmark_return``.
        """
        required = {"date", "benchmark_return"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"Cannot build alignment frame; missing columns: {missing}"
            )

        alignment = (
            df[["date", "benchmark_return"]]
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        self._log.info(
            "Alignment frame ready | rows=%d | date_range=[%s, %s]",
            len(alignment),
            alignment["date"].min(),
            alignment["date"].max(),
        )
        return alignment

    def download(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        """
        Compatibility adapter for the master market pipeline.

        ``MarketDatasetBuilder`` asks benchmark loaders for a date-bounded
        frame with ``benchmark_return``. The loader's native API stores the
        date window in ``BenchmarkConfig``, so this method updates that window
        for the current request, downloads, validates, persists, and returns
        the minimal alignment frame.
        """
        self.config.start_date = pd.Timestamp(start_date).date().isoformat()
        self.config.end_date = pd.Timestamp(end_date).date().isoformat()

        df = self.download_benchmark()
        df = self.compute_returns(df)
        self.validate_benchmark_data(df)
        self.save_parquet(df)
        return self.prepare_alignment_frame(df)

    def merge_into(
        self,
        stock_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join ``benchmark_return`` onto stock data by trading date.

        Accepts stock dates either in a ``DatetimeIndex`` or in a ``date``
        column. The stock frame's original index/shape is preserved.
        """
        merged = stock_df.copy()
        if benchmark_df.empty:
            merged["benchmark_return"] = pd.NA
            return merged

        bench = benchmark_df.copy()
        if "date" in bench.columns:
            bench_dates = pd.to_datetime(bench["date"]).dt.normalize()
        else:
            bench_dates = pd.to_datetime(bench.index).normalize()
        bench = bench.assign(_join_date=bench_dates)
        bench = bench.drop_duplicates("_join_date", keep="last")
        bench_map = bench.set_index("_join_date")["benchmark_return"].to_dict()

        if isinstance(merged.index, pd.DatetimeIndex):
            join_dates = pd.to_datetime(merged.index).normalize()
        elif "date" in merged.columns:
            join_dates = pd.to_datetime(merged["date"]).dt.normalize()
        else:
            raise KeyError("stock_df must have a DatetimeIndex or a 'date' column.")

        merged["benchmark_return"] = pd.Series(
            pd.Index(join_dates).to_numpy(), index=merged.index
        ).map(bench_map)
        merged["benchmark_return"] = merged["benchmark_return"].ffill(limit=1)
        return merged

    # ------------------------------------------------------------------
    # Convenience orchestration helper
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Full pipeline: download → compute returns → validate → save → return.

        This method loads from cache if the Parquet already exists, avoiding
        redundant network requests.

        Returns
        -------
        pd.DataFrame
            Validated, processed benchmark DataFrame.
        """
        if self.config.parquet_path.exists():
            self._log.info(
                "Cache hit — loading from %s (delete to force re-download).",
                self.config.parquet_path,
            )
            return self.load_parquet()

        df = self.download_benchmark()
        df = self.compute_returns(df)
        self.validate_benchmark_data(df)
        self.save_parquet(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the raw yfinance DataFrame into the canonical schema.

        Steps
        -----
        1. Flatten MultiIndex columns produced by yfinance (if present).
        2. Rename columns to snake_case.
        3. Strip timezone info from the DatetimeIndex → plain ``date`` column.
        4. Remove duplicate dates (keep first occurrence).
        5. Sort ascending by date.
        6. Reset to a clean RangeIndex.
        """
        df = raw.copy()

        # 1. Flatten MultiIndex (yfinance ≥ 0.2 may return one)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # 2. Rename to snake_case
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)

        # When auto_adjust=True yfinance already adjusts all OHLC columns and
        # drops the "Adj Close" column — in that case "Close" IS adjusted close.
        if "adj_close" not in df.columns and "close" in df.columns:
            self._log.debug(
                "adj_close absent (auto_adjust=True); aliasing close → adj_close."
            )
            df["adj_close"] = df["close"]

        # Keep only the columns we care about
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"]
                if c in df.columns]
        df = df[keep]

        # 3. Strip timezone → plain date column
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()          # midnight
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()   # belt-and-braces

        # 4. Drop duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["date"], keep="first")
        dropped = before - len(df)
        if dropped:
            self._log.warning("Dropped %d duplicate date row(s).", dropped)

        # 5. Sort ascending
        df = df.sort_values("date").reset_index(drop=True)

        self._log.debug(
            "Frame normalised | rows=%d | cols=%s", len(df), list(df.columns)
        )
        return df

    def _persist_parquet(
        self, df: pd.DataFrame, path: Path, label: str = "file"
    ) -> Path:
        """Write *df* to *path* as Parquet, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, str(path), compression="snappy")
        self._log.info(
            "Saved %s Parquet | rows=%d | path=%s", label, len(df), path
        )
        return path


# ---------------------------------------------------------------------------
# __main__ — demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 65)
    print("  BenchmarkLoader — demo run")
    print("=" * 65)

    # ── 1. Default config (S&P 500, recent 2-year window) ─────────────
    cfg = BenchmarkConfig(
        benchmark_ticker="^GSPC",
        start_date="2023-01-01",
        end_date="2024-12-31",
        save_raw=False,
        output_dir=Path("data/raw/market/benchmark"),
    )

    loader = BenchmarkLoader(cfg)

    # ── 2. Run full pipeline ───────────────────────────────────────────
    print("\n[1/4] Downloading & processing benchmark …")
    df = loader.run()

    # ── 3. Inspect output ─────────────────────────────────────────────
    print("\n[2/4] Sample output (first 5 rows):")
    print(df.head().to_string(index=False))

    print("\n[3/4] Sample output (last 5 rows):")
    print(df.tail().to_string(index=False))

    print(f"\n[4/4] Summary statistics:")
    desc = df[["adj_close", "benchmark_return"]].describe()
    print(desc.to_string())

    # ── 4. Alignment frame ────────────────────────────────────────────
    alignment = loader.prepare_alignment_frame(df)
    print(f"\nAlignment frame shape : {alignment.shape}")
    print(alignment.head(3).to_string(index=False))

    # ── 5. SPY alternative ────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Alternative benchmark demo: SPY ETF")
    print("─" * 65)

    spy_cfg = BenchmarkConfig(
        benchmark_ticker="SPY",
        start_date="2023-01-01",
        end_date="2024-12-31",
        output_dir=Path("data/raw/market/benchmark"),
        filename="spy_benchmark",
    )
    spy_loader = BenchmarkLoader(spy_cfg)
    spy_df = spy_loader.run()
    print(f"SPY rows downloaded   : {len(spy_df)}")
    print(f"SPY return stats:\n{spy_df['benchmark_return'].describe()}")

    print("\n✅  Demo complete.")
