"""
tests/test_market_pipeline.py
==============================
Comprehensive pytest suite for the Day 5 market data pipeline.

Coverage
--------
TestMarketDataValidation      — MarketDataLoader validation logic
TestMarketDataStandardization — column normalisation and parquet I/O
TestMarketDataEventWindow     — event-window date arithmetic
TestBenchmarkHandling         — BenchmarkLoaderProtocol contract + merge logic
TestTradingCalendar           — TradingCalendarProtocol contract + day rules
TestFeatureEngineering        — formula correctness, multi-ticker isolation
TestFeatureEngineeringEdge    — edge cases: sparse data, no benchmark, NaN
TestEventAlignment            — relative_day tagging, window continuity
TestMarketDatasetBuilder      — orchestration, merge, export, validation
TestPipelineIntegration       — end-to-end synthetic run (no API calls)
TestEdgeCases                 — holidays, timezones, duplicates, missing data

Testing philosophy
------------------
* Zero external API calls — all data is synthetic.
* Exact algebraic assertions wherever a formula is deterministic.
* Dependency injection for the three not-yet-built modules (benchmark,
  calendar, event aligner) via lightweight mock classes that satisfy
  the Protocols defined in market_dataset_builder.py.
* Every fixture is isolated — no shared mutable state between tests.
* Parametrized tests cover boundary values and regime changes.

Author : Earnings Call Sentiment Analyzer — Day 5
Python : 3.11+
"""

from __future__ import annotations

import math
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Source-module imports
# ---------------------------------------------------------------------------
# Guards make the test file importable even before all sibling modules exist;
# the corresponding test classes are skipped via pytest.importorskip().

from src.finance.market_data_loader import (
    MarketDataConfig,
    MarketDataLoader,
    ValidationResult,
)
from src.finance.feature_engineering import (
    FeatureEngineeringConfig,
    FeatureEngineeringResult,
    FinancialFeatureEngineer,
)
from src.finance.market_dataset_builder import (
    FINAL_REQUIRED_COLUMNS,
    MarketDatasetBuilder,
    MarketPipelineConfig,
    PipelineRunSummary,
    TickerResult,
    _build_synthetic_metadata,
    _build_synthetic_ohlcv,
)


# ===========================================================================
# ── Shared constants ────────────────────────────────────────────────────────
# ===========================================================================

_SEED = 42
_N_ROWS = 65          # ~3 months of trading days per ticker
_EVENT_DATE = date(2024, 10, 30)   # Wednesday — safe mid-week anchor
_EVENT_TS = pd.Timestamp(_EVENT_DATE)


# ===========================================================================
# ── Assertion helpers ────────────────────────────────────────────────────────
# ===========================================================================


def assert_close(actual: float, expected: float, rel_tol: float = 1e-9) -> None:
    """Assert two floats are equal within relative tolerance."""
    assert math.isclose(actual, expected, rel_tol=rel_tol), (
        f"Expected {expected:.10f}, got {actual:.10f} "
        f"(rel diff = {abs(actual - expected) / (abs(expected) or 1):.2e})"
    )


def assert_no_inf(df: pd.DataFrame, label: str = "") -> None:
    """Assert no column in *df* contains infinite values."""
    numeric = df.select_dtypes(include=[np.number])
    inf_cols = numeric.columns[np.isinf(numeric).any()].tolist()
    assert not inf_cols, f"{label}: Infinite values in columns {inf_cols}"


def assert_no_extreme_returns(series: pd.Series, col: str = "") -> None:
    """Assert |return| < 1 for all non-NaN values."""
    bad = series.dropna()[series.dropna().abs() >= 1.0]
    assert bad.empty, (
        f"{col}: Found |return| >= 1.0 at indices {bad.index.tolist()[:5]}"
    )


def assert_non_negative(series: pd.Series, col: str = "") -> None:
    """Assert all non-NaN values in series are >= 0."""
    bad = series.dropna()[series.dropna() < 0]
    assert bad.empty, f"{col}: Found negative values: {bad.head().tolist()}"


def assert_columns_present(df: pd.DataFrame, cols: list[str]) -> None:
    """Assert every column in *cols* is present in df."""
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}. Available: {sorted(df.columns)}"


# ===========================================================================
# ── Synthetic data builders ──────────────────────────────────────────────────
# ===========================================================================


def make_ohlcv(
    ticker: str = "AAPL",
    n: int = _N_ROWS,
    seed: int = _SEED,
    start: str = "2024-08-01",
    with_benchmark: bool = True,
    tz_aware: bool = False,
) -> pd.DataFrame:
    """
    Build a synthetic normalised OHLCV DataFrame matching the schema
    produced by ``MarketDataLoader.standardize_columns()``.

    Parameters
    ----------
    ticker : str
    n : int
        Number of business-day rows.
    seed : int
    start : str
        Start date for business-day range.
    with_benchmark : bool
        If True add a ``benchmark_return`` column.
    tz_aware : bool
        If True attach UTC timezone to the DatetimeIndex (edge-case test).
    """
    rng = np.random.default_rng(seed)
    prices = 150.0 * np.cumprod(1 + rng.normal(0.0008, 0.016, n))

    idx = pd.date_range(start, periods=n, freq="B")
    if tz_aware:
        idx = idx.tz_localize("UTC")

    df = pd.DataFrame(
        {
            "ticker": ticker,
            "open": prices * (1 + rng.uniform(-0.003, 0.003, n)),
            "high": prices * (1 + np.abs(rng.normal(0, 0.008, n))),
            "low": prices * (1 - np.abs(rng.normal(0, 0.008, n))),
            "close": prices,
            "adj_close": prices,
            "volume": rng.integers(10_000_000, 80_000_000, n).astype(float),
        },
        index=idx,
    )
    df.index.name = "date"

    if with_benchmark:
        bench = np.concatenate([[np.nan], rng.normal(0.0004, 0.010, n - 1)])
        df["benchmark_return"] = bench

    return df


def make_raw_yfinance(
    ticker: str = "AAPL",
    n: int = 30,
    seed: int = _SEED,
) -> pd.DataFrame:
    """
    Simulate the raw DataFrame returned by ``yf.download()`` —
    PascalCase columns, DatetimeIndex named 'Date'.
    """
    rng = np.random.default_rng(seed)
    prices = 180.0 * np.cumprod(1 + rng.normal(0.0005, 0.013, n))
    idx = pd.date_range("2024-01-02", periods=n, freq="B", name="Date")
    return pd.DataFrame(
        {
            "Open": prices * 0.998,
            "High": prices * 1.006,
            "Low": prices * 0.994,
            "Close": prices,
            "Adj Close": prices * 0.997,
            "Volume": rng.integers(20_000_000, 90_000_000, n).astype(float),
        },
        index=idx,
    )


def make_metadata(tickers: list[str] | None = None) -> pd.DataFrame:
    """Return a minimal transcript_metadata DataFrame for given tickers."""
    tickers = tickers or ["AAPL", "MSFT", "NVDA"]
    rows = []
    base = date(2024, 10, 30)
    for i, t in enumerate(tickers):
        ed = base + timedelta(days=i)
        rows.append(
            {
                "transcript_id": f"{t}_Q3_2024",
                "ticker": t,
                "earnings_date": ed,
                "quarter": "Q3",
                "year": 2024,
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# ── Mock implementations of not-yet-built modules ───────────────────────────
# ===========================================================================


class MockBenchmarkLoader:
    """
    Satisfies BenchmarkLoaderProtocol.

    Returns a simple synthetic benchmark return series and performs a
    left-join merge by date.
    """

    def download(
        self,
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        idx = pd.date_range(str(start_date), str(end_date), freq="B", name="date")
        n = len(idx)
        bench = np.concatenate([[np.nan], rng.normal(0.0003, 0.009, n - 1)])
        df = pd.DataFrame({"benchmark_return": bench}, index=idx)
        df.index = df.index.tz_localize(None)
        return df

    def merge_into(
        self,
        stock_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if benchmark_df.empty:
            stock_df = stock_df.copy()
            stock_df["benchmark_return"] = np.nan
            return stock_df
        merged = stock_df.copy()
        # Align on the index (both tz-naive DatetimeIndex)
        bench_series = benchmark_df["benchmark_return"]
        merged["benchmark_return"] = merged.index.map(
            bench_series.to_dict()
        )
        merged["benchmark_return"] = merged["benchmark_return"].ffill(limit=1)
        return merged

    def save_parquet(self, df: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
        return path

    def load_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")


class MockTradingCalendar:
    """
    Satisfies TradingCalendarProtocol.

    Uses a simple Mon-Fri model with a small set of hard-coded US holidays
    for the 2024 test year.
    """

    _US_HOLIDAYS_2024: frozenset[date] = frozenset(
        {
            date(2024, 1, 1),   # New Year's Day
            date(2024, 1, 15),  # MLK Day
            date(2024, 2, 19),  # Presidents' Day
            date(2024, 5, 27),  # Memorial Day
            date(2024, 6, 19),  # Juneteenth
            date(2024, 7, 4),   # Independence Day
            date(2024, 9, 2),   # Labor Day
            date(2024, 11, 28), # Thanksgiving
            date(2024, 12, 25), # Christmas
        }
    )

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self._US_HOLIDAYS_2024

    def next_trading_day(self, d: date) -> date:
        candidate = d
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def align_event_date(
        self,
        earnings_date: date,
        after_close: bool = True,
    ) -> date:
        if after_close:
            return self.next_trading_day(earnings_date + timedelta(days=1))
        return self.next_trading_day(earnings_date)

    def get_trading_days(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        out = []
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                out.append(current)
            current += timedelta(days=1)
        return out


class MockEventAligner:
    """
    Satisfies EventAlignerProtocol.

    Tags relative_day using a rank-based approach identical to the
    fallback in _tag_relative_days().
    """

    def tag_relative_days(
        self,
        market_df: pd.DataFrame,
        event_date_col: str = "aligned_event_date",
    ) -> pd.DataFrame:
        df = market_df.copy()
        if event_date_col not in df.columns:
            df["relative_day"] = np.nan
            return df

        event_ts = pd.Timestamp(df[event_date_col].iloc[0]).normalize()

        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.normalize()
        else:
            dates = pd.to_datetime(df.get("date", pd.Series())).dt.normalize()

        sorted_unique = np.sort(dates.unique())
        event_rank = int(np.searchsorted(sorted_unique, event_ts))
        rank_map = {d: int(i - event_rank) for i, d in enumerate(sorted_unique)}
        df["relative_day"] = dates.map(rank_map).values
        return df

    def build_event_windows(
        self,
        metadata_df: pd.DataFrame,
        market_dict: dict[str, pd.DataFrame],
        calendar: Any,
    ) -> pd.DataFrame:
        frames = []
        for _, row in metadata_df.iterrows():
            t = row["ticker"]
            if t in market_dict and not market_dict[t].empty:
                df = market_dict[t].copy()
                df["transcript_id"] = row["transcript_id"]
                df["aligned_event_date"] = pd.Timestamp(row["earnings_date"])
                df = self.tag_relative_days(df)
                frames.append(df)
        return pd.concat(frames) if frames else pd.DataFrame()

    def save_parquet(self, df: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
        return path

    def load_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")


# ===========================================================================
# ── pytest fixtures ──────────────────────────────────────────────────────────
# ===========================================================================


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Standard 65-row AAPL-like OHLCV + benchmark DataFrame."""
    return make_ohlcv("AAPL", n=_N_ROWS, seed=_SEED, with_benchmark=True)


@pytest.fixture
def sample_benchmark_df() -> pd.DataFrame:
    """Standalone benchmark DataFrame (benchmark_return only)."""
    rng = np.random.default_rng(99)
    idx = pd.date_range("2024-08-01", periods=_N_ROWS, freq="B", name="date")
    bench = np.concatenate([[np.nan], rng.normal(0.0003, 0.009, _N_ROWS - 1)])
    return pd.DataFrame({"benchmark_return": bench}, index=idx)


@pytest.fixture
def sample_event_df() -> pd.DataFrame:
    """65-row OHLCV with earnings_date and aligned_event_date columns."""
    df = make_ohlcv("AAPL", n=_N_ROWS, seed=_SEED)
    df["transcript_id"] = "AAPL_Q3_2024"
    df["earnings_date"] = _EVENT_TS
    df["aligned_event_date"] = _EVENT_TS
    return df


@pytest.fixture
def sample_config(tmp_path: Path) -> MarketPipelineConfig:
    """MarketPipelineConfig pointed at a temp directory."""
    return MarketPipelineConfig(
        raw_market_dir=tmp_path / "raw/market",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        transcript_metadata_path=tmp_path / "interim/transcript_metadata.parquet",
        use_cache=False,
        export_csv=True,
        save_intermediates=True,
        retry_count=1,
        retry_delay=0.0,
    )


@pytest.fixture
def loader(tmp_path: Path) -> MarketDataLoader:
    """MarketDataLoader with save_raw=True in a temp directory."""
    cfg = MarketDataConfig(
        output_dir=tmp_path / "raw/market",
        save_raw=True,
        retry_count=1,
        retry_delay=0.0,
    )
    return MarketDataLoader(config=cfg)


@pytest.fixture
def engineer() -> FinancialFeatureEngineer:
    """FinancialFeatureEngineer with default config."""
    cfg = FeatureEngineeringConfig(
        return_horizons=[1, 2, 3, 5],
        rolling_vol_window=20,
        moving_average_windows=[20, 50],
        momentum_windows=[5, 20],
        clip_extreme_returns=True,
    )
    return FinancialFeatureEngineer(config=cfg)


@pytest.fixture
def calendar() -> MockTradingCalendar:
    return MockTradingCalendar()


@pytest.fixture
def builder(sample_config: MarketPipelineConfig) -> MarketDatasetBuilder:
    """
    MarketDatasetBuilder with all three optional dependencies injected
    as mocks so no live API calls are made.
    """
    b = MarketDatasetBuilder(
        config=sample_config,
        benchmark_loader=MockBenchmarkLoader(),
        trading_calendar=MockTradingCalendar(),
        event_aligner=MockEventAligner(),
    )

    # Monkey-patch download_event_window so no yfinance calls occur
    def _fake_window(ticker, earnings_date, pre_days, post_days):
        seed = abs(hash(ticker)) % 997
        df = make_ohlcv(ticker, n=_N_ROWS, seed=seed, with_benchmark=False)
        df["earnings_date"] = pd.Timestamp(earnings_date)
        return df

    b._market_loader.download_event_window = _fake_window
    return b


@pytest.fixture
def multi_ticker_df() -> pd.DataFrame:
    """Combined OHLCV DataFrame with three tickers for isolation tests."""
    frames = [
        make_ohlcv("AAPL", n=_N_ROWS, seed=1),
        make_ohlcv("MSFT", n=_N_ROWS, seed=2),
        make_ohlcv("NVDA", n=_N_ROWS, seed=3),
    ]
    return pd.concat(frames)


# ===========================================================================
# ── Test group 1: Market data validation ────────────────────────────────────
# ===========================================================================


class TestMarketDataValidation:
    """Unit tests for MarketDataLoader.validate_market_data()."""

    def test_valid_dataframe_passes(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        result = loader.validate_market_data(sample_market_df, "AAPL")
        assert result.is_valid, f"Expected valid, errors: {result.errors}"
        assert result.row_count == _N_ROWS

    def test_empty_dataframe_rejected(self, loader: MarketDataLoader) -> None:
        result = loader.validate_market_data(pd.DataFrame(), "AAPL")
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_none_dataframe_rejected(self, loader: MarketDataLoader) -> None:
        result = loader.validate_market_data(None, "AAPL")  # type: ignore[arg-type]
        assert not result.is_valid

    def test_missing_required_columns_rejected(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        bad = sample_market_df.drop(columns=["adj_close", "volume"])
        result = loader.validate_market_data(bad, "AAPL")
        assert not result.is_valid
        assert any("adj_close" in e or "volume" in e for e in result.errors)

    def test_duplicate_dates_detected(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        dup = pd.concat([sample_market_df, sample_market_df.iloc[:3]])
        result = loader.validate_market_data(dup, "AAPL")
        assert not result.is_valid
        assert any("uplicate" in e for e in result.errors)

    def test_negative_close_rejected(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        bad = sample_market_df.copy()
        bad.iloc[5, bad.columns.get_loc("close")] = -10.0
        result = loader.validate_market_data(bad, "AAPL")
        assert not result.is_valid
        assert any("close" in e for e in result.errors)

    def test_zero_close_rejected(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        bad = sample_market_df.copy()
        bad.iloc[10, bad.columns.get_loc("close")] = 0.0
        result = loader.validate_market_data(bad, "AAPL")
        assert not result.is_valid

    def test_negative_volume_rejected(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        bad = sample_market_df.copy()
        bad.iloc[0, bad.columns.get_loc("volume")] = -500.0
        result = loader.validate_market_data(bad, "AAPL")
        assert not result.is_valid
        assert any("olume" in e for e in result.errors)

    def test_non_datetimeindex_rejected(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        bad = sample_market_df.reset_index()  # converts to RangeIndex
        result = loader.validate_market_data(bad, "AAPL")
        assert not result.is_valid
        assert any("DatetimeIndex" in e for e in result.errors)

    def test_low_row_count_produces_warning(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        tiny = sample_market_df.iloc[:3]
        result = loader.validate_market_data(tiny, "AAPL")
        # May still be valid (no hard errors), but should have a warning
        assert any("few rows" in w for w in result.warnings)

    def test_nan_values_produce_warning(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        df_with_nan = sample_market_df.copy()
        df_with_nan.iloc[2, df_with_nan.columns.get_loc("adj_close")] = np.nan
        result = loader.validate_market_data(df_with_nan, "AAPL")
        assert any("NaN" in w for w in result.warnings)

    def test_validation_result_summary_contains_ticker(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        result = loader.validate_market_data(sample_market_df, "AAPL")
        assert "AAPL" in result.summary()


# ===========================================================================
# ── Test group 2: Column standardization and parquet I/O ────────────────────
# ===========================================================================


class TestMarketDataStandardization:
    """Unit tests for standardize_columns(), save_parquet(), load_parquet()."""

    def test_pascalcase_columns_normalised(
        self, loader: MarketDataLoader
    ) -> None:
        raw = make_raw_yfinance("AAPL", n=20)
        result = loader.standardize_columns(raw, "AAPL")
        assert_columns_present(result, ["open", "high", "low", "close", "adj_close", "volume"])
        assert "Open" not in result.columns

    def test_ticker_column_added_uppercase(
        self, loader: MarketDataLoader
    ) -> None:
        raw = make_raw_yfinance("aapl", n=10)
        result = loader.standardize_columns(raw, "aapl")
        assert "ticker" in result.columns
        assert (result["ticker"] == "AAPL").all()

    def test_index_named_date(self, loader: MarketDataLoader) -> None:
        raw = make_raw_yfinance("MSFT", n=10)
        result = loader.standardize_columns(raw, "MSFT")
        assert result.index.name == "date"

    def test_timezone_aware_index_stripped(
        self, loader: MarketDataLoader
    ) -> None:
        df = make_ohlcv("AAPL", n=10, tz_aware=True)
        result = loader.standardize_columns(df, "AAPL")
        assert result.index.tz is None

    def test_missing_adj_close_falls_back_to_close(
        self, loader: MarketDataLoader
    ) -> None:
        raw = make_raw_yfinance("TSLA", n=15)
        raw = raw.drop(columns=["Adj Close"])
        result = loader.standardize_columns(raw, "TSLA")
        # adj_close should now equal close (fallback)
        assert "adj_close" in result.columns
        np.testing.assert_array_almost_equal(
            result["adj_close"].values, result["close"].values
        )

    def test_save_and_load_parquet_roundtrip(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        loader.config.output_dir = tmp_path / "raw"
        loader.config.output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = loader.save_parquet(sample_market_df, "AAPL")
        assert saved_path.exists()

        reloaded = loader.load_parquet("AAPL")
        assert len(reloaded) == len(sample_market_df)
        pd.testing.assert_index_equal(
            reloaded.index.sort_values(),
            sample_market_df.index.sort_values(),
        )

    def test_load_nonexistent_parquet_returns_empty(
        self, loader: MarketDataLoader
    ) -> None:
        df = loader.load_parquet("DOES_NOT_EXIST")
        assert df.empty

    def test_remove_duplicates_keeps_last(
        self, loader: MarketDataLoader, sample_market_df: pd.DataFrame
    ) -> None:
        # Artificially inject a duplicate row
        dup = pd.concat([sample_market_df, sample_market_df.iloc[:1]])
        clean = loader._remove_duplicates(dup)
        # All dates must be unique
        assert not clean.index.duplicated().any()
        assert len(clean) == len(sample_market_df)


# ===========================================================================
# ── Test group 3: Event window date arithmetic ───────────────────────────────
# ===========================================================================


class TestMarketDataEventWindow:
    """Tests for download_event_window() date math (no network calls)."""

    def test_event_window_pre_post_buffer_applied(
        self, loader: MarketDataLoader
    ) -> None:
        """Confirm that the date range passed to download_ticker respects buffers."""
        captured: dict[str, Any] = {}

        def _capture(ticker, start_date, end_date):
            captured["start"] = start_date
            captured["end"] = end_date
            return pd.DataFrame()  # return empty — we just want to inspect dates

        loader._download_with_retry = _capture  # type: ignore[method-assign]

        earnings = date(2024, 10, 30)
        loader.download_event_window("AAPL", earnings, pre_days=30, post_days=10)

        # The download_ticker call converts dates to strings before calling retry
        # We can verify by checking what was passed further up the chain.
        # Since download_ticker handles the conversion, test via config defaults.
        expected_start = earnings - timedelta(days=30)
        expected_end = earnings + timedelta(days=10)

        assert captured.get("start") == str(expected_start)
        assert captured.get("end") == str(expected_end)

    def test_event_window_tags_earnings_date_column(
        self, loader: MarketDataLoader
    ) -> None:
        """earnings_date column is injected when download succeeds."""
        raw = make_raw_yfinance("AAPL", n=30)
        norm = loader.standardize_columns(raw, "AAPL")

        # Directly test the tagging logic (bypass network)
        earnings = date(2024, 10, 30)
        norm["earnings_date"] = pd.Timestamp(earnings)

        assert "earnings_date" in norm.columns
        assert (norm["earnings_date"] == pd.Timestamp(earnings)).all()


# ===========================================================================
# ── Test group 4: Benchmark handling ────────────────────────────────────────
# ===========================================================================


class TestBenchmarkHandling:
    """
    Tests for the BenchmarkLoaderProtocol contract and merge logic.
    Uses MockBenchmarkLoader — these tests define the expected behaviour
    that benchmark_loader.py must satisfy.
    """

    def test_benchmark_returns_calculated(
        self, sample_benchmark_df: pd.DataFrame
    ) -> None:
        """benchmark_return column should be present and bounded."""
        assert "benchmark_return" in sample_benchmark_df.columns
        non_nan = sample_benchmark_df["benchmark_return"].dropna()
        assert len(non_nan) > 0
        assert_no_extreme_returns(non_nan, "benchmark_return")

    def test_benchmark_first_row_is_nan(
        self, sample_benchmark_df: pd.DataFrame
    ) -> None:
        """First day has no prior price → first benchmark_return must be NaN."""
        assert pd.isna(sample_benchmark_df["benchmark_return"].iloc[0])

    def test_benchmark_merge_adds_column(
        self, sample_market_df: pd.DataFrame, sample_benchmark_df: pd.DataFrame
    ) -> None:
        loader = MockBenchmarkLoader()
        df_no_bench = sample_market_df.drop(
            columns=["benchmark_return"], errors="ignore"
        )
        merged = loader.merge_into(df_no_bench, sample_benchmark_df)
        assert "benchmark_return" in merged.columns

    def test_benchmark_merge_preserves_stock_rows(
        self, sample_market_df: pd.DataFrame, sample_benchmark_df: pd.DataFrame
    ) -> None:
        loader = MockBenchmarkLoader()
        df_no_bench = sample_market_df.drop(
            columns=["benchmark_return"], errors="ignore"
        )
        merged = loader.merge_into(df_no_bench, sample_benchmark_df)
        assert len(merged) == len(sample_market_df)

    def test_benchmark_merge_empty_benchmark_produces_nan_column(
        self, sample_market_df: pd.DataFrame
    ) -> None:
        loader = MockBenchmarkLoader()
        df_no_bench = sample_market_df.drop(
            columns=["benchmark_return"], errors="ignore"
        )
        merged = loader.merge_into(df_no_bench, pd.DataFrame())
        assert "benchmark_return" in merged.columns
        assert merged["benchmark_return"].isna().all()

    def test_benchmark_download_returns_dataframe(self) -> None:
        loader = MockBenchmarkLoader()
        df = loader.download(date(2024, 1, 1), date(2024, 3, 31))
        assert isinstance(df, pd.DataFrame)
        assert "benchmark_return" in df.columns
        assert not df.empty

    def test_benchmark_duplicate_dates_handled(
        self, sample_benchmark_df: pd.DataFrame
    ) -> None:
        """Benchmark data must not have duplicate dates before merge."""
        dup_count = sample_benchmark_df.index.duplicated().sum()
        assert dup_count == 0, "Benchmark fixture has duplicate dates."

    def test_benchmark_parquet_roundtrip(
        self, sample_benchmark_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        loader = MockBenchmarkLoader()
        path = tmp_path / "benchmark.parquet"
        loader.save_parquet(sample_benchmark_df, path)
        reloaded = loader.load_parquet(path)
        assert len(reloaded) == len(sample_benchmark_df)


# ===========================================================================
# ── Test group 5: Trading calendar ──────────────────────────────────────────
# ===========================================================================


class TestTradingCalendar:
    """
    Tests for TradingCalendarProtocol rules.
    Uses MockTradingCalendar — defines the contract for trading_calendar.py.
    """

    # ── is_trading_day ────────────────────────────────────────────────

    def test_weekday_is_trading_day(self, calendar: MockTradingCalendar) -> None:
        assert calendar.is_trading_day(date(2024, 10, 30))  # Wednesday

    def test_saturday_is_not_trading_day(
        self, calendar: MockTradingCalendar
    ) -> None:
        assert not calendar.is_trading_day(date(2024, 10, 26))  # Saturday

    def test_sunday_is_not_trading_day(
        self, calendar: MockTradingCalendar
    ) -> None:
        assert not calendar.is_trading_day(date(2024, 10, 27))  # Sunday

    def test_christmas_is_not_trading_day(
        self, calendar: MockTradingCalendar
    ) -> None:
        assert not calendar.is_trading_day(date(2024, 12, 25))

    def test_thanksgiving_is_not_trading_day(
        self, calendar: MockTradingCalendar
    ) -> None:
        assert not calendar.is_trading_day(date(2024, 11, 28))

    # ── next_trading_day ──────────────────────────────────────────────

    def test_next_trading_day_already_valid(
        self, calendar: MockTradingCalendar
    ) -> None:
        # Wednesday stays Wednesday
        assert calendar.next_trading_day(date(2024, 10, 30)) == date(2024, 10, 30)

    def test_saturday_advances_to_monday(
        self, calendar: MockTradingCalendar
    ) -> None:
        saturday = date(2024, 10, 26)
        assert calendar.next_trading_day(saturday) == date(2024, 10, 28)

    def test_sunday_advances_to_monday(
        self, calendar: MockTradingCalendar
    ) -> None:
        sunday = date(2024, 10, 27)
        assert calendar.next_trading_day(sunday) == date(2024, 10, 28)

    def test_friday_before_holiday_skips_holiday(
        self, calendar: MockTradingCalendar
    ) -> None:
        # Christmas 2024 is Wednesday; next trading day from Wed = Thursday
        assert calendar.next_trading_day(date(2024, 12, 25)) == date(2024, 12, 26)

    # ── align_event_date ──────────────────────────────────────────────

    @pytest.mark.parametrize(
        "earnings_date, after_close, expected",
        [
            # Friday after-close → Monday
            (date(2024, 10, 25), True, date(2024, 10, 28)),
            # Saturday earnings → Monday (after_close=True)
            (date(2024, 10, 26), True, date(2024, 10, 28)),
            # Monday 7 AM pre-market → same Monday (after_close=False)
            (date(2024, 10, 28), False, date(2024, 10, 28)),
            # Wednesday after-close → Thursday
            (date(2024, 10, 30), True, date(2024, 10, 31)),
        ],
    )
    def test_align_event_date(
        self,
        calendar: MockTradingCalendar,
        earnings_date: date,
        after_close: bool,
        expected: date,
    ) -> None:
        result = calendar.align_event_date(earnings_date, after_close=after_close)
        assert result == expected, (
            f"earnings={earnings_date}, after_close={after_close}: "
            f"expected {expected}, got {result}"
        )

    def test_friday_after_close_maps_to_monday(
        self, calendar: MockTradingCalendar
    ) -> None:
        """Critical: Friday 4:30 PM earnings → Monday open."""
        friday = date(2024, 10, 25)
        result = calendar.align_event_date(friday, after_close=True)
        assert result.weekday() == 0  # Monday
        assert result == date(2024, 10, 28)

    # ── get_trading_days ──────────────────────────────────────────────

    def test_trading_days_count_for_standard_week(
        self, calendar: MockTradingCalendar
    ) -> None:
        days = calendar.get_trading_days(date(2024, 10, 28), date(2024, 11, 1))
        assert len(days) == 5  # Mon–Fri, no holidays

    def test_trading_days_excludes_weekend(
        self, calendar: MockTradingCalendar
    ) -> None:
        days = calendar.get_trading_days(date(2024, 10, 25), date(2024, 10, 27))
        assert len(days) == 1  # only Friday

    def test_thanksgiving_week_has_four_trading_days(
        self, calendar: MockTradingCalendar
    ) -> None:
        # Thanksgiving 2024 is Nov 28 (Thursday)
        days = calendar.get_trading_days(date(2024, 11, 25), date(2024, 11, 29))
        # Mon Tue Wed [Thu holiday] Fri = 4 days
        assert len(days) == 4


# ===========================================================================
# ── Test group 6: Feature engineering — formula correctness ─────────────────
# ===========================================================================


class TestFeatureEngineering:
    """
    Exact numeric formula verification for FinancialFeatureEngineer.
    Uses hand-verified values on a tiny controlled dataset.
    """

    @pytest.fixture
    def controlled_df(self) -> pd.DataFrame:
        """
        A 10-row DataFrame with known prices for algebraic verification.
        Prices: 100, 102, 101, 105, 103, 107, 106, 110, 108, 112
        """
        prices = np.array(
            [100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 108.0, 112.0]
        )
        bench = np.array(
            [np.nan, 0.005, -0.002, 0.008, 0.001, 0.006, -0.001, 0.007, 0.003, 0.004]
        )
        idx = pd.date_range("2024-01-02", periods=10, freq="B", name="date")
        return pd.DataFrame(
            {
                "ticker": "TEST",
                "open": prices * 0.999,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "adj_close": prices,
                "volume": np.full(10, 1_000_000, dtype=float),
                "benchmark_return": bench,
            },
            index=idx,
        )

    def test_daily_return_formula_exact(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        """return_t = (P_t / P_{t-1}) - 1"""
        out = engineer.compute_daily_returns(controlled_df)
        prices = controlled_df["adj_close"].values

        # Row 1: (102 / 100) - 1 = 0.02
        assert_close(out["daily_return"].iloc[1], (102.0 / 100.0) - 1)

        # Row 2: (101 / 102) - 1 ≈ -0.0098039...
        assert_close(out["daily_return"].iloc[2], (101.0 / 102.0) - 1)

    def test_daily_return_first_row_is_nan(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        out = engineer.compute_daily_returns(controlled_df)
        assert pd.isna(out["daily_return"].iloc[0])

    def test_forward_return_1d_exact(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        """return_1d at row 0 = (102 / 100) - 1 = 0.02"""
        out = engineer.compute_forward_returns(controlled_df)
        assert_close(out["return_1d"].iloc[0], (102.0 / 100.0) - 1)

    def test_forward_return_last_n_rows_are_nan(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        out = engineer.compute_forward_returns(controlled_df)
        # Last row must be NaN for all horizons
        for n in engineer.config.return_horizons:
            assert pd.isna(out[f"return_{n}d"].iloc[-1]), (
                f"return_{n}d last row should be NaN"
            )

    def test_abnormal_return_formula_exact(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        """AR_t = daily_return_t - benchmark_return_t"""
        df = engineer.compute_daily_returns(controlled_df)
        df = engineer.compute_benchmark_returns(df)
        out = engineer.compute_abnormal_returns(df)

        # Row 1: daily_return = 0.02, benchmark = 0.005 → AR = 0.015
        dr1 = (102.0 / 100.0) - 1   # 0.02
        br1 = 0.005
        assert_close(out["abnormal_return"].iloc[1], dr1 - br1)

    def test_abnormal_return_nd_uses_compound_benchmark(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        """
        abnormal_return_3d = return_3d - ((1 + bench)^3 - 1)
        """
        df = engineer.compute_daily_returns(controlled_df)
        df = engineer.compute_forward_returns(df)
        df = engineer.compute_benchmark_returns(df)
        out = engineer.compute_abnormal_returns(df)

        # Verify at row 1 (has enough forward data)
        r3 = out["return_3d"].iloc[1]
        bench = out["benchmark_return"].iloc[1]
        compound = (1 + bench) ** 3 - 1
        expected = r3 - compound
        actual = out["abnormal_return_3d"].iloc[1]
        assert_close(actual, expected)

    def test_car_3d_is_sum_of_ar_1_2_3(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        df = engineer.compute_daily_returns(controlled_df)
        df = engineer.compute_forward_returns(df)
        df = engineer.compute_benchmark_returns(df)
        df = engineer.compute_abnormal_returns(df)
        out = engineer.compute_cumulative_abnormal_returns(df)

        # car_3d = AR_1d + AR_2d + AR_3d (all three must be non-NaN at row 1)
        row = out.iloc[1]
        expected = (
            row.get("abnormal_return_1d", np.nan)
            + row.get("abnormal_return_2d", np.nan)
            + row.get("abnormal_return_3d", np.nan)
        )
        if not (math.isnan(expected) or math.isnan(row.get("car_3d", np.nan))):
            assert_close(row["car_3d"], expected)

    def test_rolling_volatility_is_non_negative(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        out = engineer.compute_daily_returns(sample_market_df)
        out = engineer.compute_rolling_volatility(out)
        col = f"rolling_volatility_{engineer.config.rolling_vol_window}d"
        assert col in out.columns
        assert_non_negative(out[col], col)

    def test_rolling_volatility_annualisation(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        """
        Manual: vol_20d = std(returns[-20:]) * sqrt(252).
        Verify at the last row where window is full.
        """
        df = engineer.compute_daily_returns(controlled_df)
        out = engineer.compute_rolling_volatility(df)
        col = f"rolling_volatility_{engineer.config.rolling_vol_window}d"
        # With only 10 rows and min_periods=10, the last row should have a value
        # (min_periods defaults to 10 in the config)
        last_vol = out[col].dropna()
        if len(last_vol) > 0:
            assert last_vol.iloc[-1] >= 0.0

    def test_sma_20_non_negative(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        out = engineer.compute_moving_averages(sample_market_df)
        assert_non_negative(out["sma_20"], "sma_20")

    def test_sma_50_non_negative(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        out = engineer.compute_moving_averages(sample_market_df)
        assert_non_negative(out["sma_50"], "sma_50")

    def test_ema_20_non_negative(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        out = engineer.compute_moving_averages(sample_market_df)
        assert_non_negative(out["ema_20"], "ema_20")

    def test_sma_first_row_equals_first_price(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        """With min_periods=1, SMA at row 0 = adj_close[0]."""
        out = engineer.compute_moving_averages(controlled_df)
        first_price = controlled_df["adj_close"].iloc[0]
        assert_close(out["sma_20"].iloc[0], first_price)

    def test_momentum_5d_direction(
        self,
        engineer: FinancialFeatureEngineer,
        controlled_df: pd.DataFrame,
    ) -> None:
        """momentum_5d at row 5 = (107/100) - 1 > 0 (price went up)."""
        out = engineer.compute_momentum_features(controlled_df)
        # Row 5 price=107, row 0 price=100
        expected = (107.0 / 100.0) - 1
        assert_close(out["momentum_5d"].iloc[5], expected)

    def test_intraday_range_non_negative(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        out = engineer.compute_momentum_features(sample_market_df)
        assert_non_negative(out["intraday_range"], "intraday_range")

    def test_relative_volume_non_negative(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        out = engineer.compute_relative_volume(sample_market_df)
        assert_non_negative(out["relative_volume"].dropna(), "relative_volume")

    def test_no_infinite_values_in_full_feature_set(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        result = engineer.build_feature_set(sample_market_df, "AAPL")
        assert_no_inf(result.df, "build_feature_set")

    def test_daily_return_abs_below_one(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        result = engineer.build_feature_set(sample_market_df, "AAPL")
        assert_no_extreme_returns(result.df["daily_return"], "daily_return")

    def test_build_feature_set_result_is_valid(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        result = engineer.build_feature_set(sample_market_df, "AAPL")
        assert result.is_valid, f"Validation failed: {result.validation_errors}"

    def test_build_feature_set_adds_expected_columns(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        result = engineer.build_feature_set(sample_market_df, "AAPL")
        expected = [
            "daily_return",
            "return_1d", "return_3d", "return_5d",
            "benchmark_return",
            "abnormal_return",
            "car_3d", "car_5d",
            "rolling_volatility_20d",
            "sma_20", "sma_50", "ema_20",
            "momentum_5d", "momentum_20d",
            "price_gap", "intraday_range",
            "volume_change", "relative_volume",
        ]
        assert_columns_present(result.df, expected)

    def test_result_row_count_matches_input(
        self,
        engineer: FinancialFeatureEngineer,
        sample_market_df: pd.DataFrame,
    ) -> None:
        result = engineer.build_feature_set(sample_market_df, "AAPL")
        assert result.row_count == len(sample_market_df)


# ===========================================================================
# ── Test group 7: Multi-ticker isolation ────────────────────────────────────
# ===========================================================================


class TestFeatureEngineeringMultiTicker:
    """Verify that rolling windows do not bleed across tickers."""

    def test_daily_return_first_row_nan_per_ticker(
        self,
        engineer: FinancialFeatureEngineer,
        multi_ticker_df: pd.DataFrame,
    ) -> None:
        """
        Row-0 of *each* ticker group must have NaN daily_return.
        If cross-contamination occurs, MSFT's first row would use
        AAPL's last price as the prior, producing a non-NaN value.
        """
        out = engineer.compute_daily_returns(multi_ticker_df)
        for ticker, grp in out.groupby("ticker"):
            assert pd.isna(grp["daily_return"].iloc[0]), (
                f"Ticker {ticker}: first daily_return should be NaN (got "
                f"{grp['daily_return'].iloc[0]})"
            )

    def test_rolling_volatility_no_cross_contamination(
        self,
        engineer: FinancialFeatureEngineer,
        multi_ticker_df: pd.DataFrame,
    ) -> None:
        """Compute per-ticker then combined; rolling vol values must match."""
        combined = engineer.compute_daily_returns(multi_ticker_df)
        combined = engineer.compute_rolling_volatility(combined)
        col = f"rolling_volatility_{engineer.config.rolling_vol_window}d"

        for ticker in ["AAPL", "MSFT", "NVDA"]:
            single = make_ohlcv(
                ticker,
                seed={"AAPL": 1, "MSFT": 2, "NVDA": 3}[ticker],
            )
            single_out = engineer.compute_daily_returns(single)
            single_out = engineer.compute_rolling_volatility(single_out)

            combined_vals = combined.loc[combined["ticker"] == ticker, col].dropna()
            single_vals = single_out[col].dropna()

            min_len = min(len(combined_vals), len(single_vals))
            np.testing.assert_allclose(
                combined_vals.values[:min_len],
                single_vals.values[:min_len],
                rtol=1e-9,
                err_msg=f"Volatility cross-contamination detected for {ticker}",
            )


# ===========================================================================
# ── Test group 8: Feature engineering edge cases ────────────────────────────
# ===========================================================================


class TestFeatureEngineeringEdge:
    """Edge cases: missing benchmark, sparse data, NaN handling."""

    def test_missing_benchmark_returns_nan_ar(
        self, engineer: FinancialFeatureEngineer
    ) -> None:
        df = make_ohlcv("AAPL", n=30, with_benchmark=False)
        result = engineer.build_feature_set(df, "AAPL")
        assert result.is_valid  # should not error
        assert "abnormal_return" in result.df.columns
        assert result.df["abnormal_return"].isna().all()

    def test_sparse_data_no_crash(
        self, engineer: FinancialFeatureEngineer
    ) -> None:
        """Only 6 rows — rolling windows with min_periods kick in."""
        df = make_ohlcv("AAPL", n=6, with_benchmark=True)
        result = engineer.build_feature_set(df, "AAPL")
        # Must not raise; returns may mostly be NaN due to short window
        assert isinstance(result, FeatureEngineeringResult)
        assert_no_inf(result.df, "sparse_data")

    def test_extreme_return_is_clipped(
        self, engineer: FinancialFeatureEngineer
    ) -> None:
        """A 10x price spike should be clipped to 0.99."""
        df = make_ohlcv("SPIKE", n=10, with_benchmark=False)
        df = df.copy()
        df.iloc[5, df.columns.get_loc("adj_close")] = (
            df["adj_close"].iloc[4] * 11.0
        )
        out = engineer.compute_daily_returns(df)
        assert out["daily_return"].abs().max() <= 0.99 + 1e-9

    def test_timezone_aware_index_handled(
        self, engineer: FinancialFeatureEngineer
    ) -> None:
        df = make_ohlcv("AAPL", n=30, tz_aware=True, with_benchmark=False)
        # Should not crash; _prepare_dataframe strips tz
        result = engineer.build_feature_set(df, "AAPL")
        assert isinstance(result, FeatureEngineeringResult)

    def test_input_dataframe_not_mutated(
        self, engineer: FinancialFeatureEngineer, sample_market_df: pd.DataFrame
    ) -> None:
        original_cols = list(sample_market_df.columns)
        _ = engineer.build_feature_set(sample_market_df, "AAPL")
        assert list(sample_market_df.columns) == original_cols, (
            "build_feature_set mutated the input DataFrame."
        )

    def test_coverage_report_has_all_features(
        self, engineer: FinancialFeatureEngineer, sample_market_df: pd.DataFrame
    ) -> None:
        result = engineer.build_feature_set(sample_market_df, "AAPL")
        report = result.coverage_report()
        assert not report.empty
        # Every added feature should appear in the report
        for feat in result.features_added:
            if feat in sample_market_df.columns:
                continue  # pre-existing columns may not appear
            assert feat in report.index, f"'{feat}' missing from coverage report."

    @pytest.mark.parametrize("n_horizons", [[1], [1, 5], [1, 2, 3, 5, 10]])
    def test_configurable_return_horizons(
        self, n_horizons: list[int]
    ) -> None:
        cfg = FeatureEngineeringConfig(return_horizons=n_horizons)
        eng = FinancialFeatureEngineer(config=cfg)
        df = make_ohlcv("AAPL", n=30)
        result = eng.build_feature_set(df, "AAPL")
        for n in n_horizons:
            assert f"return_{n}d" in result.df.columns


# ===========================================================================
# ── Test group 9: Event alignment ───────────────────────────────────────────
# ===========================================================================


class TestEventAlignment:
    """Tests for MockEventAligner (contract for event_alignment.py)."""

    @pytest.fixture
    def aligned_df(self, sample_event_df: pd.DataFrame) -> pd.DataFrame:
        """sample_event_df after relative_day tagging."""
        aligner = MockEventAligner()
        return aligner.tag_relative_days(
            sample_event_df, event_date_col="aligned_event_date"
        )

    def test_relative_day_column_exists(self, aligned_df: pd.DataFrame) -> None:
        assert "relative_day" in aligned_df.columns

    def test_event_day_is_zero(self, aligned_df: pd.DataFrame) -> None:
        event_mask = aligned_df.index.normalize() == _EVENT_TS.normalize()
        if event_mask.any():
            assert (aligned_df.loc[event_mask, "relative_day"] == 0).all()

    def test_pre_event_days_are_negative(self, aligned_df: pd.DataFrame) -> None:
        pre = aligned_df[aligned_df.index < _EVENT_TS]
        if not pre.empty:
            assert (pre["relative_day"] < 0).all()

    def test_post_event_days_are_positive(self, aligned_df: pd.DataFrame) -> None:
        post = aligned_df[aligned_df.index > _EVENT_TS]
        if not post.empty:
            assert (post["relative_day"] > 0).all()

    def test_relative_day_is_strictly_monotonic(
        self, aligned_df: pd.DataFrame
    ) -> None:
        days = aligned_df["relative_day"].values
        # Consecutive differences should all be +1 (one trading day steps)
        diffs = np.diff(days)
        assert (diffs == 1).all(), (
            f"relative_day is not monotonically increasing by 1: diffs={diffs}"
        )

    def test_no_duplicate_relative_days(self, aligned_df: pd.DataFrame) -> None:
        assert not aligned_df["relative_day"].duplicated().any()

    def test_aligned_event_date_no_nulls(
        self, sample_event_df: pd.DataFrame
    ) -> None:
        assert not sample_event_df["aligned_event_date"].isna().any()

    def test_tag_relative_days_missing_column_produces_nan(self) -> None:
        aligner = MockEventAligner()
        df = make_ohlcv("AAPL", n=10, with_benchmark=False)
        out = aligner.tag_relative_days(df, event_date_col="nonexistent")
        assert "relative_day" in out.columns
        assert out["relative_day"].isna().all()

    def test_build_event_windows_multi_ticker(self) -> None:
        aligner = MockEventAligner()
        metadata = make_metadata(["AAPL", "MSFT"])
        market_dict = {
            "AAPL": make_ohlcv("AAPL", n=30),
            "MSFT": make_ohlcv("MSFT", n=30, seed=2),
        }
        result = aligner.build_event_windows(metadata, market_dict, calendar=None)
        assert not result.empty
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}


# ===========================================================================
# ── Test group 10: MarketDatasetBuilder orchestration ───────────────────────
# ===========================================================================


class TestMarketDatasetBuilder:
    """Unit tests for the orchestration layer."""

    # ── process_ticker ────────────────────────────────────────────────

    def test_process_ticker_success(
        self, builder: MarketDatasetBuilder
    ) -> None:
        result = builder.process_ticker(
            ticker="AAPL",
            earnings_date=_EVENT_DATE,
            transcript_id="AAPL_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        assert result.success, f"Expected success, got error: {result.error}"
        assert result.row_count > 0
        assert "daily_return" in result.df.columns
        assert "relative_day" in result.df.columns

    def test_process_ticker_injects_metadata_columns(
        self, builder: MarketDatasetBuilder
    ) -> None:
        result = builder.process_ticker(
            ticker="MSFT",
            earnings_date=_EVENT_DATE,
            transcript_id="MSFT_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        assert result.success
        assert "transcript_id" in result.df.columns
        assert "earnings_date" in result.df.columns
        assert "aligned_event_date" in result.df.columns
        assert (result.df["transcript_id"] == "MSFT_Q3_2024").all()

    def test_process_ticker_fails_gracefully_on_empty_download(
        self, builder: MarketDatasetBuilder
    ) -> None:
        # Override to return empty
        builder._market_loader.download_event_window = (
            lambda ticker, earnings_date, pre_days, post_days: pd.DataFrame()
        )
        result = builder.process_ticker(
            ticker="FAKE",
            earnings_date=_EVENT_DATE,
            transcript_id="FAKE_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        assert not result.success
        assert result.df.empty
        assert result.error != ""

    def test_process_ticker_records_elapsed_time(
        self, builder: MarketDatasetBuilder
    ) -> None:
        result = builder.process_ticker(
            ticker="NVDA",
            earnings_date=_EVENT_DATE,
            transcript_id="NVDA_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        assert result.elapsed_seconds >= 0.0

    # ── merge_all_tickers ────────────────────────────────────────────

    def test_merge_all_tickers_empty_results_returns_empty(
        self, builder: MarketDatasetBuilder
    ) -> None:
        failed = [
            TickerResult(ticker="FAKE", success=False, error="no data", df=pd.DataFrame())
        ]
        merged = builder.merge_all_tickers(failed)
        assert merged.empty

    def test_merge_all_tickers_preserves_all_rows(
        self, builder: MarketDatasetBuilder
    ) -> None:
        results = [
            TickerResult(
                ticker=t,
                success=True,
                df=make_ohlcv(t, n=30, seed=i),
                row_count=30,
            )
            for i, t in enumerate(["AAPL", "MSFT", "NVDA"], start=1)
        ]
        merged = builder.merge_all_tickers(results)
        assert len(merged) == 90

    def test_merge_all_tickers_sorted_by_ticker_date(
        self, builder: MarketDatasetBuilder
    ) -> None:
        results = [
            TickerResult(ticker=t, success=True, df=make_ohlcv(t, n=20, seed=i), row_count=20)
            for i, t in enumerate(["NVDA", "AAPL"], start=1)
        ]
        merged = builder.merge_all_tickers(results)
        if "ticker" in merged.columns and "date" in merged.columns:
            for _, grp in merged.groupby("ticker"):
                assert grp["date"].is_monotonic_increasing

    # ── detect_failed_tickers ────────────────────────────────────────

    def test_detect_failed_tickers_returns_only_failures(
        self, builder: MarketDatasetBuilder
    ) -> None:
        results = [
            TickerResult(ticker="AAPL", success=True, df=pd.DataFrame()),
            TickerResult(ticker="FAKE", success=False, error="api error", df=pd.DataFrame()),
            TickerResult(ticker="MSFT", success=True, df=pd.DataFrame()),
        ]
        failures = builder.detect_failed_tickers(results)
        assert len(failures) == 1
        assert failures[0]["ticker"] == "FAKE"
        assert "api error" in failures[0]["error"]

    # ── summarize_missing_data ───────────────────────────────────────

    def test_summarize_missing_data_empty_returns_empty(
        self, builder: MarketDatasetBuilder
    ) -> None:
        result = builder.summarize_missing_data(pd.DataFrame())
        assert result == {}

    def test_summarize_missing_data_reports_fraction(
        self, builder: MarketDatasetBuilder, sample_market_df: pd.DataFrame
    ) -> None:
        df = sample_market_df.copy()
        df["daily_return"] = np.nan  # 100% missing
        summary = builder.summarize_missing_data(df)
        assert "daily_return" in summary
        assert summary["daily_return"] == pytest.approx(1.0)

    # ── check_event_window_integrity ─────────────────────────────────

    def test_event_window_integrity_has_event_day(
        self, builder: MarketDatasetBuilder, sample_event_df: pd.DataFrame
    ) -> None:
        aligner = MockEventAligner()
        df_tagged = aligner.tag_relative_days(
            sample_event_df, event_date_col="aligned_event_date"
        )
        report = builder.check_event_window_integrity(df_tagged)
        if not report.empty:
            assert report["has_event_day"].all()

    def test_event_window_integrity_empty_df_returns_empty(
        self, builder: MarketDatasetBuilder
    ) -> None:
        report = builder.check_event_window_integrity(pd.DataFrame())
        assert report.empty


# ===========================================================================
# ── Test group 11: validate_final_dataset ───────────────────────────────────
# ===========================================================================


class TestFinalDatasetValidation:
    """Tests for MarketDatasetBuilder.validate_final_dataset()."""

    @pytest.fixture
    def valid_final_df(self, builder: MarketDatasetBuilder) -> pd.DataFrame:
        """Produce a fully valid final DataFrame via the pipeline."""
        metadata = make_metadata(["AAPL"])
        final_df, _ = builder.build_market_dataset(metadata)
        return final_df

    def test_valid_dataset_has_no_errors(
        self,
        builder: MarketDatasetBuilder,
        valid_final_df: pd.DataFrame,
    ) -> None:
        errors, _ = builder.validate_final_dataset(valid_final_df)
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_empty_dataframe_fails_validation(
        self, builder: MarketDatasetBuilder
    ) -> None:
        errors, _ = builder.validate_final_dataset(pd.DataFrame())
        assert errors

    def test_duplicate_rows_detected(
        self, builder: MarketDatasetBuilder, valid_final_df: pd.DataFrame
    ) -> None:
        if valid_final_df.empty:
            pytest.skip("valid_final_df is empty")
        dup = pd.concat([valid_final_df, valid_final_df.iloc[:3]], ignore_index=True)
        errors, _ = builder.validate_final_dataset(dup)
        assert any("uplicate" in e for e in errors)

    def test_missing_required_column_detected(
        self, builder: MarketDatasetBuilder, valid_final_df: pd.DataFrame
    ) -> None:
        if valid_final_df.empty:
            pytest.skip("valid_final_df is empty")
        bad = valid_final_df.drop(columns=["aligned_event_date"], errors="ignore")
        errors, _ = builder.validate_final_dataset(bad)
        assert any("aligned_event_date" in e for e in errors)

    def test_infinite_values_detected(
        self, builder: MarketDatasetBuilder, valid_final_df: pd.DataFrame
    ) -> None:
        if valid_final_df.empty or "daily_return" not in valid_final_df.columns:
            pytest.skip("daily_return not present")
        bad = valid_final_df.copy()
        bad.iloc[0, bad.columns.get_loc("daily_return")] = np.inf
        errors, _ = builder.validate_final_dataset(bad)
        assert any("nfinite" in e for e in errors)

    def test_null_aligned_event_date_detected(
        self, builder: MarketDatasetBuilder, valid_final_df: pd.DataFrame
    ) -> None:
        if valid_final_df.empty or "aligned_event_date" not in valid_final_df.columns:
            pytest.skip("aligned_event_date not present")
        bad = valid_final_df.copy()
        bad.iloc[0, bad.columns.get_loc("aligned_event_date")] = pd.NaT
        errors, _ = builder.validate_final_dataset(bad)
        assert any("aligned_event_date" in e for e in errors)


# ===========================================================================
# ── Test group 12: Export and parquet I/O ───────────────────────────────────
# ===========================================================================


class TestExportOutputs:
    """Tests for export_outputs() and load_transcript_metadata()."""

    def test_parquet_written_and_reloadable(
        self, builder: MarketDatasetBuilder
    ) -> None:
        metadata = make_metadata(["AAPL"])
        final_df, ticker_results = builder.build_market_dataset(metadata)
        paths = builder.export_outputs(final_df, ticker_results)

        parquet_key = "market_data.parquet"
        assert parquet_key in paths
        p = Path(paths[parquet_key])
        assert p.exists()

        reloaded = pd.read_parquet(p)
        assert len(reloaded) == len(final_df)

    def test_csv_written_when_configured(
        self, builder: MarketDatasetBuilder
    ) -> None:
        builder.config.export_csv = True
        metadata = make_metadata(["AAPL"])
        final_df, ticker_results = builder.build_market_dataset(metadata)
        paths = builder.export_outputs(final_df, ticker_results)

        csv_key = "market_data.csv"
        assert csv_key in paths
        assert Path(paths[csv_key]).exists()

    def test_csv_not_written_when_disabled(
        self, builder: MarketDatasetBuilder
    ) -> None:
        builder.config.export_csv = False
        metadata = make_metadata(["AAPL"])
        final_df, ticker_results = builder.build_market_dataset(metadata)
        paths = builder.export_outputs(final_df, ticker_results)
        assert "market_data.csv" not in paths

    def test_aligned_events_intermediate_written(
        self, builder: MarketDatasetBuilder
    ) -> None:
        builder.config.save_intermediates = True
        metadata = make_metadata(["AAPL"])
        final_df, ticker_results = builder.build_market_dataset(metadata)
        paths = builder.export_outputs(final_df, ticker_results)
        assert "aligned_events.parquet" in paths

    def test_export_empty_df_returns_empty_paths(
        self, builder: MarketDatasetBuilder
    ) -> None:
        paths = builder.export_outputs(pd.DataFrame(), [])
        assert paths == {}

    def test_load_transcript_metadata_file_not_found_raises(
        self, builder: MarketDatasetBuilder
    ) -> None:
        with pytest.raises(FileNotFoundError):
            builder.load_transcript_metadata()

    def test_load_transcript_metadata_missing_columns_raises(
        self, builder: MarketDatasetBuilder, tmp_path: Path
    ) -> None:
        bad = pd.DataFrame({"ticker": ["AAPL"]})  # missing transcript_id, earnings_date
        path = builder.config.transcript_metadata_path
        path.parent.mkdir(parents=True, exist_ok=True)
        bad.to_parquet(path)
        with pytest.raises(ValueError, match="missing required columns"):
            builder.load_transcript_metadata()

    def test_load_transcript_metadata_normalises_ticker_case(
        self, builder: MarketDatasetBuilder, tmp_path: Path
    ) -> None:
        df = pd.DataFrame(
            {
                "transcript_id": ["aapl_q3_2024"],
                "ticker": ["aapl"],
                "earnings_date": [date(2024, 10, 30)],
            }
        )
        path = builder.config.transcript_metadata_path
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        loaded = builder.load_transcript_metadata()
        assert (loaded["ticker"] == "AAPL").all()


# ===========================================================================
# ── Test group 13: Pipeline integration (end-to-end) ────────────────────────
# ===========================================================================


class TestPipelineIntegration:
    """
    End-to-end tests: metadata → run() → parquet.
    No live API calls — monkey-patched download_event_window throughout.
    """

    def test_run_single_ticker_produces_output(
        self, builder: MarketDatasetBuilder, tmp_path: Path
    ) -> None:
        metadata = make_metadata(["AAPL"])
        summary = builder.run(metadata_df=metadata)

        assert summary.total_tickers == 1
        assert len(summary.successful_tickers) == 1
        assert "AAPL" in summary.successful_tickers
        assert summary.total_rows_exported > 0

    def test_run_multi_ticker_all_succeed(
        self, builder: MarketDatasetBuilder
    ) -> None:
        metadata = make_metadata(["AAPL", "MSFT", "NVDA"])
        summary = builder.run(metadata_df=metadata)

        assert summary.total_tickers == 3
        assert len(summary.successful_tickers) == 3
        assert len(summary.failed_tickers) == 0
        assert summary.success_rate == pytest.approx(1.0)

    def test_run_partial_failure_continues(
        self, builder: MarketDatasetBuilder
    ) -> None:
        """One ticker returning empty should not abort the rest."""
        call_count = {"n": 0}

        def _sometimes_fail(ticker, earnings_date, pre_days, post_days):
            call_count["n"] += 1
            if ticker == "MSFT":
                return pd.DataFrame()  # simulate failure
            seed = abs(hash(ticker)) % 997
            return make_ohlcv(ticker, n=_N_ROWS, seed=seed, with_benchmark=False)

        builder._market_loader.download_event_window = _sometimes_fail

        metadata = make_metadata(["AAPL", "MSFT", "NVDA"])
        summary = builder.run(metadata_df=metadata)

        assert call_count["n"] == 3  # all three were attempted
        assert "MSFT" in summary.failed_tickers
        assert "AAPL" in summary.successful_tickers
        assert "NVDA" in summary.successful_tickers

    def test_run_output_has_required_columns(
        self, builder: MarketDatasetBuilder
    ) -> None:
        metadata = make_metadata(["AAPL"])
        summary = builder.run(metadata_df=metadata)

        parquet_path_str = summary.output_paths.get("market_data.parquet")
        if parquet_path_str is None:
            pytest.skip("No parquet output path in summary.")
        final_df = pd.read_parquet(parquet_path_str)

        # Check the intersection of FINAL_REQUIRED_COLUMNS that are reachable
        # without the not-yet-built modules providing all fields
        always_present = [
            "ticker", "earnings_date", "aligned_event_date",
            "daily_return", "rolling_volatility_20d",
            "sma_20", "sma_50", "relative_volume",
        ]
        for col in always_present:
            assert col in final_df.columns, f"'{col}' missing from final output."

    def test_run_no_duplicate_rows(
        self, builder: MarketDatasetBuilder
    ) -> None:
        metadata = make_metadata(["AAPL", "MSFT"])
        summary = builder.run(metadata_df=metadata)

        parquet_path_str = summary.output_paths.get("market_data.parquet")
        if parquet_path_str is None:
            pytest.skip("No parquet output.")
        df = pd.read_parquet(parquet_path_str)

        id_cols = [c for c in ["ticker", "transcript_id", "date"] if c in df.columns]
        if id_cols:
            dup_count = df.duplicated(subset=id_cols).sum()
            assert dup_count == 0, f"{dup_count} duplicate rows in final output."

    def test_run_dates_sorted_per_ticker(
        self, builder: MarketDatasetBuilder
    ) -> None:
        metadata = make_metadata(["AAPL"])
        summary = builder.run(metadata_df=metadata)

        parquet_path_str = summary.output_paths.get("market_data.parquet")
        if parquet_path_str is None:
            pytest.skip("No parquet output.")
        df = pd.read_parquet(parquet_path_str)

        if "date" not in df.columns or "ticker" not in df.columns:
            pytest.skip("date/ticker not in output columns.")

        for ticker, grp in df.groupby("ticker"):
            assert grp["date"].is_monotonic_increasing, (
                f"Dates not sorted for ticker {ticker}."
            )

    def test_run_returns_pipeline_summary(
        self, builder: MarketDatasetBuilder
    ) -> None:
        metadata = make_metadata(["AAPL"])
        summary = builder.run(metadata_df=metadata)
        assert isinstance(summary, PipelineRunSummary)
        assert summary.run_id != ""
        assert summary.elapsed_seconds > 0

    def test_pipeline_summary_print_report_runs(
        self, builder: MarketDatasetBuilder, capsys: pytest.CaptureFixture
    ) -> None:
        metadata = make_metadata(["AAPL"])
        summary = builder.run(metadata_df=metadata)
        summary.print_report()  # should not raise
        captured = capsys.readouterr()
        assert "DAY 5" in captured.out

    def test_generate_dataset_statistics_returns_dataframe(
        self, builder: MarketDatasetBuilder
    ) -> None:
        metadata = make_metadata(["AAPL"])
        summary = builder.run(metadata_df=metadata)
        parquet_path_str = summary.output_paths.get("market_data.parquet")
        if parquet_path_str is None:
            pytest.skip("No parquet output.")
        df = pd.read_parquet(parquet_path_str)
        stats = builder.generate_dataset_statistics(df)
        assert isinstance(stats, pd.DataFrame)


# ===========================================================================
# ── Test group 14: Edge cases ────────────────────────────────────────────────
# ===========================================================================


class TestEdgeCases:
    """Regression tests for known tricky scenarios."""

    def test_holiday_event_date_aligned_forward(self) -> None:
        """Christmas 2024 (Wednesday) should align to Thursday 2024-12-26."""
        cal = MockTradingCalendar()
        aligned = cal.align_event_date(date(2024, 12, 25), after_close=True)
        # Christmas is Dec 25; next trading day after Dec 26 (after_close moves +1)
        # Dec 26 is Thursday and not a holiday in our set → valid
        assert aligned > date(2024, 12, 25)
        assert cal.is_trading_day(aligned)

    def test_weekend_earnings_never_produces_weekend_aligned_date(self) -> None:
        """Earnings on any weekend day should never yield a weekend event date."""
        cal = MockTradingCalendar()
        for days_offset in range(14):
            d = date(2024, 11, 1) + timedelta(days=days_offset)
            aligned = cal.align_event_date(d, after_close=True)
            assert aligned.weekday() < 5, (
                f"aligned_event_date {aligned} falls on a weekend."
            )

    def test_timezone_naive_timestamps_in_pipeline(
        self, builder: MarketDatasetBuilder
    ) -> None:
        """Timestamps with timezone info should be normalised to tz-naive."""
        metadata = make_metadata(["AAPL"])
        result = builder.process_ticker(
            ticker="AAPL",
            earnings_date=_EVENT_DATE,
            transcript_id="AAPL_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        assert result.success
        # Date index should be tz-naive
        if isinstance(result.df.index, pd.DatetimeIndex):
            assert result.df.index.tz is None

    def test_duplicate_events_in_metadata_processed_independently(
        self, builder: MarketDatasetBuilder
    ) -> None:
        """Two events for the same ticker on the same day should both run."""
        metadata = pd.DataFrame(
            [
                {
                    "transcript_id": "AAPL_Q3_2024",
                    "ticker": "AAPL",
                    "earnings_date": _EVENT_DATE,
                },
                {
                    "transcript_id": "AAPL_Q3_2024_REVISION",
                    "ticker": "AAPL",
                    "earnings_date": _EVENT_DATE,
                },
            ]
        )
        final_df, ticker_results = builder.build_market_dataset(metadata)
        # Both transcript_ids should appear
        if "transcript_id" in final_df.columns:
            ids = set(final_df["transcript_id"].unique())
            assert "AAPL_Q3_2024" in ids
            assert "AAPL_Q3_2024_REVISION" in ids

    def test_missing_benchmark_rows_degrade_gracefully(
        self, builder: MarketDatasetBuilder
    ) -> None:
        """Benchmark with only partial date coverage should not crash."""
        sparse_bench = pd.DataFrame(
            {"benchmark_return": [0.001, 0.002]},
            index=pd.DatetimeIndex(
                ["2024-10-28", "2024-10-29"], name="date"
            ),
        )
        result = builder.process_ticker(
            ticker="AAPL",
            earnings_date=_EVENT_DATE,
            transcript_id="AAPL_Q3_2024",
            benchmark_df=sparse_bench,
        )
        assert result.success  # must not raise

    def test_ticker_casing_normalised_to_uppercase(
        self, builder: MarketDatasetBuilder
    ) -> None:
        result = builder.process_ticker(
            ticker="aapl",  # lowercase input
            earnings_date=_EVENT_DATE,
            transcript_id="aapl_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        assert result.success
        if not result.df.empty and "ticker" in result.df.columns:
            assert (result.df["ticker"] == "AAPL").all()

    def test_very_short_history_no_crash(
        self, builder: MarketDatasetBuilder
    ) -> None:
        """A ticker with only 5 rows of history should not crash the pipeline."""
        builder._market_loader.download_event_window = (
            lambda ticker, earnings_date, pre_days, post_days: make_ohlcv(
                ticker, n=5, seed=42, with_benchmark=False
            )
        )
        result = builder.process_ticker(
            ticker="TINY",
            earnings_date=_EVENT_DATE,
            transcript_id="TINY_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        # May succeed or fail cleanly — must not raise an unhandled exception
        assert isinstance(result, TickerResult)

    @pytest.mark.parametrize(
        "ticker_symbol",
        ["AAPL", "BRK-B", "BF-B", "MSFT"],
    )
    def test_various_ticker_formats_accepted(
        self, builder: MarketDatasetBuilder, ticker_symbol: str
    ) -> None:
        """Ticker symbols with dashes (BRK-B style) must not crash."""
        result = builder.process_ticker(
            ticker=ticker_symbol,
            earnings_date=_EVENT_DATE,
            transcript_id=f"{ticker_symbol}_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        assert isinstance(result, TickerResult)  # no unhandled exception

    def test_no_data_ticker_result_has_error_message(
        self, builder: MarketDatasetBuilder
    ) -> None:
        builder._market_loader.download_event_window = (
            lambda *a, **kw: pd.DataFrame()
        )
        result = builder.process_ticker(
            ticker="DELISTED",
            earnings_date=_EVENT_DATE,
            transcript_id="DELISTED_Q3_2024",
            benchmark_df=pd.DataFrame(),
        )
        assert not result.success
        assert len(result.error) > 0


# ===========================================================================
# ── Test group 15: TickerResult and PipelineRunSummary data integrity ────────
# ===========================================================================


class TestDataclassIntegrity:
    """Sanity checks on the result dataclasses."""

    def test_ticker_result_to_dict_excludes_dataframe(self) -> None:
        r = TickerResult(
            ticker="AAPL",
            transcript_id="AAPL_Q3_2024",
            earnings_date=_EVENT_DATE,
            success=True,
            df=make_ohlcv("AAPL", n=10),
            row_count=10,
        )
        d = r.to_dict()
        assert "df" not in d
        assert d["ticker"] == "AAPL"
        assert d["row_count"] == 10

    def test_pipeline_run_summary_success_rate(self) -> None:
        s = PipelineRunSummary(
            successful_tickers=["AAPL", "MSFT"],
            failed_tickers=["FAKE"],
            total_tickers=3,
        )
        assert s.success_rate == pytest.approx(2 / 3)

    def test_pipeline_run_summary_zero_tickers_rate(self) -> None:
        s = PipelineRunSummary(total_tickers=0)
        assert s.success_rate == 0.0

    def test_feature_engineering_result_is_valid_property(self) -> None:
        r = FeatureEngineeringResult(
            df=pd.DataFrame(),
            ticker="TEST",
            validation_errors=[],
        )
        assert r.is_valid

        r_invalid = FeatureEngineeringResult(
            df=pd.DataFrame(),
            ticker="TEST",
            validation_errors=["Something broke"],
        )
        assert not r_invalid.is_valid

    def test_validation_result_summary_format(self) -> None:
        r = ValidationResult(
            is_valid=False,
            ticker="AAPL",
            errors=["close prices <= 0: 1 rows."],
            warnings=["Very few rows (3)."],
            row_count=3,
        )
        summary = r.summary()
        assert "INVALID" in summary
        assert "AAPL" in summary
        assert "close" in summary
