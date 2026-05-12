"""
src/ingestion/market_data_loader.py
=====================================
Downloads and caches OHLCV price data from yfinance.
Computes daily returns and aligns to trading calendar.

Critical design note — earnings timing alignment:
  After-market release → event_date = earnings_date + 1 business day
  Pre-market release   → event_date = earnings_date (same day)
  Getting this wrong introduces look-ahead bias into the event study.

Usage:
    loader = MarketDataLoader()
    prices = loader.download_prices(tickers, "2019-01-01", "2024-12-31")
    returns = loader.compute_returns(prices)
    loader.save_prices(prices)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.storage import save_parquet

log = get_logger(__name__)


class MarketDataLoader:
    """Downloads market price data from yfinance with caching."""

    def __init__(self) -> None:
        self.config = load_config()
        self.market_dir = Path(self.config["paths"]["raw_market"])
        self.market_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark = self.config["market"]["benchmark_ticker"]

    def download_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        include_benchmark: bool = True,
    ) -> pd.DataFrame:
        """
        Download adjusted close prices for tickers + S&P 500 benchmark.

        Args:
            tickers          : List of ticker symbols
            start            : Start date (YYYY-MM-DD)
            end              : End date (YYYY-MM-DD)
            include_benchmark: Append ^GSPC column if True

        Returns:
            DataFrame — DatetimeIndex rows, tickers as columns (adj. close)
        """
        all_tickers = list(tickers)
        if include_benchmark and self.benchmark not in all_tickers:
            all_tickers.append(self.benchmark)

        log.info(f"Downloading {len(all_tickers)} tickers: {start} → {end}")

        raw = yf.download(
            tickers=all_tickers,
            start=start,
            end=end,
            auto_adjust=True,   # prices already dividend-adjusted
            progress=False,
        )

        # Handle single-ticker vs multi-ticker column structure
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

        prices.index = pd.to_datetime(prices.index)
        prices = prices.dropna(how="all")
        log.info(f"Downloaded {prices.shape[0]:,} days × {prices.shape[1]} tickers")
        return prices

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Simple daily returns: R_t = (P_t - P_{t-1}) / P_{t-1}

        Args:
            prices: Adjusted close price DataFrame

        Returns:
            Daily returns DataFrame (same shape, first row dropped)
        """
        returns = prices.pct_change().dropna(how="all")
        log.debug(f"Returns shape: {returns.shape}")
        return returns

    def save_prices(
        self,
        prices: pd.DataFrame,
        filename: str = "prices.parquet",
    ) -> Path:
        """Save price DataFrame to Parquet."""
        out = self.market_dir / filename
        prices_reset = prices.reset_index()
        save_parquet(prices_reset, out)
        return out

    def load_prices(self, filename: str = "prices.parquet") -> pd.DataFrame:
        """Load cached prices from Parquet."""
        path = self.market_dir / filename
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
        return df

    @staticmethod
    def align_event_date(
        earnings_date: pd.Timestamp,
        timing: str = "after_market",
    ) -> pd.Timestamp:
        """
        Adjust earnings date to the correct market event date.

        Args:
            earnings_date: Raw date of the earnings call
            timing       : "after_market" or "pre_market"

        Returns:
            Adjusted event date for return measurement
        """
        if timing == "after_market":
            return earnings_date + pd.offsets.BDay(1)
        return earnings_date
