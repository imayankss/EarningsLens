"""
src/ingestion/transcript_loader.py
====================================
Loads earnings call transcripts from multiple sources and normalises
them to a consistent schema regardless of origin.

Supported sources:
  1. HuggingFace datasets (Bose345/sp500_earnings_transcripts)
  2. Local CSV / Parquet files
  3. Kaggle CSV downloads

Output schema (transcripts_raw):
  transcript_id  : str      — unique key: TICKER_YYYYMMDD
  ticker         : str      — e.g. "AAPL"
  earnings_date  : datetime — date of earnings call
  fiscal_quarter : str      — e.g. "Q1 2024" (if available)
  transcript_text: str      — full raw text
  word_count     : int      — word count of raw text
  source         : str      — origin tag

Usage:
    loader = TranscriptLoader()
    df = loader.load_from_huggingface(tickers=["AAPL", "MSFT"])
    loader.save_raw(df)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config, get_all_tickers
from src.utils.logger import get_logger
from src.utils.storage import save_parquet

log = get_logger(__name__)


class TranscriptLoader:
    """Unified transcript ingestion from multiple sources."""

    def __init__(self) -> None:
        self.config = load_config()
        self.raw_dir = Path(self.config["paths"]["raw_transcripts"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Public methods ──────────────────────────────────────────

    def load_from_huggingface(
        self,
        dataset_name: str = "Bose345/sp500_earnings_transcripts",
        tickers: list[str] | None = None,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """
        Download transcripts from a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset identifier
            tickers     : Filter to these tickers (None = keep all)
            max_rows    : Cap rows for development testing

        Returns:
            Normalised DataFrame with transcripts_raw schema
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("Run: pip install datasets")

        log.info(f"Downloading HuggingFace dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")

        if max_rows:
            dataset = dataset.select(range(min(max_rows, len(dataset))))

        df = dataset.to_pandas()
        log.info(f"Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
        log.debug(f"Columns: {list(df.columns)}")

        df = self._normalise_schema(df, source="hf_bose345")

        if tickers:
            before = len(df)
            df = df[df["ticker"].isin(tickers)].reset_index(drop=True)
            log.info(f"Filtered {before:,} → {len(df):,} rows for {len(tickers)} tickers")

        return df

    def load_from_local(self, file_path: str) -> pd.DataFrame:
        """Load from a local CSV or Parquet file."""
        path = Path(file_path)
        log.info(f"Loading local file: {path}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return self._normalise_schema(df, source="local")

    def save_raw(
        self,
        df: pd.DataFrame,
        filename: str = "transcripts_raw.parquet",
    ) -> Path:
        """Persist normalised transcripts to raw Parquet storage."""
        out_path = self.raw_dir / filename
        save_parquet(df, out_path)
        return out_path

    # ── Private helpers ─────────────────────────────────────────

    def _normalise_schema(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Map arbitrary column names to the transcripts_raw schema.
        Handles common naming variants across HuggingFace datasets.
        """
        col_map = {
            "symbol": "ticker",
            "company": "company_name",
            "date": "earnings_date",
            "text": "transcript_text",
            "content": "transcript_text",
            "transcript": "transcript_text",
            "quarter": "fiscal_quarter",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "earnings_date" in df.columns:
            df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")

        if "transcript_text" in df.columns:
            df["word_count"] = df["transcript_text"].str.split().str.len()

        df["source"] = source
        df["transcript_id"] = df.apply(
            lambda r: self._make_id(r.get("ticker", "UNK"), r.get("earnings_date")),
            axis=1,
        )

        required = [
            "transcript_id", "ticker", "earnings_date",
            "transcript_text", "word_count", "source",
        ]
        for col in required:
            if col not in df.columns:
                df[col] = None

        return df.reset_index(drop=True)

    @staticmethod
    def _make_id(ticker: str, date: object) -> str:
        """Generate deterministic transcript ID: TICKER_YYYYMMDD."""
        try:
            suffix = pd.Timestamp(date).strftime("%Y%m%d")
        except Exception:
            suffix = "UNKNOWN"
        return f"{ticker}_{suffix}"
