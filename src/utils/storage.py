"""
src/utils/storage.py
=====================
Parquet read/write helpers and DuckDB connection manager.

Architecture:
  - Write all data as Parquet (columnar, compressed, fast)
  - Query Parquet files via DuckDB in-memory (zero-copy)
  - Never write to CSV in production paths

Pattern:
    save_parquet(df, "data/processed/sentiment/aapl.parquet")
    df = load_parquet("data/processed/sentiment/aapl.parquet")
    results = query_parquet("SELECT * FROM 'data/processed/sentiment/*.parquet'")
"""
from __future__ import annotations
from pathlib import Path

import duckdb
import pandas as pd

from .config_loader import load_config
from .logger import get_logger

log = get_logger(__name__)


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    compression: str = "snappy",
) -> Path:
    """
    Save DataFrame to Parquet.

    Args:
        df         : DataFrame to save
        path       : Output file path (parent dirs created automatically)
        compression: Parquet compression codec (snappy | gzip | zstd)

    Returns:
        Resolved Path of saved file
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression=compression)
    log.info(f"Saved {len(df):,} rows → {out}")
    return out


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load Parquet file into DataFrame."""
    return pd.read_parquet(Path(path))


def query_parquet(sql: str) -> pd.DataFrame:
    """
    Run SQL directly against Parquet files using DuckDB.

    Example:
        df = query_parquet(
            "SELECT ticker, AVG(finbert_score) FROM "
            "'data/processed/sentiment/*.parquet' GROUP BY ticker"
        )
    """
    conn = duckdb.connect(":memory:")
    result = conn.execute(sql).df()
    conn.close()
    return result


def get_db_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Return a persistent DuckDB connection for the project database.

    Args:
        read_only: Open in read-only mode (safe for concurrent reads)

    Returns:
        DuckDB connection object
    """
    config = load_config()
    db_path = config["paths"]["db"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(database=db_path, read_only=read_only)
    log.debug(f"DuckDB connected: {db_path}")
    return conn
