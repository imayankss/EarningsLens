"""Small I/O helpers for project scripts and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _table_candidates(path_or_stem: str | Path) -> list[Path]:
    path = Path(path_or_stem)
    if path.suffix.lower() == ".parquet":
        return [path, path.with_suffix(".csv")]
    if path.suffix.lower() == ".csv":
        return [path, path.with_suffix(".parquet")]
    return [path.with_suffix(".parquet"), path.with_suffix(".csv")]


def read_table(
    path_or_stem: str | Path,
    required: bool = True,
    label: str | None = None,
) -> pd.DataFrame:
    """Read a parquet or CSV table, with parquet/CSV fallback by stem."""

    for candidate in _table_candidates(path_or_stem):
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".parquet":
            return pd.read_parquet(candidate)
        if candidate.suffix.lower() == ".csv":
            return pd.read_csv(candidate)

    if required:
        name = label or str(path_or_stem)
        candidates = ", ".join(str(path) for path in _table_candidates(path_or_stem))
        raise FileNotFoundError(f"Missing required table for {name}. Tried: {candidates}")
    return pd.DataFrame()


def write_table(
    df: pd.DataFrame,
    path_or_stem: str | Path,
    write_csv: bool = True,
    write_parquet: bool = True,
) -> list[Path]:
    """Write a DataFrame to CSV and/or parquet and return written paths."""

    path = Path(path_or_stem)
    written: list[Path] = []

    if path.suffix.lower() == ".csv":
        if write_csv:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            written.append(path)
        return written

    if path.suffix.lower() == ".parquet":
        if write_parquet:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            written.append(path)
        return written

    if write_parquet:
        parquet_path = path.with_suffix(".parquet")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        written.append(parquet_path)
    if write_csv:
        csv_path = path.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        written.append(csv_path)
    return written


def safe_read_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON, returning default if the file is missing or malformed."""

    fallback = {} if default is None else default
    try:
        candidate = Path(path)
        if not candidate.exists():
            return fallback
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def write_json(data: Any, path: str | Path) -> Path:
    """Write JSON with a stable, readable representation."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()


def list_files(directory: str | Path, pattern: str = "*") -> list[Path]:
    base = Path(directory)
    if not base.exists():
        return []
    return sorted(base.glob(pattern))


def format_file_size(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.exists():
        return "missing"

    size = candidate.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"
