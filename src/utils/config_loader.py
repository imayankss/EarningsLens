"""
src/utils/config_loader.py
===========================
Loads configs/config.yaml and configs/tickers.yaml.
Also reads .env via python-dotenv.
All modules should import config through here — never load YAML directly.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env from project root on first import
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Load master config from YAML.

    Args:
        config_path: Relative path from project root (default: configs/config.yaml)

    Returns:
        Parsed config dict

    Raises:
        FileNotFoundError: If config file does not exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path.resolve()}\n"
            f"Run from the project root directory."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_tickers(tickers_path: str = "configs/tickers.yaml") -> dict[str, Any]:
    """Load ticker configuration from YAML."""
    path = Path(tickers_path)
    if not path.exists():
        raise FileNotFoundError(f"Tickers config not found: {path.resolve()}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_all_tickers() -> list[str]:
    """Return the flat all_tickers list from configs/tickers.yaml."""
    config = load_tickers()
    return config.get("all_tickers", [])


def get_date_range() -> tuple[str, str]:
    """Return (start, end) date strings from tickers config."""
    config = load_tickers()
    dr = config.get("date_range", {})
    return dr.get("start", "2019-01-01"), dr.get("end", "2024-12-31")
