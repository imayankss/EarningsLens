"""
src/utils/logger.py
====================
Centralised loguru setup.
Every module calls get_logger(__name__) — never configure loguru twice.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting pipeline")
"""
from __future__ import annotations
import sys
from pathlib import Path

from loguru import logger


def get_logger(name: str, log_dir: str = "logs") -> "logger":
    """
    Returns a configured loguru logger bound to `name`.

    Args:
        name   : Module name string (use __name__)
        log_dir: Directory for log files (created if missing)

    Returns:
        Bound loguru logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler to avoid duplicate output
    logger.remove()

    # Console: coloured, human-readable
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[name]}</cyan> | "
            "{message}"
        ),
        level="INFO",
        colorize=True,
    )

    # File: full debug log, rotated at 10 MB
    safe_name = name.replace("/", ".").replace("\\", ".")
    logger.add(
        log_path / f"{safe_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    return logger.bind(name=name)
