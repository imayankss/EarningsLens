"""Lightweight stdlib logging helpers for scripts."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib logging once for CLI scripts."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
