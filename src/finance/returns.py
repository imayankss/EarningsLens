"""
src/finance/returns.py
=======================
Return computation utilities for the event study.

Formulas:
    Simple return   : R_t = (P_t - P_{t-1}) / P_{t-1}
    Abnormal return : AR  = R_stock - R_benchmark
    Cumulative AR   : CAR = Σ AR_t  (or compound version)

Usage:
    rc = ReturnComputer()
    ar  = rc.abnormal_return(stock_ret, market_ret)
    car = rc.cumulative_return(stock_returns_series)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class ReturnComputer:
    """Utility class for return and abnormal return calculations."""

    @staticmethod
    def simple_return(p_start: float, p_end: float) -> float:
        """R = (P_end - P_start) / P_start"""
        if p_start == 0:
            return np.nan
        return (p_end - p_start) / p_start

    @staticmethod
    def cumulative_return(returns: pd.Series) -> float:
        """
        Compound cumulative return over a window.
        CR = (1+r1)(1+r2)...(1+rN) − 1
        """
        return float((1 + returns.fillna(0)).prod() - 1)

    @staticmethod
    def abnormal_return(
        stock_returns : pd.Series,
        market_returns: pd.Series,
    ) -> pd.Series:
        """
        Market-adjusted abnormal return series.
        AR_t = R_stock_t − R_benchmark_t
        """
        return stock_returns - market_returns

    @staticmethod
    def cumulative_abnormal_return(ar_series: pd.Series) -> float:
        """
        Simple sum of abnormal returns (standard event-study CAR).
        CAR = Σ AR_t
        """
        return float(ar_series.sum())

    @staticmethod
    def compute_beta(
        stock_returns : pd.Series,
        market_returns: pd.Series,
    ) -> float:
        """
        OLS beta from estimation window.
        β = Cov(R_stock, R_market) / Var(R_market)

        Used for market-model abnormal returns (more rigorous than
        simple market-adjusted returns).
        """
        aligned = pd.concat(
            [stock_returns, market_returns], axis=1
        ).dropna()
        if len(aligned) < 30:
            log.warning("Fewer than 30 observations for beta estimation.")
            return 1.0   # default to market beta

        cov_matrix = aligned.cov()
        cols = aligned.columns.tolist()
        beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]
        return float(beta)
