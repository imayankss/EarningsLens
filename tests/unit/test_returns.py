"""Unit tests for return computation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pytest
from src.finance.returns import ReturnComputer


class TestReturnComputer:
    def setup_method(self):
        self.rc = ReturnComputer()

    def test_simple_return(self):
        assert abs(self.rc.simple_return(100, 110) - 0.10) < 1e-10

    def test_simple_return_zero_start(self):
        import math
        assert math.isnan(self.rc.simple_return(0, 100))

    def test_cumulative_return_compound(self):
        rets = pd.Series([0.10, -0.05, 0.08])
        expected = (1.10 * 0.95 * 1.08) - 1
        assert abs(self.rc.cumulative_return(rets) - expected) < 1e-10

    def test_abnormal_return(self):
        stock  = pd.Series([0.03, 0.02])
        market = pd.Series([0.01, 0.01])
        ar = self.rc.abnormal_return(stock, market)
        assert ar.round(2).tolist() == [0.02, 0.01]

    def test_car_is_sum(self):
        ar  = pd.Series([0.01, 0.02, -0.005])
        car = self.rc.cumulative_abnormal_return(ar)
        assert abs(car - ar.sum()) < 1e-12
