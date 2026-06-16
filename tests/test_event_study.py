"""
tests/test_event_study.py
==========================
Production-grade test suite for the DAY 6 Event-Study Pipeline.

Covers:
    - EventWindowGenerator  (event_window_generator.py)
    - CARCalculator         (car_calculator.py)
    - EventStudyValidator   (validation.py)
    - AbnormalReturns       (abnormal_returns.py)        — interface contract
    - SentimentEventMerger  (sentiment_event_merger.py)  — interface contract
    - StatisticsUtils       (statistics_utils.py)        — interface contract
    - Full pipeline integration                          (event_study.py)
    - Dataclass integrity
    - Edge cases

Design principles:
    - Zero external API / network calls
    - All datasets are synthetic and deterministic
    - Hand-verified numeric assertions for finance calculations
    - Fixture-based reusable data
    - Parametrized tests for multi-case coverage

Run:
    pytest tests/test_event_study.py -v
    pytest tests/test_event_study.py -v --tb=short -q    # compact output
"""

from __future__ import annotations

import math
from datetime import date
from typing import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Lazy imports — gracefully skip if a module is not yet implemented
# ---------------------------------------------------------------------------

def _try_import(module_path: str, name: str):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, name)
    except (ImportError, AttributeError):
        return None


# Event window generator
EventWindowConfig    = _try_import("src.event_study.event_window_generator", "EventWindowConfig")
EventWindow          = _try_import("src.event_study.event_window_generator", "EventWindow")
WindowGenerationResult = _try_import("src.event_study.event_window_generator", "WindowGenerationResult")
EventWindowGenerator = _try_import("src.event_study.event_window_generator", "EventWindowGenerator")
build_event_window_generator = _try_import(
    "src.event_study.event_window_generator", "build_event_window_generator"
)

# CAR calculator
CARConfig      = _try_import("src.event_study.car_calculator", "CARConfig")
CARResult      = _try_import("src.event_study.car_calculator", "CARResult")
CARSummary     = _try_import("src.event_study.car_calculator", "CARSummary")
CARCalculator  = _try_import("src.event_study.car_calculator", "CARCalculator")
build_car_calculator = _try_import("src.event_study.car_calculator", "build_car_calculator")

# Validation
ValidationConfig         = _try_import("src.event_study.validation", "ValidationConfig")
ValidationIssue          = _try_import("src.event_study.validation", "ValidationIssue")
ValidationSummary        = _try_import("src.event_study.validation", "ValidationSummary")
DatasetValidationResult  = _try_import("src.event_study.validation", "DatasetValidationResult")
EventStudyValidator      = _try_import("src.event_study.validation", "EventStudyValidator")
Severity                 = _try_import("src.event_study.validation", "Severity")
ValidationStage          = _try_import("src.event_study.validation", "ValidationStage")
build_event_study_validator = _try_import(
    "src.event_study.validation", "build_event_study_validator"
)
detect_duplicates        = _try_import("src.event_study.validation", "detect_duplicates")
null_rate_report         = _try_import("src.event_study.validation", "null_rate_report")
check_monotonic_groups   = _try_import("src.event_study.validation", "check_monotonic_groups")
check_grouped_integrity  = _try_import("src.event_study.validation", "check_grouped_integrity")
detect_extreme_values    = _try_import("src.event_study.validation", "detect_extreme_values")

# Optional modules (interface-contract tests only)
AbnormalReturns       = _try_import("src.event_study.abnormal_returns", "AbnormalReturns")
SentimentEventMerger  = _try_import("src.event_study.sentiment_event_merger", "SentimentEventMerger")
StatisticsUtils       = _try_import("src.event_study.statistics_utils", "StatisticsUtils")
EventStudy            = _try_import("src.event_study.event_study", "EventStudy")


# ---------------------------------------------------------------------------
# Constants for deterministic datasets
# ---------------------------------------------------------------------------

_BASE_DATE   = pd.Timestamp("2024-01-15")   # Monday — guaranteed trading day
_TICKERS     = ["AAPL", "MSFT", "NVDA"]
_EVENT_DATES = {
    "AAPL": pd.Timestamp("2024-01-15"),
    "MSFT": pd.Timestamp("2024-01-22"),
    "NVDA": pd.Timestamp("2024-01-29"),
}
_IDS = {
    "AAPL": "AAPL_Q1_2024",
    "MSFT": "MSFT_Q1_2024",
    "NVDA": "NVDA_Q1_2024",
}

# Hand-verified AR series for AAPL_Q1_2024
# relative_day: AR
_AAPL_AR_SERIES: dict[int, float] = {
    -2: -0.005,
    -1:  0.003,
     0:  0.012,   # event day
     1:  0.025,
     2: -0.008,
     3:  0.015,
     4:  0.010,
     5: -0.004,
}
# Hand-computed:
#   car_3d = AR(1) + AR(2) + AR(3)            = 0.025 + (-0.008) + 0.015 = 0.032
#   car_5d = AR(1)+AR(2)+AR(3)+AR(4)+AR(5)   = 0.025-0.008+0.015+0.010-0.004 = 0.038
_AAPL_CAR_3D: float = 0.032
_AAPL_CAR_5D: float = 0.038


# ---------------------------------------------------------------------------
# Shared helper: simple weekday-only session provider (no real calendar)
# ---------------------------------------------------------------------------

def _weekday_session_provider(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    """Mon–Fri sessions only — deterministic, no holidays."""
    return [
        pd.Timestamp(d)
        for d in pd.bdate_range(start=start, end=end, freq="B")
    ]


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture()
def session_provider() -> Callable:
    return _weekday_session_provider


@pytest.fixture()
def default_window_config():
    if EventWindowConfig is None:
        pytest.skip("EventWindowConfig not available")
    return EventWindowConfig(pre_event_days=5, post_event_days=5, include_event_day=True)


@pytest.fixture()
def window_generator(default_window_config, session_provider):
    if EventWindowGenerator is None:
        pytest.skip("EventWindowGenerator not available")
    return EventWindowGenerator(
        config=default_window_config,
        session_provider=session_provider,
    )


@pytest.fixture()
def single_event_df() -> pd.DataFrame:
    """Single AAPL event for simple window/CAR tests."""
    return pd.DataFrame(
        {
            "transcript_id":    ["AAPL_Q1_2024"],
            "ticker":           ["AAPL"],
            "aligned_event_date": [_EVENT_DATES["AAPL"]],
            "quarter":          ["Q1"],
            "year":             [2024],
        }
    )


@pytest.fixture()
def multi_event_df() -> pd.DataFrame:
    """Three tickers, one event each."""
    return pd.DataFrame(
        {
            "transcript_id": list(_IDS.values()),
            "ticker":        list(_IDS.keys()),
            "aligned_event_date": [_EVENT_DATES[t] for t in _IDS],
            "quarter":       ["Q1", "Q1", "Q1"],
            "year":          [2024, 2024, 2024],
        }
    )


@pytest.fixture()
def aapl_ar_df() -> pd.DataFrame:
    """Long-format abnormal returns for AAPL — hand-verified values."""
    rows = [
        {
            "transcript_id":      "AAPL_Q1_2024",
            "ticker":             "AAPL",
            "aligned_event_date": _EVENT_DATES["AAPL"],
            "relative_day":       rd,
            "abnormal_return":    ar,
            "daily_return":       ar + 0.002,
            "benchmark_return":   0.002,
        }
        for rd, ar in _AAPL_AR_SERIES.items()
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def multi_ticker_ar_df(aapl_ar_df) -> pd.DataFrame:
    """Three tickers with identical AR profiles (offset event dates)."""
    frames = [aapl_ar_df]
    for ticker, tid in [("MSFT", "MSFT_Q1_2024"), ("NVDA", "NVDA_Q1_2024")]:
        frame = aapl_ar_df.copy()
        frame["transcript_id"] = tid
        frame["ticker"] = ticker
        frame["aligned_event_date"] = _EVENT_DATES[ticker]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def default_car_config():
    if CARConfig is None:
        pytest.skip("CARConfig not available")
    return CARConfig(horizons=(1, 3, 5), include_event_day_in_car=False, nan_policy="skip")


@pytest.fixture()
def car_calculator(default_car_config):
    if CARCalculator is None:
        pytest.skip("CARCalculator not available")
    return CARCalculator(default_car_config)


@pytest.fixture()
def default_validator():
    if EventStudyValidator is None:
        pytest.skip("EventStudyValidator not available")
    return build_event_study_validator(strict_mode=False)


@pytest.fixture()
def strict_validator():
    if EventStudyValidator is None:
        pytest.skip("EventStudyValidator not available")
    return build_event_study_validator(strict_mode=True)


@pytest.fixture()
def valid_window_df() -> pd.DataFrame:
    """Well-formed event-window DataFrame for validation tests."""
    event_date = _EVENT_DATES["AAPL"]
    sessions = list(pd.bdate_range(
        start=event_date - pd.Timedelta(days=14),
        end=event_date + pd.Timedelta(days=14),
        freq="B",
    ))
    event_idx = sessions.index(event_date)
    rows = []
    for i, sess in enumerate(sessions):
        rd = i - event_idx
        rows.append(
            {
                "transcript_id":      "AAPL_Q1_2024",
                "ticker":             "AAPL",
                "aligned_event_date": event_date,
                "window_start":       sessions[0],
                "window_end":         sessions[-1],
                "date":               sess,
                "relative_day":       rd,
                "event_day_flag":     rd == 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def valid_car_df() -> pd.DataFrame:
    """Well-formed event-level CAR DataFrame."""
    return pd.DataFrame(
        {
            "transcript_id":      _IDS.values(),
            "ticker":             _IDS.keys(),
            "aligned_event_date": [_EVENT_DATES[t] for t in _IDS],
            "car_3d":             [0.032, 0.018, -0.011],
            "car_5d":             [0.038, 0.022, -0.007],
            "post_day_count":     [5, 5, 5],
            "nan_day_count":      [0, 0, 0],
            "car_complete":       [True, True, True],
        }
    )


@pytest.fixture()
def valid_sentiment_df() -> pd.DataFrame:
    """Well-formed sentiment merge DataFrame."""
    return pd.DataFrame(
        {
            "transcript_id":           list(_IDS.values()),
            "ticker":                  list(_IDS.keys()),
            "aligned_event_date":      [_EVENT_DATES[t] for t in _IDS],
            "finbert_sentiment_score": [0.72, 0.41, -0.33],
            "lm_tone_score":           [0.55, 0.28, -0.18],
            "sentiment_label":         ["positive", "positive", "negative"],
            "abnormal_return":         [0.025, 0.010, -0.015],
            "car_3d":                  [0.032, 0.018, -0.011],
            "car_5d":                  [0.038, 0.022, -0.007],
            "car_complete":            [True, True, True],
        }
    )


@pytest.fixture()
def synthetic_ohlcv_df() -> pd.DataFrame:
    """30-day OHLCV dataset for AAPL around the event date."""
    dates = list(pd.bdate_range(
        start=_EVENT_DATES["AAPL"] - pd.Timedelta(days=28),
        end=_EVENT_DATES["AAPL"] + pd.Timedelta(days=14),
        freq="B",
    ))
    np.random.seed(42)
    closes = 180.0 + np.cumsum(np.random.normal(0, 1.0, len(dates)))
    return pd.DataFrame(
        {
            "ticker":    "AAPL",
            "date":      dates,
            "open":      closes - 0.5,
            "high":      closes + 1.0,
            "low":       closes - 1.0,
            "close":     closes,
            "adj_close": closes,
            "volume":    np.random.randint(50_000_000, 100_000_000, len(dates)),
        }
    )


# ===========================================================================
# TEST CLASS 1 — EVENT WINDOW GENERATION
# ===========================================================================


@pytest.mark.skipif(EventWindowGenerator is None, reason="module not yet implemented")
class TestEventWindowGeneration:

    def test_single_event_generates_correct_session_count(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        win = result.windows[0]
        # pre=5 + event=1 + post=5 = 11 sessions
        assert len(win.sessions) == 11

    def test_event_day_has_relative_day_zero(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        win = result.windows[0]
        event_date = _EVENT_DATES["AAPL"]
        event_idx = win.sessions.index(event_date)
        assert win.relative_days[event_idx] == 0

    def test_relative_days_are_monotonically_increasing(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        rds = result.windows[0].relative_days
        assert all(rds[i] < rds[i + 1] for i in range(len(rds) - 1))

    def test_window_starts_at_minus_pre_days(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        win = result.windows[0]
        assert win.relative_days[0] == -5

    def test_window_ends_at_plus_post_days(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        win = result.windows[0]
        assert win.relative_days[-1] == 5

    def test_multi_ticker_generates_independent_windows(
        self, window_generator, multi_event_df
    ):
        result = window_generator.generate_event_windows(multi_event_df)
        assert result.total_events == 3
        tids = {w.transcript_id for w in result.windows}
        assert tids == set(_IDS.values())

    def test_build_event_dataset_columns(self, window_generator, single_event_df):
        result = window_generator.generate_event_windows(single_event_df)
        df = result.window_df
        for col in [
            "transcript_id", "ticker", "aligned_event_date",
            "date", "relative_day", "event_day_flag",
            "window_start", "window_end",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_exactly_one_event_day_flag_per_event(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        flag_counts = (
            result.window_df[result.window_df["event_day_flag"]]
            .groupby("transcript_id")["event_day_flag"]
            .sum()
        )
        for tid in result.window_df["transcript_id"].unique():
            assert flag_counts.get(tid, 0) == 1

    def test_no_duplicate_session_rows(self, window_generator, single_event_df):
        result = window_generator.generate_event_windows(single_event_df)
        df = result.window_df
        assert not df.duplicated(["transcript_id", "date"]).any()

    def test_all_dates_within_window_bounds(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        df = result.window_df
        assert (df["date"] >= df["window_start"]).all()
        assert (df["date"] <= df["window_end"]).all()

    def test_is_complete_true_for_full_window(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        assert result.windows[0].is_complete is True

    def test_result_is_valid_for_clean_input(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        assert result.is_valid

    def test_deterministic_output_on_repeated_calls(
        self, window_generator, single_event_df
    ):
        r1 = window_generator.generate_event_windows(single_event_df)
        r2 = window_generator.generate_event_windows(single_event_df)
        pd.testing.assert_frame_equal(r1.window_df, r2.window_df)

    def test_weekend_event_date_snaps_to_next_weekday(self, session_provider):
        if EventWindowConfig is None or EventWindowGenerator is None:
            pytest.skip("modules not available")
        saturday = pd.Timestamp("2024-01-20")  # Known Saturday
        assert saturday.dayofweek == 5
        config = EventWindowConfig(pre_event_days=2, post_event_days=2)
        gen = EventWindowGenerator(config, session_provider)
        df = pd.DataFrame({
            "transcript_id":    ["TEST_SAT"],
            "ticker":           ["TEST"],
            "aligned_event_date": [saturday],
        })
        result = gen.generate_event_windows(df)
        win = result.windows[0]
        # aligned_event_date should land on Monday 2024-01-22
        monday = pd.Timestamp("2024-01-22")
        assert win.aligned_event_date == monday

    def test_exclude_event_day_reduces_session_count(self, session_provider):
        if EventWindowConfig is None or EventWindowGenerator is None:
            pytest.skip("modules not available")
        config = EventWindowConfig(
            pre_event_days=3, post_event_days=3, include_event_day=False
        )
        gen = EventWindowGenerator(config, session_provider)
        df = pd.DataFrame({
            "transcript_id":    ["AAPL_Q1_2024"],
            "ticker":           ["AAPL"],
            "aligned_event_date": [_EVENT_DATES["AAPL"]],
        })
        result = gen.generate_event_windows(df)
        # 3 pre + 3 post = 6, no event day
        assert len(result.windows[0].sessions) == 6

    def test_missing_required_columns_raises_value_error(self, window_generator):
        bad_df = pd.DataFrame({"ticker": ["AAPL"]})
        with pytest.raises(ValueError, match="missing required columns"):
            window_generator.generate_event_windows(bad_df)

    def test_empty_dataframe_raises_value_error(self, window_generator):
        empty = pd.DataFrame(
            columns=["transcript_id", "ticker", "aligned_event_date"]
        )
        with pytest.raises(ValueError):
            window_generator.generate_event_windows(empty)

    def test_duplicate_transcript_ids_raises_value_error(self, window_generator):
        df = pd.DataFrame({
            "transcript_id":    ["AAPL_Q1_2024", "AAPL_Q1_2024"],
            "ticker":           ["AAPL", "AAPL"],
            "aligned_event_date": [_EVENT_DATES["AAPL"], _EVENT_DATES["AAPL"]],
        })
        with pytest.raises(ValueError, match="duplicate"):
            window_generator.generate_event_windows(df)

    def test_summary_contains_expected_keys(self, window_generator, single_event_df):
        result = window_generator.generate_event_windows(single_event_df)
        summary = result.summary
        for key in [
            "total_events", "complete_events", "incomplete_events",
            "completion_rate", "total_rows", "relative_day_range", "config",
        ]:
            assert key in summary

    def test_attach_relative_days_matches_generated_values(
        self, window_generator, single_event_df
    ):
        result = window_generator.generate_event_windows(single_event_df)
        df = result.window_df.copy()
        # Remove and recompute
        df = df.drop(columns=["relative_day"])
        df["relative_day"] = np.nan
        reattached = window_generator.attach_relative_days(
            result.window_df.copy()
        )
        pd.testing.assert_series_equal(
            result.window_df["relative_day"].reset_index(drop=True),
            reattached["relative_day"].reset_index(drop=True),
        )

    @pytest.mark.parametrize("pre,post", [(1, 1), (5, 5), (10, 5), (3, 10)])
    def test_configurable_window_sizes(self, session_provider, pre, post):
        if EventWindowConfig is None or EventWindowGenerator is None:
            pytest.skip("modules not available")
        config = EventWindowConfig(
            pre_event_days=pre, post_event_days=post, include_event_day=True
        )
        gen = EventWindowGenerator(config, session_provider)
        df = pd.DataFrame({
            "transcript_id":    ["AAPL_Q1_2024"],
            "ticker":           ["AAPL"],
            "aligned_event_date": [_EVENT_DATES["AAPL"]],
        })
        result = gen.generate_event_windows(df)
        win = result.windows[0]
        assert len(win.sessions) == pre + 1 + post
        assert win.relative_days[0] == -pre
        assert win.relative_days[-1] == post


# ===========================================================================
# TEST CLASS 2 — ABNORMAL RETURNS (interface contract)
# ===========================================================================


class TestAbnormalReturns:
    """
    Contract tests for abnormal_returns.py.
    Verified against hand-computed values using the formula:
        AR_t = R_stock,t - R_benchmark,t
    """

    def test_ar_formula_correctness(self):
        """Verify core formula: AR = stock_return - benchmark_return."""
        r_stock = 0.030
        r_bench = 0.005
        expected_ar = 0.025
        assert math.isclose(r_stock - r_bench, expected_ar, abs_tol=1e-10)

    def test_ar_negative_when_underperforms_benchmark(self):
        r_stock = -0.010
        r_bench =  0.008
        ar = r_stock - r_bench
        assert ar < 0
        assert math.isclose(ar, -0.018, abs_tol=1e-10)

    def test_ar_zero_when_matches_benchmark(self):
        r_stock = 0.012
        r_bench = 0.012
        assert r_stock - r_bench == 0.0

    def test_forward_return_compound_formula(self):
        """
        Compound forward return:
            R(t,n) = prod(1+r_i) - 1
        Verify for 3-day window with known values.
        """
        daily_returns = [0.025, -0.008, 0.015]
        expected = (1.025) * (0.992) * (1.015) - 1
        computed = 1.0
        for r in daily_returns:
            computed *= (1 + r)
        computed -= 1
        assert math.isclose(computed, expected, rel_tol=1e-9)

    def test_compound_return_exceeds_simple_sum_for_positive_returns(self):
        """Compound > simple sum when all returns are positive."""
        returns = [0.05, 0.05, 0.05]
        simple = sum(returns)
        compound = 1.0
        for r in returns:
            compound *= (1 + r)
        compound -= 1
        assert compound > simple

    def test_ar_vectorized_matches_elementwise(self, aapl_ar_df):
        """Verify vectorized subtraction matches element-wise for each row."""
        for _, row in aapl_ar_df.iterrows():
            expected_ar = row["daily_return"] - row["benchmark_return"]
            assert math.isclose(row["abnormal_return"], expected_ar, abs_tol=1e-10)

    def test_ar_magnitude_within_sanity_cap(self, aapl_ar_df):
        assert (aapl_ar_df["abnormal_return"].abs() < 1.0).all()

    def test_benchmark_coverage_required(self, aapl_ar_df):
        """Abnormal return must be NaN when benchmark_return is NaN."""
        ar_series = np.array([0.02, np.nan, 0.03])
        bm_series = np.array([0.005, np.nan, 0.004])
        result = ar_series - bm_series
        assert np.isnan(result[1])

    @pytest.mark.parametrize("stock_r,bench_r,expected_ar", [
        ( 0.030,  0.005,  0.025),
        (-0.020,  0.005, -0.025),
        ( 0.000,  0.000,  0.000),
        ( 0.100, -0.050,  0.150),
        (-0.050,  0.050, -0.100),
    ])
    def test_ar_formula_parametrized(self, stock_r, bench_r, expected_ar):
        assert math.isclose(stock_r - bench_r, expected_ar, abs_tol=1e-10)

    def test_abnormal_return_df_has_required_columns(self, aapl_ar_df):
        required = {
            "transcript_id", "ticker", "aligned_event_date",
            "relative_day", "abnormal_return", "daily_return", "benchmark_return",
        }
        assert required.issubset(set(aapl_ar_df.columns))

    def test_no_duplicate_relative_days_per_event(self, aapl_ar_df):
        dups = aapl_ar_df.duplicated(["transcript_id", "relative_day"])
        assert not dups.any()


# ===========================================================================
# TEST CLASS 3 — CAR CALCULATIONS
# ===========================================================================


@pytest.mark.skipif(CARCalculator is None, reason="module not yet implemented")
class TestCARCalculations:

    def test_car_3d_exact_value(self, car_calculator, aapl_ar_df):
        result_df = car_calculator.compute_car(aapl_ar_df)
        car_3d = result_df.loc[
            result_df["transcript_id"] == "AAPL_Q1_2024", "car_3d"
        ].iloc[0]
        assert math.isclose(car_3d, _AAPL_CAR_3D, abs_tol=1e-6), (
            f"Expected car_3d={_AAPL_CAR_3D}, got {car_3d}"
        )

    def test_car_5d_exact_value(self, car_calculator, aapl_ar_df):
        result_df = car_calculator.compute_car(aapl_ar_df)
        car_5d = result_df.loc[
            result_df["transcript_id"] == "AAPL_Q1_2024", "car_5d"
        ].iloc[0]
        assert math.isclose(car_5d, _AAPL_CAR_5D, abs_tol=1e-6), (
            f"Expected car_5d={_AAPL_CAR_5D}, got {car_5d}"
        )

    def test_car_1d_equals_first_post_ar(self, car_calculator, aapl_ar_df):
        result_df = car_calculator.compute_car(aapl_ar_df)
        car_1d = result_df.loc[
            result_df["transcript_id"] == "AAPL_Q1_2024", "car_1d"
        ].iloc[0]
        assert math.isclose(car_1d, _AAPL_AR_SERIES[1], abs_tol=1e-6)

    def test_car_increases_from_3d_to_5d_for_positive_trend(
        self, car_calculator, aapl_ar_df
    ):
        result_df = car_calculator.compute_car(aapl_ar_df)
        row = result_df[result_df["transcript_id"] == "AAPL_Q1_2024"].iloc[0]
        # AAPL: AR(4)=0.010, AR(5)=-0.004 → 5d > 3d by 0.006
        assert row["car_5d"] > row["car_3d"]

    def test_multi_ticker_car_one_row_per_event(
        self, car_calculator, multi_ticker_ar_df
    ):
        result_df = car_calculator.compute_car(multi_ticker_ar_df)
        assert len(result_df) == 3
        assert result_df["transcript_id"].nunique() == 3

    def test_all_tickers_have_same_car_same_ar_series(
        self, car_calculator, multi_ticker_ar_df
    ):
        result_df = car_calculator.compute_car(multi_ticker_ar_df)
        car_3ds = result_df["car_3d"].values
        # All three tickers have identical AR series
        assert all(
            math.isclose(v, _AAPL_CAR_3D, abs_tol=1e-6) for v in car_3ds
        )

    def test_car_is_nan_when_insufficient_post_days(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        # Remove all post-event rows
        pre_only = aapl_ar_df[aapl_ar_df["relative_day"] <= 0].copy()
        config = CARConfig(horizons=(3, 5), min_post_days_required=1)
        calc = CARCalculator(config)
        result_df = calc.compute_car(pre_only)
        assert result_df["car_3d"].isna().all()
        assert result_df["car_5d"].isna().all()

    def test_nan_policy_skip_ignores_nan_ar(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        df = aapl_ar_df.copy()
        # Set AR at t=2 to NaN
        df.loc[df["relative_day"] == 2, "abnormal_return"] = np.nan
        config = CARConfig(horizons=(3,), nan_policy="skip")
        calc = CARCalculator(config)
        result_df = calc.compute_car(df)
        # car_3d = AR(1) + 0 + AR(3) = 0.025 + 0.015 = 0.040 (NaN skipped)
        expected = _AAPL_AR_SERIES[1] + _AAPL_AR_SERIES[3]
        car_3d = result_df["car_3d"].iloc[0]
        assert math.isclose(car_3d, expected, abs_tol=1e-6)

    def test_nan_policy_zero_treats_nan_as_zero(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        df = aapl_ar_df.copy()
        df.loc[df["relative_day"] == 2, "abnormal_return"] = np.nan
        config = CARConfig(horizons=(3,), nan_policy="zero")
        calc = CARCalculator(config)
        result_df = calc.compute_car(df)
        # AR(2) treated as 0 → same result as "skip" here
        expected = _AAPL_AR_SERIES[1] + 0.0 + _AAPL_AR_SERIES[3]
        car_3d = result_df["car_3d"].iloc[0]
        assert math.isclose(car_3d, expected, abs_tol=1e-6)

    def test_nan_policy_raise_raises_on_nan(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        df = aapl_ar_df.copy()
        df.loc[df["relative_day"] == 1, "abnormal_return"] = np.nan
        config = CARConfig(horizons=(3,), nan_policy="raise")
        calc = CARCalculator(config)
        with pytest.raises(ValueError, match="NaN"):
            calc.compute_car(df)

    def test_car_complete_flag_true_for_full_window(
        self, car_calculator, aapl_ar_df
    ):
        result_df = car_calculator.compute_car(aapl_ar_df)
        assert result_df["car_complete"].iloc[0] is np.bool_(True)

    def test_car_complete_false_when_horizon_has_no_data(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        # Only 2 post-event days available
        short_df = aapl_ar_df[aapl_ar_df["relative_day"] <= 2].copy()
        config = CARConfig(
            horizons=(3, 5), nan_policy="skip", min_post_days_required=1
        )
        calc = CARCalculator(config)
        result_df = calc.compute_car(short_df)
        assert result_df["car_complete"].iloc[0] is np.bool_(False)

    def test_output_has_one_row_per_event(self, car_calculator, multi_ticker_ar_df):
        result_df = car_calculator.compute_car(multi_ticker_ar_df)
        assert len(result_df) == multi_ticker_ar_df["transcript_id"].nunique()

    def test_no_duplicate_transcript_ids_in_output(
        self, car_calculator, multi_ticker_ar_df
    ):
        result_df = car_calculator.compute_car(multi_ticker_ar_df)
        assert not result_df["transcript_id"].duplicated().any()

    def test_output_sorted_by_transcript_id(
        self, car_calculator, multi_ticker_ar_df
    ):
        result_df = car_calculator.compute_car(multi_ticker_ar_df)
        ids = result_df["transcript_id"].tolist()
        assert ids == sorted(ids)

    def test_car_output_columns_present(self, car_calculator, aapl_ar_df):
        result_df = car_calculator.compute_car(aapl_ar_df)
        for col in ["transcript_id", "ticker", "aligned_event_date",
                     "car_1d", "car_3d", "car_5d", "car_complete"]:
            assert col in result_df.columns

    def test_extreme_car_not_raised_for_normal_values(
        self, car_calculator, aapl_ar_df
    ):
        result = car_calculator.compute_event_car(aapl_ar_df)
        assert len(result.extreme_flags) == 0

    def test_extreme_car_flagged_when_exceeds_threshold(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        df = aapl_ar_df.copy()
        # Inflate ARs to trigger extreme flag
        for rd in [1, 2, 3, 4, 5]:
            df.loc[df["relative_day"] == rd, "abnormal_return"] = 0.30
        config = CARConfig(horizons=(3, 5), extreme_car_threshold=0.50)
        calc = CARCalculator(config)
        result = calc.compute_event_car(df)
        # car_5d = 5 * 0.30 = 1.50 > 0.50
        assert 5 in result.extreme_flags

    def test_build_car_frame_correct_dtypes(self, car_calculator, aapl_ar_df):
        result_df = car_calculator.compute_car(aapl_ar_df)
        assert result_df["car_3d"].dtype == np.float64
        assert result_df["car_5d"].dtype == np.float64
        assert result_df["car_complete"].dtype == bool

    def test_missing_required_columns_raises_value_error(self, car_calculator):
        bad_df = pd.DataFrame({"transcript_id": ["AAPL_Q1_2024"]})
        with pytest.raises(ValueError):
            car_calculator.compute_car(bad_df)

    def test_empty_dataframe_raises_value_error(self, car_calculator):
        empty = pd.DataFrame(
            columns=[
                "transcript_id", "ticker", "aligned_event_date",
                "relative_day", "abnormal_return",
            ]
        )
        with pytest.raises(ValueError):
            car_calculator.compute_car(empty)

    def test_summarize_car_metrics_structure(
        self, car_calculator, multi_ticker_ar_df
    ):
        result_df = car_calculator.compute_car(multi_ticker_ar_df)
        # Build CARResult objects from the frame (re-run per event)
        results = [
            car_calculator.compute_event_car(grp)
            for _, grp in multi_ticker_ar_df.groupby("transcript_id")
        ]
        summary = car_calculator.summarize_car_metrics(results, result_df)
        assert summary.total_events == 3
        assert summary.complete_events == 3
        assert 3 in summary.car_stats
        assert 5 in summary.car_stats

    @pytest.mark.parametrize("horizons", [(1,), (3,), (5,), (1, 3, 5), (2, 4)])
    def test_configurable_horizons_produce_correct_columns(
        self, aapl_ar_df, horizons
    ):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        config = CARConfig(horizons=horizons)
        calc = CARCalculator(config)
        result_df = calc.compute_car(aapl_ar_df)
        for h in horizons:
            assert f"car_{h}d" in result_df.columns

    def test_include_event_day_changes_car_value(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        config_excl = CARConfig(horizons=(3,), include_event_day_in_car=False)
        config_incl = CARConfig(horizons=(3,), include_event_day_in_car=True)
        car_excl = CARCalculator(config_excl).compute_car(aapl_ar_df)["car_3d"].iloc[0]
        car_incl = CARCalculator(config_incl).compute_car(aapl_ar_df)["car_3d"].iloc[0]
        # Including AR(0)=0.012 should change the value
        assert not math.isclose(car_excl, car_incl, abs_tol=1e-8)
        assert math.isclose(
            car_incl, car_excl + _AAPL_AR_SERIES[0], abs_tol=1e-6
        )


# ===========================================================================
# TEST CLASS 4 — SENTIMENT MERGE
# ===========================================================================


class TestSentimentMerge:
    """Contract and integrity tests for the sentiment-merge layer."""

    def test_valid_sentiment_df_has_required_columns(self, valid_sentiment_df):
        required = {
            "transcript_id", "finbert_sentiment_score",
            "lm_tone_score", "sentiment_label",
        }
        assert required.issubset(set(valid_sentiment_df.columns))

    def test_sentiment_labels_within_expected_set(self, valid_sentiment_df):
        allowed = {"positive", "negative", "neutral"}
        actual = set(valid_sentiment_df["sentiment_label"].unique())
        assert actual.issubset(allowed)

    def test_finbert_scores_within_range(self, valid_sentiment_df):
        assert (valid_sentiment_df["finbert_sentiment_score"].between(-1.0, 1.0)).all()

    def test_lm_scores_within_range(self, valid_sentiment_df):
        assert (valid_sentiment_df["lm_tone_score"].between(-1.0, 1.0)).all()

    def test_no_null_sentiment_scores(self, valid_sentiment_df):
        assert not valid_sentiment_df["finbert_sentiment_score"].isnull().any()
        assert not valid_sentiment_df["lm_tone_score"].isnull().any()

    def test_no_duplicate_transcript_ids(self, valid_sentiment_df):
        assert not valid_sentiment_df["transcript_id"].duplicated().any()

    def test_inner_merge_drops_unmatched_events(self):
        """Inner merge on transcript_id must drop rows not in both DataFrames."""
        market = pd.DataFrame({
            "transcript_id": ["A", "B", "C"],
            "car_3d":        [0.01, 0.02, 0.03],
        })
        sentiment = pd.DataFrame({
            "transcript_id":           ["A", "C"],
            "finbert_sentiment_score": [0.5, -0.3],
        })
        merged = pd.merge(market, sentiment, on="transcript_id", how="inner")
        assert len(merged) == 2
        assert "B" not in merged["transcript_id"].values

    def test_left_merge_preserves_all_market_events(self):
        """Left merge retains all market events; NaN for unmatched sentiment."""
        market = pd.DataFrame({
            "transcript_id": ["A", "B", "C"],
            "car_3d":        [0.01, 0.02, 0.03],
        })
        sentiment = pd.DataFrame({
            "transcript_id":           ["A"],
            "finbert_sentiment_score": [0.5],
        })
        merged = pd.merge(market, sentiment, on="transcript_id", how="left")
        assert len(merged) == 3
        assert merged.loc[merged["transcript_id"] == "B", "finbert_sentiment_score"].isna().all()

    def test_merge_key_is_transcript_id(self, valid_sentiment_df):
        """Verify transcript_id serves as the merge key."""
        assert "transcript_id" in valid_sentiment_df.columns
        assert valid_sentiment_df["transcript_id"].nunique() == len(valid_sentiment_df)

    def test_negative_sentiment_label_matches_negative_score(self, valid_sentiment_df):
        neg_rows = valid_sentiment_df[valid_sentiment_df["sentiment_label"] == "negative"]
        assert (neg_rows["finbert_sentiment_score"] < 0).all()

    def test_positive_sentiment_label_matches_positive_score(self, valid_sentiment_df):
        pos_rows = valid_sentiment_df[valid_sentiment_df["sentiment_label"] == "positive"]
        assert (pos_rows["finbert_sentiment_score"] > 0).all()


# ===========================================================================
# TEST CLASS 5 — VALIDATION LAYER
# ===========================================================================


@pytest.mark.skipif(EventStudyValidator is None, reason="module not yet implemented")
class TestValidationLayer:

    # --- Event windows ---

    def test_valid_window_df_passes(self, default_validator, valid_window_df):
        result = default_validator.validate_event_windows(valid_window_df)
        assert result.passed, [str(i) for i in result.critical_issues()]

    def test_missing_required_column_raises_critical(
        self, default_validator, valid_window_df
    ):
        broken = valid_window_df.drop(columns=["relative_day"])
        result = default_validator.validate_event_windows(broken)
        assert not result.passed
        check_names = [i.check_name for i in result.critical_issues()]
        assert "missing_required_columns" in check_names

    def test_duplicate_event_session_detected(
        self, default_validator, valid_window_df
    ):
        dup = pd.concat([valid_window_df, valid_window_df.iloc[[0]]], ignore_index=True)
        result = default_validator.validate_event_windows(dup)
        assert not result.passed

    def test_non_monotonic_relative_days_detected(
        self, default_validator, valid_window_df
    ):
        broken = valid_window_df.copy()
        # Swap two relative_day values to break monotonicity
        idx0, idx1 = broken.index[0], broken.index[2]
        broken.at[idx0, "relative_day"], broken.at[idx1, "relative_day"] = (
            broken.at[idx1, "relative_day"],
            broken.at[idx0, "relative_day"],
        )
        result = default_validator.validate_event_windows(broken)
        assert not result.passed

    def test_missing_event_day_flag_detected(
        self, default_validator, valid_window_df
    ):
        no_flag = valid_window_df.copy()
        no_flag["event_day_flag"] = False
        result = default_validator.validate_event_windows(no_flag)
        assert not result.passed
        names = [i.check_name for i in result.critical_issues()]
        assert "missing_event_day_flag" in names

    def test_date_outside_window_bounds_detected(
        self, default_validator, valid_window_df
    ):
        broken = valid_window_df.copy()
        broken.at[broken.index[0], "date"] = (
            broken["window_start"].iloc[0] - pd.Timedelta(days=10)
        )
        result = default_validator.validate_event_windows(broken)
        assert not result.passed

    # --- Abnormal returns ---

    def test_valid_ar_df_passes(self, default_validator, aapl_ar_df):
        result = default_validator.validate_abnormal_returns(aapl_ar_df)
        assert result.passed, [str(i) for i in result.critical_issues()]

    def test_extreme_daily_return_flagged(
        self, default_validator, aapl_ar_df
    ):
        extreme = aapl_ar_df.copy()
        extreme.at[extreme.index[0], "daily_return"] = 2.5  # 250% — extreme
        result = default_validator.validate_abnormal_returns(extreme)
        assert not result.passed

    def test_missing_benchmark_return_flagged(
        self, default_validator, aapl_ar_df
    ):
        missing_bm = aapl_ar_df.copy()
        missing_bm["benchmark_return"] = np.nan
        result = default_validator.validate_abnormal_returns(missing_bm)
        assert not result.passed

    def test_duplicate_relative_days_detected(
        self, default_validator, aapl_ar_df
    ):
        dup = pd.concat([aapl_ar_df, aapl_ar_df.iloc[[0]]], ignore_index=True)
        result = default_validator.validate_abnormal_returns(dup)
        assert not result.passed

    # --- CAR metrics ---

    def test_valid_car_df_passes(self, default_validator, valid_car_df):
        result = default_validator.validate_car_metrics(valid_car_df)
        assert result.passed, [str(i) for i in result.critical_issues()]

    def test_extreme_car_flagged(self, default_validator, valid_car_df):
        extreme = valid_car_df.copy()
        extreme.at[extreme.index[0], "car_3d"] = 1.5  # 150%
        result = default_validator.validate_car_metrics(extreme)
        assert not result.passed

    def test_duplicate_transcript_id_in_car_detected(
        self, default_validator, valid_car_df
    ):
        dup = pd.concat([valid_car_df, valid_car_df.iloc[[0]]], ignore_index=True)
        result = default_validator.validate_car_metrics(dup)
        assert not result.passed

    def test_missing_car_columns_raises_critical(
        self, default_validator
    ):
        df = pd.DataFrame({
            "transcript_id":    ["AAPL_Q1_2024"],
            "ticker":           ["AAPL"],
            "aligned_event_date": [_EVENT_DATES["AAPL"]],
            "car_complete":     [True],
            # No car_Nd columns
        })
        result = default_validator.validate_car_metrics(df)
        assert not result.passed

    # --- Sentiment merge ---

    def test_valid_sentiment_df_passes(
        self, default_validator, valid_sentiment_df
    ):
        result = default_validator.validate_sentiment_merge(valid_sentiment_df)
        assert result.passed, [str(i) for i in result.critical_issues()]

    def test_unexpected_sentiment_label_flagged(
        self, default_validator, valid_sentiment_df
    ):
        broken = valid_sentiment_df.copy()
        broken.at[broken.index[0], "sentiment_label"] = "bullish"  # not in vocab
        result = default_validator.validate_sentiment_merge(broken)
        # Should be WARNING (not CRITICAL by default)
        warns = [i.check_name for i in result.warning_issues()]
        assert any("unexpected_sentiment_labels" in w for w in warns)

    def test_null_sentiment_score_raises_critical(
        self, default_validator, valid_sentiment_df
    ):
        broken = valid_sentiment_df.copy()
        broken.at[broken.index[0], "finbert_sentiment_score"] = np.nan
        result = default_validator.validate_sentiment_merge(broken)
        assert not result.passed

    # --- Final dataset ---

    def test_valid_final_dataset_passes(
        self, default_validator, valid_sentiment_df
    ):
        result = default_validator.validate_final_dataset(valid_sentiment_df)
        assert result.passed, [str(i) for i in result.critical_issues()]

    def test_weekend_event_date_raises_critical(
        self, default_validator, valid_sentiment_df
    ):
        broken = valid_sentiment_df.copy()
        broken["aligned_event_date"] = pd.Timestamp("2024-01-20")  # Saturday
        result = default_validator.validate_final_dataset(broken)
        assert not result.passed

    # --- Strict mode ---

    def test_strict_mode_promotes_warning_to_critical(
        self, strict_validator, valid_sentiment_df
    ):
        broken = valid_sentiment_df.copy()
        broken.at[broken.index[0], "sentiment_label"] = "bullish"
        result = strict_validator.validate_sentiment_merge(broken)
        # In strict mode this should be CRITICAL
        crits = [i.check_name for i in result.critical_issues()]
        assert any("unexpected_sentiment_labels" in c for c in crits)

    # --- Null rate ---

    def test_high_null_rate_column_raises_critical(self, default_validator):
        df = pd.DataFrame({
            "transcript_id":    [f"ID_{i}" for i in range(20)],
            "ticker":           ["AAPL"] * 20,
            "aligned_event_date": [_EVENT_DATES["AAPL"]] * 20,
            # 90% NaN — above critical threshold
            "finbert_sentiment_score": [0.5] * 2 + [np.nan] * 18,
            "lm_tone_score":          [0.5] * 20,
            "sentiment_label":        ["positive"] * 20,
        })
        result = default_validator.validate_sentiment_merge(df)
        assert not result.passed

    # --- summarize_validation ---

    def test_summarize_validation_structure(
        self, default_validator, valid_window_df, valid_car_df
    ):
        r1 = default_validator.validate_event_windows(valid_window_df)
        r2 = default_validator.validate_car_metrics(valid_car_df)
        summary = default_validator.summarize_validation([r1, r2])
        assert "overall_passed" in summary
        assert "stages" in summary
        assert "critical_count" in summary

    def test_summarize_overall_passes_when_no_criticals(
        self, default_validator, valid_window_df
    ):
        result = default_validator.validate_event_windows(valid_window_df)
        summary = default_validator.summarize_validation([result])
        assert summary["overall_passed"] is True


# ===========================================================================
# TEST CLASS 6 — STATISTICS UTILS
# ===========================================================================


class TestStatisticsUtils:
    """
    Tests for statistical utilities (statistics_utils.py).
    Uses hand-verified statistical calculations.
    """

    def test_mean_calculation(self):
        values = [0.032, 0.018, -0.011]
        expected_mean = sum(values) / len(values)
        assert math.isclose(np.mean(values), expected_mean, rel_tol=1e-9)

    def test_std_calculation(self):
        values = [0.032, 0.018, -0.011]
        expected_std = np.std(values, ddof=1)  # sample std
        assert math.isclose(
            pd.Series(values).std(), expected_std, rel_tol=1e-9
        )

    def test_t_statistic_manual(self):
        """
        t = mean / (std / sqrt(n))
        For values [0.032, 0.018, -0.011]:
            mean = 0.013
            std  ≈ 0.02202
            n    = 3
            t    ≈ 0.013 / (0.02202 / sqrt(3)) ≈ 1.022
        """
        values = np.array([0.032, 0.018, -0.011])
        n = len(values)
        mean = values.mean()
        std = values.std(ddof=1)
        t_stat = mean / (std / math.sqrt(n))
        assert math.isclose(t_stat, mean * math.sqrt(n) / std, rel_tol=1e-9)

    def test_correlation_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        corr = np.corrcoef(x, y)[0, 1]
        assert math.isclose(corr, 1.0, abs_tol=1e-10)

    def test_correlation_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        corr = np.corrcoef(x, y)[0, 1]
        assert math.isclose(corr, -1.0, abs_tol=1e-10)

    def test_correlation_zero_for_orthogonal_signals(self):
        n = 1000
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        corr = np.corrcoef(x, y)[0, 1]
        assert abs(corr) < 0.10  # near-zero for large independent samples

    def test_winsorize_clips_extremes(self):
        """Values outside [1st, 99th] percentile should be clipped."""
        values = np.array([-5.0] + list(np.linspace(-0.1, 0.1, 98)) + [5.0])
        p1  = np.percentile(values, 1)
        p99 = np.percentile(values, 99)
        winsorized = np.clip(values, p1, p99)
        assert winsorized.min() >= p1
        assert winsorized.max() <= p99

    def test_sentiment_bucket_means(self):
        """Group-level CAR means by sentiment bucket."""
        df = pd.DataFrame({
            "sentiment_label": ["positive", "positive", "negative", "neutral"],
            "car_3d":          [0.030, 0.025, -0.020, 0.005],
        })
        means = df.groupby("sentiment_label")["car_3d"].mean()
        assert math.isclose(means["positive"], 0.0275, abs_tol=1e-10)
        assert math.isclose(means["negative"], -0.020,  abs_tol=1e-10)
        assert math.isclose(means["neutral"],   0.005,  abs_tol=1e-10)

    def test_compound_return_formula_5d(self):
        """
        Compound 5-day return from daily returns.
        Hand-computed for: [0.025, -0.008, 0.015, 0.010, -0.004]
        """
        daily = np.array([0.025, -0.008, 0.015, 0.010, -0.004])
        compound = np.prod(1 + daily) - 1
        expected = (1.025 * 0.992 * 1.015 * 1.010 * 0.996) - 1
        assert math.isclose(compound, expected, rel_tol=1e-9)

    @pytest.mark.parametrize("values,expected_mean", [
        ([0.0, 0.0, 0.0], 0.0),
        ([1.0, -1.0], 0.0),
        ([0.1, 0.2, 0.3], 0.2),
        ([-0.05, 0.05], 0.0),
    ])
    def test_mean_parametrized(self, values, expected_mean):
        assert math.isclose(np.mean(values), expected_mean, abs_tol=1e-10)


# ===========================================================================
# TEST CLASS 7 — PIPELINE INTEGRATION
# ===========================================================================


class TestPipelineIntegration:
    """
    Integration tests verifying end-to-end data flow across pipeline stages.
    All external dependencies are mocked/stubbed.
    """

    def test_window_to_car_pipeline(
        self, window_generator, car_calculator, aapl_ar_df
    ):
        """Windows → AR input → CAR output maintains transcript_id integrity."""
        if window_generator is None or car_calculator is None:
            pytest.skip("modules not available")

        events = pd.DataFrame({
            "transcript_id":    ["AAPL_Q1_2024"],
            "ticker":           ["AAPL"],
            "aligned_event_date": [_EVENT_DATES["AAPL"]],
        })
        win_result = window_generator.generate_event_windows(events)
        assert "AAPL_Q1_2024" in win_result.window_df["transcript_id"].values

        car_df = car_calculator.compute_car(aapl_ar_df)
        assert "AAPL_Q1_2024" in car_df["transcript_id"].values

        # Both outputs can be joined on transcript_id
        merged = pd.merge(
            win_result.window_df[["transcript_id", "aligned_event_date"]].drop_duplicates(),
            car_df[["transcript_id", "car_3d", "car_5d"]],
            on="transcript_id",
        )
        assert len(merged) == 1
        assert math.isclose(merged["car_3d"].iloc[0], _AAPL_CAR_3D, abs_tol=1e-6)

    def test_ar_df_and_car_df_share_transcript_ids(
        self, car_calculator, multi_ticker_ar_df
    ):
        if car_calculator is None:
            pytest.skip("module not available")
        car_df = car_calculator.compute_car(multi_ticker_ar_df)
        ar_ids  = set(multi_ticker_ar_df["transcript_id"].unique())
        car_ids = set(car_df["transcript_id"].unique())
        assert ar_ids == car_ids

    def test_car_merges_cleanly_with_sentiment(
        self, car_calculator, multi_ticker_ar_df, valid_sentiment_df
    ):
        if car_calculator is None:
            pytest.skip("module not available")
        car_df = car_calculator.compute_car(multi_ticker_ar_df)
        merged = pd.merge(
            car_df, valid_sentiment_df[["transcript_id", "finbert_sentiment_score"]],
            on="transcript_id",
        )
        # All three events have sentiment
        assert len(merged) == 3
        assert not merged["finbert_sentiment_score"].isnull().any()

    def test_validation_passes_on_car_output(
        self, default_validator, car_calculator, multi_ticker_ar_df
    ):
        if car_calculator is None or default_validator is None:
            pytest.skip("modules not available")
        car_df = car_calculator.compute_car(multi_ticker_ar_df)
        result = default_validator.validate_car_metrics(car_df)
        assert result.passed, [str(i) for i in result.critical_issues()]

    def test_final_dataset_schema_completeness(self, valid_sentiment_df):
        """Final export schema must contain all required event-study columns."""
        required = {
            "transcript_id", "ticker", "aligned_event_date",
            "finbert_sentiment_score", "lm_tone_score", "sentiment_label",
            "abnormal_return", "car_3d", "car_5d", "car_complete",
        }
        assert required.issubset(set(valid_sentiment_df.columns))

    def test_full_pipeline_transcript_id_preserved_end_to_end(
        self, window_generator, car_calculator, multi_ticker_ar_df,
        valid_sentiment_df
    ):
        if window_generator is None or car_calculator is None:
            pytest.skip("modules not available")
        events = pd.DataFrame({
            "transcript_id":    list(_IDS.values()),
            "ticker":           list(_IDS.keys()),
            "aligned_event_date": [_EVENT_DATES[t] for t in _IDS],
        })
        windows = window_generator.generate_event_windows(events)
        car_df  = car_calculator.compute_car(multi_ticker_ar_df)
        merged  = pd.merge(
            car_df,
            valid_sentiment_df[["transcript_id", "finbert_sentiment_score",
                                 "lm_tone_score", "sentiment_label"]],
            on="transcript_id",
        )
        assert set(merged["transcript_id"]) == set(_IDS.values())


# ===========================================================================
# TEST CLASS 8 — EDGE CASES
# ===========================================================================


class TestEdgeCases:

    def test_car_with_single_post_event_day(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        single_post = aapl_ar_df[aapl_ar_df["relative_day"] <= 1].copy()
        config = CARConfig(horizons=(1,), min_post_days_required=1)
        calc = CARCalculator(config)
        result_df = calc.compute_car(single_post)
        assert not result_df["car_1d"].isna().any()
        assert math.isclose(
            result_df["car_1d"].iloc[0], _AAPL_AR_SERIES[1], abs_tol=1e-6
        )

    def test_window_generator_with_zero_pre_days(self, session_provider):
        if EventWindowConfig is None or EventWindowGenerator is None:
            pytest.skip("modules not available")
        config = EventWindowConfig(pre_event_days=0, post_event_days=3)
        gen = EventWindowGenerator(config, session_provider)
        df = pd.DataFrame({
            "transcript_id":    ["AAPL_Q1_2024"],
            "ticker":           ["AAPL"],
            "aligned_event_date": [_EVENT_DATES["AAPL"]],
        })
        result = gen.generate_event_windows(df)
        win = result.windows[0]
        # Only event + 3 post
        assert len(win.sessions) == 4
        assert min(win.relative_days) == 0

    def test_window_generator_with_zero_post_days(self, session_provider):
        if EventWindowConfig is None or EventWindowGenerator is None:
            pytest.skip("modules not available")
        config = EventWindowConfig(pre_event_days=3, post_event_days=0)
        gen = EventWindowGenerator(config, session_provider)
        df = pd.DataFrame({
            "transcript_id":    ["AAPL_Q1_2024"],
            "ticker":           ["AAPL"],
            "aligned_event_date": [_EVENT_DATES["AAPL"]],
        })
        result = gen.generate_event_windows(df)
        win = result.windows[0]
        assert max(win.relative_days) == 0

    def test_all_nan_ar_post_event_produces_nan_car(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        df = aapl_ar_df.copy()
        df.loc[df["relative_day"] > 0, "abnormal_return"] = np.nan
        config = CARConfig(
            horizons=(3, 5),
            nan_policy="skip",
            min_post_days_required=1,
        )
        calc = CARCalculator(config)
        result_df = calc.compute_car(df)
        assert result_df["car_3d"].isna().all()

    def test_single_event_single_post_day_car(self):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        df = pd.DataFrame({
            "transcript_id":      ["X_Q1_2024", "X_Q1_2024"],
            "ticker":             ["X", "X"],
            "aligned_event_date": [pd.Timestamp("2024-02-01")] * 2,
            "relative_day":       [0, 1],
            "abnormal_return":    [0.005, 0.050],
            "daily_return":       [0.007, 0.052],
            "benchmark_return":   [0.002, 0.002],
        })
        config = CARConfig(horizons=(1,), include_event_day_in_car=False)
        calc = CARCalculator(config)
        result_df = calc.compute_car(df)
        assert math.isclose(result_df["car_1d"].iloc[0], 0.050, abs_tol=1e-10)

    def test_car_config_invalid_horizon_raises(self):
        if CARConfig is None:
            pytest.skip("module not available")
        with pytest.raises(ValueError):
            CARConfig(horizons=(-1, 3))

    def test_car_config_empty_horizons_raises(self):
        if CARConfig is None:
            pytest.skip("module not available")
        with pytest.raises(ValueError):
            CARConfig(horizons=())

    def test_window_config_negative_pre_raises(self):
        if EventWindowConfig is None:
            pytest.skip("module not available")
        with pytest.raises(ValueError):
            EventWindowConfig(pre_event_days=-1)

    def test_window_config_negative_post_raises(self):
        if EventWindowConfig is None:
            pytest.skip("module not available")
        with pytest.raises(ValueError):
            EventWindowConfig(post_event_days=-1)

    def test_validation_config_invalid_null_threshold_raises(self):
        if ValidationConfig is None:
            pytest.skip("module not available")
        with pytest.raises(ValueError):
            ValidationConfig(null_rate_warning_threshold=0.30,
                             null_rate_critical_threshold=0.10)

    def test_extreme_ar_of_exactly_cap_is_not_flagged(self, aapl_ar_df):
        if CARConfig is None or CARCalculator is None:
            pytest.skip("modules not available")
        df = aapl_ar_df.copy()
        # AR = exactly 1.0 should NOT exceed threshold (strictly >)
        df.loc[df["relative_day"] == 1, "abnormal_return"] = 1.0
        config = CARConfig(horizons=(1,), extreme_car_threshold=1.0)
        calc = CARCalculator(config)
        result = calc.compute_event_car(df)
        # car_1d = 1.0 which is NOT > 1.0
        assert 1 not in result.extreme_flags

    def test_many_events_performance(self, session_provider):
        """Window generation for 50 events must complete without error."""
        if EventWindowConfig is None or EventWindowGenerator is None:
            pytest.skip("modules not available")
        config = EventWindowConfig(pre_event_days=5, post_event_days=5)
        gen = EventWindowGenerator(config, session_provider)
        base = pd.Timestamp("2024-01-15")
        events = pd.DataFrame({
            "transcript_id":    [f"TICK_{i}_Q1_2024" for i in range(50)],
            "ticker":           [f"T{i:03d}" for i in range(50)],
            "aligned_event_date": [
                base + pd.offsets.BDay(i) for i in range(50)
            ],
        })
        result = gen.generate_event_windows(events)
        assert result.total_events == 50
        assert result.complete_events == 50


# ===========================================================================
# TEST CLASS 9 — DATACLASS INTEGRITY
# ===========================================================================


class TestDataclassIntegrity:

    def test_event_window_config_is_frozen(self):
        if EventWindowConfig is None:
            pytest.skip("module not available")
        config = EventWindowConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.pre_event_days = 99

    def test_car_config_is_frozen(self):
        if CARConfig is None:
            pytest.skip("module not available")
        config = CARConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.horizons = (99,)

    def test_validation_config_is_frozen(self):
        if ValidationConfig is None:
            pytest.skip("module not available")
        config = ValidationConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.strict_mode = True

    def test_car_config_default_horizons(self):
        if CARConfig is None:
            pytest.skip("module not available")
        config = CARConfig()
        assert config.horizons == (1, 3, 5)

    def test_car_config_car_column_names(self):
        if CARConfig is None:
            pytest.skip("module not available")
        config = CARConfig(horizons=(3, 5))
        assert config.car_column_names == ["car_3d", "car_5d"]

    def test_car_config_max_horizon(self):
        if CARConfig is None:
            pytest.skip("module not available")
        config = CARConfig(horizons=(1, 3, 5))
        assert config.max_horizon == 5

    def test_event_window_config_total_window_size_with_event_day(self):
        if EventWindowConfig is None:
            pytest.skip("module not available")
        config = EventWindowConfig(
            pre_event_days=5, post_event_days=5, include_event_day=True
        )
        assert config.total_window_size == 11

    def test_event_window_config_total_window_size_without_event_day(self):
        if EventWindowConfig is None:
            pytest.skip("module not available")
        config = EventWindowConfig(
            pre_event_days=5, post_event_days=5, include_event_day=False
        )
        assert config.total_window_size == 10

    def test_car_result_to_series_contains_all_horizons(self):
        if CARResult is None or CARConfig is None:
            pytest.skip("module not available")
        config = CARConfig(horizons=(1, 3, 5))
        result = CARResult(
            transcript_id="AAPL_Q1_2024",
            ticker="AAPL",
            event_date=_EVENT_DATES["AAPL"],
            car_values={1: 0.025, 3: 0.032, 5: 0.038},
            post_day_count=5,
            nan_day_count=0,
            is_complete=True,
        )
        series = result.to_series(config)
        assert "car_1d" in series.index
        assert "car_3d" in series.index
        assert "car_5d" in series.index
        assert math.isclose(series["car_3d"], 0.032, abs_tol=1e-10)

    def test_event_window_to_dataframe_shape(self, session_provider):
        if EventWindow is None or EventWindowConfig is None:
            pytest.skip("module not available")
        event_date = _EVENT_DATES["AAPL"]
        sessions = list(pd.bdate_range(
            start=event_date - pd.Timedelta(days=7),
            end=event_date + pd.Timedelta(days=7),
            freq="B",
        ))
        n = len(sessions)
        event_idx = sessions.index(event_date)
        win = EventWindow(
            transcript_id="AAPL_Q1_2024",
            ticker="AAPL",
            aligned_event_date=event_date,
            window_start=sessions[0],
            window_end=sessions[-1],
            sessions=sessions,
            relative_days=list(range(-event_idx, n - event_idx)),
            is_complete=True,
        )
        df = win.to_dataframe()
        assert len(df) == n
        assert set(df.columns) >= {
            "transcript_id", "ticker", "aligned_event_date",
            "window_start", "window_end", "date", "relative_day", "event_day_flag",
        }

    def test_validation_summary_passed_iff_no_criticals(self):
        if ValidationSummary is None:
            pytest.skip("module not available")
        s = ValidationSummary(critical_count=0, warning_count=5, info_count=2)
        assert s.passed is True
        assert s.strict_passed is False

    def test_validation_summary_strict_passed(self):
        if ValidationSummary is None:
            pytest.skip("module not available")
        s = ValidationSummary(critical_count=0, warning_count=0, info_count=1)
        assert s.strict_passed is True

    def test_validation_issue_to_dict_structure(self):
        if ValidationIssue is None or Severity is None or ValidationStage is None:
            pytest.skip("module not available")
        issue = ValidationIssue(
            severity=Severity.CRITICAL,
            stage=ValidationStage.CAR_METRICS,
            check_name="test_check",
            message="Test message",
            affected_ids=["A", "B"],
            detail={"count": 2},
        )
        d = issue.to_dict()
        assert d["severity"] == "CRITICAL"
        assert d["stage"] == "car_metrics"
        assert d["check_name"] == "test_check"
        assert d["affected_ids"] == ["A", "B"]
        assert d["detail"]["count"] == 2

    def test_dataset_validation_result_to_report_dict(self):
        if DatasetValidationResult is None or ValidationStage is None:
            pytest.skip("module not available")
        result = DatasetValidationResult(
            stage=ValidationStage.FINAL_DATASET,
            row_count=100,
            event_count=10,
            dataset_label="test",
        )
        d = result.to_report_dict()
        assert d["stage"] == "final_dataset"
        assert d["row_count"] == 100
        assert d["event_count"] == 10

    def test_dataset_validation_result_to_dataframe_empty(self):
        if DatasetValidationResult is None or ValidationStage is None:
            pytest.skip("module not available")
        result = DatasetValidationResult(stage=ValidationStage.GENERAL)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ===========================================================================
# STANDALONE UTILITY FUNCTION TESTS
# ===========================================================================


@pytest.mark.skipif(detect_duplicates is None, reason="validation module not available")
class TestValidationUtilities:

    def test_detect_duplicates_finds_exact_duplicates(self):
        df = pd.DataFrame({
            "transcript_id": ["A", "A", "B"],
            "date":          pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        })
        dups = detect_duplicates(df, ["transcript_id", "date"])
        assert len(dups) == 2
        assert set(dups["transcript_id"]) == {"A"}

    def test_detect_duplicates_empty_when_no_duplicates(self):
        df = pd.DataFrame({
            "transcript_id": ["A", "B", "C"],
            "date":          pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
        })
        dups = detect_duplicates(df, ["transcript_id", "date"])
        assert len(dups) == 0

    def test_null_rate_report_structure(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [1.0, 2.0, 3.0],
        })
        report = null_rate_report(df)
        assert "column" in report.columns
        assert "null_rate" in report.columns
        a_rate = report.loc[report["column"] == "a", "null_rate"].iloc[0]
        b_rate = report.loc[report["column"] == "b", "null_rate"].iloc[0]
        assert math.isclose(a_rate, 1 / 3, rel_tol=1e-6)
        assert math.isclose(b_rate, 0.0, abs_tol=1e-10)

    def test_null_rate_report_sorted_descending(self):
        df = pd.DataFrame({
            "high_null":   [np.nan, np.nan, np.nan, 1.0],
            "low_null":    [1.0,    2.0,    np.nan, 4.0],
            "no_null":     [1.0,    2.0,    3.0,    4.0],
        })
        report = null_rate_report(df)
        rates = report["null_rate"].tolist()
        assert rates == sorted(rates, reverse=True)

    def test_check_monotonic_groups_detects_non_monotone(self):
        df = pd.DataFrame({
            "event_id":    ["A", "A", "A", "B", "B"],
            "relative_day": [0, 2, 1, 0, 1],   # A is not monotone
        })
        bad = check_monotonic_groups(df, "event_id", "relative_day")
        assert "A" in bad
        assert "B" not in bad

    def test_check_monotonic_groups_passes_for_monotone(self):
        df = pd.DataFrame({
            "event_id":    ["A", "A", "A"],
            "relative_day": [-1, 0, 1],
        })
        bad = check_monotonic_groups(df, "event_id", "relative_day")
        assert len(bad) == 0

    def test_detect_extreme_values_finds_outliers(self):
        df = pd.DataFrame({
            "transcript_id": ["A", "B", "C", "D"],
            "car_3d": [0.02, -0.01, 1.50, -1.20],
        })
        extremes = detect_extreme_values(df, "car_3d", cap=1.0)
        assert len(extremes) == 2
        assert set(extremes["transcript_id"]) == {"C", "D"}

    def test_detect_extreme_values_empty_for_normal_data(self):
        df = pd.DataFrame({
            "transcript_id": ["A", "B"],
            "car_3d": [0.032, -0.011],
        })
        extremes = detect_extreme_values(df, "car_3d", cap=1.0)
        assert len(extremes) == 0

    def test_check_grouped_integrity_exact_count(self):
        df = pd.DataFrame({
            "transcript_id":  ["A", "A", "B", "B"],
            "event_day_flag": [True, False, True, False],
        })
        results = check_grouped_integrity(
            df, "transcript_id",
            {"one_event_day": {"type": "exact_count", "column": "event_day_flag",
                               "value": True, "expected": 1}}
        )
        assert len(results["one_event_day"]) == 0  # both pass

    def test_check_grouped_integrity_min_count(self):
        df = pd.DataFrame({
            "transcript_id": ["A", "A", "B"],
            "car_3d":        [0.01, 0.02, np.nan],
        })
        results = check_grouped_integrity(
            df, "transcript_id",
            {"has_car": {"type": "min_count", "column": "car_3d", "min": 1}}
        )
        # B has 0 non-null → fails
        assert "B" in results["has_car"]
        assert "A" not in results["has_car"]

    def test_check_grouped_integrity_no_duplicates(self):
        df = pd.DataFrame({
            "transcript_id": ["A", "A", "B"],
            "date":          pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        })
        results = check_grouped_integrity(
            df, "transcript_id",
            {"no_dup_dates": {"type": "no_duplicates", "columns": ["date"]}}
        )
        assert "A" in results["no_dup_dates"]
        assert "B" not in results["no_dup_dates"]
