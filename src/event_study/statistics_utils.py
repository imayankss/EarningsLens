"""
statistics_utils.py
===================
DAY 6/7 — Statistical Analysis Utility Layer
Earnings Call Sentiment Analyzer — Event Study Pipeline

Provides the complete reusable statistical toolkit for evaluating
sentiment-to-market-return relationships in the event-study framework.
All methods are NaN-safe, vectorised, deterministic, and export-ready.

Statistical responsibilities
----------------------------
- Descriptive statistics (mean, median, std, skew, kurtosis, percentiles)
- Pearson and Spearman correlation with p-values and confidence intervals
- One-sample and two-sample t-tests (Welch and paired)
- Sentiment-bucket construction and cross-bucket comparison
- Grouped (ticker / quarter / section) event-study aggregations
- Regression dataset preparation (feature scaling, dummy encoding,
  VIF computation, multicollinearity diagnostics)
- Full event-study metric summary table

Module boundaries
-----------------
This module does NOT:
  - compute abnormal returns          (→ abnormal_returns.py)
  - compute CAR                       (→ car_calculator.py)
  - generate event windows            (→ event_window_generator.py)
  - merge sentiment onto events       (→ sentiment_event_merger.py)
  - orchestrate the pipeline          (→ event_study.py)
  - produce plots or dashboards       (→ Day 10 visualisation layer)

Architecture position
---------------------
    event_study.py / Day 7-9 analysis scripts
                │
                ▼
        statistics_utils.py   ← THIS FILE

Python : 3.11+
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Final, Literal, Optional, Sequence

import numpy as np
import pandas as pd

# Optional heavy imports — guarded so the module loads even in lean envs.
try:
    from scipy import stats as _scipy_stats  # type: ignore[import]

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy is not installed. t-tests and Pearson-p-values will be unavailable. "
        "Install with: pip install scipy",
        ImportWarning,
        stacklevel=2,
    )

try:
    import statsmodels.api as _sm  # type: ignore[import]
    from statsmodels.stats.outliers_influence import (  # type: ignore[import]
        variance_inflation_factor as _vif,
    )

    _STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STATSMODELS_AVAILABLE = False
    warnings.warn(
        "statsmodels is not installed. OLS regression helpers and VIF "
        "computation will be unavailable. "
        "Install with: pip install statsmodels",
        ImportWarning,
        stacklevel=2,
    )

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default sentiment columns produced by the merge pipeline.
_DEFAULT_SENTIMENT_COLS: Final[tuple[str, ...]] = (
    "finbert_sentiment_score",
    "lm_tone_score",
)

# Default financial outcome columns.
_DEFAULT_OUTCOME_COLS: Final[tuple[str, ...]] = (
    "abnormal_return_1d",
    "abnormal_return_3d",
    "abnormal_return_5d",
    "car_3d",
    "car_5d",
)

# Bucket labels for three-way sentiment splits.
_BUCKET_LABELS_3: Final[tuple[str, str, str]] = ("negative", "neutral", "positive")

# Percentile thresholds for three-way bucket split (bottom / top tercile).
_TERCILE_THRESHOLDS: Final[tuple[float, float]] = (1 / 3, 2 / 3)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatisticsConfig:
    """
    Configuration for all statistical utilities.

    Parameters
    ----------
    confidence_level:
        Confidence level for intervals (default 0.95 = 95 %).
    n_buckets:
        Number of sentiment quantile buckets (default 3 = tertiles).
    min_obs_per_bucket:
        Minimum observations required in a bucket before it is reported.
        Buckets below this threshold are flagged in the output.
    winsorise_returns:
        When True, return columns are winsorised at
        (``winsorise_lower``, ``winsorise_upper``) percentiles before
        statistics are computed.
    winsorise_lower:
        Lower winsorise percentile (default 1st percentile).
    winsorise_upper:
        Upper winsorise percentile (default 99th percentile).
    regression_add_constant:
        Add an intercept column when preparing regression datasets.
    sentiment_cols:
        Tuple of sentiment score columns to use in cross-analyses.
    outcome_cols:
        Tuple of return/CAR columns to use as dependent variables.
    """

    confidence_level: float = 0.95
    n_buckets: int = 3
    min_obs_per_bucket: int = 5
    winsorise_returns: bool = False
    winsorise_lower: float = 1.0
    winsorise_upper: float = 99.0
    regression_add_constant: bool = True
    sentiment_cols: tuple[str, ...] = _DEFAULT_SENTIMENT_COLS
    outcome_cols: tuple[str, ...] = _DEFAULT_OUTCOME_COLS

    def __post_init__(self) -> None:
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be in (0, 1).")
        if self.n_buckets < 2:
            raise ValueError("n_buckets must be >= 2.")
        if self.min_obs_per_bucket < 1:
            raise ValueError("min_obs_per_bucket must be >= 1.")
        if not (0.0 <= self.winsorise_lower < self.winsorise_upper <= 100.0):
            raise ValueError(
                "winsorise_lower must be < winsorise_upper, both in [0, 100]."
            )

    @property
    def alpha(self) -> float:
        """Significance level (1 − confidence_level)."""
        return 1.0 - self.confidence_level


# ---------------------------------------------------------------------------


@dataclass
class CorrelationResult:
    """
    Output of a single correlation computation.

    Attributes
    ----------
    x_col, y_col:
        Names of the correlated columns.
    method:
        ``'pearson'`` or ``'spearman'``.
    n_obs:
        Number of valid (non-NaN) observation pairs used.
    correlation:
        Correlation coefficient.
    p_value:
        Two-tailed p-value.  NaN when scipy is unavailable.
    ci_lower, ci_upper:
        Confidence interval bounds for the correlation coefficient using
        Fisher-z transformation.  NaN when n_obs < 4.
    significant:
        True when p_value < alpha at the configured confidence level.
    """

    x_col: str = ""
    y_col: str = ""
    method: str = "pearson"
    n_obs: int = 0
    correlation: float = float("nan")
    p_value: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    significant: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "x_col": self.x_col,
            "y_col": self.y_col,
            "method": self.method,
            "n_obs": self.n_obs,
            "correlation": round(self.correlation, 6) if not np.isnan(self.correlation) else None,
            "p_value": round(self.p_value, 6) if not np.isnan(self.p_value) else None,
            "ci_lower": round(self.ci_lower, 6) if not np.isnan(self.ci_lower) else None,
            "ci_upper": round(self.ci_upper, 6) if not np.isnan(self.ci_upper) else None,
            "significant": self.significant,
        }


# ---------------------------------------------------------------------------


@dataclass
class TTestResult:
    """
    Output of a t-test computation.

    Attributes
    ----------
    label:
        Human-readable description of the test.
    test_type:
        One of ``'one_sample'``, ``'two_sample_welch'``, ``'paired'``.
    group_a_label, group_b_label:
        Names of the groups being compared.
    n_a, n_b:
        Sample sizes (n_b = 0 for one-sample tests).
    mean_a, mean_b:
        Group means.
    std_a, std_b:
        Group standard deviations.
    t_statistic:
        Computed t statistic.
    p_value:
        Two-tailed p-value.
    degrees_of_freedom:
        Effective degrees of freedom.
    ci_lower, ci_upper:
        Confidence interval for the mean difference (or mean in one-sample).
    significant:
        True when p_value < alpha.
    effect_size_cohens_d:
        Cohen's d effect size.  NaN for one-sample tests.
    """

    label: str = ""
    test_type: str = "two_sample_welch"
    group_a_label: str = "group_a"
    group_b_label: str = "group_b"
    n_a: int = 0
    n_b: int = 0
    mean_a: float = float("nan")
    mean_b: float = float("nan")
    std_a: float = float("nan")
    std_b: float = float("nan")
    t_statistic: float = float("nan")
    p_value: float = float("nan")
    degrees_of_freedom: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    significant: bool = False
    effect_size_cohens_d: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        return {k: (round(v, 6) if isinstance(v, float) and not np.isnan(v) else v)
                for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------


@dataclass
class RegressionPreparationResult:
    """
    Output of prepare_regression_dataset().

    Attributes
    ----------
    X:
        Feature matrix (DataFrame) ready for OLS / ML models.
    y:
        Target Series.
    feature_names:
        Ordered list of feature column names in X.
    n_obs:
        Number of complete-case observations.
    n_dropped_nan:
        Rows dropped due to NaN in X or y.
    vif_df:
        Variance Inflation Factor table (feature → VIF).
        Empty DataFrame when statsmodels is unavailable.
    high_vif_features:
        Feature names with VIF > 10 (potential multicollinearity).
    scaler_params:
        Dict of {col: {'mean': …, 'std': …}} for standardised features.
    """

    X: pd.DataFrame = field(default_factory=pd.DataFrame)
    y: pd.Series = field(default_factory=pd.Series)
    feature_names: list[str] = field(default_factory=list)
    n_obs: int = 0
    n_dropped_nan: int = 0
    vif_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    high_vif_features: list[str] = field(default_factory=list)
    scaler_params: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"RegressionPreparationResult",
            f"  n_obs           : {self.n_obs}",
            f"  n_dropped_nan   : {self.n_dropped_nan}",
            f"  features ({len(self.feature_names)}): {self.feature_names}",
            f"  high_vif        : {self.high_vif_features}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------


@dataclass
class StatisticsSummary:
    """
    Comprehensive summary produced by summarize_event_study_metrics().

    Attributes
    ----------
    descriptive_stats:
        DataFrame of descriptive statistics for all numeric columns.
    correlation_matrix:
        Pearson correlation heatmap DataFrame (sentiment × outcomes).
    correlation_results:
        List of CorrelationResult objects for each pair.
    bucket_summary:
        DataFrame of mean metrics per sentiment bucket.
    t_test_results:
        List of TTestResult objects comparing positive vs negative buckets.
    n_events:
        Total number of events analysed.
    columns_analysed:
        Columns included in the summary.
    """

    descriptive_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_results: list[CorrelationResult] = field(default_factory=list)
    bucket_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    t_test_results: list[TTestResult] = field(default_factory=list)
    n_events: int = 0
    columns_analysed: list[str] = field(default_factory=list)

    def print_report(self) -> None:
        sep = "─" * 68
        print(f"\n{'═' * 68}")
        print("  EVENT STUDY STATISTICS SUMMARY")
        print(f"{'═' * 68}")
        print(f"  Events analysed : {self.n_events}")
        print(f"  Columns         : {self.columns_analysed}\n")

        if not self.descriptive_stats.empty:
            print(f"  {sep}")
            print("  DESCRIPTIVE STATISTICS")
            print(f"  {sep}")
            print(self.descriptive_stats.to_string())

        if not self.correlation_matrix.empty:
            print(f"\n  {sep}")
            print("  CORRELATION MATRIX  (Pearson)")
            print(f"  {sep}")
            print(self.correlation_matrix.round(4).to_string())

        if self.correlation_results:
            print(f"\n  {sep}")
            print("  PAIRWISE CORRELATIONS")
            print(f"  {sep}")
            for r in self.correlation_results:
                sig = "***" if r.p_value < 0.01 else ("**" if r.p_value < 0.05 else ("*" if r.p_value < 0.10 else ""))
                print(
                    f"  {r.x_col:<35s} × {r.y_col:<28s} "
                    f"r={r.correlation:+.4f}  p={r.p_value:.4f}  n={r.n_obs}  {sig}"
                )

        if not self.bucket_summary.empty:
            print(f"\n  {sep}")
            print("  SENTIMENT BUCKET SUMMARY")
            print(f"  {sep}")
            print(self.bucket_summary.to_string())

        if self.t_test_results:
            print(f"\n  {sep}")
            print("  T-TEST RESULTS  (positive vs negative bucket)")
            print(f"  {sep}")
            for t in self.t_test_results:
                sig = "***" if t.p_value < 0.01 else ("**" if t.p_value < 0.05 else ("*" if t.p_value < 0.10 else "ns"))
                print(
                    f"  {t.label:<40s}  "
                    f"t={t.t_statistic:+.4f}  p={t.p_value:.4f}  "
                    f"d={t.effect_size_cohens_d:.4f}  {sig}"
                )

        print(f"\n{'═' * 68}\n")


# ---------------------------------------------------------------------------
# Internal utility functions
# ---------------------------------------------------------------------------


def _require_scipy(method_name: str) -> None:
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            f"{method_name} requires scipy.  Install with: pip install scipy"
        )


def _require_statsmodels(method_name: str) -> None:
    if not _STATSMODELS_AVAILABLE:
        raise ImportError(
            f"{method_name} requires statsmodels.  "
            "Install with: pip install statsmodels"
        )


def _drop_nan_pair(
    x: pd.Series, y: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Drop rows where either x or y is NaN. Returns aligned pair."""
    mask = x.notna() & y.notna()
    return x[mask], y[mask]


def _fisher_z_ci(r: float, n: int, alpha: float) -> tuple[float, float]:
    """
    Compute confidence interval for a Pearson/Spearman correlation
    using Fisher's z-transformation.

    Returns (lower, upper) or (nan, nan) when n < 4.
    """
    if n < 4 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    if _SCIPY_AVAILABLE:
        z_crit = float(_scipy_stats.norm.ppf(1.0 - alpha / 2))
    else:
        z_crit = 1.96  # approximate 95 % critical value
    lo = float(np.tanh(z - z_crit * se))
    hi = float(np.tanh(z + z_crit * se))
    return lo, hi


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's d effect size between two independent samples."""
    na, nb = len(a.dropna()), len(b.dropna())
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def _winsorise(series: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
    """Clip a Series to [lower_pct, upper_pct] percentile bounds."""
    lo = np.nanpercentile(series.dropna(), lower_pct)
    hi = np.nanpercentile(series.dropna(), upper_pct)
    return series.clip(lower=lo, upper=hi)


def _safe_float(v: object) -> float:
    """Convert a value to float; return NaN on failure."""
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# StatisticsUtils
# ---------------------------------------------------------------------------


class StatisticsUtils:
    """
    Statistical analysis toolkit for the event-study pipeline.

    Provides NaN-safe, vectorised implementations of all statistical
    methods required by DAY 6/7 analysis workflows.  The class is
    stateless beyond its configuration: every method accepts DataFrames
    explicitly and returns typed result objects.

    Parameters
    ----------
    config:
        StatisticsConfig controlling confidence levels, bucket counts,
        winsorisation, and default column selections.

    Examples
    --------
    >>> su = StatisticsUtils()
    >>> corr = su.compute_correlations(
    ...     df, x_cols=["finbert_sentiment_score"], y_cols=["car_3d"]
    ... )
    >>> ttest = su.run_t_test(group_a=pos_returns, group_b=neg_returns)
    """

    def __init__(
        self,
        config: Optional[StatisticsConfig] = None,
        **kwargs: object,
    ) -> None:
        if config is None:
            config = StatisticsConfig()
        self.config: StatisticsConfig = config
        logger.debug("StatisticsUtils initialised — config=%s", config)

    # ------------------------------------------------------------------
    # 1. Descriptive statistics
    # ------------------------------------------------------------------

    def describe(
        self,
        df: pd.DataFrame,
        cols: Optional[Sequence[str]] = None,
        percentiles: Sequence[float] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    ) -> pd.DataFrame:
        """
        Extended descriptive statistics for numeric columns.

        Supplements pandas .describe() with skewness, excess kurtosis,
        number of NaN values, and configurable percentile rows.

        Parameters
        ----------
        df:
            Source DataFrame.
        cols:
            Columns to describe.  Defaults to all numeric columns.
        percentiles:
            Quantile levels to include in the output.

        Returns
        -------
        pd.DataFrame
            Rows = statistics, columns = requested columns.
        """
        if cols is None:
            cols = list(df.select_dtypes(include="number").columns)
        present = [c for c in cols if c in df.columns]
        if not present:
            logger.warning("describe(): no matching numeric columns found.")
            return pd.DataFrame()

        sub = df[present]
        base = sub.describe(percentiles=list(percentiles)).rename(
            index=lambda x: str(x)
        )
        extra = pd.DataFrame(
            {
                col: {
                    "skew": _safe_float(sub[col].skew()),
                    "kurtosis": _safe_float(sub[col].kurtosis()),
                    "nan_count": int(sub[col].isna().sum()),
                    "nan_pct": round(float(sub[col].isna().mean()) * 100, 2),
                }
                for col in present
            }
        )
        result = pd.concat([base, extra])
        logger.debug("describe() — shape=%s", result.shape)
        return result

    # ------------------------------------------------------------------
    # 2. Correlation analysis
    # ------------------------------------------------------------------

    def compute_correlations(
        self,
        df: pd.DataFrame,
        x_cols: Optional[Sequence[str]] = None,
        y_cols: Optional[Sequence[str]] = None,
        method: Literal["pearson", "spearman", "both"] = "both",
    ) -> list[CorrelationResult]:
        """
        Compute pairwise correlations between sentiment and return columns.

        Parameters
        ----------
        df:
            Merged event-study DataFrame.
        x_cols:
            Predictor columns (default: config.sentiment_cols).
        y_cols:
            Outcome columns (default: config.outcome_cols).
        method:
            ``'pearson'``, ``'spearman'``, or ``'both'``.

        Returns
        -------
        list[CorrelationResult]
            One result per (x, y, method) combination.
        """
        x_cols = list(x_cols or self.config.sentiment_cols)
        y_cols = list(y_cols or self.config.outcome_cols)
        methods: list[str] = (
            ["pearson", "spearman"] if method == "both" else [method]
        )

        results: list[CorrelationResult] = []
        for m in methods:
            for xc in x_cols:
                for yc in y_cols:
                    if xc not in df.columns or yc not in df.columns:
                        logger.debug(
                            "compute_correlations: column missing (%s or %s) — skip",
                            xc, yc,
                        )
                        continue
                    results.append(
                        self._single_correlation(df[xc], df[yc], m, xc, yc)
                    )

        logger.info(
            "compute_correlations — %d pairs × %d methods = %d results",
            len(x_cols) * len(y_cols),
            len(methods),
            len(results),
        )
        return results

    def correlation_matrix(
        self,
        df: pd.DataFrame,
        cols: Optional[Sequence[str]] = None,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> pd.DataFrame:
        """
        Build a square correlation matrix for *cols*.

        Parameters
        ----------
        df:
            Source DataFrame.
        cols:
            Columns to include.  Defaults to sentiment + outcome columns.
        method:
            Correlation method.

        Returns
        -------
        pd.DataFrame
            Square correlation matrix with NaN on missing pairs.
        """
        if cols is None:
            cols = [
                c for c in list(self.config.sentiment_cols) + list(self.config.outcome_cols)
                if c in df.columns
            ]
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.DataFrame()

        corr = df[present].corr(method=method, numeric_only=True)
        logger.debug("correlation_matrix() — shape=%s  method=%s", corr.shape, method)
        return corr

    def _single_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        method: str,
        x_col: str,
        y_col: str,
    ) -> CorrelationResult:
        """Compute correlation for a single (x, y) pair."""
        xc, yc = _drop_nan_pair(x, y)
        n = len(xc)
        result = CorrelationResult(
            x_col=x_col, y_col=y_col, method=method, n_obs=n
        )

        if n < 3:
            logger.debug(
                "_single_correlation: n=%d too small for %s × %s", n, x_col, y_col
            )
            return result

        if method == "pearson":
            if _SCIPY_AVAILABLE:
                r, p = _scipy_stats.pearsonr(xc, yc)
            else:
                r = float(xc.corr(yc, method="pearson"))
                p = float("nan")
        else:  # spearman
            if _SCIPY_AVAILABLE:
                r, p = _scipy_stats.spearmanr(xc, yc)
            else:
                r = float(xc.corr(yc, method="spearman"))
                p = float("nan")

        r, p = float(r), float(p)
        ci_lo, ci_hi = _fisher_z_ci(r, n, self.config.alpha)

        result.correlation = r
        result.p_value = p
        result.ci_lower = ci_lo
        result.ci_upper = ci_hi
        result.significant = (not np.isnan(p)) and (p < self.config.alpha)
        return result

    # ------------------------------------------------------------------
    # 3. Significance testing
    # ------------------------------------------------------------------

    def run_t_test(
        self,
        group_a: pd.Series,
        group_b: Optional[pd.Series] = None,
        *,
        label: str = "",
        group_a_label: str = "group_a",
        group_b_label: str = "group_b",
        popmean: float = 0.0,
        test_type: Literal["one_sample", "two_sample_welch", "paired"] = "two_sample_welch",
    ) -> TTestResult:
        """
        Run a t-test and return a fully populated TTestResult.

        Parameters
        ----------
        group_a:
            First sample (or the only sample for one-sample tests).
        group_b:
            Second sample.  Required for ``'two_sample_welch'`` and
            ``'paired'``.
        label:
            Human-readable description for reporting.
        group_a_label, group_b_label:
            Short labels for the two groups.
        popmean:
            Hypothesised population mean for one-sample tests.
        test_type:
            Test variant.

        Returns
        -------
        TTestResult
        """
        _require_scipy("run_t_test")
        a = group_a.dropna()
        result = TTestResult(
            label=label or f"{group_a_label} vs {group_b_label}",
            test_type=test_type,
            group_a_label=group_a_label,
            group_b_label=group_b_label,
            n_a=len(a),
            mean_a=float(a.mean()) if len(a) else float("nan"),
            std_a=float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
        )

        if test_type == "one_sample":
            if len(a) < 2:
                logger.warning("run_t_test: insufficient data for one-sample test.")
                return result
            t_stat, p_val = _scipy_stats.ttest_1samp(a, popmean=popmean)
            df_val = len(a) - 1
            ci = _scipy_stats.t.interval(
                self.config.confidence_level, df=df_val,
                loc=float(a.mean()), scale=_scipy_stats.sem(a)
            )
            result.t_statistic = float(t_stat)
            result.p_value = float(p_val)
            result.degrees_of_freedom = float(df_val)
            result.ci_lower, result.ci_upper = float(ci[0]), float(ci[1])
            result.significant = float(p_val) < self.config.alpha

        else:
            if group_b is None:
                raise ValueError(
                    f"group_b is required for test_type='{test_type}'."
                )
            b = group_b.dropna()
            result.n_b = len(b)
            result.mean_b = float(b.mean()) if len(b) else float("nan")
            result.std_b = float(b.std(ddof=1)) if len(b) > 1 else float("nan")

            if len(a) < 2 or len(b) < 2:
                logger.warning(
                    "run_t_test: insufficient data (n_a=%d, n_b=%d).",
                    len(a), len(b),
                )
                return result

            if test_type == "paired":
                min_n = min(len(a), len(b))
                t_stat, p_val = _scipy_stats.ttest_rel(a.iloc[:min_n], b.iloc[:min_n])
                df_val = min_n - 1
            else:  # two_sample_welch
                t_stat, p_val = _scipy_stats.ttest_ind(a, b, equal_var=False)
                # Welch–Satterthwaite degrees of freedom
                s1, s2 = a.var(ddof=1), b.var(ddof=1)
                n1, n2 = len(a), len(b)
                num = (s1 / n1 + s2 / n2) ** 2
                denom = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
                df_val = float(num / denom) if denom != 0 else float(n1 + n2 - 2)

            # CI for the difference in means
            mean_diff = float(a.mean() - b.mean())
            se_diff = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            t_crit = float(
                _scipy_stats.t.ppf(1 - self.config.alpha / 2, df=df_val)
            )
            result.t_statistic = float(t_stat)
            result.p_value = float(p_val)
            result.degrees_of_freedom = df_val
            result.ci_lower = mean_diff - t_crit * se_diff
            result.ci_upper = mean_diff + t_crit * se_diff
            result.significant = float(p_val) < self.config.alpha
            result.effect_size_cohens_d = _cohens_d(a, b)

        logger.debug(
            "run_t_test [%s] — t=%.4f  p=%.4f  sig=%s",
            result.label, result.t_statistic, result.p_value, result.significant,
        )
        return result

    # ------------------------------------------------------------------
    # 4. Confidence intervals
    # ------------------------------------------------------------------

    def calculate_confidence_intervals(
        self,
        series: pd.Series,
        *,
        label: str = "",
    ) -> dict[str, float]:
        """
        Compute the confidence interval for the mean of *series*.

        Uses the t-distribution (appropriate for small samples) when
        scipy is available; falls back to a normal approximation otherwise.

        Parameters
        ----------
        series:
            Numeric Series.  NaN values are dropped.
        label:
            Optional name for logging.

        Returns
        -------
        dict
            Keys: ``mean``, ``ci_lower``, ``ci_upper``, ``std_err``,
            ``n_obs``, ``confidence_level``.
        """
        s = series.dropna()
        n = len(s)
        mean = float(s.mean()) if n > 0 else float("nan")
        se = float(s.sem()) if n > 1 else float("nan")
        ci_lo, ci_hi = float("nan"), float("nan")

        if n > 1:
            if _SCIPY_AVAILABLE:
                ci = _scipy_stats.t.interval(
                    self.config.confidence_level, df=n - 1, loc=mean, scale=se
                )
                ci_lo, ci_hi = float(ci[0]), float(ci[1])
            else:
                # Normal approximation
                z = 1.96
                ci_lo = mean - z * se
                ci_hi = mean + z * se

        out = {
            "label": label,
            "n_obs": n,
            "mean": round(mean, 8),
            "ci_lower": round(ci_lo, 8) if not np.isnan(ci_lo) else float("nan"),
            "ci_upper": round(ci_hi, 8) if not np.isnan(ci_hi) else float("nan"),
            "std_err": round(se, 8) if not np.isnan(se) else float("nan"),
            "confidence_level": self.config.confidence_level,
        }
        logger.debug(
            "calculate_confidence_intervals [%s] — mean=%.5f  CI=[%.5f, %.5f]  n=%d",
            label, mean, ci_lo, ci_hi, n,
        )
        return out

    # ------------------------------------------------------------------
    # 5. Sentiment bucket analysis
    # ------------------------------------------------------------------

    def build_sentiment_buckets(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "finbert_sentiment_score",
        n_buckets: Optional[int] = None,
        bucket_col_name: Optional[str] = None,
        labels: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Assign each event to a sentiment quantile bucket.

        Adds a new column ``<bucket_col_name>`` to the DataFrame
        containing the bucket label for each row.

        Parameters
        ----------
        df:
            Event-study DataFrame containing *sentiment_col*.
        sentiment_col:
            Column to bucket on.
        n_buckets:
            Number of buckets.  Defaults to ``config.n_buckets``.
        bucket_col_name:
            Name for the new bucket column.  Defaults to
            ``<sentiment_col>_bucket``.
        labels:
            Optional custom label list of length *n_buckets*.
            For 3 buckets defaults to ``('negative', 'neutral', 'positive')``.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with the bucket column appended.
        """
        n = n_buckets or self.config.n_buckets
        col_name = bucket_col_name or f"{sentiment_col}_bucket"

        if sentiment_col not in df.columns:
            logger.warning(
                "build_sentiment_buckets: column '%s' not found — bucket col set to NaN.",
                sentiment_col,
            )
            out = df.copy()
            out[col_name] = float("nan")
            return out

        if labels is None:
            if n == 3:
                labels = list(_BUCKET_LABELS_3)
            else:
                labels = [f"bucket_{i + 1}" for i in range(n)]

        if len(labels) != n:
            raise ValueError(
                f"len(labels)={len(labels)} must equal n_buckets={n}."
            )

        valid = df[sentiment_col].notna()
        out = df.copy()
        out[col_name] = pd.NA

        if valid.sum() > 0:
            try:
                out.loc[valid, col_name] = pd.qcut(
                    df.loc[valid, sentiment_col],
                    q=n,
                    labels=labels,
                    duplicates="drop",
                )
            except ValueError as exc:
                logger.warning(
                    "build_sentiment_buckets: pd.qcut failed (%s) — "
                    "falling back to pd.cut with equal-width bins.",
                    exc,
                )
                out.loc[valid, col_name] = pd.cut(
                    df.loc[valid, sentiment_col],
                    bins=n,
                    labels=labels,
                )

        logger.info(
            "build_sentiment_buckets — col='%s'  n=%d  dist=%s",
            sentiment_col, n,
            dict(out[col_name].value_counts().items()),
        )
        return out

    # ------------------------------------------------------------------
    # 6. Group statistics
    # ------------------------------------------------------------------

    def compute_group_statistics(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_cols: Optional[Sequence[str]] = None,
        agg_funcs: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute grouped summary statistics.

        Parameters
        ----------
        df:
            Source DataFrame.
        group_col:
            Column to group by (e.g. ``'ticker'``, ``'quarter'``,
            ``'finbert_sentiment_score_bucket'``).
        value_cols:
            Numeric columns to aggregate.  Defaults to outcome columns.
        agg_funcs:
            Aggregation functions to apply.  Defaults to
            ``['count', 'mean', 'median', 'std', 'min', 'max']``.

        Returns
        -------
        pd.DataFrame
            Multi-level column DataFrame of group statistics.
        """
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in DataFrame.")

        value_cols = list(
            value_cols
            or [c for c in self.config.outcome_cols if c in df.columns]
        )
        if not value_cols:
            logger.warning("compute_group_statistics: no value columns found.")
            return pd.DataFrame()

        agg_funcs = list(agg_funcs or ["count", "mean", "median", "std", "min", "max"])

        grouped = (
            df.groupby(group_col, observed=True)[value_cols]
            .agg(agg_funcs)
        )
        logger.info(
            "compute_group_statistics — group='%s'  groups=%d  value_cols=%s",
            group_col, grouped.shape[0], value_cols,
        )
        return grouped

    # ------------------------------------------------------------------
    # 7. Event-study metric summary
    # ------------------------------------------------------------------

    def summarize_event_study_metrics(
        self,
        df: pd.DataFrame,
        sentiment_cols: Optional[Sequence[str]] = None,
        outcome_cols: Optional[Sequence[str]] = None,
        bucket_col: str = "finbert_sentiment_score",
    ) -> StatisticsSummary:
        """
        Full event-study analysis summary combining descriptive stats,
        correlation analysis, bucket comparison, and t-tests.

        Parameters
        ----------
        df:
            Merged event-study DataFrame (output of SentimentEventMerger).
        sentiment_cols:
            Sentiment predictor columns.
        outcome_cols:
            Return / CAR outcome columns.
        bucket_col:
            Column used to build the sentiment-bucket split.

        Returns
        -------
        StatisticsSummary
        """
        s_cols = list(sentiment_cols or self.config.sentiment_cols)
        o_cols = list(outcome_cols or self.config.outcome_cols)
        all_cols = [c for c in s_cols + o_cols if c in df.columns]

        logger.info(
            "summarize_event_study_metrics — n=%d  s_cols=%s  o_cols=%s",
            len(df), s_cols, o_cols,
        )

        # ── Optionally winsorise return columns ───────────────────────
        analysis_df = df.copy()
        if self.config.winsorise_returns:
            for col in o_cols:
                if col in analysis_df.columns:
                    analysis_df[col] = _winsorise(
                        analysis_df[col],
                        self.config.winsorise_lower,
                        self.config.winsorise_upper,
                    )
            logger.debug(
                "Winsorised return columns at [%.1f%%, %.1f%%]",
                self.config.winsorise_lower,
                self.config.winsorise_upper,
            )

        # ── Descriptive statistics ────────────────────────────────────
        desc = self.describe(analysis_df, cols=all_cols)

        # ── Correlation matrix ────────────────────────────────────────
        corr_mat = self.correlation_matrix(analysis_df, cols=all_cols)

        # ── Pairwise correlation results ──────────────────────────────
        corr_results = self.compute_correlations(
            analysis_df,
            x_cols=s_cols,
            y_cols=o_cols,
            method="both",
        )

        # ── Sentiment bucket summary ──────────────────────────────────
        bucketed = self.build_sentiment_buckets(
            analysis_df, sentiment_col=bucket_col
        )
        bucket_grp_col = f"{bucket_col}_bucket"
        bucket_summary = pd.DataFrame()
        if bucket_grp_col in bucketed.columns:
            bucket_summary = self.compute_group_statistics(
                bucketed,
                group_col=bucket_grp_col,
                value_cols=o_cols,
                agg_funcs=["count", "mean", "median", "std"],
            )

        # ── T-tests: positive vs negative bucket ─────────────────────
        t_results: list[TTestResult] = []
        if _SCIPY_AVAILABLE and bucket_grp_col in bucketed.columns:
            pos_mask = bucketed[bucket_grp_col] == "positive"
            neg_mask = bucketed[bucket_grp_col] == "negative"
            for col in o_cols:
                if col not in bucketed.columns:
                    continue
                if pos_mask.sum() >= 2 and neg_mask.sum() >= 2:
                    t_res = self.run_t_test(
                        group_a=bucketed.loc[pos_mask, col],
                        group_b=bucketed.loc[neg_mask, col],
                        label=f"{col}",
                        group_a_label="positive",
                        group_b_label="negative",
                    )
                    t_results.append(t_res)

        summary = StatisticsSummary(
            descriptive_stats=desc,
            correlation_matrix=corr_mat,
            correlation_results=corr_results,
            bucket_summary=bucket_summary,
            t_test_results=t_results,
            n_events=len(analysis_df),
            columns_analysed=all_cols,
        )
        return summary

    # ------------------------------------------------------------------
    # 8. Regression preparation
    # ------------------------------------------------------------------

    def prepare_regression_dataset(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[Sequence[str]] = None,
        *,
        standardise_features: bool = False,
        add_constant: Optional[bool] = None,
        dummy_cols: Optional[Sequence[str]] = None,
        compute_vif: bool = True,
    ) -> RegressionPreparationResult:
        """
        Prepare a clean feature matrix and target vector for OLS / ML.

        Operations performed
        --------------------
        1. Subset to *feature_cols* + *target_col*.
        2. One-hot encode *dummy_cols* (categorical → dummies, drop first).
        3. Drop rows with any NaN in X or y.
        4. Optionally standardise numeric features (z-score).
        5. Add intercept column if configured.
        6. Compute VIF for multicollinearity diagnostics.

        Parameters
        ----------
        df:
            Merged event-study DataFrame.
        target_col:
            Dependent variable column (e.g. ``'abnormal_return_3d'``).
        feature_cols:
            Independent variable columns.  Defaults to sentiment_cols.
        standardise_features:
            When True, apply z-score standardisation to numeric features.
        add_constant:
            Override ``config.regression_add_constant``.
        dummy_cols:
            Categorical columns to one-hot encode before regression.
        compute_vif:
            Compute VIF table.  Requires statsmodels.

        Returns
        -------
        RegressionPreparationResult
        """
        feature_cols = list(feature_cols or self.config.sentiment_cols)
        add_const = add_constant if add_constant is not None else self.config.regression_add_constant

        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame.")

        required_cols = [target_col] + [c for c in feature_cols if c in df.columns]
        sub = df[required_cols].copy()

        # Dummy encode categorical columns
        if dummy_cols:
            present_dummy = [c for c in dummy_cols if c in sub.columns]
            if present_dummy:
                sub = pd.get_dummies(sub, columns=present_dummy, drop_first=True)
                logger.debug("Dummy-encoded columns: %s", present_dummy)

        n_before = len(sub)
        sub = sub.dropna()
        n_dropped = n_before - len(sub)
        if n_dropped:
            logger.info(
                "prepare_regression_dataset: dropped %d rows with NaN.", n_dropped
            )

        y = sub[target_col]
        feature_names = [c for c in sub.columns if c != target_col]
        X = sub[feature_names].copy()

        # Standardise
        scaler_params: dict[str, dict[str, float]] = {}
        if standardise_features:
            numeric_feats = X.select_dtypes(include="number").columns.tolist()
            for col in numeric_feats:
                mean = float(X[col].mean())
                std = float(X[col].std(ddof=1))
                scaler_params[col] = {"mean": mean, "std": std}
                if std > 0:
                    X[col] = (X[col] - mean) / std
                else:
                    logger.warning(
                        "prepare_regression_dataset: std=0 for '%s' — not standardised.",
                        col,
                    )

        # Add intercept
        if add_const:
            X.insert(0, "const", 1.0)
            feature_names = ["const"] + [c for c in X.columns if c != "const"]

        feature_names = list(X.columns)

        # VIF
        vif_df = pd.DataFrame()
        high_vif: list[str] = []
        if compute_vif and _STATSMODELS_AVAILABLE and len(X) > len(feature_names):
            vif_data = []
            X_arr = X.values.astype(float)
            for i, col in enumerate(X.columns):
                try:
                    vif_val = float(_vif(X_arr, i))
                except Exception:
                    vif_val = float("nan")
                vif_data.append({"feature": col, "VIF": round(vif_val, 4)})
            vif_df = pd.DataFrame(vif_data).set_index("feature")
            high_vif = vif_df[vif_df["VIF"] > 10].index.tolist()
            if high_vif:
                logger.warning(
                    "prepare_regression_dataset: high VIF (>10) features: %s",
                    high_vif,
                )

        result = RegressionPreparationResult(
            X=X,
            y=y,
            feature_names=feature_names,
            n_obs=len(X),
            n_dropped_nan=n_dropped,
            vif_df=vif_df,
            high_vif_features=high_vif,
            scaler_params=scaler_params,
        )
        logger.info(
            "prepare_regression_dataset — target='%s'  n_obs=%d  "
            "n_dropped=%d  features=%s",
            target_col, result.n_obs, result.n_dropped_nan, feature_names,
        )
        return result

    # ------------------------------------------------------------------
    # 9. Z-score helpers
    # ------------------------------------------------------------------

    def compute_z_scores(
        self,
        df: pd.DataFrame,
        cols: Optional[Sequence[str]] = None,
        suffix: str = "_zscore",
    ) -> pd.DataFrame:
        """
        Append z-score columns for each numeric column in *cols*.

        Parameters
        ----------
        df:
            Source DataFrame.
        cols:
            Columns to z-score.  Defaults to all numeric columns.
        suffix:
            Appended to each column name for the z-score column.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with additional z-score columns.
        """
        if cols is None:
            cols = list(df.select_dtypes(include="number").columns)
        out = df.copy()
        for col in cols:
            if col not in out.columns:
                continue
            mean = out[col].mean()
            std = out[col].std(ddof=1)
            z_col = f"{col}{suffix}"
            if std and not np.isnan(std):
                out[z_col] = (out[col] - mean) / std
            else:
                out[z_col] = float("nan")
                logger.debug(
                    "compute_z_scores: std=0 or NaN for '%s' — z-score set to NaN.",
                    col,
                )
        return out

    # ------------------------------------------------------------------
    # 10. Percentile bucketing
    # ------------------------------------------------------------------

    def percentile_rank(
        self,
        series: pd.Series,
        ascending: bool = True,
    ) -> pd.Series:
        """
        Compute the percentile rank (0–100) for each element in *series*.

        NaN values receive a NaN rank.

        Parameters
        ----------
        series:
            Numeric series to rank.
        ascending:
            When True (default), higher values get higher percentile ranks.

        Returns
        -------
        pd.Series
            Percentile ranks in [0, 100].
        """
        return series.rank(pct=True, ascending=ascending, na_option="keep") * 100

    def assign_quintiles(
        self,
        series: pd.Series,
        label_prefix: str = "Q",
    ) -> pd.Series:
        """
        Assign events to quintiles (5 equal-frequency buckets).

        Parameters
        ----------
        series:
            Numeric series.
        label_prefix:
            Prefix for quintile labels (e.g. ``'Q'`` → Q1 … Q5).

        Returns
        -------
        pd.Series
            Categorical series of quintile labels.
        """
        labels = [f"{label_prefix}{i}" for i in range(1, 6)]
        try:
            return pd.qcut(series, q=5, labels=labels, duplicates="drop")
        except ValueError:
            logger.warning(
                "assign_quintiles: could not form 5 equal-frequency bins — "
                "falling back to 5 equal-width bins."
            )
            return pd.cut(series, bins=5, labels=labels)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def quick_describe(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convenience wrapper: describe *cols* with default config."""
    return StatisticsUtils().describe(df, cols=cols)


def quick_correlations(
    df: pd.DataFrame,
    x_cols: Optional[Sequence[str]] = None,
    y_cols: Optional[Sequence[str]] = None,
) -> list[CorrelationResult]:
    """Convenience wrapper: compute Pearson + Spearman correlations."""
    return StatisticsUtils().compute_correlations(df, x_cols=x_cols, y_cols=y_cols)


def quick_t_test(
    group_a: pd.Series,
    group_b: pd.Series,
    label: str = "",
) -> TTestResult:
    """Convenience wrapper: Welch t-test between two independent samples."""
    return StatisticsUtils().run_t_test(
        group_a=group_a, group_b=group_b, label=label
    )


# ---------------------------------------------------------------------------
# Self-test / demo block
# ---------------------------------------------------------------------------


def _make_demo_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic merged event-study DataFrame for the demo."""
    rng = np.random.default_rng(seed)
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    finbert = rng.uniform(-1, 1, n)
    lm = rng.uniform(-0.5, 0.5, n)

    # Inject weak positive relationship: positive sentiment → positive AR
    ar1d = 0.3 * finbert + rng.normal(0, 0.02, n)
    ar3d = 0.25 * finbert + rng.normal(0, 0.03, n)
    ar5d = 0.20 * finbert + rng.normal(0, 0.04, n)
    car3 = ar1d + ar3d
    car5 = ar1d + ar3d + ar5d

    # Introduce a few NaNs
    for arr in (finbert, lm, ar1d):
        arr[rng.choice(n, 5, replace=False)] = np.nan

    return pd.DataFrame(
        {
            "transcript_id": [f"T{i:04d}" for i in range(n)],
            "ticker": rng.choice(tickers, n),
            "quarter": rng.choice(quarters, n),
            "finbert_sentiment_score": finbert,
            "lm_tone_score": lm,
            "abnormal_return_1d": ar1d,
            "abnormal_return_3d": ar3d,
            "abnormal_return_5d": ar5d,
            "car_3d": car3,
            "car_5d": car5,
        }
    )


def _run_demo() -> None:
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger.info("=" * 60)
    logger.info("SELF-TEST / DEMO — statistics_utils.py")
    logger.info("=" * 60)

    df = _make_demo_df(n=100)
    su = StatisticsUtils(
        config=StatisticsConfig(
            confidence_level=0.95,
            n_buckets=3,
            winsorise_returns=True,
        )
    )

    # ── Test 1: describe() ─────────────────────────────────────────────
    logger.info("\n--- Test 1: describe() ---")
    desc = su.describe(df)
    assert not desc.empty
    assert "skew" in desc.index
    assert "nan_count" in desc.index
    print("[Test 1] describe() — shape:", desc.shape)
    print(desc.round(4).to_string())

    # ── Test 2: compute_correlations() ────────────────────────────────
    logger.info("\n--- Test 2: compute_correlations() ---")
    corrs = su.compute_correlations(df, method="both")
    assert len(corrs) > 0
    for c in corrs:
        assert isinstance(c, CorrelationResult)
        assert c.n_obs > 0
    print(f"\n[Test 2] {len(corrs)} correlation results")
    for c in corrs:
        sig = "***" if (not np.isnan(c.p_value) and c.p_value < 0.01) else ""
        print(
            f"  {c.method:<9s}  {c.x_col:<28s} × {c.y_col:<25s} "
            f"r={c.correlation:+.4f}  p={c.p_value:.4f}  {sig}"
        )

    # ── Test 3: correlation_matrix() ──────────────────────────────────
    logger.info("\n--- Test 3: correlation_matrix() ---")
    corr_mat = su.correlation_matrix(df)
    assert not corr_mat.empty
    assert corr_mat.shape[0] == corr_mat.shape[1]
    print("\n[Test 3] Correlation matrix:")
    print(corr_mat.round(4).to_string())

    # ── Test 4: run_t_test() ──────────────────────────────────────────
    if _SCIPY_AVAILABLE:
        logger.info("\n--- Test 4: run_t_test() ---")
        pos_ar = df.loc[df["finbert_sentiment_score"] > 0.2, "abnormal_return_3d"]
        neg_ar = df.loc[df["finbert_sentiment_score"] < -0.2, "abnormal_return_3d"]
        tt = su.run_t_test(
            group_a=pos_ar,
            group_b=neg_ar,
            label="positive vs negative (AR 3d)",
            group_a_label="positive",
            group_b_label="negative",
        )
        assert isinstance(tt, TTestResult)
        assert not np.isnan(tt.t_statistic)
        print(
            f"\n[Test 4] t-test: t={tt.t_statistic:+.4f}  p={tt.p_value:.4f}  "
            f"d={tt.effect_size_cohens_d:.4f}  sig={tt.significant}"
        )

        # One-sample t-test (AR should be ≠ 0?)
        tt_one = su.run_t_test(
            group_a=df["abnormal_return_1d"].dropna(),
            test_type="one_sample",
            label="AR_1d vs 0",
            popmean=0.0,
        )
        assert not np.isnan(tt_one.t_statistic)
        print(
            f"[Test 4] one-sample: t={tt_one.t_statistic:+.4f}  p={tt_one.p_value:.4f}  "
            f"ci=[{tt_one.ci_lower:.5f}, {tt_one.ci_upper:.5f}]"
        )

    # ── Test 5: calculate_confidence_intervals() ──────────────────────
    logger.info("\n--- Test 5: calculate_confidence_intervals() ---")
    ci = su.calculate_confidence_intervals(df["car_3d"], label="car_3d")
    assert "mean" in ci and "ci_lower" in ci and "ci_upper" in ci
    print(f"\n[Test 5] CI for car_3d: mean={ci['mean']:.5f}  "
          f"CI=[{ci['ci_lower']:.5f}, {ci['ci_upper']:.5f}]")

    # ── Test 6: build_sentiment_buckets() ─────────────────────────────
    logger.info("\n--- Test 6: build_sentiment_buckets() ---")
    bucketed = su.build_sentiment_buckets(df, sentiment_col="finbert_sentiment_score")
    assert "finbert_sentiment_score_bucket" in bucketed.columns
    dist = bucketed["finbert_sentiment_score_bucket"].value_counts()
    print(f"\n[Test 6] Bucket distribution:\n{dist.to_string()}")

    # ── Test 7: compute_group_statistics() ────────────────────────────
    logger.info("\n--- Test 7: compute_group_statistics() ---")
    grp = su.compute_group_statistics(df, group_col="ticker")
    assert not grp.empty
    print(f"\n[Test 7] Grouped stats (by ticker) — shape: {grp.shape}")
    print(grp.round(4).to_string())

    # ── Test 8: summarize_event_study_metrics() ───────────────────────
    logger.info("\n--- Test 8: summarize_event_study_metrics() ---")
    summary = su.summarize_event_study_metrics(df)
    assert isinstance(summary, StatisticsSummary)
    assert summary.n_events == len(df)
    assert not summary.descriptive_stats.empty
    assert not summary.correlation_matrix.empty
    summary.print_report()

    # ── Test 9: prepare_regression_dataset() ──────────────────────────
    logger.info("\n--- Test 9: prepare_regression_dataset() ---")
    reg = su.prepare_regression_dataset(
        df,
        target_col="abnormal_return_3d",
        feature_cols=["finbert_sentiment_score", "lm_tone_score"],
        standardise_features=True,
    )
    assert reg.n_obs > 0
    assert "finbert_sentiment_score" in reg.feature_names
    assert len(reg.X) == len(reg.y)
    print(f"\n[Test 9] Regression prep — n_obs={reg.n_obs}  "
          f"n_dropped={reg.n_dropped_nan}")
    print(reg.summary())
    if not reg.vif_df.empty:
        print("VIF:\n", reg.vif_df.to_string())

    # ── Test 10: compute_z_scores() ────────────────────────────────────
    logger.info("\n--- Test 10: compute_z_scores() ---")
    z_df = su.compute_z_scores(
        df, cols=["finbert_sentiment_score", "abnormal_return_1d"]
    )
    assert "finbert_sentiment_score_zscore" in z_df.columns
    assert "abnormal_return_1d_zscore" in z_df.columns
    mean_z = z_df["finbert_sentiment_score_zscore"].mean()
    print(f"\n[Test 10] Z-score mean ≈ 0: {mean_z:.8f}  ✓")

    # ── Test 11: percentile_rank() and assign_quintiles() ─────────────
    logger.info("\n--- Test 11: percentile_rank() + assign_quintiles() ---")
    pct = su.percentile_rank(df["finbert_sentiment_score"].dropna())
    assert pct.min() >= 0 and pct.max() <= 100
    quints = su.assign_quintiles(df["finbert_sentiment_score"].dropna())
    assert quints.nunique() <= 5
    print(f"\n[Test 11] Percentile rank range: [{pct.min():.1f}, {pct.max():.1f}]")
    print(f"[Test 11] Quintile dist:\n{quints.value_counts().sort_index().to_string()}")

    # ── Test 12: module-level convenience wrappers ─────────────────────
    logger.info("\n--- Test 12: convenience wrappers ---")
    desc2 = quick_describe(df, cols=["finbert_sentiment_score"])
    assert not desc2.empty
    corrs2 = quick_correlations(
        df,
        x_cols=["finbert_sentiment_score"],
        y_cols=["car_5d"],
    )
    assert len(corrs2) >= 1
    if _SCIPY_AVAILABLE:
        tt2 = quick_t_test(
            df["abnormal_return_1d"].dropna().head(30),
            df["abnormal_return_1d"].dropna().tail(30),
            label="head vs tail",
        )
        assert not np.isnan(tt2.t_statistic)
    print("[Test 12] Convenience wrappers ✓")

    # ── Test 13: StatisticsConfig validation ──────────────────────────
    logger.info("\n--- Test 13: StatisticsConfig validation ---")
    for bad_kwargs, exc_type in [
        ({"confidence_level": 1.5}, ValueError),
        ({"n_buckets": 1}, ValueError),
        ({"winsorise_lower": 80.0, "winsorise_upper": 20.0}, ValueError),
    ]:
        try:
            StatisticsConfig(**bad_kwargs)  # type: ignore[arg-type]
            assert False, f"Should have raised {exc_type.__name__}"
        except exc_type:
            pass
    print("[Test 13] Config validation ✓")

    print("\n" + "=" * 60)
    print("  SELF-TEST PASSED ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _run_demo()
