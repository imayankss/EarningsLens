"""
src/event_study/validation.py
==============================
Validation and integrity-checking engine for the DAY 6 event-study pipeline.

Responsibilities:
    - Validate event-study datasets at every pipeline stage
    - Enforce finance and data-integrity rules
    - Detect malformed event windows, abnormal returns, CAR metrics
    - Validate sentiment/market merges and benchmark alignment
    - Generate structured, export-ready validation reports

Scope:
    - Validation and integrity checking ONLY
    - No finance metric computation
    - No abnormal return formulas
    - No CAR logic
    - No orchestration
    - No trading-calendar logic

Severity levels:
    CRITICAL  — pipeline should not proceed; data is fundamentally broken
    WARNING   — degraded quality; pipeline may proceed with caution
    INFO      — informational observation; no action required

Author: Earnings Call Sentiment Analyzer Pipeline
Python: 3.11+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SANE_RETURN: Final[float] = 1.0          # |daily return| cap
_MAX_SANE_CAR: Final[float] = 1.0             # |CAR| cap
_DEFAULT_NULL_RATE_WARNING: Final[float] = 0.05   # 5 %
_DEFAULT_NULL_RATE_CRITICAL: Final[float] = 0.20  # 20 %
_DEFAULT_MIN_COVERAGE: Final[float] = 0.80        # 80 % events must be complete
_DEFAULT_MAX_EXTREME_RATE: Final[float] = 0.02    # 2 % extreme-value ceiling

# Column-name constants
_TRANSCRIPT_ID: Final[str] = "transcript_id"
_TICKER: Final[str] = "ticker"
_ALIGNED_EVENT_DATE: Final[str] = "aligned_event_date"
_RELATIVE_DAY: Final[str] = "relative_day"
_ABNORMAL_RETURN: Final[str] = "abnormal_return"
_BENCHMARK_RETURN: Final[str] = "benchmark_return"
_DAILY_RETURN: Final[str] = "daily_return"
_EVENT_DAY_FLAG: Final[str] = "event_day_flag"
_FINBERT_SCORE: Final[str] = "finbert_sentiment_score"
_LM_SCORE: Final[str] = "lm_tone_score"
_SENTIMENT_LABEL: Final[str] = "sentiment_label"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Severity level of a validation issue."""
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    INFO     = "INFO"


class ValidationStage(str, Enum):
    """Pipeline stage a validation issue belongs to."""
    EVENT_WINDOWS      = "event_windows"
    ABNORMAL_RETURNS   = "abnormal_returns"
    CAR_METRICS        = "car_metrics"
    SENTIMENT_MERGE    = "sentiment_merge"
    BENCHMARK          = "benchmark"
    FINAL_DATASET      = "final_dataset"
    GENERAL            = "general"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """
    Immutable configuration for the event-study validation engine.

    Parameters
    ----------
    null_rate_warning_threshold:
        Fraction of nulls in any column that triggers a WARNING.
        Default: 0.05 (5 %).
    null_rate_critical_threshold:
        Fraction of nulls in any column that triggers a CRITICAL issue.
        Default: 0.20 (20 %).
    min_event_coverage:
        Minimum fraction of events that must have complete windows / CAR.
        Below this a CRITICAL issue is raised.  Default: 0.80.
    max_extreme_return_rate:
        Maximum fraction of rows allowed to have |return| > return_cap
        before a CRITICAL issue is raised.  Default: 0.02.
    return_cap:
        Absolute daily-return threshold above which a value is "extreme".
        Default: 1.0 (100 %).
    car_cap:
        Absolute CAR threshold above which a value is "extreme".
        Default: 1.0 (100 %).
    required_post_horizons:
        Post-event relative days that every event must cover.
        Default: (1, 3, 5).
    require_event_day_row:
        If True, validate that each event has exactly one row where
        ``relative_day == 0``.  Default: True.
    strict_mode:
        If True, WARNING-level issues are promoted to CRITICAL.
        Default: False.
    """

    null_rate_warning_threshold: float  = _DEFAULT_NULL_RATE_WARNING
    null_rate_critical_threshold: float = _DEFAULT_NULL_RATE_CRITICAL
    min_event_coverage: float           = _DEFAULT_MIN_COVERAGE
    max_extreme_return_rate: float      = _DEFAULT_MAX_EXTREME_RATE
    return_cap: float                   = _MAX_SANE_RETURN
    car_cap: float                      = _MAX_SANE_CAR
    required_post_horizons: tuple[int, ...] = (1, 3, 5)
    require_event_day_row: bool         = True
    strict_mode: bool                   = False

    def __post_init__(self) -> None:
        if not 0 < self.null_rate_warning_threshold <= 1:
            raise ValueError("null_rate_warning_threshold must be in (0, 1].")
        if not 0 < self.null_rate_critical_threshold <= 1:
            raise ValueError("null_rate_critical_threshold must be in (0, 1].")
        if self.null_rate_warning_threshold > self.null_rate_critical_threshold:
            raise ValueError(
                "null_rate_warning_threshold must be <= null_rate_critical_threshold."
            )
        if not 0 < self.min_event_coverage <= 1:
            raise ValueError("min_event_coverage must be in (0, 1].")
        if not 0 < self.max_extreme_return_rate <= 1:
            raise ValueError("max_extreme_return_rate must be in (0, 1].")
        if self.return_cap <= 0:
            raise ValueError("return_cap must be > 0.")
        if self.car_cap <= 0:
            raise ValueError("car_cap must be > 0.")

    def effective_severity(self, base: Severity) -> Severity:
        """Promote severity if strict_mode is active."""
        if self.strict_mode and base == Severity.WARNING:
            return Severity.CRITICAL
        return base


@dataclass(slots=True)
class ValidationIssue:
    """
    A single validation finding.

    Attributes
    ----------
    severity:   CRITICAL / WARNING / INFO
    stage:      Pipeline stage where the issue was detected.
    check_name: Short machine-readable identifier for the check.
    message:    Human-readable description of the issue.
    affected_ids: Optional list of transcript_ids affected (truncated to 10).
    detail:     Optional additional structured detail dict.
    """

    severity: Severity
    stage: ValidationStage
    check_name: str
    message: str
    affected_ids: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity":     self.severity.value,
            "stage":        self.stage.value,
            "check_name":   self.check_name,
            "message":      self.message,
            "affected_ids": self.affected_ids[:10],
            "detail":       self.detail,
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] [{self.stage.value}] "
            f"{self.check_name}: {self.message}"
        )


@dataclass(slots=True)
class ValidationSummary:
    """
    Aggregate counts and pass/fail verdict for a validation run.

    Attributes
    ----------
    total_checks:       Number of individual checks attempted.
    critical_count:     CRITICAL-severity issues found.
    warning_count:      WARNING-severity issues found.
    info_count:         INFO-severity issues found.
    passed:             True iff zero CRITICAL issues exist.
    strict_passed:      True iff zero CRITICAL or WARNING issues exist.
    """

    total_checks: int   = 0
    critical_count: int = 0
    warning_count: int  = 0
    info_count: int     = 0

    @property
    def passed(self) -> bool:
        return self.critical_count == 0

    @property
    def strict_passed(self) -> bool:
        return self.critical_count == 0 and self.warning_count == 0

    @property
    def total_issues(self) -> int:
        return self.critical_count + self.warning_count + self.info_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_checks":   self.total_checks,
            "critical_count": self.critical_count,
            "warning_count":  self.warning_count,
            "info_count":     self.info_count,
            "total_issues":   self.total_issues,
            "passed":         self.passed,
            "strict_passed":  self.strict_passed,
        }


@dataclass(slots=True)
class DatasetValidationResult:
    """
    Full output of a validation run against one pipeline dataset.

    Attributes
    ----------
    stage:          Which pipeline stage was validated.
    issues:         All detected :class:`ValidationIssue` instances.
    summary:        Aggregate :class:`ValidationSummary`.
    row_count:      Number of rows in the validated DataFrame.
    event_count:    Number of unique transcript_id values.
    validated_at:   ISO timestamp of when validation ran.
    dataset_label:  Optional human-readable label for the dataset.
    """

    stage: ValidationStage
    issues: list[ValidationIssue]       = field(default_factory=list)
    summary: ValidationSummary          = field(default_factory=ValidationSummary)
    row_count: int                      = 0
    event_count: int                    = 0
    validated_at: str                   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    dataset_label: str                  = ""

    @property
    def passed(self) -> bool:
        return self.summary.passed

    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    def warning_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "stage":         self.stage.value,
            "dataset_label": self.dataset_label,
            "validated_at":  self.validated_at,
            "row_count":     self.row_count,
            "event_count":   self.event_count,
            "summary":       self.summary.to_dict(),
            "issues":        [i.to_dict() for i in self.issues],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export issue list as a flat DataFrame for logging/parquet export."""
        if not self.issues:
            return pd.DataFrame(
                columns=[
                    "severity", "stage", "check_name",
                    "message", "affected_ids", "detail",
                ]
            )
        return pd.DataFrame([i.to_dict() for i in self.issues])


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------


class EventStudyValidator:
    """
    Validates event-study datasets at every stage of the DAY 6 pipeline.

    Parameters
    ----------
    config:
        :class:`ValidationConfig` controlling thresholds and strictness.

    Usage
    -----
    ::

        from src.event_study.validation import EventStudyValidator, ValidationConfig

        validator = EventStudyValidator(ValidationConfig())

        result = validator.validate_event_windows(window_df)
        if not result.passed:
            for issue in result.critical_issues():
                logger.error(issue)
    """

    def __init__(self, config: ValidationConfig) -> None:
        self._cfg = config
        logger.info(
            "EventStudyValidator initialised | strict=%s null_warn=%.0f%% "
            "null_crit=%.0f%% coverage=%.0f%%",
            config.strict_mode,
            config.null_rate_warning_threshold * 100,
            config.null_rate_critical_threshold * 100,
            config.min_event_coverage * 100,
        )

    # ------------------------------------------------------------------
    # Public stage validators
    # ------------------------------------------------------------------

    def validate_event_windows(self, df: pd.DataFrame) -> DatasetValidationResult:
        """
        Validate the long-format event-window DataFrame.

        Expected columns (minimum):
            ``transcript_id``, ``ticker``, ``aligned_event_date``,
            ``relative_day``, ``date``, ``event_day_flag``,
            ``window_start``, ``window_end``.

        Checks:
            - Required columns present
            - No duplicate (transcript_id, date) pairs
            - Monotonic relative_day per event
            - Exactly one event_day_flag=True row per event
            - date within [window_start, window_end]
            - No null values in key columns
            - Null-rate thresholds across all columns
        """
        result = self._make_result(ValidationStage.EVENT_WINDOWS, df, "event_windows")
        issues: list[ValidationIssue] = []

        required = {
            _TRANSCRIPT_ID, _TICKER, _ALIGNED_EVENT_DATE,
            _RELATIVE_DAY, "date", _EVENT_DAY_FLAG, "window_start", "window_end",
        }
        if not self._check_required_columns(df, required, ValidationStage.EVENT_WINDOWS, issues):
            return self._finalise(result, issues)

        issues += self._check_null_rates(df, ValidationStage.EVENT_WINDOWS)
        issues += self._check_no_null_key_columns(
            df, [_TRANSCRIPT_ID, _TICKER, _ALIGNED_EVENT_DATE, _RELATIVE_DAY, "date"],
            ValidationStage.EVENT_WINDOWS,
        )
        issues += self._check_duplicate_rows(
            df, [_TRANSCRIPT_ID, "date"], ValidationStage.EVENT_WINDOWS,
            "duplicate_event_session_rows",
        )
        issues += self._check_monotonic_relative_days(df, issues)
        issues += self._check_event_day_flag(df)
        issues += self._check_date_within_bounds(df)
        issues += self._check_event_coverage(
            df, ValidationStage.EVENT_WINDOWS, "event_window_coverage"
        )

        return self._finalise(result, issues)

    def validate_abnormal_returns(self, df: pd.DataFrame) -> DatasetValidationResult:
        """
        Validate the long-format abnormal-return DataFrame.

        Expected columns (minimum):
            ``transcript_id``, ``ticker``, ``aligned_event_date``,
            ``relative_day``, ``abnormal_return``,
            ``daily_return``, ``benchmark_return``.

        Checks:
            - Required columns present
            - No duplicate (transcript_id, relative_day)
            - Extreme |daily_return| values
            - Extreme |abnormal_return| values
            - Benchmark coverage
            - Null rates
            - Required post horizons present per event
        """
        result = self._make_result(
            ValidationStage.ABNORMAL_RETURNS, df, "abnormal_returns"
        )
        issues: list[ValidationIssue] = []

        required = {
            _TRANSCRIPT_ID, _TICKER, _ALIGNED_EVENT_DATE,
            _RELATIVE_DAY, _ABNORMAL_RETURN, _DAILY_RETURN, _BENCHMARK_RETURN,
        }
        if not self._check_required_columns(
            df, required, ValidationStage.ABNORMAL_RETURNS, issues
        ):
            return self._finalise(result, issues)

        issues += self._check_null_rates(df, ValidationStage.ABNORMAL_RETURNS)
        issues += self._check_no_null_key_columns(
            df, [_TRANSCRIPT_ID, _TICKER, _RELATIVE_DAY],
            ValidationStage.ABNORMAL_RETURNS,
        )
        issues += self._check_duplicate_rows(
            df, [_TRANSCRIPT_ID, _RELATIVE_DAY],
            ValidationStage.ABNORMAL_RETURNS,
            "duplicate_abnormal_return_rows",
        )
        issues += self._check_extreme_returns(
            df, _DAILY_RETURN, self._cfg.return_cap, ValidationStage.ABNORMAL_RETURNS
        )
        issues += self._check_extreme_returns(
            df, _ABNORMAL_RETURN, self._cfg.return_cap, ValidationStage.ABNORMAL_RETURNS
        )
        issues += self._check_benchmark_coverage(df)
        issues += self._check_required_post_horizons(df)

        return self._finalise(result, issues)

    def validate_car_metrics(self, df: pd.DataFrame) -> DatasetValidationResult:
        """
        Validate the event-level CAR metrics DataFrame.

        Expected columns (minimum):
            ``transcript_id``, ``ticker``, ``aligned_event_date``,
            at least one ``car_Nd`` column, ``car_complete``.

        Checks:
            - Required columns present
            - No duplicate transcript_ids
            - Extreme |CAR| values
            - NaN rates per CAR column
            - car_complete dtype
            - CAR coverage rate
        """
        result = self._make_result(
            ValidationStage.CAR_METRICS, df, "car_metrics"
        )
        issues: list[ValidationIssue] = []

        required = {_TRANSCRIPT_ID, _TICKER, _ALIGNED_EVENT_DATE, "car_complete"}
        if not self._check_required_columns(
            df, required, ValidationStage.CAR_METRICS, issues
        ):
            return self._finalise(result, issues)

        car_cols = [c for c in df.columns if c.startswith("car_") and c != "car_complete"]
        if not car_cols:
            issues.append(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    stage=ValidationStage.CAR_METRICS,
                    check_name="missing_car_columns",
                    message="No CAR metric columns (car_Nd) found in dataset.",
                )
            )
            return self._finalise(result, issues)

        issues += self._check_duplicate_rows(
            df, [_TRANSCRIPT_ID],
            ValidationStage.CAR_METRICS,
            "duplicate_car_event_rows",
        )
        issues += self._check_no_null_key_columns(
            df, [_TRANSCRIPT_ID, _TICKER, _ALIGNED_EVENT_DATE],
            ValidationStage.CAR_METRICS,
        )

        for col in car_cols:
            issues += self._check_extreme_returns(
                df, col, self._cfg.car_cap, ValidationStage.CAR_METRICS
            )
            issues += self._check_column_null_rate(
                df, col, ValidationStage.CAR_METRICS, f"null_rate_{col}"
            )

        issues += self._check_car_complete_dtype(df)
        issues += self._check_car_coverage(df, car_cols)

        return self._finalise(result, issues)

    def validate_sentiment_merge(self, df: pd.DataFrame) -> DatasetValidationResult:
        """
        Validate that sentiment scores are correctly merged onto event rows.

        Expected columns (minimum):
            ``transcript_id``, ``finbert_sentiment_score``,
            ``lm_tone_score``, ``sentiment_label``.

        Checks:
            - Required columns present
            - No null sentiment scores
            - sentiment_label values in expected set
            - No duplicate transcript_ids
            - Null rates for sentiment columns
        """
        result = self._make_result(
            ValidationStage.SENTIMENT_MERGE, df, "sentiment_merge"
        )
        issues: list[ValidationIssue] = []

        required = {
            _TRANSCRIPT_ID, _FINBERT_SCORE, _LM_SCORE, _SENTIMENT_LABEL,
        }
        if not self._check_required_columns(
            df, required, ValidationStage.SENTIMENT_MERGE, issues
        ):
            return self._finalise(result, issues)

        issues += self._check_duplicate_rows(
            df, [_TRANSCRIPT_ID],
            ValidationStage.SENTIMENT_MERGE,
            "duplicate_sentiment_event_rows",
        )
        issues += self._check_no_null_key_columns(
            df, [_FINBERT_SCORE, _LM_SCORE, _SENTIMENT_LABEL],
            ValidationStage.SENTIMENT_MERGE,
        )
        issues += self._check_null_rates(
            df, ValidationStage.SENTIMENT_MERGE,
            columns=[_FINBERT_SCORE, _LM_SCORE, _SENTIMENT_LABEL],
        )
        issues += self._check_sentiment_label_values(df)
        issues += self._check_sentiment_score_range(df)

        return self._finalise(result, issues)

    def validate_final_dataset(self, df: pd.DataFrame) -> DatasetValidationResult:
        """
        Validate the fully-merged event-study export dataset.

        Expected columns (minimum):
            ``transcript_id``, ``ticker``, ``aligned_event_date`` (or
            ``event_date``), ``finbert_sentiment_score``, ``lm_tone_score``,
            ``sentiment_label``, ``abnormal_return``, at least one ``car_Nd``.

        Runs all relevant sub-checks across the unified dataset.
        """
        result = self._make_result(
            ValidationStage.FINAL_DATASET, df, "event_study_final"
        )
        issues: list[ValidationIssue] = []

        # Normalise event_date / aligned_event_date
        if "event_date" in df.columns and _ALIGNED_EVENT_DATE not in df.columns:
            df = df.rename(columns={"event_date": _ALIGNED_EVENT_DATE})

        required = {
            _TRANSCRIPT_ID, _TICKER, _ALIGNED_EVENT_DATE,
            _FINBERT_SCORE, _LM_SCORE, _SENTIMENT_LABEL, _ABNORMAL_RETURN,
        }
        if not self._check_required_columns(
            df, required, ValidationStage.FINAL_DATASET, issues
        ):
            return self._finalise(result, issues)

        car_cols = [c for c in df.columns if c.startswith("car_") and c != "car_complete"]
        if not car_cols:
            issues.append(ValidationIssue(
                severity=self._cfg.effective_severity(Severity.WARNING),
                stage=ValidationStage.FINAL_DATASET,
                check_name="missing_car_columns",
                message="No CAR columns (car_Nd) found in final dataset.",
            ))

        issues += self._check_duplicate_rows(
            df, [_TRANSCRIPT_ID],
            ValidationStage.FINAL_DATASET,
            "duplicate_final_event_rows",
        )
        issues += self._check_no_null_key_columns(
            df,
            [_TRANSCRIPT_ID, _TICKER, _ALIGNED_EVENT_DATE,
             _FINBERT_SCORE, _LM_SCORE, _SENTIMENT_LABEL],
            ValidationStage.FINAL_DATASET,
        )
        issues += self._check_null_rates(df, ValidationStage.FINAL_DATASET)
        issues += self._check_extreme_returns(
            df, _ABNORMAL_RETURN, self._cfg.return_cap, ValidationStage.FINAL_DATASET
        )
        for col in car_cols:
            issues += self._check_extreme_returns(
                df, col, self._cfg.car_cap, ValidationStage.FINAL_DATASET
            )
        issues += self._check_sentiment_label_values(df)
        issues += self._check_sentiment_score_range(df)
        issues += self._check_event_date_not_weekend(df, _ALIGNED_EVENT_DATE)

        return self._finalise(result, issues)

    def validate_event_dataset(self, df: pd.DataFrame) -> DatasetValidationResult:
        """
        General-purpose event dataset validator.

        Runs the common integrity checks applicable to any event-level or
        long-format DataFrame in the pipeline:
            - Required identifier columns
            - Duplicate rows
            - Null rates
            - Extreme return values (if return columns present)
            - Monotonic relative_day ordering (if column present)
        """
        result = self._make_result(
            ValidationStage.GENERAL, df, "event_dataset"
        )
        issues: list[ValidationIssue] = []

        required = {_TRANSCRIPT_ID, _TICKER}
        if not self._check_required_columns(
            df, required, ValidationStage.GENERAL, issues
        ):
            return self._finalise(result, issues)

        issues += self._check_null_rates(df, ValidationStage.GENERAL)
        issues += self._check_no_null_key_columns(
            df, [_TRANSCRIPT_ID, _TICKER], ValidationStage.GENERAL
        )
        issues += self._check_duplicate_rows(
            df, [_TRANSCRIPT_ID] + (
                [_RELATIVE_DAY] if _RELATIVE_DAY in df.columns else []
            ),
            ValidationStage.GENERAL,
            "duplicate_event_rows",
        )

        for ret_col in [_DAILY_RETURN, _ABNORMAL_RETURN]:
            if ret_col in df.columns:
                issues += self._check_extreme_returns(
                    df, ret_col, self._cfg.return_cap, ValidationStage.GENERAL
                )

        if _RELATIVE_DAY in df.columns and _TRANSCRIPT_ID in df.columns:
            issues += self._check_monotonic_relative_days(df, [])

        return self._finalise(result, issues)

    def summarize_validation(
        self, results: Sequence[DatasetValidationResult]
    ) -> dict[str, Any]:
        """
        Produce a structured summary dict across multiple validation results.

        Parameters
        ----------
        results:
            One :class:`DatasetValidationResult` per pipeline stage.

        Returns
        -------
        dict with keys: ``overall_passed``, ``stages``, ``total_issues``,
        ``critical_count``, ``warning_count``, ``info_count``.
        """
        all_issues = [i for r in results for i in r.issues]
        critical = sum(1 for i in all_issues if i.severity == Severity.CRITICAL)
        warning  = sum(1 for i in all_issues if i.severity == Severity.WARNING)
        info     = sum(1 for i in all_issues if i.severity == Severity.INFO)

        stages_summary = {}
        for r in results:
            stages_summary[r.stage.value] = {
                "dataset_label": r.dataset_label,
                "passed":        r.passed,
                "row_count":     r.row_count,
                "event_count":   r.event_count,
                **r.summary.to_dict(),
            }

        summary = {
            "overall_passed":  critical == 0,
            "total_issues":    len(all_issues),
            "critical_count":  critical,
            "warning_count":   warning,
            "info_count":      info,
            "stages":          stages_summary,
            "validated_at":    datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Validation summary | overall=%s critical=%d warning=%d info=%d stages=%d",
            "PASS" if summary["overall_passed"] else "FAIL",
            critical, warning, info, len(results),
        )
        return summary

    # ------------------------------------------------------------------
    # Private check implementations
    # ------------------------------------------------------------------

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        required: set[str],
        stage: ValidationStage,
        issues: list[ValidationIssue],
    ) -> bool:
        """Return True iff all required columns are present."""
        missing = required - set(df.columns)
        if missing:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                stage=stage,
                check_name="missing_required_columns",
                message=(
                    f"Required columns missing: {sorted(missing)}. "
                    f"Pipeline cannot proceed."
                ),
                detail={"missing": sorted(missing)},
            ))
            return False
        return True

    def _check_null_rates(
        self,
        df: pd.DataFrame,
        stage: ValidationStage,
        columns: list[str] | None = None,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        cols = columns if columns is not None else list(df.columns)
        for col in cols:
            if col not in df.columns:
                continue
            rate = float(df[col].isnull().mean())
            if rate >= self._cfg.null_rate_critical_threshold:
                issues.append(ValidationIssue(
                    severity=Severity.CRITICAL,
                    stage=stage,
                    check_name=f"high_null_rate_{col}",
                    message=(
                        f"Column '{col}' has {rate:.1%} null values — "
                        f"exceeds critical threshold "
                        f"({self._cfg.null_rate_critical_threshold:.1%})."
                    ),
                    detail={"column": col, "null_rate": round(rate, 4)},
                ))
            elif rate >= self._cfg.null_rate_warning_threshold:
                issues.append(ValidationIssue(
                    severity=self._cfg.effective_severity(Severity.WARNING),
                    stage=stage,
                    check_name=f"elevated_null_rate_{col}",
                    message=(
                        f"Column '{col}' has {rate:.1%} null values — "
                        f"exceeds warning threshold "
                        f"({self._cfg.null_rate_warning_threshold:.1%})."
                    ),
                    detail={"column": col, "null_rate": round(rate, 4)},
                ))
        return issues

    def _check_column_null_rate(
        self,
        df: pd.DataFrame,
        column: str,
        stage: ValidationStage,
        check_name: str,
    ) -> list[ValidationIssue]:
        return self._check_null_rates(df, stage, columns=[column])

    def _check_no_null_key_columns(
        self,
        df: pd.DataFrame,
        columns: list[str],
        stage: ValidationStage,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for col in columns:
            if col not in df.columns:
                continue
            null_count = int(df[col].isnull().sum())
            if null_count:
                issues.append(ValidationIssue(
                    severity=Severity.CRITICAL,
                    stage=stage,
                    check_name=f"null_key_column_{col}",
                    message=(
                        f"Key column '{col}' has {null_count} null value(s). "
                        f"All key columns must be non-null."
                    ),
                    detail={"column": col, "null_count": null_count},
                ))
        return issues

    def _check_duplicate_rows(
        self,
        df: pd.DataFrame,
        subset: list[str],
        stage: ValidationStage,
        check_name: str,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        valid_subset = [c for c in subset if c in df.columns]
        if not valid_subset:
            return issues
        dup_mask = df.duplicated(subset=valid_subset)
        n_dup = int(dup_mask.sum())
        if n_dup:
            affected = (
                df.loc[dup_mask, _TRANSCRIPT_ID].unique().tolist()[:10]
                if _TRANSCRIPT_ID in df.columns
                else []
            )
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                stage=stage,
                check_name=check_name,
                message=(
                    f"Found {n_dup} duplicate rows on columns "
                    f"{valid_subset}."
                ),
                affected_ids=[str(x) for x in affected],
                detail={"duplicate_count": n_dup, "subset": valid_subset},
            ))
        return issues

    def _check_monotonic_relative_days(
        self,
        df: pd.DataFrame,
        _pre_issues: list[ValidationIssue],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if _TRANSCRIPT_ID not in df.columns or _RELATIVE_DAY not in df.columns:
            return issues

        bad_events: list[str] = []
        for tid, grp in df.groupby(_TRANSCRIPT_ID, sort=False):
            ordered_rd = grp[_RELATIVE_DAY]
            diffs = ordered_rd.diff().dropna()
            if not (diffs >= 1).all():
                bad_events.append(str(tid))

        if bad_events:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                stage=ValidationStage.EVENT_WINDOWS,
                check_name="non_monotonic_relative_days",
                message=(
                    f"{len(bad_events)} event(s) have non-monotonically "
                    f"increasing relative_day sequences."
                ),
                affected_ids=bad_events[:10],
                detail={"affected_count": len(bad_events)},
            ))
        return issues

    def _check_event_day_flag(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if _EVENT_DAY_FLAG not in df.columns or _TRANSCRIPT_ID not in df.columns:
            return issues
        if not self._cfg.require_event_day_row:
            return issues

        flag_counts = (
            df[df[_EVENT_DAY_FLAG].astype(bool)]
            .groupby(_TRANSCRIPT_ID)[_EVENT_DAY_FLAG]
            .count()
        )
        all_ids = set(df[_TRANSCRIPT_ID].unique())
        flagged_ids = set(flag_counts.index)

        missing_flag = all_ids - flagged_ids
        if missing_flag:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                stage=ValidationStage.EVENT_WINDOWS,
                check_name="missing_event_day_flag",
                message=(
                    f"{len(missing_flag)} event(s) have no "
                    f"event_day_flag=True row (t=0 missing)."
                ),
                affected_ids=[str(x) for x in list(missing_flag)[:10]],
                detail={"affected_count": len(missing_flag)},
            ))

        multi_flag = flag_counts[flag_counts > 1]
        if not multi_flag.empty:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                stage=ValidationStage.EVENT_WINDOWS,
                check_name="multiple_event_day_flags",
                message=(
                    f"{len(multi_flag)} event(s) have more than one "
                    f"event_day_flag=True row."
                ),
                affected_ids=[str(x) for x in multi_flag.index.tolist()[:10]],
                detail={"affected_count": len(multi_flag)},
            ))
        return issues

    def _check_date_within_bounds(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for col in ["window_start", "window_end", "date"]:
            if col not in df.columns:
                return issues

        out = df[
            (df["date"] < df["window_start"]) | (df["date"] > df["window_end"])
        ]
        if not out.empty:
            affected = (
                df.loc[out.index, _TRANSCRIPT_ID].unique().tolist()[:10]
                if _TRANSCRIPT_ID in df.columns else []
            )
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                stage=ValidationStage.EVENT_WINDOWS,
                check_name="date_out_of_window_bounds",
                message=(
                    f"{len(out)} rows have 'date' outside "
                    f"[window_start, window_end]."
                ),
                affected_ids=[str(x) for x in affected],
                detail={"affected_row_count": len(out)},
            ))
        return issues

    def _check_event_coverage(
        self,
        df: pd.DataFrame,
        stage: ValidationStage,
        check_name: str,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if _TRANSCRIPT_ID not in df.columns:
            return issues

        total = df[_TRANSCRIPT_ID].nunique()
        if total == 0:
            return issues

        # Coverage = fraction of events with at least min_post_days_required post days
        min_required = min(self._cfg.required_post_horizons)
        coverage_counts = (
            df[df.get(_RELATIVE_DAY, pd.Series(dtype=int)) >= min_required]
            .groupby(_TRANSCRIPT_ID)[_TRANSCRIPT_ID]
            .count()
            if _RELATIVE_DAY in df.columns
            else pd.Series(dtype=int)
        )
        covered = len(coverage_counts)
        rate = covered / total if total else 0.0

        if rate < self._cfg.min_event_coverage:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                stage=stage,
                check_name=check_name,
                message=(
                    f"Only {rate:.1%} of events have post-event data "
                    f"(min required: {self._cfg.min_event_coverage:.1%})."
                ),
                detail={"coverage_rate": round(rate, 4), "total_events": total},
            ))
        return issues

    def _check_extreme_returns(
        self,
        df: pd.DataFrame,
        column: str,
        cap: float,
        stage: ValidationStage,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if column not in df.columns:
            return issues

        extreme_mask = df[column].abs() > cap
        n_extreme = int(extreme_mask.sum())
        if n_extreme == 0:
            return issues

        rate = n_extreme / len(df)
        severity = (
            Severity.CRITICAL
            if rate > self._cfg.max_extreme_return_rate
            else self._cfg.effective_severity(Severity.WARNING)
        )
        affected = (
            df.loc[extreme_mask, _TRANSCRIPT_ID].unique().tolist()[:10]
            if _TRANSCRIPT_ID in df.columns else []
        )
        issues.append(ValidationIssue(
            severity=severity,
            stage=stage,
            check_name=f"extreme_values_{column}",
            message=(
                f"Column '{column}' has {n_extreme} rows ({rate:.1%}) "
                f"with |value| > {cap:.2f}."
            ),
            affected_ids=[str(x) for x in affected],
            detail={
                "column": column,
                "extreme_count": n_extreme,
                "extreme_rate": round(rate, 4),
                "cap": cap,
            },
        ))
        return issues

    def _check_benchmark_coverage(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if _BENCHMARK_RETURN not in df.columns:
            return issues

        null_count = int(df[_BENCHMARK_RETURN].isnull().sum())
        if null_count == 0:
            return issues

        rate = null_count / len(df)
        severity = (
            Severity.CRITICAL
            if rate >= self._cfg.null_rate_critical_threshold
            else self._cfg.effective_severity(Severity.WARNING)
        )
        issues.append(ValidationIssue(
            severity=severity,
            stage=ValidationStage.BENCHMARK,
            check_name="missing_benchmark_returns",
            message=(
                f"{null_count} rows ({rate:.1%}) are missing benchmark_return. "
                f"Abnormal return calculations will be incorrect."
            ),
            detail={"null_count": null_count, "null_rate": round(rate, 4)},
        ))
        return issues

    def _check_required_post_horizons(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if _RELATIVE_DAY not in df.columns or _TRANSCRIPT_ID not in df.columns:
            return issues

        incomplete: list[str] = []
        max_horizon = max(self._cfg.required_post_horizons)

        for tid, grp in df.groupby(_TRANSCRIPT_ID, sort=False):
            present_days = set(grp[_RELATIVE_DAY].unique())
            for h in self._cfg.required_post_horizons:
                if h not in present_days:
                    incomplete.append(str(tid))
                    break

        if incomplete:
            rate = len(incomplete) / df[_TRANSCRIPT_ID].nunique()
            severity = (
                Severity.CRITICAL
                if rate > (1.0 - self._cfg.min_event_coverage)
                else self._cfg.effective_severity(Severity.WARNING)
            )
            issues.append(ValidationIssue(
                severity=severity,
                stage=ValidationStage.ABNORMAL_RETURNS,
                check_name="missing_required_post_horizons",
                message=(
                    f"{len(incomplete)} event(s) are missing at least one "
                    f"required post-event horizon from "
                    f"{list(self._cfg.required_post_horizons)}."
                ),
                affected_ids=incomplete[:10],
                detail={
                    "incomplete_count": len(incomplete),
                    "required_horizons": list(self._cfg.required_post_horizons),
                },
            ))
        return issues

    def _check_car_complete_dtype(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if "car_complete" not in df.columns:
            return issues
        if df["car_complete"].dtype != bool:
            issues.append(ValidationIssue(
                severity=self._cfg.effective_severity(Severity.WARNING),
                stage=ValidationStage.CAR_METRICS,
                check_name="car_complete_dtype",
                message=(
                    f"Column 'car_complete' has dtype '{df['car_complete'].dtype}'; "
                    f"expected bool."
                ),
                detail={"actual_dtype": str(df["car_complete"].dtype)},
            ))
        return issues

    def _check_car_coverage(
        self, df: pd.DataFrame, car_cols: list[str]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        total = len(df)
        if total == 0:
            return issues

        for col in car_cols:
            valid = df[col].notna().sum()
            rate = valid / total
            if rate < self._cfg.min_event_coverage:
                issues.append(ValidationIssue(
                    severity=Severity.CRITICAL,
                    stage=ValidationStage.CAR_METRICS,
                    check_name=f"low_car_coverage_{col}",
                    message=(
                        f"Only {rate:.1%} of events have a valid '{col}' "
                        f"(min required: {self._cfg.min_event_coverage:.1%})."
                    ),
                    detail={
                        "column": col,
                        "coverage_rate": round(rate, 4),
                        "valid_count": int(valid),
                        "total_events": total,
                    },
                ))
        return issues

    def _check_sentiment_label_values(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if _SENTIMENT_LABEL not in df.columns:
            return issues

        expected = {"positive", "negative", "neutral"}
        actual = set(df[_SENTIMENT_LABEL].dropna().unique())
        unexpected = actual - expected
        if unexpected:
            issues.append(ValidationIssue(
                severity=self._cfg.effective_severity(Severity.WARNING),
                stage=ValidationStage.SENTIMENT_MERGE,
                check_name="unexpected_sentiment_labels",
                message=(
                    f"Unexpected sentiment_label values found: {sorted(unexpected)}. "
                    f"Expected: {sorted(expected)}."
                ),
                detail={
                    "unexpected_values": sorted(str(v) for v in unexpected),
                    "expected_values":   sorted(expected),
                },
            ))
        return issues

    def _check_sentiment_score_range(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for col, lo, hi in [
            (_FINBERT_SCORE, -1.0, 1.0),
            (_LM_SCORE,      -1.0, 1.0),
        ]:
            if col not in df.columns:
                continue
            out_of_range = df[col].dropna()
            out_of_range = out_of_range[(out_of_range < lo) | (out_of_range > hi)]
            if not out_of_range.empty:
                issues.append(ValidationIssue(
                    severity=self._cfg.effective_severity(Severity.WARNING),
                    stage=ValidationStage.SENTIMENT_MERGE,
                    check_name=f"out_of_range_{col}",
                    message=(
                        f"{len(out_of_range)} values in '{col}' are outside "
                        f"[{lo}, {hi}]."
                    ),
                    detail={
                        "column":          col,
                        "out_of_range_count": len(out_of_range),
                        "min_value":       float(out_of_range.min()),
                        "max_value":       float(out_of_range.max()),
                    },
                ))
        return issues

    def _check_event_date_not_weekend(
        self,
        df: pd.DataFrame,
        date_col: str,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if date_col not in df.columns:
            return issues

        dates = pd.to_datetime(df[date_col]).dropna()
        weekend_mask = dates.dt.dayofweek >= 5
        n_weekend = int(weekend_mask.sum())
        if n_weekend:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                stage=ValidationStage.FINAL_DATASET,
                check_name="event_date_on_weekend",
                message=(
                    f"{n_weekend} event(s) have an aligned_event_date on a weekend. "
                    f"All event dates must be valid NYSE trading sessions."
                ),
                detail={"weekend_count": n_weekend},
            ))
        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_result(
        stage: ValidationStage,
        df: pd.DataFrame,
        label: str,
    ) -> DatasetValidationResult:
        return DatasetValidationResult(
            stage=stage,
            row_count=len(df),
            event_count=(
                int(df[_TRANSCRIPT_ID].nunique())
                if _TRANSCRIPT_ID in df.columns
                else 0
            ),
            dataset_label=label,
        )

    def _finalise(
        self,
        result: DatasetValidationResult,
        issues: list[ValidationIssue],
    ) -> DatasetValidationResult:
        result.issues = issues
        result.summary = _build_summary(issues)

        for issue in issues:
            log_fn = (
                logger.error   if issue.severity == Severity.CRITICAL
                else logger.warning if issue.severity == Severity.WARNING
                else logger.info
            )
            log_fn("%s", issue)

        logger.info(
            "Validation finished | stage=%s label=%s rows=%d events=%d "
            "critical=%d warning=%d info=%d passed=%s",
            result.stage.value,
            result.dataset_label,
            result.row_count,
            result.event_count,
            result.summary.critical_count,
            result.summary.warning_count,
            result.summary.info_count,
            result.passed,
        )
        return result


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def detect_duplicates(
    df: pd.DataFrame,
    subset: list[str],
) -> pd.DataFrame:
    """
    Return the subset of *df* that contains duplicate rows on *subset* columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    subset:
        Column names to check for duplication.

    Returns
    -------
    pd.DataFrame
        All rows involved in at least one duplicate (keep=False).
    """
    valid = [c for c in subset if c in df.columns]
    if not valid:
        return pd.DataFrame()
    return df[df.duplicated(subset=valid, keep=False)].copy()


def null_rate_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column null rates for *df*.

    Returns
    -------
    pd.DataFrame
        Columns: ``column``, ``null_count``, ``null_rate``, ``total_rows``.
        Sorted descending by ``null_rate``.
    """
    total = len(df)
    records = [
        {
            "column":     col,
            "null_count": int(df[col].isnull().sum()),
            "null_rate":  round(float(df[col].isnull().mean()), 6),
            "total_rows": total,
        }
        for col in df.columns
    ]
    return (
        pd.DataFrame(records)
        .sort_values("null_rate", ascending=False)
        .reset_index(drop=True)
    )


def check_monotonic_groups(
    df: pd.DataFrame,
    group_col: str,
    order_col: str,
    strict: bool = True,
) -> list[str]:
    """
    Return group keys from *group_col* where *order_col* is not monotonically
    increasing within the group.

    Parameters
    ----------
    df:
        Long-format DataFrame.
    group_col:
        Column whose unique values define groups.
    order_col:
        Column that should be monotonically increasing within each group.
    strict:
        If True, require strictly increasing (diffs >= 1).
        If False, allow non-decreasing (diffs >= 0).

    Returns
    -------
    list[str]
        Group keys that violate the ordering constraint.
    """
    threshold = 1 if strict else 0
    bad: list[str] = []
    for key, grp in df.groupby(group_col, sort=False):
        ordered_vals = grp[order_col]
        diffs = ordered_vals.diff().dropna()
        if not (diffs >= threshold).all():
            bad.append(str(key))
    return bad


def check_grouped_integrity(
    df: pd.DataFrame,
    group_col: str,
    checks: dict[str, Any],
) -> dict[str, list[str]]:
    """
    Run configurable per-group integrity checks.

    Parameters
    ----------
    df:
        Long-format DataFrame.
    group_col:
        Column that defines the groups (e.g. ``transcript_id``).
    checks:
        Dict of ``check_name → check_spec`` where check_spec is one of:

        ``{"type": "exact_count", "column": col, "value": True, "expected": 1}``
            Each group must have exactly ``expected`` rows where ``column == value``.

        ``{"type": "min_count", "column": col, "min": N}``
            Each group must have at least ``min`` non-null rows in ``column``.

        ``{"type": "no_duplicates", "columns": [col1, col2]}``
            No duplicate rows on given columns within each group.

    Returns
    -------
    dict[str, list[str]]
        Mapping check_name → list of group-key strings that failed the check.
    """
    failures: dict[str, list[str]] = {name: [] for name in checks}

    for key, grp in df.groupby(group_col, sort=False):
        for name, spec in checks.items():
            ctype = spec.get("type")

            if ctype == "exact_count":
                col = spec["column"]
                val = spec["value"]
                expected = spec.get("expected", 1)
                actual = int((grp[col] == val).sum()) if col in grp.columns else 0
                if actual != expected:
                    failures[name].append(str(key))

            elif ctype == "min_count":
                col = spec["column"]
                mn = spec.get("min", 1)
                actual = int(grp[col].notna().sum()) if col in grp.columns else 0
                if actual < mn:
                    failures[name].append(str(key))

            elif ctype == "no_duplicates":
                cols = [c for c in spec.get("columns", []) if c in grp.columns]
                if cols and grp.duplicated(subset=cols).any():
                    failures[name].append(str(key))

    return failures


def detect_extreme_values(
    df: pd.DataFrame,
    column: str,
    cap: float,
) -> pd.DataFrame:
    """
    Return rows of *df* where ``|column| > cap``.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Numeric column to inspect.
    cap:
        Absolute threshold.

    Returns
    -------
    pd.DataFrame
        Filtered rows, or empty DataFrame if none found.
    """
    if column not in df.columns:
        return pd.DataFrame()
    return df[df[column].abs() > cap].copy()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_summary(issues: list[ValidationIssue]) -> ValidationSummary:
    summary = ValidationSummary(total_checks=len(issues))
    for issue in issues:
        if issue.severity == Severity.CRITICAL:
            summary.critical_count += 1
        elif issue.severity == Severity.WARNING:
            summary.warning_count += 1
        else:
            summary.info_count += 1
    return summary


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_event_study_validator(
    null_rate_warning: float  = _DEFAULT_NULL_RATE_WARNING,
    null_rate_critical: float = _DEFAULT_NULL_RATE_CRITICAL,
    min_event_coverage: float = _DEFAULT_MIN_COVERAGE,
    return_cap: float         = _MAX_SANE_RETURN,
    car_cap: float            = _MAX_SANE_CAR,
    required_post_horizons: tuple[int, ...] = (1, 3, 5),
    strict_mode: bool         = False,
) -> EventStudyValidator:
    """
    Convenience factory for :class:`EventStudyValidator`.

    Returns
    -------
    EventStudyValidator
    """
    config = ValidationConfig(
        null_rate_warning_threshold=null_rate_warning,
        null_rate_critical_threshold=null_rate_critical,
        min_event_coverage=min_event_coverage,
        return_cap=return_cap,
        car_cap=car_cap,
        required_post_horizons=required_post_horizons,
        strict_mode=strict_mode,
    )
    return EventStudyValidator(config)
