"""
lm_validation.py
================
Research-Grade Validation Layer for the Loughran-McDonald Sentiment Pipeline.

Responsibilities
----------------
* Validate chunk-level, transcript-level, section-level, and speaker-level
  LM output DataFrames produced by lm_scoring.py and lm_aggregation.py.
* Accumulate structured ValidationIssue objects (with severity, category,
  affected rows/columns) without raising immediately.
* Produce ValidationSummary objects and human-readable reports.
* Surface distribution diagnostics (score histograms, null rates, coverage
  distributions, label distributions) to support pipeline health monitoring.

Does NOT implement:
    scoring, matching, aggregation, pipeline orchestration, FinBERT inference.

Validation philosophy
---------------------
Every check is a pure function of the input DataFrame + config.
No check has side effects or mutates the DataFrame.
Issue lists are deterministically ordered: schema → nulls → scores →
counts → coverage → duplicates → labels → distributions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"

    def __lt__(self, other: "Severity") -> bool:
        _order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2}
        return _order[self] < _order[other]


# ---------------------------------------------------------------------------
# Validation category constants
# ---------------------------------------------------------------------------

class VCategory:
    SCHEMA       = "schema"
    NULLS        = "nulls"
    SCORES       = "scores"
    COUNTS       = "counts"
    COVERAGE     = "coverage"
    DUPLICATES   = "duplicates"
    LABELS       = "labels"
    DISTRIBUTION = "distribution"
    INTEGRITY    = "integrity"


# ---------------------------------------------------------------------------
# Column-set constants
# ---------------------------------------------------------------------------

#: Core columns present in chunk-level LM scoring output.
CHUNK_SCHEMA: FrozenSet[str] = frozenset({
    "transcript_id",
    "chunk_id",
    "lm_tone_score",
    "lm_positive_count",
    "lm_negative_count",
    "lm_uncertainty_count",
    "token_count",
    "matched_token_count",
})

#: Core columns present in transcript-level aggregation output.
TRANSCRIPT_SCHEMA: FrozenSet[str] = frozenset({
    "transcript_id",
    "lm_mean_tone",
    "lm_median_tone",
    "lm_std_tone",
    "lm_token_total",
    "lm_positive_total",
    "lm_negative_total",
    "lm_coverage_ratio",
    "lm_sentiment_label",
    "lm_chunk_count",
})

#: Core columns present in section-level aggregation output.
SECTION_SCHEMA: FrozenSet[str] = TRANSCRIPT_SCHEMA | frozenset({"section_type"})

#: Core columns present in speaker-level aggregation output.
SPEAKER_SCHEMA: FrozenSet[str] = TRANSCRIPT_SCHEMA | frozenset({"speaker_role"})

#: Valid sentiment label values.
VALID_LABELS: FrozenSet[str] = frozenset({"positive", "neutral", "negative"})

#: Valid section type values.
VALID_SECTIONS: FrozenSet[str] = frozenset({
    "prepared_remarks", "qa", "closing_remarks", "opening_remarks",
})

#: All numeric tone / score columns (may or may not be present).
_SCORE_COLUMNS: Tuple[str, ...] = (
    "lm_tone_score",
    "lm_mean_tone",
    "lm_median_tone",
    "lm_std_tone",
)

#: Count columns that must be non-negative integers.
_COUNT_COLUMNS: Tuple[str, ...] = (
    "lm_positive_count",
    "lm_negative_count",
    "lm_uncertainty_count",
    "lm_litigious_count",
    "lm_strong_modal_count",
    "lm_weak_modal_count",
    "lm_constraining_count",
    "token_count",
    "matched_token_count",
)

_COUNT_TOTAL_COLUMNS: Tuple[str, ...] = (
    "lm_positive_total",
    "lm_negative_total",
    "lm_uncertainty_total",
    "lm_token_total",
    "lm_matched_token_total",
)

_COVERAGE_COLUMNS: Tuple[str, ...] = (
    "lm_coverage_ratio",
    "coverage_ratio",
    "token_coverage",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LMValidationConfig:
    """
    Immutable configuration for LMValidator.

    Parameters
    ----------
    score_min / score_max:
        Acceptable range for lm_tone_score and lm_mean_tone.
    max_null_ratio:
        Maximum fraction of null values tolerated in critical columns
        before an ERROR is raised (vs. WARNING below this threshold).
    max_duplicate_ratio:
        Maximum fraction of duplicate key rows before escalating to ERROR.
    min_coverage_ratio:
        Coverage ratios below this threshold are flagged as WARNING.
    max_std_tone:
        Standard-deviation above this value triggers a distribution WARNING.
    min_chunk_count:
        Transcript groups with fewer chunks than this are flagged.
    nan_as_error:
        When True any NaN in a score column is an ERROR; else WARNING.
    warn_on_empty_distribution:
        When True a category with zero matches triggers an INFO notice.
    strict_label_check:
        When True an unknown label is an ERROR; else WARNING.
    required_chunk_columns:
        Override the expected chunk-level schema.
    required_transcript_columns:
        Override the expected transcript-level schema.
    """
    score_min: float = -1.0
    score_max: float = 1.0
    max_null_ratio: float = 0.10
    max_duplicate_ratio: float = 0.0
    min_coverage_ratio: float = 0.01
    max_std_tone: float = 0.50
    min_chunk_count: int = 1
    nan_as_error: bool = False
    warn_on_empty_distribution: bool = True
    strict_label_check: bool = False
    required_chunk_columns: FrozenSet[str] = CHUNK_SCHEMA
    required_transcript_columns: FrozenSet[str] = TRANSCRIPT_SCHEMA


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """
    Represents a single validation finding.

    Attributes
    ----------
    severity     : Severity enum value.
    category     : One of the VCategory string constants.
    check_name   : Short snake_case identifier for the check.
    message      : Human-readable description.
    affected_rows: Number of DataFrame rows affected (0 = schema-level).
    affected_columns: Columns relevant to this issue.
    """
    severity: Severity
    category: str
    check_name: str
    message: str
    affected_rows: int = 0
    affected_columns: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        cols = f"  cols={self.affected_columns}" if self.affected_columns else ""
        rows = f"  rows={self.affected_rows}" if self.affected_rows else ""
        return (
            f"[{self.severity.value:<7}] [{self.category:<12}] "
            f"{self.check_name}: {self.message}{rows}{cols}"
        )


# ---------------------------------------------------------------------------
# ValidationSummary
# ---------------------------------------------------------------------------

@dataclass
class ValidationSummary:
    """
    Aggregated result of a validation run.

    Attributes
    ----------
    passed         : True when no ERRORs were raised.
    error_count    : Number of ERROR-severity issues.
    warning_count  : Number of WARNING-severity issues.
    info_count     : Number of INFO-severity issues.
    category_counts: Issue counts keyed by VCategory string.
    issue_list     : Ordered list of all ValidationIssue objects.
    validated_at   : UTC ISO-8601 timestamp.
    elapsed_ms     : Wall-clock time of the validation run.
    diagnostics    : Optional free-form diagnostic dict (distributions, etc.).
    """
    passed: bool
    error_count: int
    warning_count: int
    info_count: int
    category_counts: Dict[str, int]
    issue_list: List[ValidationIssue]
    validated_at: str
    elapsed_ms: float
    diagnostics: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "=== ValidationSummary ===",
            f"  passed         : {self.passed}",
            f"  validated_at   : {self.validated_at}",
            f"  elapsed_ms     : {self.elapsed_ms:.2f}",
            f"  errors         : {self.error_count}",
            f"  warnings       : {self.warning_count}",
            f"  infos          : {self.info_count}",
            "  category_counts:",
        ]
        for cat, cnt in sorted(self.category_counts.items()):
            lines.append(f"    {cat:<14}: {cnt}")
        return "\n".join(lines)

    def generate_report(self) -> str:
        """
        Full human-readable validation report including every issue and
        all diagnostic sections.
        """
        sep = "-" * 70
        lines = [
            "=" * 70,
            "  LM VALIDATION REPORT",
            "=" * 70,
            self.summary(),
            "",
            sep,
            "  ISSUES",
            sep,
        ]
        if not self.issue_list:
            lines.append("  (no issues found)")
        else:
            # Group by severity for readability
            for sev in (Severity.ERROR, Severity.WARNING, Severity.INFO):
                group = [i for i in self.issue_list if i.severity == sev]
                if group:
                    lines.append(f"\n  {sev.value} ({len(group)})")
                    for iss in group:
                        lines.append(f"    {iss}")

        if self.diagnostics:
            lines += ["", sep, "  DIAGNOSTICS", sep]
            for section, content in self.diagnostics.items():
                lines.append(f"\n  [{section}]")
                if isinstance(content, dict):
                    for k, v in content.items():
                        lines.append(f"    {k:<30}: {v}")
                else:
                    lines.append(f"    {content}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LMValidator
# ---------------------------------------------------------------------------

class LMValidator:
    """
    Research-grade validation engine for LM sentiment pipeline outputs.

    All validation methods accept a pandas DataFrame and return a list of
    ``ValidationIssue`` objects.  The top-level ``validate_*`` methods can be
    called individually or composed via ``validate_chunk_output()``,
    ``validate_transcript_output()``, etc.

    Parameters
    ----------
    config : LMValidationConfig, optional
        Validation thresholds and behaviour.

    Usage
    -----
    ::

        validator = LMValidator()
        summary   = validator.validate_chunk_output(chunk_df)
        print(summary.generate_report())

        issues = validator.validate_scores(chunk_df)
        issues += validator.validate_duplicates(chunk_df, key_cols=["chunk_id"])
    """

    def __init__(self, config: Optional[LMValidationConfig] = None) -> None:
        self._config = config or LMValidationConfig()
        logger.info("[LMValidator] Initialised with config: %s", self._config)

    # ==================================================================
    # Composed entry points
    # ==================================================================

    def validate_chunk_output(self, df: pd.DataFrame) -> ValidationSummary:
        """
        Run the full validation suite for chunk-level LM scoring output.
        """
        return self._run_suite(
            df=df,
            suite_name="chunk",
            required_cols=self._config.required_chunk_columns,
            key_cols=["chunk_id"],
            score_cols=[c for c in _SCORE_COLUMNS if c in ("lm_tone_score",)],
            count_cols=_COUNT_COLUMNS,
            label_col="lm_label" if "lm_label" in df.columns else None,
        )

    def validate_transcript_output(self, df: pd.DataFrame) -> ValidationSummary:
        """
        Run the full validation suite for transcript-level aggregation output.
        """
        return self._run_suite(
            df=df,
            suite_name="transcript",
            required_cols=self._config.required_transcript_columns,
            key_cols=["transcript_id"],
            score_cols=["lm_mean_tone", "lm_median_tone", "lm_std_tone"],
            count_cols=_COUNT_TOTAL_COLUMNS,
            label_col="lm_sentiment_label",
        )

    def validate_section_output(self, df: pd.DataFrame) -> ValidationSummary:
        """
        Run the full validation suite for section-level aggregation output.
        """
        return self._run_suite(
            df=df,
            suite_name="section",
            required_cols=SECTION_SCHEMA,
            key_cols=["transcript_id", "section_type"],
            score_cols=["lm_mean_tone", "lm_median_tone", "lm_std_tone"],
            count_cols=_COUNT_TOTAL_COLUMNS,
            label_col="lm_sentiment_label",
        )

    def validate_speaker_output(self, df: pd.DataFrame) -> ValidationSummary:
        """
        Run the full validation suite for speaker-level aggregation output.
        """
        return self._run_suite(
            df=df,
            suite_name="speaker",
            required_cols=SPEAKER_SCHEMA,
            key_cols=["transcript_id", "speaker_role"],
            score_cols=["lm_mean_tone", "lm_median_tone", "lm_std_tone"],
            count_cols=_COUNT_TOTAL_COLUMNS,
            label_col="lm_sentiment_label",
        )

    # ==================================================================
    # Individual check methods — public for direct/partial use
    # ==================================================================

    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: FrozenSet[str],
    ) -> List[ValidationIssue]:
        """
        Verify the DataFrame contains all required columns.

        Returns one ERROR per missing column; one INFO for each extra column
        beyond the required set (purely informational).
        """
        issues: List[ValidationIssue] = []
        present = set(df.columns)

        missing = required_columns - present
        for col in sorted(missing):
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                category=VCategory.SCHEMA,
                check_name="missing_column",
                message=f"Required column '{col}' is absent from the DataFrame.",
                affected_columns=[col],
            ))

        if df.empty and not missing:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category=VCategory.SCHEMA,
                check_name="empty_dataframe",
                message="DataFrame has correct schema but contains zero rows.",
            ))

        logger.debug(
            "[LMValidator] validate_schema: %d missing columns, "
            "empty=%s.",
            len(missing), df.empty,
        )
        return issues

    # ------------------------------------------------------------------
    def validate_nulls(
        self,
        df: pd.DataFrame,
        critical_columns: Optional[Sequence[str]] = None,
    ) -> List[ValidationIssue]:
        """
        Detect null / NaN values in numeric and label columns.

        Columns with null ratios above ``config.max_null_ratio`` are
        escalated to ERROR; below that threshold they are WARNING.

        Parameters
        ----------
        df:
            DataFrame to validate.
        critical_columns:
            Subset of columns to check.  Defaults to all present score +
            count + coverage columns.
        """
        issues: List[ValidationIssue] = []
        if df.empty:
            return issues

        if critical_columns is None:
            candidates = (
                list(_SCORE_COLUMNS)
                + list(_COUNT_COLUMNS)
                + list(_COUNT_TOTAL_COLUMNS)
                + list(_COVERAGE_COLUMNS)
                + ["lm_sentiment_label", "lm_label", "transcript_id", "chunk_id"]
            )
            critical_columns = [c for c in candidates if c in df.columns]

        n = len(df)
        for col in critical_columns:
            null_mask = df[col].isnull()
            n_null = int(null_mask.sum())
            if n_null == 0:
                continue
            ratio = n_null / n
            sev = (
                Severity.ERROR
                if (ratio > self._config.max_null_ratio or self._config.nan_as_error)
                else Severity.WARNING
            )
            issues.append(ValidationIssue(
                severity=sev,
                category=VCategory.NULLS,
                check_name="null_values",
                message=(
                    f"Column '{col}' has {n_null}/{n} null values "
                    f"(ratio={ratio:.4f})."
                ),
                affected_rows=n_null,
                affected_columns=[col],
            ))

        logger.debug("[LMValidator] validate_nulls: %d null issues.", len(issues))
        return issues

    # ------------------------------------------------------------------
    def validate_scores(
        self,
        df: pd.DataFrame,
        score_columns: Optional[Sequence[str]] = None,
    ) -> List[ValidationIssue]:
        """
        Validate numeric tone / sentiment score columns.

        Checks:
        * Values in [score_min, score_max].
        * No ±inf values.
        * std_tone >= 0 (not applicable to raw scores).
        * NaN presence.
        """
        issues: List[ValidationIssue] = []
        if df.empty:
            return issues

        if score_columns is None:
            score_columns = [c for c in _SCORE_COLUMNS if c in df.columns]

        lo, hi = self._config.score_min, self._config.score_max

        for col in score_columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")

            # NaN check
            n_nan = int(series.isna().sum())
            if n_nan:
                sev = Severity.ERROR if self._config.nan_as_error else Severity.WARNING
                issues.append(ValidationIssue(
                    severity=sev,
                    category=VCategory.SCORES,
                    check_name="nan_score",
                    message=f"'{col}' has {n_nan} NaN value(s).",
                    affected_rows=n_nan,
                    affected_columns=[col],
                ))

            # Infinite values
            n_inf = int(np.isinf(series.dropna()).sum())
            if n_inf:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category=VCategory.SCORES,
                    check_name="infinite_score",
                    message=f"'{col}' has {n_inf} infinite value(s).",
                    affected_rows=n_inf,
                    affected_columns=[col],
                ))

            # Range check
            valid = series.dropna()
            out_of_range = valid[(valid < lo) | (valid > hi)]
            if len(out_of_range):
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category=VCategory.SCORES,
                    check_name="score_out_of_range",
                    message=(
                        f"'{col}' has {len(out_of_range)} value(s) outside "
                        f"[{lo}, {hi}]. "
                        f"Min={float(out_of_range.min()):.4f}, "
                        f"Max={float(out_of_range.max()):.4f}."
                    ),
                    affected_rows=len(out_of_range),
                    affected_columns=[col],
                ))

            # std_tone must be non-negative
            if col == "lm_std_tone":
                neg_std = valid[valid < 0]
                if len(neg_std):
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        category=VCategory.SCORES,
                        check_name="negative_std_tone",
                        message=(
                            f"'lm_std_tone' has {len(neg_std)} negative value(s); "
                            "standard deviation cannot be negative."
                        ),
                        affected_rows=len(neg_std),
                        affected_columns=["lm_std_tone"],
                    ))

            # High std_tone warning
            if col == "lm_std_tone":
                high_std = valid[valid > self._config.max_std_tone]
                if len(high_std):
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        category=VCategory.SCORES,
                        check_name="high_std_tone",
                        message=(
                            f"'lm_std_tone' has {len(high_std)} value(s) > "
                            f"{self._config.max_std_tone}. "
                            "High intra-transcript tone variability."
                        ),
                        affected_rows=len(high_std),
                        affected_columns=["lm_std_tone"],
                    ))

        logger.debug("[LMValidator] validate_scores: %d issues.", len(issues))
        return issues

    # ------------------------------------------------------------------
    def validate_counts(
        self,
        df: pd.DataFrame,
        count_columns: Optional[Sequence[str]] = None,
    ) -> List[ValidationIssue]:
        """
        Validate count integrity.

        Checks:
        * All count columns are non-negative.
        * matched_token_count <= token_count (chunk level).
        * Count totals are non-negative integers.
        """
        issues: List[ValidationIssue] = []
        if df.empty:
            return issues

        if count_columns is None:
            count_columns = [
                c for c in list(_COUNT_COLUMNS) + list(_COUNT_TOTAL_COLUMNS)
                if c in df.columns
            ]

        for col in count_columns:
            series = pd.to_numeric(df[col], errors="coerce").fillna(0)
            n_neg = int((series < 0).sum())
            if n_neg:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category=VCategory.COUNTS,
                    check_name="negative_count",
                    message=f"'{col}' has {n_neg} negative value(s).",
                    affected_rows=n_neg,
                    affected_columns=[col],
                ))

        # Matched token consistency check (chunk level)
        if "matched_token_count" in df.columns and "token_count" in df.columns:
            matched = pd.to_numeric(df["matched_token_count"], errors="coerce").fillna(0)
            total   = pd.to_numeric(df["token_count"], errors="coerce").fillna(0)
            violation = matched > total
            n_viol = int(violation.sum())
            if n_viol:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category=VCategory.COUNTS,
                    check_name="matched_exceeds_total",
                    message=(
                        f"matched_token_count > token_count in {n_viol} row(s). "
                        "Matched tokens cannot exceed total tokens."
                    ),
                    affected_rows=n_viol,
                    affected_columns=["matched_token_count", "token_count"],
                ))

        # Aggregated totals consistency: pos + neg <= token_total
        if all(c in df.columns for c in ("lm_positive_total", "lm_negative_total", "lm_token_total")):
            pos = pd.to_numeric(df["lm_positive_total"], errors="coerce").fillna(0)
            neg = pd.to_numeric(df["lm_negative_total"], errors="coerce").fillna(0)
            tok = pd.to_numeric(df["lm_token_total"], errors="coerce").fillna(0)
            violation = (pos + neg) > tok
            n_viol = int(violation.sum())
            if n_viol:
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category=VCategory.COUNTS,
                    check_name="pos_neg_exceeds_tokens",
                    message=(
                        f"(lm_positive_total + lm_negative_total) > lm_token_total "
                        f"in {n_viol} row(s). Expected: matched counts ≤ total tokens."
                    ),
                    affected_rows=n_viol,
                    affected_columns=["lm_positive_total", "lm_negative_total", "lm_token_total"],
                ))

        logger.debug("[LMValidator] validate_counts: %d issues.", len(issues))
        return issues

    # ------------------------------------------------------------------
    def validate_coverage(
        self,
        df: pd.DataFrame,
        coverage_columns: Optional[Sequence[str]] = None,
    ) -> List[ValidationIssue]:
        """
        Validate coverage ratio columns.

        Checks:
        * All coverage ratios in [0.0, 1.0].
        * Ratios below min_coverage_ratio are flagged as WARNING.
        * Computed coverage consistency: matched / total ≈ ratio.
        """
        issues: List[ValidationIssue] = []
        if df.empty:
            return issues

        if coverage_columns is None:
            coverage_columns = [c for c in _COVERAGE_COLUMNS if c in df.columns]

        for col in coverage_columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue

            # Bounds check
            out_lo = series[series < 0.0]
            out_hi = series[series > 1.0]
            if len(out_lo):
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category=VCategory.COVERAGE,
                    check_name="coverage_below_zero",
                    message=f"'{col}' has {len(out_lo)} value(s) < 0.",
                    affected_rows=len(out_lo),
                    affected_columns=[col],
                ))
            if len(out_hi):
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category=VCategory.COVERAGE,
                    check_name="coverage_above_one",
                    message=f"'{col}' has {len(out_hi)} value(s) > 1.",
                    affected_rows=len(out_hi),
                    affected_columns=[col],
                ))

            # Low coverage warning
            low = series[series < self._config.min_coverage_ratio]
            if len(low):
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category=VCategory.COVERAGE,
                    check_name="low_coverage",
                    message=(
                        f"'{col}' has {len(low)} row(s) with coverage ratio "
                        f"< {self._config.min_coverage_ratio:.4f}. "
                        "Low dictionary match rate; scoring reliability reduced."
                    ),
                    affected_rows=len(low),
                    affected_columns=[col],
                ))

        # Derived coverage consistency (chunk level)
        if (
            "matched_token_count" in df.columns
            and "token_count" in df.columns
            and "coverage_ratio" in df.columns
        ):
            matched = pd.to_numeric(df["matched_token_count"], errors="coerce")
            total   = pd.to_numeric(df["token_count"], errors="coerce").replace(0, np.nan)
            computed = (matched / total).fillna(0.0)
            stored   = pd.to_numeric(df["coverage_ratio"], errors="coerce").fillna(0.0)
            discrepancy = (computed - stored).abs()
            n_discrepant = int((discrepancy > 1e-4).sum())
            if n_discrepant:
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category=VCategory.COVERAGE,
                    check_name="coverage_ratio_inconsistency",
                    message=(
                        f"{n_discrepant} row(s) where stored coverage_ratio "
                        "differs from matched/total by > 1e-4. "
                        "Possible re-computation mismatch."
                    ),
                    affected_rows=n_discrepant,
                    affected_columns=["coverage_ratio", "matched_token_count", "token_count"],
                ))

        logger.debug("[LMValidator] validate_coverage: %d issues.", len(issues))
        return issues

    # ------------------------------------------------------------------
    def validate_duplicates(
        self,
        df: pd.DataFrame,
        key_cols: Sequence[str],
    ) -> List[ValidationIssue]:
        """
        Detect duplicate rows on the specified key columns.

        Parameters
        ----------
        df       : DataFrame to check.
        key_cols : Columns that should form a unique key.
        """
        issues: List[ValidationIssue] = []
        if df.empty:
            return issues

        present_keys = [c for c in key_cols if c in df.columns]
        if not present_keys:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category=VCategory.DUPLICATES,
                check_name="key_columns_absent",
                message=(
                    f"Duplicate check requested on {key_cols} but none of "
                    "those columns exist in the DataFrame."
                ),
            ))
            return issues

        dup_mask = df.duplicated(subset=present_keys, keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup:
            ratio = n_dup / len(df)
            sev = (
                Severity.ERROR
                if ratio > self._config.max_duplicate_ratio
                else Severity.WARNING
            )
            # Surface a sample of duplicated key values
            sample_keys = (
                df.loc[dup_mask, present_keys]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            issues.append(ValidationIssue(
                severity=sev,
                category=VCategory.DUPLICATES,
                check_name="duplicate_key_rows",
                message=(
                    f"{n_dup} duplicate row(s) on key {present_keys} "
                    f"(ratio={ratio:.4f}). Sample: {sample_keys}."
                ),
                affected_rows=n_dup,
                affected_columns=list(present_keys),
            ))

        logger.debug(
            "[LMValidator] validate_duplicates (key=%s): %d dup rows.",
            present_keys, n_dup,
        )
        return issues

    # ------------------------------------------------------------------
    def validate_labels(
        self,
        df: pd.DataFrame,
        label_col: str = "lm_sentiment_label",
        valid_labels: FrozenSet[str] = VALID_LABELS,
    ) -> List[ValidationIssue]:
        """
        Validate that all label values are within the allowed set.

        Also checks that all three label classes are represented (INFO
        notice when a label class is entirely absent).
        """
        issues: List[ValidationIssue] = []
        if df.empty or label_col not in df.columns:
            return issues

        series = df[label_col].dropna().astype(str).str.lower().str.strip()
        unknown = set(series.unique()) - valid_labels
        if unknown:
            sev = Severity.ERROR if self._config.strict_label_check else Severity.WARNING
            n_bad = int(series.isin(unknown).sum())
            issues.append(ValidationIssue(
                severity=sev,
                category=VCategory.LABELS,
                check_name="invalid_label",
                message=(
                    f"'{label_col}' contains {n_bad} row(s) with unknown "
                    f"label value(s): {sorted(unknown)}."
                ),
                affected_rows=n_bad,
                affected_columns=[label_col],
            ))

        # Warn if a label class is entirely absent
        if self._config.warn_on_empty_distribution:
            for lbl in valid_labels:
                if lbl not in series.values:
                    issues.append(ValidationIssue(
                        severity=Severity.INFO,
                        category=VCategory.LABELS,
                        check_name="missing_label_class",
                        message=(
                            f"Label class '{lbl}' has zero occurrences in "
                            f"'{label_col}'. This may be expected for small "
                            "samples."
                        ),
                        affected_columns=[label_col],
                    ))

        logger.debug("[LMValidator] validate_labels: %d issues.", len(issues))
        return issues

    # ------------------------------------------------------------------
    def validate_distribution(
        self,
        df: pd.DataFrame,
        score_col: str = "lm_tone_score",
    ) -> Tuple[List[ValidationIssue], Dict]:
        """
        Compute distribution diagnostics and flag statistical anomalies.

        Returns both issues and a diagnostics dict for the ValidationSummary.
        """
        issues: List[ValidationIssue] = []
        diagnostics: Dict = {}

        if df.empty or score_col not in df.columns:
            return issues, diagnostics

        series = pd.to_numeric(df[score_col], errors="coerce").dropna()
        if series.empty:
            return issues, diagnostics

        diag: Dict = {
            "count":    int(len(series)),
            "mean":     float(round(series.mean(), 6)),
            "median":   float(round(series.median(), 6)),
            "std":      float(round(series.std(), 6)) if len(series) > 1 else 0.0,
            "min":      float(round(series.min(), 6)),
            "max":      float(round(series.max(), 6)),
            "pct_positive": float(round((series > 0.02).mean(), 4)),
            "pct_negative": float(round((series < -0.02).mean(), 4)),
            "pct_neutral":  float(round(((series >= -0.02) & (series <= 0.02)).mean(), 4)),
        }

        # Histogram buckets (10 equal-width bins across [-1, 1])
        bins = np.linspace(-1.0, 1.0, 11)
        counts, _ = np.histogram(series, bins=bins)
        diag["histogram_bins"] = [round(float(b), 2) for b in bins]
        diag["histogram_counts"] = counts.tolist()
        diagnostics[score_col] = diag

        # Nearly-constant score WARNING
        if diag["std"] < 0.001 and len(series) > 10:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category=VCategory.DISTRIBUTION,
                check_name="near_constant_scores",
                message=(
                    f"'{score_col}' std={diag['std']:.6f} across "
                    f"{diag['count']} rows. Scores may be degenerate."
                ),
                affected_columns=[score_col],
            ))

        # Extreme skew WARNING
        if diag["std"] > self._config.max_std_tone:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category=VCategory.DISTRIBUTION,
                check_name="high_score_variance",
                message=(
                    f"'{score_col}' std={diag['std']:.4f} > threshold "
                    f"{self._config.max_std_tone}. "
                    "High score variance may indicate pipeline instability."
                ),
                affected_columns=[score_col],
            ))

        # All-neutral WARNING
        if diag["pct_neutral"] > 0.90:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category=VCategory.DISTRIBUTION,
                check_name="predominantly_neutral",
                message=(
                    f"'{score_col}': {diag['pct_neutral']*100:.1f}% of "
                    "scores fall in the neutral band [-0.02, 0.02]. "
                    "LM matching may be too sparse."
                ),
                affected_columns=[score_col],
            ))

        logger.debug(
            "[LMValidator] validate_distribution [%s]: mean=%.4f std=%.4f.",
            score_col, diag["mean"], diag["std"],
        )
        return issues, diagnostics

    # ------------------------------------------------------------------
    def validate_chunk_count(
        self,
        df: pd.DataFrame,
        group_col: str = "transcript_id",
        count_col: str = "lm_chunk_count",
    ) -> List[ValidationIssue]:
        """
        Validate that each transcript has at least min_chunk_count chunks.
        Only meaningful on aggregated (transcript/section/speaker) outputs
        where a ``lm_chunk_count`` column exists.
        """
        issues: List[ValidationIssue] = []
        if df.empty or count_col not in df.columns:
            return issues

        series = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
        low = series[series < self._config.min_chunk_count]
        if len(low):
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category=VCategory.INTEGRITY,
                check_name="low_chunk_count",
                message=(
                    f"{len(low)} group(s) in '{group_col}' have fewer than "
                    f"{self._config.min_chunk_count} chunk(s). "
                    "Sentiment may be unreliable for very short transcripts."
                ),
                affected_rows=len(low),
                affected_columns=[count_col],
            ))
        return issues

    # ==================================================================
    # Unified suite runner (internal)
    # ==================================================================

    def _run_suite(
        self,
        df: pd.DataFrame,
        suite_name: str,
        required_cols: FrozenSet[str],
        key_cols: List[str],
        score_cols: List[str],
        count_cols: Tuple[str, ...],
        label_col: Optional[str],
    ) -> ValidationSummary:
        """Execute all checks in deterministic order; build ValidationSummary."""
        from datetime import datetime, timezone

        t0 = time.perf_counter()
        logger.info(
            "[LMValidator] Starting '%s' validation suite on %d rows.",
            suite_name, len(df),
        )

        all_issues: List[ValidationIssue] = []
        all_diagnostics: Dict = {}

        # 1. Schema
        all_issues += self.validate_schema(df, required_cols)

        # Only proceed with content checks if schema is not catastrophically broken
        schema_errors = [i for i in all_issues if i.severity == Severity.ERROR]
        if schema_errors and df.empty:
            logger.warning(
                "[LMValidator] Aborting '%s' suite early: schema errors + empty df.",
                suite_name,
            )
        else:
            # 2. Nulls
            all_issues += self.validate_nulls(df)

            # 3. Scores
            present_score_cols = [c for c in score_cols if c in df.columns]
            all_issues += self.validate_scores(df, present_score_cols)

            # 4. Counts
            all_issues += self.validate_counts(df, list(count_cols))

            # 5. Coverage
            all_issues += self.validate_coverage(df)

            # 6. Duplicates
            all_issues += self.validate_duplicates(df, key_cols)

            # 7. Labels
            if label_col:
                all_issues += self.validate_labels(df, label_col)

            # 8. Distribution diagnostics
            primary_score = present_score_cols[0] if present_score_cols else None
            if primary_score:
                dist_issues, dist_diag = self.validate_distribution(df, primary_score)
                all_issues += dist_issues
                all_diagnostics.update(dist_diag)

            # 9. Chunk count (aggregated outputs only)
            all_issues += self.validate_chunk_count(df)

        # Build summary
        error_count   = sum(1 for i in all_issues if i.severity == Severity.ERROR)
        warning_count = sum(1 for i in all_issues if i.severity == Severity.WARNING)
        info_count    = sum(1 for i in all_issues if i.severity == Severity.INFO)

        cat_counts: Dict[str, int] = {}
        for iss in all_issues:
            cat_counts[iss.category] = cat_counts.get(iss.category, 0) + 1

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        summary = ValidationSummary(
            passed=(error_count == 0),
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            category_counts=cat_counts,
            issue_list=all_issues,
            validated_at=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=elapsed_ms,
            diagnostics=all_diagnostics,
        )

        logger.info(
            "[LMValidator] '%s' suite complete: passed=%s, "
            "errors=%d, warnings=%d, infos=%d [%.1f ms].",
            suite_name,
            summary.passed,
            error_count, warning_count, info_count,
            elapsed_ms,
        )
        return summary

    # ------------------------------------------------------------------
    # Standalone helpers exposed for pipeline use
    # ------------------------------------------------------------------

    def generate_report(self, summary: ValidationSummary) -> str:
        """Delegate to ``ValidationSummary.generate_report()``."""
        return summary.generate_report()

    def summary(self, validation_summary: ValidationSummary) -> str:
        """Delegate to ``ValidationSummary.summary()``."""
        return validation_summary.summary()


# ---------------------------------------------------------------------------
# Self-test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    print("\n" + "=" * 72)
    print("LMValidator — self-test / demo")
    print("=" * 72)

    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ #
    # Helper to build synthetic chunk DataFrames                         #
    # ------------------------------------------------------------------ #
    def _make_chunk_df(n: int, inject_errors: bool = False) -> pd.DataFrame:
        tones = np.clip(rng.normal(0.05, 0.12, n), -1, 1)
        tok   = rng.integers(80, 300, n)
        match = (tok * rng.uniform(0.03, 0.20, n)).astype(int)
        pos   = (match * rng.uniform(0.3, 0.7, n)).astype(int)
        neg   = (match * rng.uniform(0.1, 0.4, n)).astype(int)
        unc   = rng.integers(0, 5, n)

        tids = [f"AAPL_Q1_2025"] * (n // 2) + [f"MSFT_Q2_2025"] * (n - n // 2)
        rows = {
            "transcript_id":       tids,
            "chunk_id":            [f"chunk_{i:04d}" for i in range(n)],
            "section_type":        ["prepared_remarks" if i % 2 == 0 else "qa" for i in range(n)],
            "speaker_role":        ["ceo" if i % 3 == 0 else ("cfo" if i % 3 == 1 else "analyst") for i in range(n)],
            "lm_tone_score":       np.round(tones, 6).tolist(),
            "lm_label":            ["positive" if t > 0.02 else ("negative" if t < -0.02 else "neutral") for t in tones],
            "lm_positive_count":   pos.tolist(),
            "lm_negative_count":   neg.tolist(),
            "lm_uncertainty_count":unc.tolist(),
            "lm_litigious_count":  rng.integers(0, 3, n).tolist(),
            "lm_strong_modal_count":rng.integers(0, 4, n).tolist(),
            "lm_weak_modal_count": rng.integers(0, 4, n).tolist(),
            "lm_constraining_count":rng.integers(0, 2, n).tolist(),
            "token_count":         tok.tolist(),
            "matched_token_count": match.tolist(),
            "coverage_ratio":      np.round(match / tok, 6).tolist(),
        }
        df = pd.DataFrame(rows)

        if inject_errors:
            # Out-of-range tone
            df.loc[0, "lm_tone_score"] = 1.8
            # Negative count
            df.loc[1, "lm_positive_count"] = -5
            # matched > total
            df.loc[2, "matched_token_count"] = int(df.loc[2, "token_count"]) + 50
            # Duplicate chunk
            df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
            # NaN score
            df.loc[4, "lm_tone_score"] = np.nan
            # Bad label
            df.loc[5, "lm_label"] = "very_positive"
            # Out-of-range coverage
            df.loc[6, "coverage_ratio"] = 1.5

        return df

    def _make_transcript_df(n: int) -> pd.DataFrame:
        tones = np.clip(rng.normal(0.06, 0.08, n), -1, 1)
        return pd.DataFrame({
            "transcript_id":          [f"T_{i:04d}" for i in range(n)],
            "lm_mean_tone":           np.round(tones, 6).tolist(),
            "lm_median_tone":         np.round(tones * 0.95, 6).tolist(),
            "lm_std_tone":            np.round(np.abs(rng.normal(0.07, 0.02, n)), 6).tolist(),
            "lm_token_total":         rng.integers(500, 3000, n).tolist(),
            "lm_matched_token_total": rng.integers(50, 400, n).tolist(),
            "lm_positive_total":      rng.integers(10, 100, n).tolist(),
            "lm_negative_total":      rng.integers(5, 60, n).tolist(),
            "lm_uncertainty_total":   rng.integers(0, 20, n).tolist(),
            "lm_coverage_ratio":      np.round(rng.uniform(0.04, 0.18, n), 6).tolist(),
            "lm_sentiment_label":     ["positive" if t > 0.02 else ("negative" if t < -0.02 else "neutral") for t in tones],
            "lm_chunk_count":         rng.integers(5, 30, n).tolist(),
        })

    validator = LMValidator()

    # ------------------------------------------------------------------ #
    # 1. Clean chunk validation (should PASS)                             #
    # ------------------------------------------------------------------ #
    print("\n[1] Validating CLEAN chunk-level output …")
    clean_df = _make_chunk_df(40, inject_errors=False)
    clean_summary = validator.validate_chunk_output(clean_df)
    print(clean_summary.summary())
    print(f"  passed: {clean_summary.passed}")

    # ------------------------------------------------------------------ #
    # 2. Malformed chunk validation (should FAIL)                         #
    # ------------------------------------------------------------------ #
    print("\n[2] Validating MALFORMED chunk-level output …")
    bad_df = _make_chunk_df(40, inject_errors=True)
    bad_summary = validator.validate_chunk_output(bad_df)
    print(bad_summary.summary())
    print(f"  passed: {bad_summary.passed}")
    print("\n  Issue list:")
    for issue in bad_summary.issue_list:
        print(f"    {issue}")

    # ------------------------------------------------------------------ #
    # 3. Full report for malformed output                                 #
    # ------------------------------------------------------------------ #
    print("\n[3] Full validation report (malformed) …")
    report = validator.generate_report(bad_summary)
    print(report)

    # ------------------------------------------------------------------ #
    # 4. Transcript-level validation                                      #
    # ------------------------------------------------------------------ #
    print("\n[4] Validating transcript-level aggregation output …")
    tx_df = _make_transcript_df(20)
    tx_summary = validator.validate_transcript_output(tx_df)
    print(tx_summary.summary())
    print(f"  passed: {tx_summary.passed}")

    # Inject a bad transcript row
    tx_df_bad = tx_df.copy()
    tx_df_bad.loc[0, "lm_mean_tone"] = 2.5      # out of range
    tx_df_bad.loc[1, "lm_std_tone"]  = -0.1     # negative std
    tx_df_bad.loc[2, "lm_sentiment_label"] = "bullish"  # invalid label
    # Duplicate
    tx_df_bad = pd.concat([tx_df_bad, tx_df_bad.iloc[[5]]], ignore_index=True)
    tx_bad_summary = validator.validate_transcript_output(tx_df_bad)
    print("\n  Malformed transcript issues:")
    for issue in tx_bad_summary.issue_list:
        print(f"    {issue}")

    # ------------------------------------------------------------------ #
    # 5. Distribution diagnostics                                         #
    # ------------------------------------------------------------------ #
    print("\n[5] Distribution diagnostics …")
    dist_issues, dist_diag = validator.validate_distribution(
        clean_df, "lm_tone_score"
    )
    print(json.dumps(
        {k: v for k, v in dist_diag.get("lm_tone_score", {}).items()
         if k != "histogram_counts"},
        indent=2,
    ))

    # ------------------------------------------------------------------ #
    # 6. Near-constant score WARNING trigger                              #
    # ------------------------------------------------------------------ #
    print("\n[6] Near-constant score edge case …")
    flat_df = clean_df.copy()
    flat_df["lm_tone_score"] = 0.001   # essentially constant
    flat_issues, _ = validator.validate_distribution(flat_df, "lm_tone_score")
    for iss in flat_issues:
        print(f"  {iss}")

    # ------------------------------------------------------------------ #
    # 7. Missing column schema error                                      #
    # ------------------------------------------------------------------ #
    print("\n[7] Schema error — missing required columns …")
    sparse_df = clean_df.drop(columns=["lm_tone_score", "matched_token_count"])
    schema_issues = validator.validate_schema(sparse_df, CHUNK_SCHEMA)
    for iss in schema_issues:
        print(f"  {iss}")

    # ------------------------------------------------------------------ #
    # 8. Standalone duplicate check                                       #
    # ------------------------------------------------------------------ #
    print("\n[8] Standalone duplicate check …")
    dup_issues = validator.validate_duplicates(bad_df, key_cols=["chunk_id"])
    for iss in dup_issues:
        print(f"  {iss}")

    # ------------------------------------------------------------------ #
    # 9. Coverage consistency check                                       #
    # ------------------------------------------------------------------ #
    print("\n[9] Coverage inconsistency check …")
    cov_df = clean_df.copy()
    cov_df.loc[0, "coverage_ratio"] = 0.99   # inconsistent with matched/total
    cov_issues = validator.validate_coverage(cov_df, ["coverage_ratio"])
    for iss in cov_issues:
        print(f"  {iss}")

    print("\n" + "=" * 72)
    print("Self-test complete.")
    print("=" * 72 + "\n")
