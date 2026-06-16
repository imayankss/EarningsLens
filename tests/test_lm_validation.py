"""
tests/test_lm_validation.py
===========================
Comprehensive unit test suite for lm_validation.py.

Coverage
--------
1.  LMValidationConfig     — construction, defaults, immutability
2.  ValidationIssue        — fields, severity ordering, string repr
3.  ValidationSummary      — aggregation, summary(), generate_report()
4.  LMValidator.validate_schema       — missing/extra columns
5.  LMValidator.validate_nulls        — null ratios, escalation
6.  LMValidator.validate_scores       — range, inf, NaN, std checks
7.  LMValidator.validate_counts       — negatives, matched > total
8.  LMValidator.validate_coverage     — bounds, low-coverage flags, consistency
9.  LMValidator.validate_duplicates   — key-column duplicates, ratio escalation
10. LMValidator.validate_labels       — unknown labels, missing classes
11. LMValidator.validate_distribution — histogram, near-constant, all-neutral
12. LMValidator.validate_chunk_count  — min chunk threshold
13. Composed suite runners            — validate_chunk/transcript/section/speaker_output
14. Severity escalation               — configurable thresholds respected
15. Determinism                       — repeated calls produce identical issue lists
16. Edge cases                        — empty df, all-null df, infinite values

Design principles
-----------------
* All fixtures are pure functions returning fresh DataFrames — no shared state.
* Expected issue counts use >= guards where implementation may split a single
  finding into multiple issues; exact counts are used only when the test is
  tightly coupled to a specific single check.
* Floating-point comparisons that appear in diagnostic dicts use pytest.approx.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
try:
    from src.sentiment.lm_validation import (
        LMValidationConfig,
        ValidationIssue,
        ValidationSummary,
        LMValidator,
        Severity,
        VCategory,
        CHUNK_SCHEMA,
        TRANSCRIPT_SCHEMA,
        SECTION_SCHEMA,
        SPEAKER_SCHEMA,
        VALID_LABELS,
    )
except ImportError:
    from lm_validation import (  # type: ignore[no-redef]
        LMValidationConfig,
        ValidationIssue,
        ValidationSummary,
        LMValidator,
        Severity,
        VCategory,
        CHUNK_SCHEMA,
        TRANSCRIPT_SCHEMA,
        SECTION_SCHEMA,
        SPEAKER_SCHEMA,
        VALID_LABELS,
    )


# ===========================================================================
# ── Fixtures ─────────────────────────────────────────────────────────────────
# ===========================================================================

RNG = np.random.default_rng(42)


def _chunk_df(
    n: int = 40,
    inject_score_error: bool = False,
    inject_negative_count: bool = False,
    inject_matched_exceeds: bool = False,
    inject_duplicate_chunk: bool = False,
    inject_null_score: bool = False,
    inject_inf_score: bool = False,
    inject_bad_label: bool = False,
    inject_low_coverage: bool = False,
    n_transcripts: int = 2,
) -> pd.DataFrame:
    """
    Build a synthetic chunk-level LM scoring DataFrame.
    Pass inject_* flags to introduce specific fault conditions.
    """
    tones = np.clip(RNG.normal(0.05, 0.08, n), -1, 1)
    tok   = RNG.integers(80, 250, n)
    match = np.clip((tok * RNG.uniform(0.05, 0.20, n)).astype(int), 1, tok)
    pos   = np.clip((match * RNG.uniform(0.3, 0.7, n)).astype(int), 0, match)
    neg   = np.clip((match * RNG.uniform(0.1, 0.4, n)).astype(int), 0, match - pos)
    unc   = RNG.integers(0, 5, n)

    tid_cycle = [f"TX_{i % n_transcripts:04d}" for i in range(n)]

    df = pd.DataFrame({
        "transcript_id":       tid_cycle,
        "chunk_id":            [f"CK_{i:06d}" for i in range(n)],
        "section_type":        ["prepared_remarks" if i % 2 == 0 else "qa" for i in range(n)],
        "speaker_role":        ["ceo" if i % 3 == 0 else ("cfo" if i % 3 == 1 else "analyst") for i in range(n)],
        "lm_tone_score":       np.round(tones, 6).tolist(),
        "lm_label":            ["positive" if t > 0.02 else ("negative" if t < -0.02 else "neutral") for t in tones],
        "lm_positive_count":   pos.tolist(),
        "lm_negative_count":   neg.tolist(),
        "lm_uncertainty_count": unc.tolist(),
        "lm_litigious_count":  RNG.integers(0, 3, n).tolist(),
        "lm_strong_modal_count": RNG.integers(0, 4, n).tolist(),
        "lm_weak_modal_count": RNG.integers(0, 4, n).tolist(),
        "lm_constraining_count": RNG.integers(0, 2, n).tolist(),
        "token_count":         tok.tolist(),
        "matched_token_count": match.tolist(),
        "coverage_ratio":      np.round(match / tok, 6).tolist(),
    })

    if inject_score_error:
        df.loc[0, "lm_tone_score"] = 2.5
        df.loc[1, "lm_tone_score"] = -1.8

    if inject_negative_count:
        df.loc[2, "lm_positive_count"] = -10
        df.loc[3, "lm_negative_count"] = -5

    if inject_matched_exceeds:
        df.loc[4, "matched_token_count"] = int(df.loc[4, "token_count"]) + 100

    if inject_duplicate_chunk:
        dup = df.iloc[[5]].copy()
        df = pd.concat([df, dup], ignore_index=True)

    if inject_null_score:
        df.loc[6, "lm_tone_score"] = np.nan
        df.loc[7, "lm_tone_score"] = np.nan

    if inject_inf_score:
        df.loc[8, "lm_tone_score"] = float("inf")
        df.loc[9, "lm_tone_score"] = float("-inf")

    if inject_bad_label:
        df.loc[10, "lm_label"] = "very_positive"
        df.loc[11, "lm_label"] = "bullish"

    if inject_low_coverage:
        df.loc[12, "coverage_ratio"] = 0.0
        df.loc[12, "matched_token_count"] = 0

    return df.reset_index(drop=True)


def _transcript_df(
    n: int = 20,
    inject_score_error: bool = False,
    inject_negative_std: bool = False,
    inject_bad_label: bool = False,
    inject_duplicate: bool = False,
    inject_null: bool = False,
) -> pd.DataFrame:
    tones = np.clip(RNG.normal(0.05, 0.07, n), -1, 1)
    df = pd.DataFrame({
        "transcript_id":          [f"TX_{i:04d}" for i in range(n)],
        "lm_mean_tone":           np.round(tones, 6).tolist(),
        "lm_median_tone":         np.round(tones * 0.95, 6).tolist(),
        "lm_std_tone":            np.round(np.abs(RNG.normal(0.06, 0.02, n)), 6).tolist(),
        "lm_token_total":         RNG.integers(400, 3000, n).tolist(),
        "lm_matched_token_total": RNG.integers(40, 400, n).tolist(),
        "lm_positive_total":      RNG.integers(10, 120, n).tolist(),
        "lm_negative_total":      RNG.integers(5, 80, n).tolist(),
        "lm_uncertainty_total":   RNG.integers(0, 30, n).tolist(),
        "lm_coverage_ratio":      np.round(RNG.uniform(0.05, 0.18, n), 6).tolist(),
        "lm_sentiment_label":     ["positive" if t > 0.02 else ("negative" if t < -0.02 else "neutral") for t in tones],
        "lm_chunk_count":         RNG.integers(5, 30, n).tolist(),
    })

    if inject_score_error:
        df.loc[0, "lm_mean_tone"] = 3.0
        df.loc[1, "lm_mean_tone"] = -2.5

    if inject_negative_std:
        df.loc[2, "lm_std_tone"] = -0.05

    if inject_bad_label:
        df.loc[3, "lm_sentiment_label"] = "extreme_positive"

    if inject_duplicate:
        dup = df.iloc[[4]].copy()
        df = pd.concat([df, dup], ignore_index=True)

    if inject_null:
        df.loc[5, "lm_mean_tone"] = np.nan
        df.loc[6, "lm_mean_tone"] = np.nan

    return df.reset_index(drop=True)


@pytest.fixture
def default_validator() -> LMValidator:
    return LMValidator()


@pytest.fixture
def strict_validator() -> LMValidator:
    cfg = LMValidationConfig(
        nan_as_error=True,
        strict_label_check=True,
        max_null_ratio=0.05,
        max_duplicate_ratio=0.0,
    )
    return LMValidator(cfg)


@pytest.fixture
def clean_chunk_df() -> pd.DataFrame:
    return _chunk_df(n=40)


@pytest.fixture
def clean_transcript_df() -> pd.DataFrame:
    return _transcript_df(n=20)


# ===========================================================================
# ── 1. LMValidationConfig ────────────────────────────────────────────────────
# ===========================================================================

class TestLMValidationConfig:

    def test_default_construction(self):
        cfg = LMValidationConfig()
        assert cfg.score_min == pytest.approx(-1.0)
        assert cfg.score_max == pytest.approx(1.0)
        assert 0.0 < cfg.max_null_ratio <= 1.0
        assert cfg.max_duplicate_ratio >= 0.0
        assert cfg.min_coverage_ratio >= 0.0

    def test_immutability(self):
        cfg = LMValidationConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.score_min = -99.0  # type: ignore[misc]

    def test_custom_thresholds_accepted(self):
        cfg = LMValidationConfig(
            score_min=-1.0,
            score_max=1.0,
            max_null_ratio=0.20,
            max_duplicate_ratio=0.05,
            min_coverage_ratio=0.02,
        )
        assert cfg.max_null_ratio == pytest.approx(0.20)
        assert cfg.max_duplicate_ratio == pytest.approx(0.05)

    def test_nan_as_error_flag(self):
        cfg = LMValidationConfig(nan_as_error=True)
        assert cfg.nan_as_error is True

    def test_strict_label_check_flag(self):
        cfg = LMValidationConfig(strict_label_check=True)
        assert cfg.strict_label_check is True

    def test_boolean_flags_default_values(self):
        cfg = LMValidationConfig()
        assert isinstance(cfg.nan_as_error, bool)
        assert isinstance(cfg.strict_label_check, bool)
        assert isinstance(cfg.warn_on_empty_distribution, bool)


# ===========================================================================
# ── 2. ValidationIssue ───────────────────────────────────────────────────────
# ===========================================================================

class TestValidationIssue:

    def _make(self, sev: Severity = Severity.WARNING) -> ValidationIssue:
        return ValidationIssue(
            severity=sev,
            category=VCategory.SCORES,
            check_name="test_check",
            message="Test message.",
            affected_rows=5,
            affected_columns=["lm_tone_score"],
        )

    def test_required_fields_present(self):
        issue = self._make()
        assert hasattr(issue, "severity")
        assert hasattr(issue, "category")
        assert hasattr(issue, "check_name")
        assert hasattr(issue, "message")
        assert hasattr(issue, "affected_rows")
        assert hasattr(issue, "affected_columns")

    def test_severity_enum_values(self):
        for sev in (Severity.INFO, Severity.WARNING, Severity.ERROR):
            issue = self._make(sev=sev)
            assert issue.severity == sev

    def test_str_representation_contains_severity(self):
        issue = self._make(Severity.ERROR)
        s = str(issue)
        assert "ERROR" in s

    def test_str_representation_contains_message(self):
        issue = self._make()
        s = str(issue)
        assert "Test message" in s

    def test_str_representation_contains_check_name(self):
        issue = self._make()
        s = str(issue)
        assert "test_check" in s

    def test_affected_rows_zero_for_schema_issue(self):
        issue = ValidationIssue(
            severity=Severity.ERROR,
            category=VCategory.SCHEMA,
            check_name="missing_column",
            message="Column X absent.",
        )
        assert issue.affected_rows == 0

    def test_severity_ordering(self):
        assert Severity.INFO < Severity.WARNING
        assert Severity.WARNING < Severity.ERROR
        assert not (Severity.ERROR < Severity.WARNING)


# ===========================================================================
# ── 3. ValidationSummary ─────────────────────────────────────────────────────
# ===========================================================================

class TestValidationSummary:

    def _make_summary(
        self,
        errors: int = 2,
        warnings: int = 3,
        infos: int = 1,
    ) -> ValidationSummary:
        issues: List[ValidationIssue] = []
        for _ in range(errors):
            issues.append(ValidationIssue(
                severity=Severity.ERROR, category=VCategory.SCORES,
                check_name="score_range", message="Out of range.",
            ))
        for _ in range(warnings):
            issues.append(ValidationIssue(
                severity=Severity.WARNING, category=VCategory.COVERAGE,
                check_name="low_coverage", message="Low coverage.",
            ))
        for _ in range(infos):
            issues.append(ValidationIssue(
                severity=Severity.INFO, category=VCategory.LABELS,
                check_name="missing_class", message="Missing label class.",
            ))
        from datetime import datetime, timezone
        return ValidationSummary(
            passed=(errors == 0),
            error_count=errors,
            warning_count=warnings,
            info_count=infos,
            category_counts={
                VCategory.SCORES: errors,
                VCategory.COVERAGE: warnings,
                VCategory.LABELS: infos,
            },
            issue_list=issues,
            validated_at=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=12.5,
            diagnostics={},
        )

    def test_passed_false_when_errors_present(self):
        s = self._make_summary(errors=1, warnings=0, infos=0)
        assert s.passed is False

    def test_passed_true_when_no_errors(self):
        s = self._make_summary(errors=0, warnings=2, infos=1)
        assert s.passed is True

    def test_error_count_matches(self):
        s = self._make_summary(errors=3)
        assert s.error_count == 3

    def test_warning_count_matches(self):
        s = self._make_summary(warnings=4)
        assert s.warning_count == 4

    def test_info_count_matches(self):
        s = self._make_summary(infos=2)
        assert s.info_count == 2

    def test_issue_list_length(self):
        s = self._make_summary(errors=2, warnings=3, infos=1)
        assert len(s.issue_list) == 6

    def test_summary_str_contains_counts(self):
        s = self._make_summary(errors=2, warnings=3, infos=1)
        text = s.summary()
        assert "2" in text
        assert "3" in text

    def test_summary_str_contains_passed(self):
        s = self._make_summary(errors=0)
        text = s.summary()
        assert "True" in text or "passed" in text.lower()

    def test_generate_report_contains_all_severities(self):
        s = self._make_summary(errors=1, warnings=1, infos=1)
        report = s.generate_report()
        assert "ERROR" in report
        assert "WARNING" in report
        assert "INFO" in report

    def test_generate_report_is_string(self):
        s = self._make_summary()
        assert isinstance(s.generate_report(), str)

    def test_category_counts_populated(self):
        s = self._make_summary(errors=2, warnings=3)
        assert s.category_counts.get(VCategory.SCORES, 0) == 2
        assert s.category_counts.get(VCategory.COVERAGE, 0) == 3

    def test_elapsed_ms_present(self):
        s = self._make_summary()
        assert s.elapsed_ms == pytest.approx(12.5)

    def test_zero_issues_report(self):
        s = self._make_summary(errors=0, warnings=0, infos=0)
        report = s.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0


# ===========================================================================
# ── 4. validate_schema ───────────────────────────────────────────────────────
# ===========================================================================

class TestValidateSchema:

    def test_valid_chunk_schema_no_issues(self, default_validator, clean_chunk_df):
        issues = default_validator.validate_schema(clean_chunk_df, CHUNK_SCHEMA)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_missing_single_column_is_error(self, default_validator, clean_chunk_df):
        df = clean_chunk_df.drop(columns=["lm_tone_score"])
        issues = default_validator.validate_schema(df, CHUNK_SCHEMA)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert any("lm_tone_score" in i.message for i in errors)

    def test_missing_multiple_columns_each_reported(self, default_validator, clean_chunk_df):
        df = clean_chunk_df.drop(columns=["lm_tone_score", "token_count"])
        issues = default_validator.validate_schema(df, CHUNK_SCHEMA)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 2
        missing_cols = {c for i in errors for c in i.affected_columns}
        assert "lm_tone_score" in missing_cols
        assert "token_count" in missing_cols

    def test_empty_dataframe_with_correct_schema(self, default_validator, clean_chunk_df):
        empty = clean_chunk_df.iloc[0:0].copy()
        issues = default_validator.validate_schema(empty, CHUNK_SCHEMA)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_completely_empty_schema_raises_errors(self, default_validator):
        df = pd.DataFrame()
        issues = default_validator.validate_schema(df, CHUNK_SCHEMA)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1

    def test_transcript_schema_validation(self, default_validator, clean_transcript_df):
        issues = default_validator.validate_schema(clean_transcript_df, TRANSCRIPT_SCHEMA)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_extra_columns_do_not_cause_errors(self, default_validator, clean_chunk_df):
        df = clean_chunk_df.copy()
        df["extra_col"] = 42
        issues = default_validator.validate_schema(df, CHUNK_SCHEMA)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0


# ===========================================================================
# ── 5. validate_nulls ────────────────────────────────────────────────────────
# ===========================================================================

class TestValidateNulls:

    def test_no_nulls_no_issues(self, default_validator, clean_chunk_df):
        issues = default_validator.validate_nulls(clean_chunk_df)
        assert len(issues) == 0

    def test_null_in_score_column_detected(self, default_validator):
        df = _chunk_df(n=20)
        df.loc[0, "lm_tone_score"] = np.nan
        issues = default_validator.validate_nulls(df, critical_columns=["lm_tone_score"])
        assert len(issues) >= 1
        assert any("lm_tone_score" in i.message for i in issues)

    def test_null_ratio_below_threshold_is_warning(self, default_validator):
        df = _chunk_df(n=100)
        # 5 nulls out of 100 = 5% < default max_null_ratio (10%)
        for i in range(5):
            df.loc[i, "lm_tone_score"] = np.nan
        issues = default_validator.validate_nulls(df, critical_columns=["lm_tone_score"])
        assert any(i.severity == Severity.WARNING for i in issues)
        assert not any(i.severity == Severity.ERROR for i in issues)

    def test_null_ratio_above_threshold_is_error(self):
        cfg = LMValidationConfig(max_null_ratio=0.05)
        validator = LMValidator(cfg)
        df = _chunk_df(n=100)
        # 20 nulls out of 100 = 20% > 5%
        for i in range(20):
            df.loc[i, "lm_tone_score"] = np.nan
        issues = validator.validate_nulls(df, critical_columns=["lm_tone_score"])
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_nan_as_error_escalates_any_nan(self, strict_validator):
        df = _chunk_df(n=20)
        df.loc[0, "lm_tone_score"] = np.nan
        issues = strict_validator.validate_nulls(df, critical_columns=["lm_tone_score"])
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_all_null_column_is_error(self, default_validator):
        df = _chunk_df(n=20)
        df["lm_tone_score"] = np.nan
        issues = default_validator.validate_nulls(df, critical_columns=["lm_tone_score"])
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_affected_rows_count_correct(self, default_validator):
        df = _chunk_df(n=30)
        n_null = 4
        for i in range(n_null):
            df.loc[i, "lm_tone_score"] = np.nan
        issues = default_validator.validate_nulls(df, critical_columns=["lm_tone_score"])
        null_issues = [i for i in issues if "lm_tone_score" in i.message]
        assert len(null_issues) >= 1
        assert any(i.affected_rows == n_null for i in null_issues)

    def test_empty_dataframe_no_null_issues(self, default_validator):
        df = _chunk_df(n=0)
        issues = default_validator.validate_nulls(df)
        assert len(issues) == 0


# ===========================================================================
# ── 6. validate_scores ───────────────────────────────────────────────────────
# ===========================================================================

class TestValidateScores:

    def test_valid_scores_no_issues(self, default_validator, clean_chunk_df):
        issues = default_validator.validate_scores(
            clean_chunk_df, score_columns=["lm_tone_score"]
        )
        assert len(issues) == 0

    def test_score_above_max_is_error(self, default_validator):
        df = _chunk_df(inject_score_error=True)
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert any("range" in i.check_name or "out_of_range" in i.check_name for i in errors)

    def test_score_below_min_is_error(self, default_validator):
        df = _chunk_df()
        df.loc[0, "lm_tone_score"] = -1.5
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1

    def test_infinite_score_is_error(self, default_validator):
        df = _chunk_df(inject_inf_score=True)
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert any("infinite" in i.check_name or "inf" in i.message.lower() for i in errors)

    def test_nan_score_is_warning_by_default(self, default_validator):
        df = _chunk_df(inject_null_score=True)
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        nan_issues = [i for i in issues if "nan" in i.check_name.lower()]
        assert len(nan_issues) >= 1
        assert all(i.severity in (Severity.WARNING, Severity.ERROR) for i in nan_issues)

    def test_nan_score_is_error_when_configured(self, strict_validator):
        df = _chunk_df(inject_null_score=True)
        issues = strict_validator.validate_scores(df, score_columns=["lm_tone_score"])
        nan_issues = [i for i in issues if "nan" in i.check_name.lower()]
        assert any(i.severity == Severity.ERROR for i in nan_issues)

    def test_negative_std_tone_is_error(self, default_validator):
        df = _transcript_df(n=10, inject_negative_std=True)
        issues = default_validator.validate_scores(
            df, score_columns=["lm_mean_tone", "lm_std_tone"]
        )
        std_errors = [
            i for i in issues
            if "std" in i.check_name.lower() or "negative" in i.check_name.lower()
        ]
        assert len(std_errors) >= 1
        assert any(i.severity == Severity.ERROR for i in std_errors)

    def test_high_std_tone_is_warning(self, default_validator):
        df = _transcript_df(n=10)
        df["lm_std_tone"] = 0.80
        issues = default_validator.validate_scores(df, score_columns=["lm_std_tone"])
        high_std = [i for i in issues if "high" in i.check_name.lower() or "std" in i.check_name.lower()]
        assert len(high_std) >= 1
        assert any(i.severity == Severity.WARNING for i in high_std)

    def test_exactly_minus_one_is_valid(self, default_validator):
        df = _chunk_df(n=10)
        df["lm_tone_score"] = -1.0
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        errors = [i for i in issues if i.severity == Severity.ERROR and "range" in i.check_name]
        assert len(errors) == 0

    def test_exactly_plus_one_is_valid(self, default_validator):
        df = _chunk_df(n=10)
        df["lm_tone_score"] = 1.0
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        errors = [i for i in issues if i.severity == Severity.ERROR and "range" in i.check_name]
        assert len(errors) == 0

    def test_affected_rows_reflects_count(self, default_validator):
        df = _chunk_df(n=20)
        df.loc[0, "lm_tone_score"] = 1.5
        df.loc[1, "lm_tone_score"] = -1.5
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        range_issues = [i for i in issues if "range" in i.check_name]
        assert any(i.affected_rows == 2 for i in range_issues)

    def test_empty_df_no_score_issues(self, default_validator):
        df = _chunk_df(n=0)
        issues = default_validator.validate_scores(df)
        assert len(issues) == 0


# ===========================================================================
# ── 7. validate_counts ───────────────────────────────────────────────────────
# ===========================================================================

class TestValidateCounts:

    def test_valid_counts_no_issues(self, default_validator, clean_chunk_df):
        issues = default_validator.validate_counts(clean_chunk_df)
        assert len(issues) == 0

    def test_negative_positive_count_is_error(self, default_validator):
        df = _chunk_df(inject_negative_count=True)
        issues = default_validator.validate_counts(df)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert any("negative" in i.check_name.lower() or "negative" in i.message.lower() for i in errors)

    def test_negative_negative_count_is_error(self, default_validator):
        df = _chunk_df(n=10)
        df.loc[0, "lm_negative_count"] = -3
        issues = default_validator.validate_counts(df)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1

    def test_matched_exceeds_total_is_error(self, default_validator):
        df = _chunk_df(inject_matched_exceeds=True)
        issues = default_validator.validate_counts(df)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert any("exceed" in i.check_name.lower() or "exceed" in i.message.lower() for i in errors)

    def test_zero_counts_are_valid(self, default_validator):
        df = _chunk_df(n=10)
        df["lm_positive_count"] = 0
        df["lm_negative_count"] = 0
        issues = default_validator.validate_counts(df)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_all_count_columns_checked(self, default_validator):
        df = _chunk_df(n=10)
        df.loc[0, "lm_uncertainty_count"] = -1
        issues = default_validator.validate_counts(df)
        assert len(issues) >= 1

    def test_token_count_negative_is_error(self, default_validator):
        df = _chunk_df(n=10)
        df.loc[0, "token_count"] = -50
        issues = default_validator.validate_counts(df)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1

    def test_pos_plus_neg_exceeds_total_is_warning(self, default_validator):
        df = _transcript_df(n=10)
        df.loc[0, "lm_positive_total"] = 9000
        df.loc[0, "lm_negative_total"] = 9000
        df.loc[0, "lm_token_total"] = 100
        issues = default_validator.validate_counts(df)
        warn_or_err = [
            i for i in issues
            if i.severity in (Severity.WARNING, Severity.ERROR)
            and "exceed" in i.check_name.lower()
        ]
        assert len(warn_or_err) >= 1

    def test_empty_df_no_count_issues(self, default_validator):
        df = _chunk_df(n=0)
        issues = default_validator.validate_counts(df)
        assert len(issues) == 0


# ===========================================================================
# ── 8. validate_coverage ─────────────────────────────────────────────────────
# ===========================================================================

class TestValidateCoverage:

    def test_valid_coverage_no_issues(self, default_validator, clean_chunk_df):
        issues = default_validator.validate_coverage(clean_chunk_df)
        assert len(issues) == 0

    def test_coverage_below_zero_is_error(self, default_validator):
        df = _chunk_df(n=10)
        df.loc[0, "coverage_ratio"] = -0.05
        issues = default_validator.validate_coverage(df, coverage_columns=["coverage_ratio"])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert any("below_zero" in i.check_name or "< 0" in i.message for i in errors)

    def test_coverage_above_one_is_error(self, default_validator):
        df = _chunk_df(n=10)
        df.loc[0, "coverage_ratio"] = 1.5
        issues = default_validator.validate_coverage(df, coverage_columns=["coverage_ratio"])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert any("above_one" in i.check_name or "> 1" in i.message for i in errors)

    def test_low_coverage_below_threshold_is_warning(self, default_validator):
        df = _chunk_df(n=20)
        df.loc[0, "coverage_ratio"] = 0.0
        df.loc[1, "coverage_ratio"] = 0.0
        issues = default_validator.validate_coverage(df, coverage_columns=["coverage_ratio"])
        warnings = [i for i in issues if i.severity == Severity.WARNING and "low" in i.check_name.lower()]
        assert len(warnings) >= 1

    def test_coverage_consistency_inconsistency_is_warning(self, default_validator):
        df = _chunk_df(n=10)
        # Deliberately set coverage_ratio inconsistent with matched/total
        df.loc[0, "coverage_ratio"] = 0.99
        df.loc[0, "matched_token_count"] = 1
        df.loc[0, "token_count"] = 100
        issues = default_validator.validate_coverage(df, coverage_columns=["coverage_ratio"])
        consistency_issues = [
            i for i in issues
            if "inconsistency" in i.check_name.lower() or "inconsistent" in i.message.lower()
        ]
        assert len(consistency_issues) >= 1

    def test_coverage_exactly_zero_is_flagged(self, default_validator):
        df = _chunk_df(n=10)
        df["coverage_ratio"] = 0.0
        df["matched_token_count"] = 0
        issues = default_validator.validate_coverage(df, coverage_columns=["coverage_ratio"])
        low_issues = [i for i in issues if "low" in i.check_name.lower()]
        assert len(low_issues) >= 1

    def test_coverage_exactly_one_is_valid(self, default_validator):
        df = _chunk_df(n=10)
        df["coverage_ratio"] = 1.0
        df["matched_token_count"] = df["token_count"]
        issues = default_validator.validate_coverage(df, coverage_columns=["coverage_ratio"])
        errors = [
            i for i in issues
            if i.severity == Severity.ERROR
            and ("below_zero" in i.check_name or "above_one" in i.check_name)
        ]
        assert len(errors) == 0

    def test_lm_coverage_ratio_column_checked(self, default_validator):
        df = _transcript_df(n=10)
        df.loc[0, "lm_coverage_ratio"] = -0.1
        issues = default_validator.validate_coverage(df)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1


# ===========================================================================
# ── 9. validate_duplicates ───────────────────────────────────────────────────
# ===========================================================================

class TestValidateDuplicates:

    def test_no_duplicates_no_issues(self, default_validator, clean_chunk_df):
        issues = default_validator.validate_duplicates(clean_chunk_df, key_cols=["chunk_id"])
        assert len(issues) == 0

    def test_duplicate_chunk_id_detected(self, default_validator):
        df = _chunk_df(inject_duplicate_chunk=True)
        issues = default_validator.validate_duplicates(df, key_cols=["chunk_id"])
        assert len(issues) >= 1
        assert any("duplicate" in i.check_name.lower() for i in issues)

    def test_duplicate_transcript_id_detected(self, default_validator):
        df = _transcript_df(inject_duplicate=True)
        issues = default_validator.validate_duplicates(df, key_cols=["transcript_id"])
        assert len(issues) >= 1

    def test_duplicate_ratio_zero_triggers_error(self, strict_validator):
        """max_duplicate_ratio=0.0 means any duplicate is an ERROR."""
        df = _chunk_df(inject_duplicate_chunk=True)
        issues = strict_validator.validate_duplicates(df, key_cols=["chunk_id"])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1

    def test_duplicate_below_ratio_threshold_is_warning(self):
        """When duplicate_ratio > 0 a small number of dupes is a WARNING."""
        cfg = LMValidationConfig(max_duplicate_ratio=0.50)
        validator = LMValidator(cfg)
        df = _chunk_df(n=100)
        # Duplicate 1 row → ratio = 2/101 ≈ 2% < 50%
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        issues = validator.validate_duplicates(df, key_cols=["chunk_id"])
        assert any(i.severity == Severity.WARNING for i in issues)
        assert not any(i.severity == Severity.ERROR for i in issues)

    def test_missing_key_column_warns(self, default_validator):
        df = _chunk_df(n=10)
        issues = default_validator.validate_duplicates(df, key_cols=["nonexistent_col"])
        assert any(i.severity == Severity.WARNING for i in issues)

    def test_composite_key_duplicate_detected(self, default_validator):
        df = _transcript_df(n=10)
        df_section = df.copy()
        df_section["section_type"] = "prepared_remarks"
        dup = df_section.iloc[[0]].copy()
        df_dup = pd.concat([df_section, dup], ignore_index=True)
        issues = default_validator.validate_duplicates(
            df_dup, key_cols=["transcript_id", "section_type"]
        )
        assert len(issues) >= 1

    def test_affected_rows_reported(self, default_validator):
        df = _chunk_df(n=20)
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        issues = default_validator.validate_duplicates(df, key_cols=["chunk_id"])
        dup_issues = [i for i in issues if "duplicate" in i.check_name.lower()]
        assert any(i.affected_rows >= 2 for i in dup_issues)


# ===========================================================================
# ── 10. validate_labels ──────────────────────────────────────────────────────
# ===========================================================================

class TestValidateLabels:

    def test_valid_labels_no_issues(self, default_validator, clean_chunk_df):
        issues = default_validator.validate_labels(clean_chunk_df, label_col="lm_label")
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_invalid_label_is_warning_by_default(self, default_validator):
        df = _chunk_df(inject_bad_label=True)
        issues = default_validator.validate_labels(df, label_col="lm_label")
        label_issues = [i for i in issues if "invalid" in i.check_name.lower() or "label" in i.check_name.lower()]
        assert len(label_issues) >= 1
        assert any(i.severity == Severity.WARNING for i in label_issues)

    def test_invalid_label_is_error_when_strict(self, strict_validator):
        df = _chunk_df(inject_bad_label=True)
        issues = strict_validator.validate_labels(df, label_col="lm_label")
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1

    def test_missing_label_class_is_info(self, default_validator):
        df = _chunk_df(n=30)
        # Force all labels to "positive" — "negative" and "neutral" absent
        df["lm_label"] = "positive"
        issues = default_validator.validate_labels(df, label_col="lm_label")
        info_issues = [
            i for i in issues
            if i.severity == Severity.INFO and "missing" in i.check_name.lower()
        ]
        assert len(info_issues) >= 1

    def test_all_three_labels_present_no_missing_class_info(self, default_validator):
        df = _chunk_df(n=30)
        df.loc[0, "lm_label"] = "positive"
        df.loc[1, "lm_label"] = "negative"
        df.loc[2, "lm_label"] = "neutral"
        issues = default_validator.validate_labels(df, label_col="lm_label")
        missing = [
            i for i in issues
            if i.severity == Severity.INFO and "missing" in i.check_name.lower()
        ]
        assert len(missing) == 0

    def test_absent_label_column_returns_empty(self, default_validator, clean_chunk_df):
        issues = default_validator.validate_labels(clean_chunk_df, label_col="nonexistent_label_col")
        assert len(issues) == 0

    def test_empty_df_label_validation(self, default_validator):
        df = _chunk_df(n=0)
        issues = default_validator.validate_labels(df, label_col="lm_label")
        assert len(issues) == 0

    def test_all_valid_label_values_accepted(self, default_validator):
        for label in ("positive", "neutral", "negative"):
            df = _chunk_df(n=5)
            df["lm_label"] = label
            issues = default_validator.validate_labels(df, label_col="lm_label")
            invalid = [
                i for i in issues
                if "invalid" in i.check_name.lower()
                and i.severity in (Severity.ERROR, Severity.WARNING)
            ]
            assert len(invalid) == 0, f"Label '{label}' wrongly flagged"


# ===========================================================================
# ── 11. validate_distribution ────────────────────────────────────────────────
# ===========================================================================

class TestValidateDistribution:

    def test_valid_distribution_returns_diagnostics(self, default_validator, clean_chunk_df):
        issues, diagnostics = default_validator.validate_distribution(
            clean_chunk_df, score_col="lm_tone_score"
        )
        assert "lm_tone_score" in diagnostics
        diag = diagnostics["lm_tone_score"]
        assert "mean" in diag
        assert "std" in diag
        assert "count" in diag
        assert "histogram_counts" in diag

    def test_near_constant_scores_warning(self, default_validator):
        df = _chunk_df(n=40)
        df["lm_tone_score"] = 0.0001   # near-constant
        issues, _ = default_validator.validate_distribution(df, score_col="lm_tone_score")
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert len(warnings) >= 1
        assert any("constant" in i.check_name.lower() for i in warnings)

    def test_high_variance_warning(self, default_validator):
        df = _chunk_df(n=40)
        # Alternate extreme values for high std
        df["lm_tone_score"] = [1.0 if i % 2 == 0 else -1.0 for i in range(len(df))]
        issues, _ = default_validator.validate_distribution(df, score_col="lm_tone_score")
        high_var = [i for i in issues if "variance" in i.check_name.lower() or "std" in i.check_name.lower()]
        assert len(high_var) >= 1

    def test_all_neutral_scores_warning(self, default_validator):
        df = _chunk_df(n=40)
        df["lm_tone_score"] = 0.005   # all in [-0.02, 0.02] neutral band
        issues, _ = default_validator.validate_distribution(df, score_col="lm_tone_score")
        neutral_warns = [
            i for i in issues
            if "neutral" in i.check_name.lower() or "neutral" in i.message.lower()
        ]
        assert len(neutral_warns) >= 1

    def test_histogram_bins_correct_length(self, default_validator, clean_chunk_df):
        _, diagnostics = default_validator.validate_distribution(
            clean_chunk_df, score_col="lm_tone_score"
        )
        diag = diagnostics.get("lm_tone_score", {})
        bins = diag.get("histogram_bins", [])
        counts = diag.get("histogram_counts", [])
        assert len(bins) == len(counts) + 1

    def test_histogram_counts_sum_to_n(self, default_validator):
        df = _chunk_df(n=50)
        _, diagnostics = default_validator.validate_distribution(df, score_col="lm_tone_score")
        diag = diagnostics.get("lm_tone_score", {})
        counts = diag.get("histogram_counts", [])
        assert sum(counts) == diag.get("count", -1)

    def test_distribution_percentages_sum_to_one(self, default_validator, clean_chunk_df):
        _, diagnostics = default_validator.validate_distribution(
            clean_chunk_df, score_col="lm_tone_score"
        )
        diag = diagnostics["lm_tone_score"]
        total_pct = diag["pct_positive"] + diag["pct_negative"] + diag["pct_neutral"]
        assert total_pct == pytest.approx(1.0, abs=1e-6)

    def test_empty_df_returns_empty(self, default_validator):
        df = _chunk_df(n=0)
        issues, diagnostics = default_validator.validate_distribution(df, score_col="lm_tone_score")
        assert issues == []
        assert diagnostics == {}

    def test_missing_score_col_returns_empty(self, default_validator, clean_chunk_df):
        issues, diagnostics = default_validator.validate_distribution(
            clean_chunk_df, score_col="nonexistent_col"
        )
        assert issues == []
        assert diagnostics == {}


# ===========================================================================
# ── 12. validate_chunk_count ─────────────────────────────────────────────────
# ===========================================================================

class TestValidateChunkCount:

    def test_sufficient_chunk_count_no_issues(self, default_validator, clean_transcript_df):
        issues = default_validator.validate_chunk_count(clean_transcript_df)
        assert len(issues) == 0

    def test_zero_chunk_count_is_warning(self, default_validator):
        df = _transcript_df(n=10)
        df.loc[0, "lm_chunk_count"] = 0
        issues = default_validator.validate_chunk_count(df)
        assert len(issues) >= 1
        assert any(i.severity == Severity.WARNING for i in issues)

    def test_below_min_chunk_count_detected(self):
        cfg = LMValidationConfig(min_chunk_count=5)
        validator = LMValidator(cfg)
        df = _transcript_df(n=10)
        df.loc[0:3, "lm_chunk_count"] = 2  # 4 rows below threshold
        issues = validator.validate_chunk_count(df)
        assert len(issues) >= 1
        low_issues = [i for i in issues if "chunk" in i.check_name.lower()]
        assert any(i.affected_rows >= 4 for i in low_issues)

    def test_absent_chunk_count_column_no_issues(self, default_validator, clean_chunk_df):
        # Chunk-level df has no lm_chunk_count — should return no issues
        issues = default_validator.validate_chunk_count(clean_chunk_df)
        assert len(issues) == 0


# ===========================================================================
# ── 13. Composed Suite Runners ───────────────────────────────────────────────
# ===========================================================================

class TestComposedSuites:

    def test_chunk_output_clean_passes(self, default_validator, clean_chunk_df):
        summary = default_validator.validate_chunk_output(clean_chunk_df)
        assert summary.passed is True
        assert summary.error_count == 0

    def test_transcript_output_clean_passes(self, default_validator, clean_transcript_df):
        summary = default_validator.validate_transcript_output(clean_transcript_df)
        assert summary.passed is True
        assert summary.error_count == 0

    def test_chunk_output_with_score_error_fails(self, default_validator):
        df = _chunk_df(inject_score_error=True)
        summary = default_validator.validate_chunk_output(df)
        assert summary.passed is False
        assert summary.error_count >= 1

    def test_chunk_output_with_duplicate_fails_strictly(self, strict_validator):
        df = _chunk_df(inject_duplicate_chunk=True)
        summary = strict_validator.validate_chunk_output(df)
        assert summary.passed is False

    def test_transcript_output_with_bad_label_warnings(self, default_validator):
        df = _transcript_df(inject_bad_label=True)
        summary = default_validator.validate_transcript_output(df)
        assert summary.warning_count >= 1 or summary.error_count >= 1

    def test_section_output_validation(self, default_validator):
        df = _transcript_df(n=10)
        df["section_type"] = "prepared_remarks"
        summary = default_validator.validate_section_output(df)
        assert isinstance(summary, ValidationSummary)

    def test_speaker_output_validation(self, default_validator):
        df = _transcript_df(n=10)
        df["speaker_role"] = "ceo"
        summary = default_validator.validate_speaker_output(df)
        assert isinstance(summary, ValidationSummary)

    def test_summary_issue_list_is_list(self, default_validator, clean_chunk_df):
        summary = default_validator.validate_chunk_output(clean_chunk_df)
        assert isinstance(summary.issue_list, list)

    def test_summary_contains_validated_at(self, default_validator, clean_chunk_df):
        summary = default_validator.validate_chunk_output(clean_chunk_df)
        assert isinstance(summary.validated_at, str)
        assert len(summary.validated_at) > 0

    def test_summary_elapsed_ms_positive(self, default_validator, clean_chunk_df):
        summary = default_validator.validate_chunk_output(clean_chunk_df)
        assert summary.elapsed_ms >= 0.0

    def test_summary_diagnostics_dict(self, default_validator, clean_chunk_df):
        summary = default_validator.validate_chunk_output(clean_chunk_df)
        assert isinstance(summary.diagnostics, dict)

    def test_category_counts_populated_on_failure(self, default_validator):
        df = _chunk_df(
            inject_score_error=True,
            inject_duplicate_chunk=True,
            inject_null_score=True,
        )
        summary = default_validator.validate_chunk_output(df)
        assert len(summary.category_counts) >= 1
        assert sum(summary.category_counts.values()) == len(summary.issue_list)


# ===========================================================================
# ── 14. Severity Escalation ──────────────────────────────────────────────────
# ===========================================================================

class TestSeverityEscalation:

    def test_nan_warning_escalates_to_error_with_nan_as_error(self):
        cfg = LMValidationConfig(nan_as_error=True)
        validator = LMValidator(cfg)
        df = _chunk_df(inject_null_score=True)
        issues = validator.validate_nulls(df, critical_columns=["lm_tone_score"])
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_label_warning_escalates_to_error_with_strict_labels(self):
        cfg = LMValidationConfig(strict_label_check=True)
        validator = LMValidator(cfg)
        df = _chunk_df(inject_bad_label=True)
        issues = validator.validate_labels(df, label_col="lm_label")
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_duplicate_escalation_by_ratio(self):
        cfg_strict  = LMValidationConfig(max_duplicate_ratio=0.0)
        cfg_lenient = LMValidationConfig(max_duplicate_ratio=0.50)
        df = _chunk_df(n=100)
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        issues_strict  = LMValidator(cfg_strict).validate_duplicates(df, ["chunk_id"])
        issues_lenient = LMValidator(cfg_lenient).validate_duplicates(df, ["chunk_id"])
        assert any(i.severity == Severity.ERROR   for i in issues_strict)
        assert any(i.severity == Severity.WARNING for i in issues_lenient)
        assert not any(i.severity == Severity.ERROR for i in issues_lenient)

    def test_null_ratio_escalation_by_threshold(self):
        df = _chunk_df(n=100)
        # Set 12% null rate
        for i in range(12):
            df.loc[i, "lm_tone_score"] = np.nan

        cfg_tight = LMValidationConfig(max_null_ratio=0.05)  # 12% > 5% → ERROR
        cfg_loose = LMValidationConfig(max_null_ratio=0.20)  # 12% < 20% → WARNING

        issues_tight = LMValidator(cfg_tight).validate_nulls(df, critical_columns=["lm_tone_score"])
        issues_loose = LMValidator(cfg_loose).validate_nulls(df, critical_columns=["lm_tone_score"])

        assert any(i.severity == Severity.ERROR   for i in issues_tight)
        assert not any(i.severity == Severity.ERROR for i in issues_loose)
        assert any(i.severity == Severity.WARNING for i in issues_loose)


# ===========================================================================
# ── 15. Determinism ──────────────────────────────────────────────────────────
# ===========================================================================

class TestDeterminism:

    def test_same_df_same_issue_count(self, default_validator):
        df = _chunk_df(inject_score_error=True, inject_duplicate_chunk=True)
        s1 = default_validator.validate_chunk_output(df)
        s2 = default_validator.validate_chunk_output(df)
        assert s1.error_count   == s2.error_count
        assert s1.warning_count == s2.warning_count
        assert s1.info_count    == s2.info_count

    def test_same_df_same_category_counts(self, default_validator):
        df = _chunk_df(inject_null_score=True)
        s1 = default_validator.validate_chunk_output(df)
        s2 = default_validator.validate_chunk_output(df)
        assert s1.category_counts == s2.category_counts

    def test_same_df_same_check_names(self, default_validator):
        df = _chunk_df(inject_score_error=True)
        s1 = default_validator.validate_chunk_output(df)
        s2 = default_validator.validate_chunk_output(df)
        names1 = [i.check_name for i in s1.issue_list]
        names2 = [i.check_name for i in s2.issue_list]
        assert names1 == names2

    def test_validator_stateless_between_runs(self, default_validator):
        """Running on a clean df after a dirty df must not leak state."""
        dirty = _chunk_df(inject_score_error=True)
        clean = _chunk_df()
        s_dirty = default_validator.validate_chunk_output(dirty)
        s_clean = default_validator.validate_chunk_output(clean)
        assert s_dirty.error_count >= 1
        assert s_clean.error_count == 0


# ===========================================================================
# ── 16. Edge Cases ────────────────────────────────────────────────────────────
# ===========================================================================

class TestEdgeCases:

    def test_completely_empty_df(self, default_validator):
        df = pd.DataFrame()
        issues = default_validator.validate_schema(df, CHUNK_SCHEMA)
        assert len(issues) >= 1

    def test_empty_df_chunk_suite(self, default_validator):
        empty = _chunk_df(n=0)
        summary = default_validator.validate_chunk_output(empty)
        assert isinstance(summary, ValidationSummary)
        assert summary.error_count == 0

    def test_single_row_df(self, default_validator):
        df = _chunk_df(n=1)
        summary = default_validator.validate_chunk_output(df)
        assert isinstance(summary, ValidationSummary)

    def test_all_null_score_df(self, default_validator):
        df = _chunk_df(n=20)
        df["lm_tone_score"] = np.nan
        summary = default_validator.validate_chunk_output(df)
        assert summary.error_count >= 1 or summary.warning_count >= 1

    def test_infinite_score_df(self, default_validator):
        df = _chunk_df(inject_inf_score=True)
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1

    def test_all_same_transcript_id_no_false_positives(self, default_validator):
        """A section-level df with repeated transcript_id + section combos."""
        df = _transcript_df(n=5)
        df["section_type"] = ["prepared_remarks", "qa",
                               "prepared_remarks", "qa", "closing_remarks"]
        # All (transcript_id, section_type) combinations are unique
        issues = default_validator.validate_duplicates(
            df, key_cols=["transcript_id", "section_type"]
        )
        assert len(issues) == 0

    def test_mixed_valid_invalid_rows(self, default_validator):
        """Only the invalid rows should be counted in affected_rows."""
        df = _chunk_df(n=50)
        n_bad = 3
        for i in range(n_bad):
            df.loc[i, "lm_tone_score"] = 2.0 + i
        issues = default_validator.validate_scores(df, score_columns=["lm_tone_score"])
        range_issues = [i for i in issues if "range" in i.check_name]
        assert any(i.affected_rows == n_bad for i in range_issues)

    def test_report_generation_never_raises(self, default_validator):
        """generate_report must not raise for any input DataFrame."""
        for inject_kwarg in [
            {},
            {"inject_score_error": True},
            {"inject_duplicate_chunk": True},
            {"inject_null_score": True},
            {"inject_bad_label": True},
        ]:
            df = _chunk_df(**inject_kwarg)
            summary = default_validator.validate_chunk_output(df)
            try:
                report = summary.generate_report()
                assert isinstance(report, str)
            except Exception as exc:
                pytest.fail(f"generate_report raised for {inject_kwarg}: {exc}")

    def test_generate_report_helper_matches_summary_method(self, default_validator, clean_chunk_df):
        summary = default_validator.validate_chunk_output(clean_chunk_df)
        report_via_validator = default_validator.generate_report(summary)
        report_via_summary   = summary.generate_report()
        assert report_via_validator == report_via_summary

    def test_summary_helper_matches_method(self, default_validator, clean_chunk_df):
        summary = default_validator.validate_chunk_output(clean_chunk_df)
        via_validator = default_validator.summary(summary)
        via_summary   = summary.summary()
        assert via_validator == via_summary


# ===========================================================================
# ── Standalone execution
# ===========================================================================

if __name__ == "__main__":
    import subprocess

    print("Running test_lm_validation.py via pytest …\n")
    ret = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    if ret.returncode == 0:
        print("\n✅  All tests passed.")
    else:
        print(f"\n❌  Tests failed (exit code {ret.returncode}).")
    sys.exit(ret.returncode)
