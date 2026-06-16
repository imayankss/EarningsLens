"""
sentiment_validation.py
=======================
Centralized validation framework for the financial NLP earnings call
sentiment pipeline.

Part of: Earnings Call Sentiment Analyzer — DAY 8 Pipeline
Stage:   Validation Layer (post-inference quality gate)

Responsibilities
----------------
This module is the single source of truth for data quality across every
sentiment artifact produced by the pipeline.  It validates:

  * **Chunk-level** outputs (from finbert_pipeline.py)
  * **Sentence-level** outputs (from sentence_sentiment.py)
  * **Transcript-level** aggregations (from aggregation.py)
  * **Hierarchical** aggregation outputs (from hierarchical_aggregation.py)
  * **Cross-layer** consistency (chunk ↔ transcript, sentence ↔ transcript)

Validation categories
---------------------
  SCHEMA        — required columns present, correct dtypes
  PROBABILITY   — P(pos)+P(neu)+P(neg) ≈ 1; each ∈ [0,1]
  ORDERING      — chunk_order / sentence_order monotonic, no gaps
  DUPLICATE     — unique IDs across the dataset
  NULL_RATE     — configurable warn/error thresholds per column
  RANGE         — numeric fields within expected bounds
  LINKAGE       — every chunk/sentence traces back to a known transcript
  HIERARCHY     — parent-child consistency across aggregation levels
  COMPLETENESS  — every expected record is present; no silent losses
  AGGREGATION   — cross-layer score consistency checks

Design principles
-----------------
* Validators NEVER mutate source DataFrames.
* All checks are individually addressable and combinable via
  ``ValidationConfig.run_*`` flags.
* Issue severity escalates from INFO → WARNING → ERROR.
* Every issue carries an actionable message, affected row count,
  and up to ``sample_size`` example IDs for debugging.
* The :class:`SentimentValidator` is the single entry point;
  specialized helpers are internal to keep the public API clean.
* Partial validation runs are fully supported via config flags.

Integration notes
-----------------
This module has zero dependencies on FinBERT, chunk generators, or any
other inference component.  It operates entirely on pandas DataFrames,
making it usable independently of the rest of the pipeline.

Expected schemas
----------------
Chunk DataFrame (finbert_pipeline._OUTPUT_COLUMNS):
    chunk_id, transcript_id, chunk_order, positive_prob, neutral_prob,
    negative_prob, sentiment_score, confidence, predicted_label,
    token_count, section_type, dominant_speaker, model_name,
    was_skipped, skip_reason

Sentence DataFrame (sentence_sentiment._OUTPUT_COLUMNS):
    sentence_id, transcript_id, sentence_order, sentence_score,
    positive_prob, neutral_prob, negative_prob, confidence,
    predicted_label, local_sentiment_shift, rolling_sentiment_mean,
    rolling_sentiment_std, is_spike, cumulative_sentiment, ema_sentiment,
    speaker, speaker_role, section_type, token_estimate, model_name,
    was_skipped, skip_reason

Transcript DataFrame (aggregation output):
    transcript_id, chunk_count / sentence_count, sentiment_score,
    sentiment_label, positive_probability, neutral_probability,
    negative_probability, sentiment_std, sentiment_median,
    confidence_mean

Hierarchy DataFrame (hierarchical_aggregation output):
    transcript_id, level, group_key, sentiment_score, sentence_count,
    confidence_mean  (and optional section_type / speaker / speaker_role)

Usage
-----
    from sentiment_validation import SentimentValidator, ValidationConfig

    config    = ValidationConfig(max_null_rate_error=0.10)
    validator = SentimentValidator(config)

    chunk_summary     = validator.validate_chunks(chunk_df)
    sentence_summary  = validator.validate_sentences(sentence_df)
    transcript_summary= validator.validate_transcripts(transcript_df)
    hierarchy_summary = validator.validate_hierarchy(hier_df)

    report = validator.generate_validation_report(
        chunk_summary, sentence_summary, transcript_summary, hierarchy_summary
    )
    print(report)

Python: 3.11+
"""

from __future__ import annotations

import json
import logging
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Issue severity level.

    ERROR   — data quality problem that must be fixed before downstream use.
    WARNING — potential problem; pipeline can proceed but results may be unreliable.
    INFO    — informational; check passed or advisory note.
    """
    ERROR   = "ERROR"
    WARNING = "WARNING"
    INFO    = "INFO"


class Category(str, Enum):
    """Validation category tags for grouping and filtering issues."""
    SCHEMA        = "schema"
    PROBABILITY   = "probability"
    ORDERING      = "ordering"
    DUPLICATE     = "duplicate"
    NULL_RATE     = "null_rate"
    RANGE         = "range"
    LINKAGE       = "linkage"
    HIERARCHY     = "hierarchy"
    COMPLETENESS  = "completeness"
    AGGREGATION   = "aggregation"


class ValidationTarget(str, Enum):
    """Which pipeline artifact is being validated."""
    CHUNK      = "chunk"
    SENTENCE   = "sentence"
    TRANSCRIPT = "transcript"
    HIERARCHY  = "hierarchy"
    CROSS      = "cross"          # cross-layer consistency


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """All tunable parameters for :class:`SentimentValidator`.

    Thresholds
    ----------
    prob_sum_tolerance : float
        Maximum allowed |P(pos)+P(neu)+P(neg) − 1|.  Default 1e-4
        matches the tolerance used in finbert_pipeline and sentence_sentiment.
    max_null_rate_warning : float
        Null rate above which a WARNING is raised (per column).
    max_null_rate_error : float
        Null rate above which an ERROR is raised (per column).
    max_token_length : int
        Hard maximum token count per chunk.  Must not exceed 512.
    sentiment_score_min / max : float
        Expected range for sentiment_score  (P(pos)−P(neg) ∈ [−1, 1]).
    prob_min / max : float
        Expected range for individual probability columns.
    confidence_min / max : float
        Expected range for the confidence column.
    aggregation_score_tolerance : float
        |reported_score − computed_mean|  beyond which to raise a WARNING
        on transcript-level aggregation consistency.
    max_sample_ids : int
        Maximum number of example IDs to include in each ValidationIssue.
    skip_column : str
        Column name used to filter out skipped records before validation.

    Feature flags
    -------------
    run_schema_checks / run_probability_checks / … : bool
        Enable/disable individual check groups for partial validation runs.
    warn_on_high_skip_rate : bool
        Raise a WARNING when > skip_rate_threshold rows are marked skipped.
    skip_rate_threshold : float
        Fraction of skipped rows that triggers the skip-rate warning.
    """

    # --- Thresholds ---
    prob_sum_tolerance          : float = 1e-4
    max_null_rate_warning       : float = 0.05
    max_null_rate_error         : float = 0.20
    max_token_length            : int   = 512
    sentiment_score_min         : float = -1.0
    sentiment_score_max         : float =  1.0
    prob_min                    : float =  0.0
    prob_max                    : float =  1.0
    confidence_min              : float =  0.0
    confidence_max              : float =  1.0
    aggregation_score_tolerance : float =  0.05

    # --- Sampling ---
    max_sample_ids              : int   = 5

    # --- Column names ---
    skip_column                 : str   = "was_skipped"

    # --- Feature flags ---
    run_schema_checks           : bool  = True
    run_probability_checks      : bool  = True
    run_ordering_checks         : bool  = True
    run_duplicate_checks        : bool  = True
    run_null_rate_checks        : bool  = True
    run_range_checks            : bool  = True
    run_linkage_checks          : bool  = True
    run_hierarchy_checks        : bool  = True
    run_completeness_checks     : bool  = True
    run_aggregation_checks      : bool  = True

    # --- Skip-rate alerting ---
    warn_on_high_skip_rate      : bool  = True
    skip_rate_threshold         : float = 0.30

    def __post_init__(self) -> None:
        if self.max_token_length > 512:
            raise ValueError(
                f"max_token_length={self.max_token_length} exceeds the BERT "
                "hard limit of 512."
            )
        if not 0.0 <= self.max_null_rate_warning <= self.max_null_rate_error <= 1.0:
            raise ValueError(
                "Null-rate thresholds must satisfy "
                "0 ≤ warning ≤ error ≤ 1."
            )


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A single identified data-quality issue.

    Attributes
    ----------
    severity   : Severity
        ERROR / WARNING / INFO.
    category   : Category
        Validation category (probability, ordering, …).
    target     : ValidationTarget
        Which pipeline layer this issue belongs to.
    field      : str
        The DataFrame column(s) involved, e.g. ``"positive_prob"``.
    message    : str
        Human-readable description with enough context to act on.
    count      : int
        Number of affected rows.
    sample_ids : list[str]
        Up to ``ValidationConfig.max_sample_ids`` example identifiers
        (chunk_id / sentence_id / transcript_id) for debugging.
    check_name : str
        Internal name of the check that generated this issue, useful for
        filtering in automated test assertions.
    """

    severity   : str
    category   : str
    target     : str
    field      : str
    message    : str
    count      : int        = 0
    sample_ids : list[str]  = field(default_factory=list)
    check_name : str        = ""

    # Convenience constructor for passed checks
    @classmethod
    def passed(
        cls,
        target     : str,
        check_name : str,
        message    : str,
        field      : str = "",
    ) -> "ValidationIssue":
        """Create an INFO-severity record for a check that passed."""
        return cls(
            severity   = Severity.INFO.value,
            category   = Category.COMPLETENESS.value,
            target     = target,
            field      = field,
            message    = message,
            count      = 0,
            sample_ids = [],
            check_name = check_name,
        )

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"ValidationIssue("
            f"[{self.severity}] {self.target}.{self.field}: "
            f"{self.message[:80]!r}, count={self.count})"
        )


# ---------------------------------------------------------------------------
# ValidationSummary
# ---------------------------------------------------------------------------

@dataclass
class ValidationSummary:
    """Aggregated result of validating one pipeline layer.

    Attributes
    ----------
    target        : str
        Which layer was validated (chunk / sentence / transcript / hierarchy).
    total_rows    : int
        Number of rows in the validated DataFrame.
    valid_rows    : int
        Rows not marked as skipped.
    error_count   : int
    warning_count : int
    info_count    : int
    checks_run    : int
    passed        : bool
        True only if error_count == 0.
    issues        : list[ValidationIssue]
        All issues, sorted by severity then category.
    validated_at  : str
        ISO-8601 UTC timestamp of when the summary was produced.
    """

    target        : str
    total_rows    : int
    valid_rows    : int
    error_count   : int
    warning_count : int
    info_count    : int
    checks_run    : int
    passed        : bool
    issues        : list[ValidationIssue]
    validated_at  : str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        return self.warning_count > 0

    @property
    def health_score(self) -> float:
        """0–100 quality score.  Errors cost 10 pts each, warnings 2 pts."""
        penalty = self.error_count * 10 + self.warning_count * 2
        return max(0.0, 100.0 - float(penalty))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR.value]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING.value]

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Flat DataFrame of all issues (one row per issue)."""
        if not self.issues:
            return pd.DataFrame()
        rows = [i.to_dict() for i in self.issues]
        return pd.DataFrame(rows)

    def to_dict(self, include_issues: bool = False) -> dict:
        """Return a summary dict for logging / export."""
        d = {
            "target"        : self.target,
            "total_rows"    : self.total_rows,
            "valid_rows"    : self.valid_rows,
            "error_count"   : self.error_count,
            "warning_count" : self.warning_count,
            "checks_run"    : self.checks_run,
            "passed"        : self.passed,
            "health_score"  : round(self.health_score, 1),
            "validated_at"  : self.validated_at,
        }
        if include_issues:
            d["issues"] = [i.to_dict() for i in self.issues]
        return d

    def __repr__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (
            f"ValidationSummary({status} | target={self.target!r} | "
            f"rows={self.total_rows} | E={self.error_count} "
            f"W={self.warning_count} | score={self.health_score:.0f}/100)"
        )


# ---------------------------------------------------------------------------
# Schema definitions for each validated artifact
# ---------------------------------------------------------------------------

# fmt: off
_CHUNK_REQUIRED_COLS: frozenset[str] = frozenset({
    "chunk_id", "transcript_id", "chunk_order",
    "positive_prob", "neutral_prob", "negative_prob",
    "sentiment_score", "confidence", "predicted_label",
})
_CHUNK_OPTIONAL_COLS: frozenset[str] = frozenset({
    "token_count", "section_type", "dominant_speaker",
    "model_name", "was_skipped", "skip_reason",
})

_SENTENCE_REQUIRED_COLS: frozenset[str] = frozenset({
    "sentence_id", "transcript_id", "sentence_order",
    "sentence_score", "positive_prob", "neutral_prob", "negative_prob",
    "confidence", "predicted_label",
})
_SENTENCE_OPTIONAL_COLS: frozenset[str] = frozenset({
    "local_sentiment_shift", "rolling_sentiment_mean", "rolling_sentiment_std",
    "is_spike", "cumulative_sentiment", "ema_sentiment",
    "speaker", "speaker_role", "section_type", "token_estimate",
    "model_name", "was_skipped", "skip_reason",
})

_TRANSCRIPT_REQUIRED_COLS: frozenset[str] = frozenset({
    "transcript_id", "sentiment_score",
})
_TRANSCRIPT_OPTIONAL_COLS: frozenset[str] = frozenset({
    "chunk_count", "sentence_count",
    "sentiment_label",
    "positive_probability", "neutral_probability", "negative_probability",
    "sentiment_std", "sentiment_median", "confidence_mean",
})

_HIERARCHY_REQUIRED_COLS: frozenset[str] = frozenset({
    "transcript_id", "level", "sentiment_score",
})
_HIERARCHY_OPTIONAL_COLS: frozenset[str] = frozenset({
    "group_key", "section_type", "speaker", "speaker_role",
    "sentence_count", "chunk_count", "confidence_mean",
    "positive_probability", "neutral_probability", "negative_probability",
})

_VALID_LABELS: frozenset[str] = frozenset({
    "positive", "neutral", "negative", "unknown",
})
_VALID_HIERARCHY_LEVELS: frozenset[str] = frozenset({
    "transcript", "section", "speaker", "chunk", "sentence",
})
# fmt: on


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample(values: pd.Series, n: int) -> list[str]:
    """Return up to *n* unique non-null string values from *values*."""
    vals = values.dropna().astype(str).unique()[:n]
    return list(vals)


def _active_rows(df: pd.DataFrame, skip_col: str) -> pd.DataFrame:
    """Return rows not marked as skipped.  Works even if skip_col is absent."""
    if skip_col in df.columns:
        return df[~df[skip_col].fillna(False).astype(bool)]
    return df


def _skip_rate(df: pd.DataFrame, skip_col: str) -> float:
    if skip_col not in df.columns or df.empty:
        return 0.0
    return float(df[skip_col].fillna(False).astype(bool).mean())


# ---------------------------------------------------------------------------
# Core validator class
# ---------------------------------------------------------------------------

class SentimentValidator:
    """Centralized, configurable validation engine for sentiment pipeline outputs.

    Parameters
    ----------
    config : ValidationConfig, optional
        If ``None``, sensible defaults are used.

    Public API
    ----------
    validate_chunks(df)        → ValidationSummary
    validate_sentences(df)     → ValidationSummary
    validate_transcripts(df)   → ValidationSummary
    validate_hierarchy(df)     → ValidationSummary
    validate_cross_layer(...)  → ValidationSummary
    generate_validation_report(...) → str
    """

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        logger.info(
            "SentimentValidator initialised | "
            "prob_tol=%.1e | null_warn=%.0f%% | null_err=%.0f%%",
            self.config.prob_sum_tolerance,
            self.config.max_null_rate_warning * 100,
            self.config.max_null_rate_error   * 100,
        )

    # ===================================================================
    # PUBLIC ENTRY POINTS
    # ===================================================================

    def validate_chunks(self, df: pd.DataFrame) -> ValidationSummary:
        """Validate a chunk-level sentiment DataFrame.

        Runs all applicable checks from :class:`ValidationConfig` against the
        schema produced by ``finbert_pipeline.py``.

        Parameters
        ----------
        df : pd.DataFrame
            Chunk-level sentiment DataFrame.

        Returns
        -------
        ValidationSummary
        """
        logger.info("Validating chunks | rows=%d", len(df))
        issues: list[ValidationIssue] = []
        cfg   = self.config
        target = ValidationTarget.CHUNK.value
        active = _active_rows(df, cfg.skip_column)

        if cfg.run_schema_checks:
            issues += self._check_schema(
                df, _CHUNK_REQUIRED_COLS, target, "chunk_id"
            )

        if df.empty:
            return self._build_summary(target, df, active, issues)

        if cfg.warn_on_high_skip_rate:
            issues += self._check_skip_rate(df, target)

        if cfg.run_duplicate_checks:
            issues += self._check_duplicates(
                df, id_col="chunk_id", target=target
            )

        if cfg.run_probability_checks and not active.empty:
            issues += self._check_prob_sums(active, target, id_col="chunk_id")
            issues += self._check_prob_range(active, target)

        if cfg.run_range_checks and not active.empty:
            issues += self._check_numeric_range(
                active, "sentiment_score",
                cfg.sentiment_score_min, cfg.sentiment_score_max,
                target, id_col="chunk_id",
            )
            issues += self._check_numeric_range(
                active, "confidence",
                cfg.confidence_min, cfg.confidence_max,
                target, id_col="chunk_id",
            )
            if "token_count" in df.columns:
                issues += self._check_token_limit(df, target)

        if cfg.run_ordering_checks and not active.empty:
            issues += self._check_ordering_per_group(
                active, group_col="transcript_id",
                order_col="chunk_order", id_col="chunk_id",
                target=target,
            )

        if cfg.run_null_rate_checks:
            issues += self._check_null_rates(
                df, _CHUNK_REQUIRED_COLS | _CHUNK_OPTIONAL_COLS, target
            )

        if cfg.run_range_checks and "predicted_label" in active.columns:
            issues += self._check_label_values(active, target, id_col="chunk_id")

        return self._build_summary(target, df, active, issues)

    # -------------------------------------------------------------------

    def validate_sentences(self, df: pd.DataFrame) -> ValidationSummary:
        """Validate a sentence-level sentiment DataFrame.

        Runs all applicable checks against the schema produced by
        ``sentence_sentiment.py``.

        Parameters
        ----------
        df : pd.DataFrame
            Sentence-level sentiment DataFrame.

        Returns
        -------
        ValidationSummary
        """
        logger.info("Validating sentences | rows=%d", len(df))
        issues: list[ValidationIssue] = []
        cfg   = self.config
        target = ValidationTarget.SENTENCE.value
        active = _active_rows(df, cfg.skip_column)

        if cfg.run_schema_checks:
            issues += self._check_schema(
                df, _SENTENCE_REQUIRED_COLS, target, "sentence_id"
            )

        if df.empty:
            return self._build_summary(target, df, active, issues)

        if cfg.warn_on_high_skip_rate:
            issues += self._check_skip_rate(df, target)

        if cfg.run_duplicate_checks:
            issues += self._check_duplicates(df, "sentence_id", target)

        if cfg.run_probability_checks and not active.empty:
            issues += self._check_prob_sums(active, target, id_col="sentence_id")
            issues += self._check_prob_range(active, target)

        if cfg.run_ordering_checks and not active.empty:
            issues += self._check_ordering_per_group(
                active,
                group_col   = "transcript_id",
                order_col   = "sentence_order",
                id_col      = "sentence_id",
                target      = target,
                check_gaps  = True,
            )

        if cfg.run_range_checks and not active.empty:
            score_col = (
                "sentence_score" if "sentence_score" in active.columns
                else "sentiment_score"
            )
            if score_col in active.columns:
                issues += self._check_numeric_range(
                    active, score_col,
                    cfg.sentiment_score_min, cfg.sentiment_score_max,
                    target, id_col="sentence_id",
                )
            issues += self._check_numeric_range(
                active, "confidence",
                cfg.confidence_min, cfg.confidence_max,
                target, id_col="sentence_id",
            )

        if cfg.run_range_checks and "predicted_label" in active.columns:
            issues += self._check_label_values(
                active, target, id_col="sentence_id"
            )

        if cfg.run_null_rate_checks:
            issues += self._check_null_rates(
                df, _SENTENCE_REQUIRED_COLS | _SENTENCE_OPTIONAL_COLS, target
            )

        if cfg.run_completeness_checks and not active.empty:
            issues += self._check_shift_continuity(active, target)

        return self._build_summary(target, df, active, issues)

    # -------------------------------------------------------------------

    def validate_transcripts(self, df: pd.DataFrame) -> ValidationSummary:
        """Validate a transcript-level aggregation DataFrame.

        Validates the output of ``aggregation.py`` or any DataFrame that
        has one row per transcript with aggregated sentiment metrics.

        Parameters
        ----------
        df : pd.DataFrame
            One row per transcript.

        Returns
        -------
        ValidationSummary
        """
        logger.info("Validating transcripts | rows=%d", len(df))
        issues: list[ValidationIssue] = []
        cfg   = self.config
        target = ValidationTarget.TRANSCRIPT.value
        active = df  # no skip column at transcript level

        if cfg.run_schema_checks:
            issues += self._check_schema(
                df, _TRANSCRIPT_REQUIRED_COLS, target, "transcript_id"
            )

        if df.empty:
            return self._build_summary(target, df, active, issues)

        if cfg.run_duplicate_checks:
            issues += self._check_duplicates(df, "transcript_id", target)

        if cfg.run_probability_checks:
            issues += self._check_transcript_probs(df, target)

        if cfg.run_range_checks:
            issues += self._check_numeric_range(
                df, "sentiment_score",
                cfg.sentiment_score_min, cfg.sentiment_score_max,
                target, id_col="transcript_id",
            )
            if "confidence_mean" in df.columns:
                issues += self._check_numeric_range(
                    df, "confidence_mean",
                    cfg.confidence_min, cfg.confidence_max,
                    target, id_col="transcript_id",
                )

        if cfg.run_range_checks and "sentiment_label" in df.columns:
            issues += self._check_label_values(
                df, target, id_col="transcript_id",
                label_col="sentiment_label",
            )

        if cfg.run_null_rate_checks:
            issues += self._check_null_rates(
                df,
                _TRANSCRIPT_REQUIRED_COLS | _TRANSCRIPT_OPTIONAL_COLS,
                target,
            )

        return self._build_summary(target, df, active, issues)

    # -------------------------------------------------------------------

    def validate_hierarchy(self, df: pd.DataFrame) -> ValidationSummary:
        """Validate a hierarchical aggregation DataFrame.

        Validates the output of ``hierarchical_aggregation.py``, which
        contains sentiment metrics at multiple levels (transcript, section,
        speaker, chunk, sentence) in a single long-format DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            One row per (transcript × level) combination.

        Returns
        -------
        ValidationSummary
        """
        logger.info("Validating hierarchy | rows=%d", len(df))
        issues: list[ValidationIssue] = []
        cfg   = self.config
        target = ValidationTarget.HIERARCHY.value

        if cfg.run_schema_checks:
            issues += self._check_schema(
                df, _HIERARCHY_REQUIRED_COLS, target, "transcript_id"
            )

        if df.empty:
            return self._build_summary(target, df, df, issues)

        if cfg.run_hierarchy_checks:
            issues += self._check_hierarchy_levels(df, target)
            issues += self._check_hierarchy_duplicates(df, target)

        if cfg.run_range_checks:
            issues += self._check_numeric_range(
                df, "sentiment_score",
                cfg.sentiment_score_min, cfg.sentiment_score_max,
                target, id_col="transcript_id",
            )

        if cfg.run_probability_checks:
            issues += self._check_transcript_probs(
                df, target,
                pos_col="positive_probability",
                neu_col="neutral_probability",
                neg_col="negative_probability",
            )

        if cfg.run_null_rate_checks:
            issues += self._check_null_rates(
                df,
                _HIERARCHY_REQUIRED_COLS | _HIERARCHY_OPTIONAL_COLS,
                target,
            )

        return self._build_summary(target, df, df, issues)

    # -------------------------------------------------------------------

    def validate_cross_layer(
        self,
        *,
        chunk_df     : Optional[pd.DataFrame] = None,
        sentence_df  : Optional[pd.DataFrame] = None,
        transcript_df: Optional[pd.DataFrame] = None,
        hierarchy_df : Optional[pd.DataFrame] = None,
    ) -> ValidationSummary:
        """Validate consistency ACROSS pipeline layers.

        Checks that:
        * Every transcript referenced in chunk_df appears in transcript_df.
        * Every transcript in sentence_df appears in transcript_df.
        * Chunk counts in transcript_df match actual chunk_df counts.
        * Sentence counts in transcript_df match actual sentence_df counts.
        * Transcript-level scores are consistent with chunk-level means.

        All arguments are optional; only available layers are checked.

        Returns
        -------
        ValidationSummary  with target="cross"
        """
        logger.info("Validating cross-layer consistency")
        issues: list[ValidationIssue] = []
        cfg   = self.config
        target = ValidationTarget.CROSS.value

        # Build reference set from transcript_df
        tx_ids: Optional[set[str]] = None
        if transcript_df is not None and not transcript_df.empty:
            if "transcript_id" in transcript_df.columns:
                tx_ids = set(transcript_df["transcript_id"].dropna().astype(str))

        if cfg.run_linkage_checks:
            if chunk_df is not None and not chunk_df.empty and tx_ids is not None:
                issues += self._check_linkage(
                    chunk_df, tx_ids, "transcript_id",
                    target, "chunk→transcript"
                )
            if sentence_df is not None and not sentence_df.empty and tx_ids is not None:
                issues += self._check_linkage(
                    sentence_df, tx_ids, "transcript_id",
                    target, "sentence→transcript"
                )

        if cfg.run_completeness_checks:
            if chunk_df is not None and transcript_df is not None:
                issues += self._check_count_consistency(
                    fine_df     = chunk_df,
                    coarse_df   = transcript_df,
                    fine_id_col = "chunk_id",
                    count_col   = "chunk_count",
                    target      = target,
                )
            if sentence_df is not None and transcript_df is not None:
                issues += self._check_count_consistency(
                    fine_df     = sentence_df,
                    coarse_df   = transcript_df,
                    fine_id_col = "sentence_id",
                    count_col   = "sentence_count",
                    target      = target,
                )

        if cfg.run_aggregation_checks:
            if chunk_df is not None and transcript_df is not None:
                issues += self._check_aggregation_consistency(
                    fine_df      = chunk_df,
                    coarse_df    = transcript_df,
                    fine_score   = "sentiment_score",
                    coarse_score = "sentiment_score",
                    target       = target,
                    layer_label  = "chunk→transcript",
                )

        # Hierarchy completeness against transcript list
        if cfg.run_hierarchy_checks:
            if hierarchy_df is not None and tx_ids is not None and not hierarchy_df.empty:
                issues += self._check_linkage(
                    hierarchy_df, tx_ids, "transcript_id",
                    target, "hierarchy→transcript"
                )

        all_dfs = [
            d for d in [chunk_df, sentence_df, transcript_df, hierarchy_df]
            if d is not None
        ]
        total_rows = sum(len(d) for d in all_dfs)
        placeholder = pd.DataFrame({"transcript_id": list(tx_ids or [])})

        return self._build_summary(target, placeholder, placeholder, issues)

    # ===================================================================
    # REPORT GENERATION
    # ===================================================================

    def generate_validation_report(
        self,
        *summaries: ValidationSummary,
        title    : str = "Earnings Call Sentiment Pipeline — Validation Report",
        as_json  : bool = False,
    ) -> str:
        """Generate a human-readable (or JSON) validation report.

        Parameters
        ----------
        *summaries : ValidationSummary
            One or more summaries to include in the report.
        title : str
            Report title line.
        as_json : bool
            If True, return a JSON string instead of formatted text.

        Returns
        -------
        str
        """
        if as_json:
            return self._build_json_report(list(summaries), title)
        return self._build_text_report(list(summaries), title)

    def save_report(
        self,
        report : str,
        path   : Union[str, Path],
    ) -> Path:
        """Write a text or JSON report to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        logger.info("Validation report saved → %s", out)
        return out

    def save_issues_csv(
        self,
        *summaries: ValidationSummary,
        path: Union[str, Path],
    ) -> Path:
        """Write all issues from multiple summaries to a CSV file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = [s.to_dataframe() for s in summaries if s.issues]
        if frames:
            combined = pd.concat(frames, ignore_index=True)
        else:
            combined = pd.DataFrame()
        combined.to_csv(out, index=False)
        logger.info("Issues CSV saved → %s (%d rows)", out, len(combined))
        return out

    # ===================================================================
    # INTERNAL CHECK METHODS
    # ===================================================================

    # --- Schema ---

    def _check_schema(
        self,
        df          : pd.DataFrame,
        required    : frozenset[str],
        target      : str,
        primary_key : str,
    ) -> list[ValidationIssue]:
        issues = []
        missing = required - set(df.columns)
        if missing:
            issues.append(ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.SCHEMA.value,
                target     = target,
                field      = ", ".join(sorted(missing)),
                message    = (
                    f"Missing required columns: {sorted(missing)}. "
                    f"DataFrame has: {sorted(df.columns.tolist())}"
                ),
                count      = len(missing),
                check_name = "required_columns",
            ))
        else:
            issues.append(ValidationIssue.passed(
                target, "required_columns",
                f"All {len(required)} required columns present.",
            ))

        if df.empty:
            issues.append(ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.COMPLETENESS.value,
                target     = target,
                field      = "rows",
                message    = "DataFrame is empty — no rows to validate.",
                check_name = "non_empty",
            ))
        else:
            issues.append(ValidationIssue.passed(
                target, "non_empty",
                f"DataFrame is non-empty ({len(df)} rows).",
            ))

        return issues

    # --- Skip rate ---

    def _check_skip_rate(
        self, df: pd.DataFrame, target: str
    ) -> list[ValidationIssue]:
        rate = _skip_rate(df, self.config.skip_column)
        if rate > self.config.skip_rate_threshold:
            return [ValidationIssue(
                severity   = Severity.WARNING.value,
                category   = Category.COMPLETENESS.value,
                target     = target,
                field      = self.config.skip_column,
                message    = (
                    f"High skip rate: {rate:.1%} of rows are marked "
                    f"was_skipped=True (threshold {self.config.skip_rate_threshold:.0%}). "
                    "Check upstream preprocessing for data losses."
                ),
                count      = int(df[self.config.skip_column].fillna(False).sum()),
                check_name = "skip_rate",
            )]
        return [ValidationIssue.passed(
            target, "skip_rate",
            f"Skip rate {rate:.1%} within threshold "
            f"({self.config.skip_rate_threshold:.0%}).",
            field=self.config.skip_column,
        )]

    # --- Probability sum ---

    def _check_prob_sums(
        self,
        df     : pd.DataFrame,
        target : str,
        id_col : str,
        pos_col: str = "positive_prob",
        neu_col: str = "neutral_prob",
        neg_col: str = "negative_prob",
    ) -> list[ValidationIssue]:
        # Only run if all three columns present
        needed = {pos_col, neu_col, neg_col}
        if not needed.issubset(set(df.columns)):
            return []

        sums = df[pos_col] + df[neu_col] + df[neg_col]
        bad  = (sums - 1.0).abs() > self.config.prob_sum_tolerance
        n_bad = int(bad.sum())

        if n_bad > 0:
            ids = _sample(
                df.loc[bad, id_col] if id_col in df.columns else pd.Series(dtype=str),
                self.config.max_sample_ids,
            )
            return [ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.PROBABILITY.value,
                target     = target,
                field      = f"{pos_col}+{neu_col}+{neg_col}",
                message    = (
                    f"{n_bad} row(s) have probability sums outside "
                    f"[1 ± {self.config.prob_sum_tolerance:.0e}]. "
                    f"Range: [{sums.min():.6f}, {sums.max():.6f}]."
                ),
                count      = n_bad,
                sample_ids = ids,
                check_name = "prob_sum",
            )]

        return [ValidationIssue.passed(
            target, "prob_sum",
            f"All {len(df)} probability sums within tolerance.",
            field=f"{pos_col}+{neu_col}+{neg_col}",
        )]

    # --- Probability range ---

    def _check_prob_range(
        self,
        df    : pd.DataFrame,
        target: str,
    ) -> list[ValidationIssue]:
        issues = []
        prob_cols = [c for c in ["positive_prob", "neutral_prob", "negative_prob"]
                     if c in df.columns]
        for col in prob_cols:
            out = (df[col] < self.config.prob_min - 1e-9) | \
                  (df[col] > self.config.prob_max + 1e-9)
            n_out = int(out.sum())
            if n_out > 0:
                issues.append(ValidationIssue(
                    severity   = Severity.ERROR.value,
                    category   = Category.RANGE.value,
                    target     = target,
                    field      = col,
                    message    = (
                        f"{n_out} values outside [{self.config.prob_min}, "
                        f"{self.config.prob_max}]. "
                        f"Min={df[col].min():.6f}, Max={df[col].max():.6f}."
                    ),
                    count      = n_out,
                    check_name = f"prob_range_{col}",
                ))
            else:
                issues.append(ValidationIssue.passed(
                    target, f"prob_range_{col}",
                    f"{col} all within [0, 1].",
                    field=col,
                ))
        return issues

    # --- Transcript probability columns ---

    def _check_transcript_probs(
        self,
        df     : pd.DataFrame,
        target : str,
        pos_col: str = "positive_probability",
        neu_col: str = "neutral_probability",
        neg_col: str = "negative_probability",
    ) -> list[ValidationIssue]:
        issues = []
        # Only run if all three probability columns present
        if all(c in df.columns for c in (pos_col, neu_col, neg_col)):
            issues += self._check_prob_sums(
                df, target, id_col="transcript_id",
                pos_col=pos_col, neu_col=neu_col, neg_col=neg_col,
            )
        # Check confidence_mean range if present
        if "confidence_mean" in df.columns:
            issues += self._check_numeric_range(
                df, "confidence_mean",
                self.config.confidence_min, self.config.confidence_max,
                target, id_col="transcript_id",
            )
        return issues

    # --- Numeric range ---

    def _check_numeric_range(
        self,
        df    : pd.DataFrame,
        col   : str,
        lo    : float,
        hi    : float,
        target: str,
        id_col: str = "",
    ) -> list[ValidationIssue]:
        if col not in df.columns:
            return []
        series = df[col].dropna()
        if series.empty:
            return []
        out = (series < lo - 1e-9) | (series > hi + 1e-9)
        n_out = int(out.sum())
        if n_out > 0:
            ids = []
            if id_col and id_col in df.columns:
                ids = _sample(df.loc[out.index[out], id_col], self.config.max_sample_ids)
            return [ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.RANGE.value,
                target     = target,
                field      = col,
                message    = (
                    f"{n_out} values in '{col}' outside [{lo}, {hi}]. "
                    f"Min={series.min():.6f}, Max={series.max():.6f}."
                ),
                count      = n_out,
                sample_ids = ids,
                check_name = f"range_{col}",
            )]
        return [ValidationIssue.passed(
            target, f"range_{col}",
            f"'{col}' all within [{lo}, {hi}].",
            field=col,
        )]

    # --- Token limit ---

    def _check_token_limit(
        self,
        df    : pd.DataFrame,
        target: str,
        col   : str = "token_count",
    ) -> list[ValidationIssue]:
        if col not in df.columns:
            return []
        over = df[col] > self.config.max_token_length
        n_over = int(over.sum())
        if n_over > 0:
            ids = _sample(
                df.loc[over, "chunk_id"] if "chunk_id" in df.columns
                else pd.Series(dtype=str),
                self.config.max_sample_ids,
            )
            return [ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.RANGE.value,
                target     = target,
                field      = col,
                message    = (
                    f"{n_over} chunk(s) exceed the token limit of "
                    f"{self.config.max_token_length}. "
                    f"Max observed: {int(df[col].max())}."
                ),
                count      = n_over,
                sample_ids = ids,
                check_name = "token_limit",
            )]
        return [ValidationIssue.passed(
            target, "token_limit",
            f"All token counts ≤ {self.config.max_token_length}.",
            field=col,
        )]

    # --- Duplicate detection ---

    def _check_duplicates(
        self,
        df    : pd.DataFrame,
        id_col: str,
        target: str,
    ) -> list[ValidationIssue]:
        if id_col not in df.columns:
            return []
        dup_mask = df.duplicated(id_col, keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup > 0:
            ids = _sample(df.loc[dup_mask, id_col], self.config.max_sample_ids)
            return [ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.DUPLICATE.value,
                target     = target,
                field      = id_col,
                message    = (
                    f"{n_dup} rows have duplicate '{id_col}' values. "
                    f"Unique duplicated IDs: "
                    f"{df.loc[dup_mask, id_col].unique()[:3].tolist()}."
                ),
                count      = n_dup,
                sample_ids = ids,
                check_name = f"duplicate_{id_col}",
            )]
        return [ValidationIssue.passed(
            target, f"duplicate_{id_col}",
            f"All {len(df)} '{id_col}' values are unique.",
            field=id_col,
        )]

    # --- Ordering ---

    def _check_ordering_per_group(
        self,
        df         : pd.DataFrame,
        group_col  : str,
        order_col  : str,
        id_col     : str,
        target     : str,
        check_gaps : bool = False,
    ) -> list[ValidationIssue]:
        if group_col not in df.columns or order_col not in df.columns:
            return []

        bad_monotonic : list[str] = []
        bad_dup_order : list[str] = []
        gap_groups    : list[str] = []

        for tid, grp in df.groupby(group_col, sort=False):
            grp_sorted = grp.sort_values(order_col)
            orders = grp_sorted[order_col].tolist()

            # Monotonic check
            if orders != sorted(orders):
                bad_monotonic.append(str(tid))

            # Duplicate order check
            if len(orders) != len(set(orders)):
                bad_dup_order.append(str(tid))

            # Gap detection (optional — meaningful for sentences, not chunks)
            if check_gaps and len(orders) > 1:
                expected = list(range(int(min(orders)), int(max(orders)) + 1))
                if sorted(orders) != expected:
                    gap_groups.append(str(tid))

        issues = []
        if bad_monotonic:
            issues.append(ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.ORDERING.value,
                target     = target,
                field      = order_col,
                message    = (
                    f"{len(bad_monotonic)} transcript(s) have non-monotonic "
                    f"'{order_col}'. Affected: {bad_monotonic[:3]}."
                ),
                count      = len(bad_monotonic),
                sample_ids = bad_monotonic[:self.config.max_sample_ids],
                check_name = f"ordering_{order_col}",
            ))
        else:
            issues.append(ValidationIssue.passed(
                target, f"ordering_{order_col}",
                f"All '{order_col}' values are monotonic per transcript.",
                field=order_col,
            ))

        if bad_dup_order:
            issues.append(ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.DUPLICATE.value,
                target     = target,
                field      = order_col,
                message    = (
                    f"{len(bad_dup_order)} transcript(s) have duplicate "
                    f"'{order_col}' values: {bad_dup_order[:3]}."
                ),
                count      = len(bad_dup_order),
                sample_ids = bad_dup_order[:self.config.max_sample_ids],
                check_name = f"dup_order_{order_col}",
            ))

        if check_gaps and gap_groups:
            issues.append(ValidationIssue(
                severity   = Severity.WARNING.value,
                category   = Category.COMPLETENESS.value,
                target     = target,
                field      = order_col,
                message    = (
                    f"{len(gap_groups)} transcript(s) have gaps in "
                    f"'{order_col}' (possible sentence loss). "
                    f"Affected: {gap_groups[:3]}."
                ),
                count      = len(gap_groups),
                sample_ids = gap_groups[:self.config.max_sample_ids],
                check_name = f"gaps_{order_col}",
            ))

        return issues

    # --- Label validation ---

    def _check_label_values(
        self,
        df        : pd.DataFrame,
        target    : str,
        id_col    : str,
        label_col : str = "predicted_label",
    ) -> list[ValidationIssue]:
        if label_col not in df.columns:
            return []
        bad = ~df[label_col].isin(_VALID_LABELS)
        n_bad = int(bad.sum())
        if n_bad > 0:
            ids = _sample(
                df.loc[bad, id_col] if id_col in df.columns else pd.Series(dtype=str),
                self.config.max_sample_ids,
            )
            return [ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.RANGE.value,
                target     = target,
                field      = label_col,
                message    = (
                    f"{n_bad} rows have invalid '{label_col}' values: "
                    f"{df.loc[bad, label_col].unique()[:5].tolist()}. "
                    f"Expected one of: {sorted(_VALID_LABELS)}."
                ),
                count      = n_bad,
                sample_ids = ids,
                check_name = f"label_values_{label_col}",
            )]
        return [ValidationIssue.passed(
            target, f"label_values_{label_col}",
            f"All '{label_col}' values are valid.",
            field=label_col,
        )]

    # --- Null rates ---

    def _check_null_rates(
        self,
        df          : pd.DataFrame,
        cols_to_check: frozenset[str],
        target      : str,
    ) -> list[ValidationIssue]:
        issues = []
        present_cols = [c for c in cols_to_check if c in df.columns]
        if not present_cols or df.empty:
            return []
        null_rates = df[present_cols].isnull().mean()

        for col, rate in null_rates.items():
            if rate <= self.config.max_null_rate_warning:
                # Only emit INFO for high-value required columns
                if col in _CHUNK_REQUIRED_COLS | _SENTENCE_REQUIRED_COLS | \
                           _TRANSCRIPT_REQUIRED_COLS:
                    issues.append(ValidationIssue.passed(
                        target, f"null_rate_{col}",
                        f"Null rate for '{col}' is {rate:.2%} (OK).",
                        field=col,
                    ))
            elif rate <= self.config.max_null_rate_error:
                n_null = int((df[col].isnull()).sum())
                issues.append(ValidationIssue(
                    severity   = Severity.WARNING.value,
                    category   = Category.NULL_RATE.value,
                    target     = target,
                    field      = col,
                    message    = (
                        f"Column '{col}' has {rate:.1%} null values "
                        f"({n_null} rows) — above the warning threshold "
                        f"of {self.config.max_null_rate_warning:.0%}."
                    ),
                    count      = n_null,
                    check_name = f"null_rate_{col}",
                ))
            else:
                n_null = int((df[col].isnull()).sum())
                issues.append(ValidationIssue(
                    severity   = Severity.ERROR.value,
                    category   = Category.NULL_RATE.value,
                    target     = target,
                    field      = col,
                    message    = (
                        f"Column '{col}' has {rate:.1%} null values "
                        f"({n_null} rows) — above the error threshold "
                        f"of {self.config.max_null_rate_error:.0%}."
                    ),
                    count      = n_null,
                    check_name = f"null_rate_{col}",
                ))
        return issues

    # --- Shift continuity ---

    def _check_shift_continuity(
        self,
        df    : pd.DataFrame,
        target: str,
        shift_col: str = "local_sentiment_shift",
        order_col: str = "sentence_order",
        group_col: str = "transcript_id",
        id_col   : str = "sentence_id",
    ) -> list[ValidationIssue]:
        """Verify: first sentence per transcript has shift=NaN, rest are finite."""
        if shift_col not in df.columns:
            return []
        if group_col not in df.columns or order_col not in df.columns:
            return []

        bad_first_not_nan  : list[str] = []
        bad_rest_are_nan   : list[str] = []

        for tid, grp in df.groupby(group_col, sort=False):
            grp_sorted = grp.sort_values(order_col)
            shifts = grp_sorted[shift_col].tolist()
            if not shifts:
                continue
            # First sentence should be NaN
            if not (pd.isna(shifts[0]) or shifts[0] is None):
                bad_first_not_nan.append(str(tid))
            # Remaining should not be NaN
            for s in shifts[1:]:
                if pd.isna(s):
                    bad_rest_are_nan.append(str(tid))
                    break

        issues = []
        if bad_first_not_nan:
            issues.append(ValidationIssue(
                severity   = Severity.WARNING.value,
                category   = Category.COMPLETENESS.value,
                target     = target,
                field      = shift_col,
                message    = (
                    f"{len(bad_first_not_nan)} transcript(s) have a non-NaN "
                    f"'{shift_col}' on the first sentence (expected NaN). "
                    f"Affected: {bad_first_not_nan[:3]}."
                ),
                count      = len(bad_first_not_nan),
                sample_ids = bad_first_not_nan[:self.config.max_sample_ids],
                check_name = "shift_first_nan",
            ))
        if bad_rest_are_nan:
            issues.append(ValidationIssue(
                severity   = Severity.WARNING.value,
                category   = Category.COMPLETENESS.value,
                target     = target,
                field      = shift_col,
                message    = (
                    f"{len(bad_rest_are_nan)} transcript(s) have unexpected "
                    f"NaN in '{shift_col}' beyond the first sentence. "
                    f"Affected: {bad_rest_are_nan[:3]}."
                ),
                count      = len(bad_rest_are_nan),
                sample_ids = bad_rest_are_nan[:self.config.max_sample_ids],
                check_name = "shift_interior_nan",
            ))
        if not issues:
            issues.append(ValidationIssue.passed(
                target, "shift_continuity",
                f"'{shift_col}' continuity looks correct "
                "(first=NaN, rest=finite).",
                field=shift_col,
            ))
        return issues

    # --- Linkage ---

    def _check_linkage(
        self,
        df        : pd.DataFrame,
        ref_ids   : set[str],
        key_col   : str,
        target    : str,
        label     : str,
    ) -> list[ValidationIssue]:
        if key_col not in df.columns:
            return []
        child_ids = set(df[key_col].dropna().astype(str).unique())
        orphans   = child_ids - ref_ids
        if orphans:
            return [ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.LINKAGE.value,
                target     = target,
                field      = key_col,
                message    = (
                    f"{len(orphans)} transcript_id(s) in {label} have no "
                    f"corresponding record in the reference layer: "
                    f"{sorted(orphans)[:5]}."
                ),
                count      = len(orphans),
                sample_ids = sorted(orphans)[:self.config.max_sample_ids],
                check_name = f"linkage_{label.replace('→','_to_')}",
            )]
        return [ValidationIssue.passed(
            target, f"linkage_{label.replace('→','_to_')}",
            f"All {label} transcript_id references are valid.",
            field=key_col,
        )]

    # --- Count consistency ---

    def _check_count_consistency(
        self,
        fine_df    : pd.DataFrame,
        coarse_df  : pd.DataFrame,
        fine_id_col: str,
        count_col  : str,
        target     : str,
    ) -> list[ValidationIssue]:
        """Verify that transcript-level count columns match actual fine-grained row counts."""
        if count_col not in coarse_df.columns:
            return []
        if "transcript_id" not in fine_df.columns:
            return []

        actual_counts = fine_df.groupby("transcript_id")[fine_id_col].count()
        mismatches = []
        for _, row in coarse_df.iterrows():
            tid      = str(row["transcript_id"])
            expected = int(row[count_col])
            actual   = int(actual_counts.get(tid, 0))
            if expected != actual:
                mismatches.append(
                    f"{tid}: expected={expected}, actual={actual}"
                )

        if mismatches:
            return [ValidationIssue(
                severity   = Severity.WARNING.value,
                category   = Category.AGGREGATION.value,
                target     = target,
                field      = count_col,
                message    = (
                    f"{len(mismatches)} transcript(s) have '{count_col}' "
                    f"mismatch between reported and actual counts: "
                    f"{mismatches[:3]}."
                ),
                count      = len(mismatches),
                sample_ids = mismatches[:self.config.max_sample_ids],
                check_name = f"count_consistency_{count_col}",
            )]
        return [ValidationIssue.passed(
            target, f"count_consistency_{count_col}",
            f"All '{count_col}' values match actual counts.",
            field=count_col,
        )]

    # --- Aggregation consistency ---

    def _check_aggregation_consistency(
        self,
        fine_df     : pd.DataFrame,
        coarse_df   : pd.DataFrame,
        fine_score  : str,
        coarse_score: str,
        target      : str,
        layer_label : str,
    ) -> list[ValidationIssue]:
        """Check that transcript-level scores are plausibly consistent with chunk means."""
        if fine_score not in fine_df.columns or coarse_score not in coarse_df.columns:
            return []
        if "transcript_id" not in fine_df.columns:
            return []

        skip_col = self.config.skip_column
        active_fine = _active_rows(fine_df, skip_col)
        if active_fine.empty:
            return []

        computed_means = active_fine.groupby("transcript_id")[fine_score].mean()
        mismatches = []

        for _, row in coarse_df.iterrows():
            tid      = str(row["transcript_id"])
            reported = float(row[coarse_score])
            computed = float(computed_means.get(tid, float("nan")))
            if math.isnan(computed):
                continue
            if abs(reported - computed) > self.config.aggregation_score_tolerance:
                mismatches.append(
                    f"{tid}: reported={reported:.4f}, "
                    f"chunk_mean={computed:.4f}, "
                    f"delta={abs(reported-computed):.4f}"
                )

        if mismatches:
            return [ValidationIssue(
                severity   = Severity.WARNING.value,
                category   = Category.AGGREGATION.value,
                target     = target,
                field      = coarse_score,
                message    = (
                    f"{len(mismatches)} transcript(s) have {layer_label} "
                    f"score inconsistency > {self.config.aggregation_score_tolerance}: "
                    f"{mismatches[:2]}. "
                    "(Expected if weighted aggregation differs from uniform mean.)"
                ),
                count      = len(mismatches),
                sample_ids = mismatches[:self.config.max_sample_ids],
                check_name = f"aggregation_consistency_{layer_label}",
            )]
        return [ValidationIssue.passed(
            target, f"aggregation_consistency_{layer_label}",
            f"{layer_label} score consistency OK "
            f"(all within ±{self.config.aggregation_score_tolerance}).",
            field=coarse_score,
        )]

    # --- Hierarchy levels ---

    def _check_hierarchy_levels(
        self, df: pd.DataFrame, target: str
    ) -> list[ValidationIssue]:
        if "level" not in df.columns:
            return []
        bad = ~df["level"].isin(_VALID_HIERARCHY_LEVELS)
        n_bad = int(bad.sum())
        if n_bad > 0:
            return [ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.HIERARCHY.value,
                target     = target,
                field      = "level",
                message    = (
                    f"{n_bad} rows have invalid hierarchy 'level' values: "
                    f"{df.loc[bad, 'level'].unique()[:5].tolist()}. "
                    f"Expected: {sorted(_VALID_HIERARCHY_LEVELS)}."
                ),
                count      = n_bad,
                check_name = "hierarchy_levels",
            )]
        # Verify each transcript has a transcript-level record
        has_transcript_level = (
            df[df["level"] == "transcript"]["transcript_id"]
            .unique()
        )
        all_tids = df["transcript_id"].unique()
        missing_top = set(all_tids) - set(has_transcript_level)
        issues: list[ValidationIssue] = []
        if missing_top:
            issues.append(ValidationIssue(
                severity   = Severity.WARNING.value,
                category   = Category.HIERARCHY.value,
                target     = target,
                field      = "level",
                message    = (
                    f"{len(missing_top)} transcript(s) have no top-level "
                    f"('transcript') hierarchy record: "
                    f"{sorted(missing_top)[:5]}."
                ),
                count      = len(missing_top),
                sample_ids = sorted(missing_top)[:self.config.max_sample_ids],
                check_name = "hierarchy_top_level",
            ))
        else:
            issues.append(ValidationIssue.passed(
                target, "hierarchy_levels",
                f"All level values valid; all transcripts have a top-level record.",
                field="level",
            ))
        return issues

    def _check_hierarchy_duplicates(
        self, df: pd.DataFrame, target: str
    ) -> list[ValidationIssue]:
        """Each (transcript_id, level, group_key) combination should be unique."""
        key_cols = [c for c in ["transcript_id", "level", "group_key"]
                    if c in df.columns]
        if len(key_cols) < 2:
            return []
        dup = df.duplicated(key_cols, keep=False)
        n_dup = int(dup.sum())
        if n_dup > 0:
            return [ValidationIssue(
                severity   = Severity.ERROR.value,
                category   = Category.DUPLICATE.value,
                target     = target,
                field      = "+".join(key_cols),
                message    = (
                    f"{n_dup} duplicate ({'+'.join(key_cols)}) combinations "
                    "in hierarchy DataFrame."
                ),
                count      = n_dup,
                check_name = "hierarchy_duplicates",
            )]
        return [ValidationIssue.passed(
            target, "hierarchy_duplicates",
            f"All ({'+'.join(key_cols)}) combinations are unique.",
            field="+".join(key_cols),
        )]

    # ===================================================================
    # SUMMARY BUILDER
    # ===================================================================

    def _build_summary(
        self,
        target    : str,
        df        : pd.DataFrame,
        active    : pd.DataFrame,
        issues    : list[ValidationIssue],
    ) -> ValidationSummary:
        # Sort: ERRORs first, then WARNINGs, then INFO
        _order = {
            Severity.ERROR.value  : 0,
            Severity.WARNING.value: 1,
            Severity.INFO.value   : 2,
        }
        issues_sorted = sorted(
            issues,
            key=lambda i: (_order.get(i.severity, 9), i.category, i.field),
        )
        errors   = sum(1 for i in issues if i.severity == Severity.ERROR.value)
        warnings = sum(1 for i in issues if i.severity == Severity.WARNING.value)
        infos    = sum(1 for i in issues if i.severity == Severity.INFO.value)

        summary = ValidationSummary(
            target        = target,
            total_rows    = len(df),
            valid_rows    = len(active),
            error_count   = errors,
            warning_count = warnings,
            info_count    = infos,
            checks_run    = len(issues),
            passed        = errors == 0,
            issues        = issues_sorted,
        )
        status = "✅ PASSED" if summary.passed else "❌ FAILED"
        logger.info(
            "Validation [%s] %s | E=%d W=%d | score=%.0f/100",
            target, status, errors, warnings, summary.health_score,
        )
        return summary

    # ===================================================================
    # REPORT RENDERING
    # ===================================================================

    _SEP_HEAVY = "═" * 68
    _SEP_LIGHT = "─" * 68

    def _build_text_report(
        self,
        summaries: list[ValidationSummary],
        title    : str,
    ) -> str:
        lines: list[str] = []
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines += [
            self._SEP_HEAVY,
            f"  {title}",
            f"  Generated: {ts}",
            self._SEP_HEAVY,
        ]

        # High-level dashboard
        lines.append("\n  LAYER SUMMARY")
        lines.append(self._SEP_LIGHT)
        header = f"  {'Layer':<14s}  {'Rows':>6s}  {'E':>4s}  {'W':>4s}  {'Score':>7s}  Status"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        all_passed = True
        for s in summaries:
            status = "✅ PASS" if s.passed else "❌ FAIL"
            bar = _health_bar(s.health_score)
            lines.append(
                f"  {s.target:<14s}  {s.total_rows:>6d}  "
                f"{s.error_count:>4d}  {s.warning_count:>4d}  "
                f"{s.health_score:>5.0f}/100  {status}  {bar}"
            )
            if not s.passed:
                all_passed = False

        overall = "✅ ALL LAYERS PASSED" if all_passed else "❌ VALIDATION FAILURES DETECTED"
        lines += ["", f"  Overall: {overall}", ""]

        # Per-layer detail
        for s in summaries:
            lines += [
                self._SEP_LIGHT,
                f"  [{s.target.upper()}]  rows={s.total_rows}  "
                f"valid={s.valid_rows}  "
                f"checks={s.checks_run}  "
                f"health={s.health_score:.0f}/100",
                self._SEP_LIGHT,
            ]

            # Errors
            errors = [i for i in s.issues if i.severity == Severity.ERROR.value]
            if errors:
                lines.append(f"  ❌ ERRORS ({len(errors)})")
                for iss in errors:
                    lines += _format_issue(iss)
            else:
                lines.append("  ❌ ERRORS  : none")

            # Warnings
            warnings = [i for i in s.issues if i.severity == Severity.WARNING.value]
            if warnings:
                lines.append(f"\n  ⚠  WARNINGS ({len(warnings)})")
                for iss in warnings:
                    lines += _format_issue(iss)
            else:
                lines.append("  ⚠  WARNINGS: none")

            # Info (only show if few)
            infos = [i for i in s.issues if i.severity == Severity.INFO.value]
            if infos and len(infos) <= 10:
                lines.append(f"\n  ℹ  PASSED CHECKS ({len(infos)})")
                for iss in infos:
                    lines.append(f"     ✓ [{iss.check_name}] {iss.message}")
            elif infos:
                lines.append(
                    f"\n  ℹ  PASSED CHECKS: {len(infos)} "
                    "(omitted for brevity)"
                )
            lines.append("")

        lines.append(self._SEP_HEAVY)
        return "\n".join(lines)

    def _build_json_report(
        self,
        summaries: list[ValidationSummary],
        title    : str,
    ) -> str:
        report = {
            "title"    : title,
            "generated": datetime.now(timezone.utc).isoformat(),
            "overall_passed": all(s.passed for s in summaries),
            "layers"   : [s.to_dict(include_issues=True) for s in summaries],
        }
        return json.dumps(report, indent=2, default=str)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _health_bar(score: float, width: int = 10) -> str:
    """ASCII health bar: e.g. '████░░░░░░ 70'"""
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _format_issue(iss: ValidationIssue, indent: int = 5) -> list[str]:
    """Format a single ValidationIssue into indented display lines."""
    pad = " " * indent
    lines = [
        f"{pad}[{iss.category.upper()}] {iss.target}.{iss.field}",
        f"{pad}  {iss.message}",
    ]
    if iss.count:
        lines.append(f"{pad}  count={iss.count}")
    if iss.sample_ids:
        lines.append(f"{pad}  sample_ids={iss.sample_ids}")
    return lines


# ---------------------------------------------------------------------------
# Convenience wrappers (functional API)
# ---------------------------------------------------------------------------

def quick_validate_chunks(df: pd.DataFrame, **cfg_kwargs) -> ValidationSummary:
    """One-line chunk validation with optional config overrides."""
    return SentimentValidator(ValidationConfig(**cfg_kwargs)).validate_chunks(df)


def quick_validate_sentences(df: pd.DataFrame, **cfg_kwargs) -> ValidationSummary:
    """One-line sentence validation with optional config overrides."""
    return SentimentValidator(ValidationConfig(**cfg_kwargs)).validate_sentences(df)


def quick_validate_transcripts(df: pd.DataFrame, **cfg_kwargs) -> ValidationSummary:
    """One-line transcript validation with optional config overrides."""
    return SentimentValidator(ValidationConfig(**cfg_kwargs)).validate_transcripts(df)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%H:%M:%S",
    )

    print("=" * 68)
    print("  SentimentValidator — Self-Test & Demo")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # Config validation
    # -----------------------------------------------------------------------
    print("\n[CONFIG]")
    try:
        ValidationConfig(max_token_length=600)
    except ValueError as e:
        print(f"  max_token_length > 512 rejected: ✅")
    try:
        ValidationConfig(max_null_rate_warning=0.5, max_null_rate_error=0.2)
    except ValueError as e:
        print(f"  null_rate threshold ordering rejected: ✅")
    cfg = ValidationConfig(
        prob_sum_tolerance    = 1e-4,
        max_null_rate_warning = 0.05,
        max_null_rate_error   = 0.20,
        aggregation_score_tolerance = 0.05,
        max_sample_ids        = 3,
    )
    print(f"  Config OK | prob_tol={cfg.prob_sum_tolerance:.0e} | "
          f"null_warn={cfg.max_null_rate_warning:.0%}")

    validator = SentimentValidator(cfg)

    # -----------------------------------------------------------------------
    # 1.  CHUNK VALIDATION
    # -----------------------------------------------------------------------
    print("\n[CHUNK VALIDATION]")

    def _make_chunk_df(n: int = 20, inject_errors: bool = False) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        pos = rng.dirichlet(np.ones(3), n)[:, 0]
        neg = rng.dirichlet(np.ones(3), n)[:, 2]
        neu = 1.0 - pos - neg
        neu = np.clip(neu, 0, 1)
        df = pd.DataFrame({
            "chunk_id"        : [f"AAPL_c{i:03d}" for i in range(n)],
            "transcript_id"   : ["AAPL_Q1_2025"] * (n // 2) + ["MSFT_Q2_2025"] * (n - n // 2),
            "chunk_order"     : list(range(n // 2)) + list(range(n - n // 2)),
            "positive_prob"   : np.round(pos, 6),
            "neutral_prob"    : np.round(neu, 6),
            "negative_prob"   : np.round(neg, 6),
            "sentiment_score" : np.round(pos - neg, 6),
            "confidence"      : np.round(np.maximum(pos, np.maximum(neu, neg)), 6),
            "predicted_label" : ["positive" if p > n_ and p > g else
                                  "neutral" if n_ > g else "negative"
                                  for p, n_, g in zip(pos, neu, neg)],
            "token_count"     : rng.integers(50, 450, n),
            "section_type"    : ["prepared_remarks"] * (n // 2) + ["qa"] * (n - n // 2),
            "dominant_speaker": ["Tim Cook"] * (n // 2) + ["Analyst"] * (n - n // 2),
            "model_name"      : ["ProsusAI/finbert"] * n,
            "was_skipped"     : [False] * n,
            "skip_reason"     : [""] * n,
        })
        if inject_errors:
            # Inject: prob sum error
            df.loc[0, "positive_prob"] = 0.9
            df.loc[0, "neutral_prob"]  = 0.5
            df.loc[0, "negative_prob"] = 0.3  # sum = 1.7
            # Inject: duplicate chunk_id
            df.loc[n - 1, "chunk_id"] = df.loc[0, "chunk_id"]
            # Inject: token limit violation
            df.loc[2, "token_count"] = 600
            # Inject: bad label
            df.loc[3, "predicted_label"] = "VERY_POSITIVE"
            # Inject: non-monotonic order in MSFT
            df.loc[n // 2, "chunk_order"] = 999
        return df

    chunk_clean = _make_chunk_df(n=20, inject_errors=False)
    chunk_dirty = _make_chunk_df(n=20, inject_errors=True)

    s_clean = validator.validate_chunks(chunk_clean)
    s_dirty = validator.validate_chunks(chunk_dirty)

    print(f"  Clean: {s_clean}")
    print(f"  Dirty: {s_dirty}")
    assert s_clean.passed,          "Clean chunk DF should pass"
    assert not s_dirty.passed,      "Dirty chunk DF should fail"
    assert s_dirty.error_count >= 3, f"Expected ≥ 3 errors, got {s_dirty.error_count}"
    print("  ✅ Chunk validation assertions passed.")

    # -----------------------------------------------------------------------
    # 2.  SENTENCE VALIDATION
    # -----------------------------------------------------------------------
    print("\n[SENTENCE VALIDATION]")

    def _make_sentence_df(n: int = 15, inject_errors: bool = False) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        pos  = rng.uniform(0.1, 0.8, n)
        neg  = rng.uniform(0.0, 0.5, n)
        neu  = 1.0 - pos - neg
        neu  = np.clip(neu, 0, 1)
        # Re-normalise
        tot  = pos + neg + neu
        pos, neg, neu = pos/tot, neg/tot, neu/tot

        scores = pos - neg
        shifts = [None] + list(np.diff(scores))

        n_a = n // 2
        n_b = n - n_a
        df = pd.DataFrame({
            "sentence_id"            : [f"sent_AAPL_{i:06d}_aa" for i in range(n)],
            "transcript_id"          : ["AAPL_Q1_2025"] * n_a + ["MSFT_Q2_2025"] * n_b,
            "sentence_order"         : list(range(n_a)) + list(range(n_b)),
            "sentence_score"         : np.round(scores, 6),
            "positive_prob"          : np.round(pos, 6),
            "neutral_prob"           : np.round(neu, 6),
            "negative_prob"          : np.round(neg, 6),
            "confidence"             : np.round(np.maximum(pos, np.maximum(neu, neg)), 6),
            "predicted_label"        : ["positive" if p > n_ and p > g else
                                        "neutral" if n_ > g else "negative"
                                        for p, n_, g in zip(pos, neu, neg)],
            "local_sentiment_shift"  : [float("nan") if s is None else float(s)
                                        for s in shifts],
            "rolling_sentiment_mean" : np.round(scores, 6),
            "rolling_sentiment_std"  : np.round(np.abs(scores) * 0.1, 6),
            "is_spike"               : [False] * n,
            "cumulative_sentiment"   : np.round(np.cumsum(scores), 6),
            "ema_sentiment"          : np.round(scores * 0.9, 6),
            "speaker"                : ["Tim Cook"] * n_a + ["Satya Nadella"] * n_b,
            "speaker_role"           : ["CEO"] * n,
            "section_type"           : ["prepared_remarks"] * n,
            "token_estimate"         : rng.integers(5, 50, n),
            "model_name"             : ["ProsusAI/finbert"] * n,
            "was_skipped"            : [False] * n,
            "skip_reason"            : [""] * n,
        })
        if inject_errors:
            # Duplicate sentence_id
            df.loc[n - 1, "sentence_id"] = df.loc[0, "sentence_id"]
            # Bad label
            df.loc[1, "predicted_label"] = "extreme_positive"
            # Out-of-range score
            df.loc[2, "sentence_score"] = 1.5
        return df

    sent_clean = _make_sentence_df(n=15, inject_errors=False)
    sent_dirty = _make_sentence_df(n=15, inject_errors=True)

    s_sent_clean = validator.validate_sentences(sent_clean)
    s_sent_dirty = validator.validate_sentences(sent_dirty)

    print(f"  Clean: {s_sent_clean}")
    print(f"  Dirty: {s_sent_dirty}")
    assert s_sent_clean.passed,     "Clean sentence DF should pass"
    assert not s_sent_dirty.passed, "Dirty sentence DF should fail"
    print("  ✅ Sentence validation assertions passed.")

    # -----------------------------------------------------------------------
    # 3.  TRANSCRIPT VALIDATION
    # -----------------------------------------------------------------------
    print("\n[TRANSCRIPT VALIDATION]")

    tx_df_clean = pd.DataFrame({
        "transcript_id"       : ["AAPL_Q1_2025", "MSFT_Q2_2025"],
        "sentiment_score"     : [0.35, 0.12],
        "sentiment_label"     : ["positive", "neutral"],
        "positive_probability": [0.60, 0.45],
        "neutral_probability" : [0.25, 0.35],
        "negative_probability": [0.15, 0.20],
        "chunk_count"         : [10, 8],
        "sentence_count"      : [45, 30],
        "sentiment_std"       : [0.18, 0.22],
        "sentiment_median"    : [0.37, 0.10],
        "confidence_mean"     : [0.72, 0.68],
    })
    tx_df_dirty = tx_df_clean.copy()
    # Inject: duplicate transcript
    tx_df_dirty = pd.concat([tx_df_dirty, tx_df_dirty.iloc[[0]]], ignore_index=True)
    # Inject: bad label
    tx_df_dirty.loc[0, "sentiment_label"] = "SUPER_POSITIVE"
    # Inject: prob sum error
    tx_df_dirty.loc[1, "positive_probability"] = 0.9

    s_tx_clean = validator.validate_transcripts(tx_df_clean)
    s_tx_dirty = validator.validate_transcripts(tx_df_dirty)

    print(f"  Clean: {s_tx_clean}")
    print(f"  Dirty: {s_tx_dirty}")
    assert s_tx_clean.passed,     "Clean transcript DF should pass"
    assert not s_tx_dirty.passed, "Dirty transcript DF should fail"
    print("  ✅ Transcript validation assertions passed.")

    # -----------------------------------------------------------------------
    # 4.  HIERARCHY VALIDATION
    # -----------------------------------------------------------------------
    print("\n[HIERARCHY VALIDATION]")

    hier_df_clean = pd.DataFrame({
        "transcript_id"       : ["AAPL_Q1_2025"] * 3 + ["MSFT_Q2_2025"] * 3,
        "level"               : ["transcript", "section", "speaker"] * 2,
        "group_key"           : [
            "AAPL_Q1_2025", "prepared_remarks", "Tim Cook",
            "MSFT_Q2_2025", "qa",               "Satya Nadella",
        ],
        "sentiment_score"     : [0.35, 0.42, 0.38, 0.12, 0.05, 0.15],
        "sentence_count"      : [45, 20, 25, 30, 10, 20],
        "confidence_mean"     : [0.72, 0.75, 0.70, 0.68, 0.65, 0.71],
    })
    hier_df_dirty = hier_df_clean.copy()
    # Inject: invalid level
    hier_df_dirty.loc[0, "level"] = "sub_atomic"
    # Inject: duplicate
    hier_df_dirty = pd.concat(
        [hier_df_dirty, hier_df_dirty.iloc[[1]]], ignore_index=True
    )

    s_hier_clean = validator.validate_hierarchy(hier_df_clean)
    s_hier_dirty = validator.validate_hierarchy(hier_df_dirty)

    print(f"  Clean: {s_hier_clean}")
    print(f"  Dirty: {s_hier_dirty}")
    assert s_hier_clean.passed,     "Clean hierarchy DF should pass"
    assert not s_hier_dirty.passed, "Dirty hierarchy DF should fail"
    print("  ✅ Hierarchy validation assertions passed.")

    # -----------------------------------------------------------------------
    # 5.  CROSS-LAYER VALIDATION
    # -----------------------------------------------------------------------
    print("\n[CROSS-LAYER VALIDATION]")

    # Introduce an orphan transcript in chunks not present in transcript_df
    chunk_with_orphan = chunk_clean.copy()
    chunk_with_orphan.loc[0, "transcript_id"] = "ORPHAN_T1"

    s_cross = validator.validate_cross_layer(
        chunk_df      = chunk_with_orphan,
        sentence_df   = sent_clean,
        transcript_df = tx_df_clean,
    )
    print(f"  Cross-layer: {s_cross}")
    orphan_issues = [i for i in s_cross.issues if "orphan" in i.message.lower()
                     or "linkage" in i.category.lower()]
    assert not s_cross.passed, "Cross-layer should fail with orphan transcript"
    print("  ✅ Cross-layer validation assertions passed.")

    # -----------------------------------------------------------------------
    # 6.  REPORT GENERATION
    # -----------------------------------------------------------------------
    print("\n[REPORT GENERATION]")
    report_text = validator.generate_validation_report(
        s_clean, s_sent_clean, s_tx_clean, s_hier_clean,
        title="Demo: All Clean Layers",
    )
    # Spot-check
    assert "PASSED" in report_text
    assert "CHUNK" in report_text.upper()
    assert "SENTENCE" in report_text.upper()
    print("  Text report generated ✅  "
          f"({len(report_text)} chars, {report_text.count(chr(10))} lines)")

    report_json_str = validator.generate_validation_report(
        s_clean, s_sent_clean,
        title="Demo JSON", as_json=True,
    )
    report_json = json.loads(report_json_str)
    assert "layers" in report_json
    assert report_json["overall_passed"] is True
    print(f"  JSON report generated ✅  ({len(report_json_str)} chars)")

    # -----------------------------------------------------------------------
    # 7.  DIRTY REPORT — verify failures surfaced
    # -----------------------------------------------------------------------
    report_dirty = validator.generate_validation_report(
        s_dirty, s_sent_dirty, s_tx_dirty, s_hier_dirty,
        title="Demo: Injected Failures Report",
    )
    assert "FAIL" in report_dirty
    print(f"  Dirty report generated ✅  (failures detected)")

    # -----------------------------------------------------------------------
    # 8.  to_dataframe
    # -----------------------------------------------------------------------
    print("\n[DATAFRAME EXPORT]")
    df_issues = s_dirty.to_dataframe()
    print(f"  Dirty chunk issues DataFrame: {df_issues.shape}")
    assert "severity" in df_issues.columns
    assert "message"  in df_issues.columns
    error_rows = df_issues[df_issues["severity"] == "ERROR"]
    print(f"  Error rows: {len(error_rows)}")

    # -----------------------------------------------------------------------
    # 9.  Functional API
    # -----------------------------------------------------------------------
    print("\n[FUNCTIONAL API]")
    s_qc = quick_validate_chunks(chunk_clean)
    assert s_qc.passed
    s_qs = quick_validate_sentences(sent_clean)
    assert s_qs.passed
    s_qt = quick_validate_transcripts(tx_df_clean)
    assert s_qt.passed
    print("  quick_validate_* functions: ✅")

    # -----------------------------------------------------------------------
    # 10. Edge cases
    # -----------------------------------------------------------------------
    print("\n[EDGE CASES]")
    s_empty = validator.validate_chunks(pd.DataFrame())
    assert not s_empty.passed, "Empty DataFrame should fail"
    print(f"  Empty DataFrame: {s_empty} ✅")

    s_missing_col = validator.validate_chunks(
        pd.DataFrame({"chunk_id": ["c1"], "transcript_id": ["T1"]})
    )
    assert not s_missing_col.passed, "Missing required cols should fail"
    print(f"  Missing required cols: {s_missing_col} ✅")

    # Single-row valid
    single = chunk_clean.iloc[[0]].copy()
    single["chunk_order"] = 0
    s_single = validator.validate_chunks(single)
    assert s_single.passed, "Single valid row should pass"
    print(f"  Single row: {s_single} ✅")

    # -----------------------------------------------------------------------
    # 11. Health score range check
    # -----------------------------------------------------------------------
    print("\n[HEALTH SCORE]")
    for summary, label in [
        (s_clean,       "clean chunks"),
        (s_dirty,       "dirty chunks"),
        (s_tx_clean,    "clean transcripts"),
        (s_hier_dirty,  "dirty hierarchy"),
    ]:
        bar = _health_bar(summary.health_score)
        print(f"  {label:<20s}  {summary.health_score:>5.0f}/100  {bar}")

    print("\n" + "=" * 68)
    print("  Self-test complete — all assertions passed ✅")
    print("=" * 68)
