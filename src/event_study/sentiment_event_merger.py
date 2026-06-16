"""
sentiment_event_merger.py
=========================
DAY 6 — Sentiment-to-Event Merge Engine
Earnings Call Sentiment Analyzer — Event Study Pipeline

Merges FinBERT transformer sentiment scores and Loughran-McDonald (LM)
dictionary sentiment scores onto the financial event-study dataset.
Produces a single, analysis-ready DataFrame keyed on ``transcript_id``
that unifies all NLP outputs with market reaction data.

Module boundaries
-----------------
This module does NOT:
  - compute abnormal returns         (→ abnormal_returns.py)
  - compute CAR                      (→ car_calculator.py)
  - generate event windows           (→ event_window_generator.py)
  - download market / benchmark data (→ market_data_loader.py)
  - orchestrate the pipeline         (→ event_study.py)

Architecture position
---------------------
    event_study.py (orchestrator)
          │
          ├── abnormal_returns.py
          ├── car_calculator.py
          │
          ▼
    sentiment_event_merger.py   ← THIS FILE
          │
          ▼
    validation.py  →  final event_study.parquet

Merge strategy
--------------
Both sentiment sources are LEFT-JOINed onto the event frame using
``transcript_id`` as the primary key.  This preserves every financial
event even when sentiment scores are absent (they appear as NaN and are
reported in MergeCoverageReport).  No event rows are ever dropped silently.

Python : 3.11+
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Final, Optional

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MERGE_KEY: Final[str] = "transcript_id"

# Canonical FinBERT output columns that this module expects / normalises to.
_FINBERT_CANONICAL: Final[dict[str, str]] = {
    # possible source name  →  canonical name
    "sentiment_score": "finbert_sentiment_score",
    "score": "finbert_sentiment_score",
    "finbert_score": "finbert_sentiment_score",
    "finbert_sentiment_score": "finbert_sentiment_score",
    "sentiment_label": "finbert_sentiment_label",
    "label": "finbert_sentiment_label",
    "finbert_label": "finbert_sentiment_label",
    "finbert_sentiment_label": "finbert_sentiment_label",
    "positive_score": "finbert_positive_score",
    "pos_score": "finbert_positive_score",
    "finbert_positive": "finbert_positive_score",
    "finbert_positive_score": "finbert_positive_score",
    "negative_score": "finbert_negative_score",
    "neg_score": "finbert_negative_score",
    "finbert_negative": "finbert_negative_score",
    "finbert_negative_score": "finbert_negative_score",
    "neutral_score": "finbert_neutral_score",
    "neu_score": "finbert_neutral_score",
    "finbert_neutral": "finbert_neutral_score",
    "finbert_neutral_score": "finbert_neutral_score",
}

# Canonical LM output columns.
_LM_CANONICAL: Final[dict[str, str]] = {
    "tone_score": "lm_tone_score",
    "lm_score": "lm_tone_score",
    "lm_tone": "lm_tone_score",
    "lm_tone_score": "lm_tone_score",
    "positive_count": "lm_positive_count",
    "lm_positive": "lm_positive_count",
    "lm_pos_count": "lm_positive_count",
    "lm_positive_count": "lm_positive_count",
    "negative_count": "lm_negative_count",
    "lm_negative": "lm_negative_count",
    "lm_neg_count": "lm_negative_count",
    "lm_negative_count": "lm_negative_count",
}

# All columns that must appear in the final merged output (NaN is acceptable).
_REQUIRED_OUTPUT_COLS: Final[tuple[str, ...]] = (
    "finbert_sentiment_score",
    "finbert_sentiment_label",
    "finbert_positive_score",
    "finbert_negative_score",
    "finbert_neutral_score",
    "lm_tone_score",
    "lm_positive_count",
    "lm_negative_count",
)

# Minimum acceptable sentiment coverage (fraction of events with non-NaN
# finbert_sentiment_score) below which a warning is always emitted.
_WARN_COVERAGE_THRESHOLD: Final[float] = 0.50


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentimentMergeConfig:
    """
    Configuration for SentimentEventMerger.

    Parameters
    ----------
    merge_key:
        Primary join key shared across event, FinBERT, and LM datasets.
        Must be a unique identifier per earnings call.
    how:
        pandas merge strategy for both joins.  ``'left'`` (default) keeps
        every financial event regardless of sentiment availability.
        ``'inner'`` drops events with no sentiment — use with caution.
    drop_duplicate_transcripts:
        When True, duplicate ``transcript_id`` rows in the sentiment source
        frames are resolved by keeping the first occurrence (after logging).
        When False a ValueError is raised on duplicates.
    fill_missing_finbert:
        When True, NaN FinBERT scores are filled with 0.0 and the label
        with ``'neutral'``.  When False (default), NaNs are preserved.
    fill_missing_lm:
        When True, NaN LM scores/counts are filled with 0.0.
    warn_low_coverage:
        Emit a warning when FinBERT coverage falls below
        ``_WARN_COVERAGE_THRESHOLD``.
    normalize_columns:
        Apply the canonical column-name mapping tables before merging so
        that non-standard source column names are accepted.
    """

    merge_key: str = _MERGE_KEY
    how: str = "left"
    drop_duplicate_transcripts: bool = True
    fill_missing_finbert: bool = False
    fill_missing_lm: bool = False
    warn_low_coverage: bool = True
    normalize_columns: bool = True

    def __post_init__(self) -> None:
        if self.how not in {"left", "inner", "outer", "right"}:
            raise ValueError(
                f"how='{self.how}' is not a valid pandas merge strategy."
            )
        if not self.merge_key:
            raise ValueError("merge_key must be a non-empty string.")


# ---------------------------------------------------------------------------


@dataclass
class MergeCoverageReport:
    """
    Diagnostic statistics produced after each merge stage.

    Attributes
    ----------
    source:
        Label identifying which sentiment source was merged (e.g. 'FinBERT').
    events_in:
        Number of unique transcript_ids in the event frame before merging.
    sentiment_rows_in:
        Number of rows in the sentiment source before deduplication.
    sentiment_unique:
        Number of unique transcript_ids in the sentiment source.
    duplicates_dropped:
        Number of duplicate transcript_id rows removed from the sentiment
        source before the merge.
    matched:
        Number of event transcript_ids that found a matching sentiment row.
    unmatched:
        Number of event transcript_ids with no matching sentiment row (NaN).
    coverage_rate:
        Fraction of events that received sentiment scores (0–1).
    null_rates:
        Column-level NaN rate in the merged output (column → fraction).
    row_count_after:
        Total rows in the merged frame (should equal events_in for LEFT join).
    """

    source: str = ""
    events_in: int = 0
    sentiment_rows_in: int = 0
    sentiment_unique: int = 0
    duplicates_dropped: int = 0
    matched: int = 0
    unmatched: int = 0
    row_count_after: int = 0
    null_rates: dict[str, float] = field(default_factory=dict)

    @property
    def coverage_rate(self) -> float:
        if self.events_in == 0:
            return 0.0
        return self.matched / self.events_in

    def log_summary(self) -> None:
        logger.info(
            "[%s] events_in=%d  matched=%d  unmatched=%d  "
            "coverage=%.1f%%  duplicates_dropped=%d  rows_after=%d",
            self.source,
            self.events_in,
            self.matched,
            self.unmatched,
            self.coverage_rate * 100,
            self.duplicates_dropped,
            self.row_count_after,
        )
        for col, rate in self.null_rates.items():
            if rate > 0:
                logger.debug(
                    "  [%s] null_rate  %-40s: %.1f%%",
                    self.source,
                    col,
                    rate * 100,
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "events_in": self.events_in,
            "sentiment_rows_in": self.sentiment_rows_in,
            "sentiment_unique": self.sentiment_unique,
            "duplicates_dropped": self.duplicates_dropped,
            "matched": self.matched,
            "unmatched": self.unmatched,
            "coverage_rate_pct": round(self.coverage_rate * 100, 2),
            "row_count_after": self.row_count_after,
            "null_rates": {k: round(v, 4) for k, v in self.null_rates.items()},
        }


# ---------------------------------------------------------------------------


@dataclass
class SentimentMergeResult:
    """
    Final output of a completed SentimentEventMerger.merge() call.

    Attributes
    ----------
    merged_df:
        Analysis-ready DataFrame containing all event finance columns plus
        FinBERT and LM sentiment columns.
    finbert_report:
        Coverage diagnostics for the FinBERT merge stage.
    lm_report:
        Coverage diagnostics for the LM merge stage.
    columns_added:
        List of sentiment columns that were appended to the event frame.
    total_events:
        Number of unique transcript_ids in the merged output.
    overall_finbert_coverage:
        Fraction of events with non-NaN finbert_sentiment_score.
    overall_lm_coverage:
        Fraction of events with non-NaN lm_tone_score.
    warnings_issued:
        Human-readable list of warning messages generated during the merge.
    """

    merged_df: pd.DataFrame
    finbert_report: MergeCoverageReport
    lm_report: MergeCoverageReport
    columns_added: list[str] = field(default_factory=list)
    total_events: int = 0
    overall_finbert_coverage: float = 0.0
    overall_lm_coverage: float = 0.0
    warnings_issued: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, object]:
        return {
            "total_events": self.total_events,
            "columns_added": self.columns_added,
            "finbert_coverage_pct": round(self.overall_finbert_coverage * 100, 2),
            "lm_coverage_pct": round(self.overall_lm_coverage * 100, 2),
            "warnings_count": len(self.warnings_issued),
            "finbert_report": self.finbert_report.as_dict(),
            "lm_report": self.lm_report.as_dict(),
        }

    def log_summary(self) -> None:
        logger.info(
            "SentimentMergeResult — events=%d  finbert_cov=%.1f%%  "
            "lm_cov=%.1f%%  cols_added=%d  warnings=%d",
            self.total_events,
            self.overall_finbert_coverage * 100,
            self.overall_lm_coverage * 100,
            len(self.columns_added),
            len(self.warnings_issued),
        )
        for w in self.warnings_issued:
            logger.warning("  %s", w)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _normalize_column_names(
    df: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Rename DataFrame columns according to *mapping* (source → canonical).

    Only columns present in both the DataFrame and the mapping keys are
    renamed.  All other columns are preserved unchanged.

    Parameters
    ----------
    df:
        Source DataFrame.
    mapping:
        Dict of {possible_source_name: canonical_name}.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns renamed where applicable.
    """
    rename_map = {
        col: mapping[col]
        for col in df.columns
        if col in mapping and col != mapping[col]
    }
    if rename_map:
        logger.debug("Normalising column names: %s", rename_map)
        df = df.rename(columns=rename_map)
        if df.columns.has_duplicates:
            normalised = pd.DataFrame(index=df.index)
            for col in pd.unique(df.columns):
                same_name = df.loc[:, df.columns == col]
                if same_name.shape[1] == 1:
                    normalised[col] = same_name.iloc[:, 0]
                else:
                    normalised[col] = same_name.bfill(axis=1).iloc[:, 0]
            df = normalised
    return df


def _deduplicate_on_key(
    df: pd.DataFrame,
    key: str,
    source_label: str,
    keep: str = "first",
) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows by *key*, keeping the first occurrence.

    Parameters
    ----------
    df:
        Source DataFrame potentially containing duplicate key values.
    key:
        Column name to deduplicate on.
    source_label:
        Human-readable label for log messages.
    keep:
        Which duplicate to keep (``'first'`` or ``'last'``).

    Returns
    -------
    (deduplicated_df, n_dropped)
        Tuple of the cleaned frame and the count of rows dropped.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates(subset=[key], keep=keep)
    n_dropped = n_before - len(df_clean)
    if n_dropped:
        logger.warning(
            "[%s] Dropped %d duplicate '%s' rows (kept %s occurrence).",
            source_label,
            n_dropped,
            key,
            keep,
        )
    return df_clean, n_dropped


def _compute_null_rates(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, float]:
    """Return a {column: null_fraction} mapping for *columns* in *df*."""
    rates: dict[str, float] = {}
    for col in columns:
        if col in df.columns and len(df) > 0:
            rates[col] = float(df[col].isna().mean())
    return rates


def _ensure_columns_exist(
    df: pd.DataFrame,
    required: tuple[str, ...],
    fill_numeric: float = float("nan"),
    fill_str: str = "",
) -> pd.DataFrame:
    """
    Add any *required* columns that are absent from *df*.

    Numeric columns (those with numeric canonical names based on
    ``_REQUIRED_OUTPUT_COLS``) are filled with *fill_numeric*; the label
    column is filled with *fill_str*.
    """
    label_cols = {"finbert_sentiment_label"}
    for col in required:
        if col not in df.columns:
            logger.debug("Adding missing output column '%s' as NaN/empty.", col)
            if col in label_cols:
                df[col] = fill_str if fill_str else np.nan
            else:
                df[col] = fill_numeric
    return df


# ---------------------------------------------------------------------------
# SentimentEventMerger
# ---------------------------------------------------------------------------


class SentimentEventMerger:
    """
    Merge FinBERT and Loughran-McDonald sentiment scores onto event data.

    This is the single owner of sentiment-merge logic in the project.
    It is called by ``EventStudyEngine`` (event_study.py) after CAR metrics
    have been computed and before final schema enforcement.

    The merge is intentionally non-destructive: every financial event row
    survives the merge regardless of sentiment availability.  Missing
    sentiment values appear as NaN and are reported in MergeCoverageReport.

    Parameters
    ----------
    config:
        SentimentMergeConfig controlling join strategy, deduplication
        behaviour, and fill policies.  Defaults to SentimentMergeConfig().

    Examples
    --------
    >>> merger = SentimentEventMerger()
    >>> result = merger.merge(
    ...     event_df=event_df,
    ...     finbert_df=finbert_df,
    ...     lm_df=lm_df,
    ...     merge_key="transcript_id",
    ... )
    >>> result.merged_df.columns.tolist()
    [..., 'finbert_sentiment_score', 'lm_tone_score', ...]
    """

    def __init__(
        self,
        config: Optional[SentimentMergeConfig] = None,
        **kwargs: object,
    ) -> None:
        if config is None:
            config = SentimentMergeConfig()
        self.config: SentimentMergeConfig = config
        logger.debug("SentimentEventMerger initialised — config=%s", config)

    # ------------------------------------------------------------------
    # Primary public interface  (called by EventStudyEngine)
    # ------------------------------------------------------------------

    def merge(
        self,
        event_df: pd.DataFrame,
        finbert_df: pd.DataFrame,
        lm_df: pd.DataFrame,
        merge_key: str = _MERGE_KEY,
    ) -> pd.DataFrame:
        """
        Orchestrate entry point matching the interface expected by EventStudyEngine.

        Delegates to :meth:`merge_sentiment_data` and returns the merged
        DataFrame directly (without the full SentimentMergeResult wrapper)
        so the orchestrator can proceed immediately to the next stage.

        Parameters
        ----------
        event_df:
            Event-level DataFrame (one row per earnings call event).
        finbert_df:
            FinBERT sentiment outputs keyed on *merge_key*.
        lm_df:
            LM dictionary scores keyed on *merge_key*.
        merge_key:
            Primary join key.  Overrides ``config.merge_key`` when provided.

        Returns
        -------
        pd.DataFrame
            Merged analysis-ready DataFrame.
        """
        effective_key = merge_key or self.config.merge_key
        result = self.merge_sentiment_data(
            event_df=event_df,
            finbert_df=finbert_df,
            lm_df=lm_df,
            merge_key=effective_key,
        )
        result.log_summary()
        return result.merged_df

    # ------------------------------------------------------------------

    def merge_sentiment_data(
        self,
        event_df: pd.DataFrame,
        finbert_df: pd.DataFrame,
        lm_df: pd.DataFrame,
        merge_key: Optional[str] = None,
    ) -> SentimentMergeResult:
        """
        Full merge pipeline returning a SentimentMergeResult.

        Sequence
        --------
        1. Validate event frame has the merge key.
        2. Merge FinBERT scores  → merge_finbert_scores()
        3. Merge LM scores       → merge_lm_scores()
        4. Ensure all required output columns exist.
        5. Apply optional fill policies.
        6. Validate merge integrity.
        7. Compute coverage statistics.
        8. Return SentimentMergeResult.

        Parameters
        ----------
        event_df:
            Financial event frame.  Preserved as-is; columns added via join.
        finbert_df:
            FinBERT output frame.  May use non-canonical column names —
            they are normalised automatically when ``config.normalize_columns``
            is True.
        lm_df:
            LM output frame.  Same normalisation rules as *finbert_df*.
        merge_key:
            Overrides ``config.merge_key``.

        Returns
        -------
        SentimentMergeResult
        """
        key = merge_key or self.config.merge_key
        logger.info(
            "merge_sentiment_data — events=%d  finbert_rows=%d  lm_rows=%d  key='%s'",
            len(event_df),
            len(finbert_df),
            len(lm_df),
            key,
        )

        if key not in event_df.columns:
            raise ValueError(
                f"event_df is missing the merge key column '{key}'. "
                "Ensure the event-study pipeline produced a transcript_id column."
            )

        issued_warnings: list[str] = []
        columns_before = set(event_df.columns)

        # ── Stage A: FinBERT merge ────────────────────────────────────
        merged_df, finbert_report = self.merge_finbert_scores(
            event_df=event_df,
            finbert_df=finbert_df,
            merge_key=key,
        )

        # ── Stage B: LM merge ─────────────────────────────────────────
        merged_df, lm_report = self.merge_lm_scores(
            event_df=merged_df,
            lm_df=lm_df,
            merge_key=key,
        )

        # ── Stage C: Ensure all required output columns are present ───
        merged_df = _ensure_columns_exist(merged_df, _REQUIRED_OUTPUT_COLS)

        # ── Stage D: Fill policies ────────────────────────────────────
        merged_df = self._apply_fill_policies(merged_df)

        # ── Stage E: Validate integrity ───────────────────────────────
        self.validate_merge_integrity(
            original_event_df=event_df,
            merged_df=merged_df,
            merge_key=key,
        )

        # ── Stage F: Coverage & warnings ─────────────────────────────
        finbert_cov = self._coverage_rate(merged_df, "finbert_sentiment_score")
        lm_cov = self._coverage_rate(merged_df, "lm_tone_score")

        if self.config.warn_low_coverage:
            if finbert_cov < _WARN_COVERAGE_THRESHOLD:
                msg = (
                    f"FinBERT coverage is low: {finbert_cov:.1%} of events have "
                    "finbert_sentiment_score.  Check that call_level_sentiment.parquet "
                    "transcript_ids align with the event dataset."
                )
                warnings.warn(msg, stacklevel=3)
                issued_warnings.append(msg)

            if lm_cov < _WARN_COVERAGE_THRESHOLD:
                msg = (
                    f"LM coverage is low: {lm_cov:.1%} of events have lm_tone_score. "
                    "Check that lm_scores.parquet transcript_ids align."
                )
                warnings.warn(msg, stacklevel=3)
                issued_warnings.append(msg)

        columns_added = sorted(set(merged_df.columns) - columns_before)

        result = SentimentMergeResult(
            merged_df=merged_df,
            finbert_report=finbert_report,
            lm_report=lm_report,
            columns_added=columns_added,
            total_events=merged_df[key].nunique(),
            overall_finbert_coverage=finbert_cov,
            overall_lm_coverage=lm_cov,
            warnings_issued=issued_warnings,
        )
        return result

    # ------------------------------------------------------------------

    def merge_finbert_scores(
        self,
        event_df: pd.DataFrame,
        finbert_df: pd.DataFrame,
        merge_key: Optional[str] = None,
    ) -> tuple[pd.DataFrame, MergeCoverageReport]:
        """
        Left-join FinBERT sentiment scores onto *event_df*.

        Applies column normalisation, deduplication, and the configured
        join strategy.  Returns a coverage report for diagnostics.

        Parameters
        ----------
        event_df:
            Event frame to receive sentiment columns.
        finbert_df:
            FinBERT outputs.  May have non-canonical column names.
        merge_key:
            Join key column name.

        Returns
        -------
        (merged_df, MergeCoverageReport)
        """
        key = merge_key or self.config.merge_key
        report = MergeCoverageReport(source="FinBERT")
        report.events_in = event_df[key].nunique()
        report.sentiment_rows_in = len(finbert_df)

        if len(finbert_df) == 0:
            logger.warning(
                "[FinBERT] Source frame is empty — all FinBERT columns will be NaN."
            )
            report.unmatched = report.events_in
            report.row_count_after = len(event_df)
            return event_df.copy(), report

        # Normalise column names
        if self.config.normalize_columns:
            finbert_df = _normalize_column_names(finbert_df, _FINBERT_CANONICAL)

        # Ensure merge key present
        if key not in finbert_df.columns:
            raise ValueError(
                f"FinBERT DataFrame missing merge key '{key}'. "
                f"Available columns: {list(finbert_df.columns)}"
            )

        # Deduplication
        if self.config.drop_duplicate_transcripts:
            finbert_df, n_dropped = _deduplicate_on_key(finbert_df, key, "FinBERT")
            report.duplicates_dropped = n_dropped
        else:
            dupes = finbert_df.duplicated(subset=[key]).sum()
            if dupes:
                raise ValueError(
                    f"FinBERT DataFrame contains {dupes} duplicate '{key}' rows. "
                    "Set drop_duplicate_transcripts=True to auto-resolve."
                )

        report.sentiment_unique = finbert_df[key].nunique()

        # Select only the key + canonical sentiment columns that exist
        finbert_cols = [key] + [
            c for c in finbert_df.columns
            if c in set(_FINBERT_CANONICAL.values()) and c != key
        ]
        finbert_slim = finbert_df[finbert_cols].copy()

        # Merge
        merged = event_df.merge(finbert_slim, on=key, how=self.config.how)

        # Coverage counts
        if "finbert_sentiment_score" in merged.columns:
            report.matched = int(merged["finbert_sentiment_score"].notna().sum())
        else:
            report.matched = 0
        report.unmatched = report.events_in - report.matched
        report.row_count_after = len(merged)
        report.null_rates = _compute_null_rates(
            merged,
            [c for c in _FINBERT_CANONICAL.values() if c in merged.columns],
        )
        report.log_summary()
        return merged, report

    # ------------------------------------------------------------------

    def merge_lm_scores(
        self,
        event_df: pd.DataFrame,
        lm_df: pd.DataFrame,
        merge_key: Optional[str] = None,
    ) -> tuple[pd.DataFrame, MergeCoverageReport]:
        """
        Left-join Loughran-McDonald sentiment scores onto *event_df*.

        Handles column normalisation, deduplication, and join.  The
        ``lm_tone_score`` field is the primary LM metric; word-count
        fields (lm_positive_count, lm_negative_count) are included when
        available.

        Parameters
        ----------
        event_df:
            Event frame (already enriched with FinBERT columns from
            merge_finbert_scores()).
        lm_df:
            LM dictionary outputs.
        merge_key:
            Join key column name.

        Returns
        -------
        (merged_df, MergeCoverageReport)
        """
        key = merge_key or self.config.merge_key
        report = MergeCoverageReport(source="LM")
        report.events_in = event_df[key].nunique()
        report.sentiment_rows_in = len(lm_df)

        if len(lm_df) == 0:
            logger.warning(
                "[LM] Source frame is empty — all LM columns will be NaN."
            )
            report.unmatched = report.events_in
            report.row_count_after = len(event_df)
            return event_df.copy(), report

        # Normalise
        if self.config.normalize_columns:
            lm_df = _normalize_column_names(lm_df, _LM_CANONICAL)

        if key not in lm_df.columns:
            raise ValueError(
                f"LM DataFrame missing merge key '{key}'. "
                f"Available columns: {list(lm_df.columns)}"
            )

        # Deduplication
        if self.config.drop_duplicate_transcripts:
            lm_df, n_dropped = _deduplicate_on_key(lm_df, key, "LM")
            report.duplicates_dropped = n_dropped
        else:
            dupes = lm_df.duplicated(subset=[key]).sum()
            if dupes:
                raise ValueError(
                    f"LM DataFrame contains {dupes} duplicate '{key}' rows."
                )

        report.sentiment_unique = lm_df[key].nunique()

        lm_cols = [key] + [
            c for c in lm_df.columns
            if c in set(_LM_CANONICAL.values()) and c != key
        ]
        # Guard: avoid merging a column that already exists in event_df
        # (except the key itself) to prevent _x/_y suffix collisions.
        existing_non_key = set(event_df.columns) - {key}
        lm_cols = [
            c for c in lm_cols
            if c == key or c not in existing_non_key
        ]
        lm_slim = lm_df[lm_cols].copy()

        merged = event_df.merge(lm_slim, on=key, how=self.config.how)

        if "lm_tone_score" in merged.columns:
            report.matched = int(merged["lm_tone_score"].notna().sum())
        else:
            report.matched = 0
        report.unmatched = report.events_in - report.matched
        report.row_count_after = len(merged)
        report.null_rates = _compute_null_rates(
            merged,
            [c for c in _LM_CANONICAL.values() if c in merged.columns],
        )
        report.log_summary()
        return merged, report

    # ------------------------------------------------------------------

    def validate_merge_integrity(
        self,
        original_event_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        merge_key: Optional[str] = None,
    ) -> None:
        """
        Assert that the merged DataFrame preserves all original events and
        has not introduced unexpected duplicates or row explosions.

        Checks performed
        ----------------
        1. Row count consistency: LEFT join must not lose or gain event rows.
        2. No duplicate merge-key rows in the merged output.
        3. No new NaN values in columns that were fully populated pre-merge.

        Parameters
        ----------
        original_event_df:
            The event frame before any sentiment merging.
        merged_df:
            The fully merged output.
        merge_key:
            Join key column.

        Raises
        ------
        ValueError
            When any integrity check fails (row explosion, row loss,
            unexpected duplicates).
        """
        key = merge_key or self.config.merge_key
        logger.debug("validate_merge_integrity — key='%s'", key)

        original_count = len(original_event_df)
        merged_count = len(merged_df)

        # 1. Row count
        if self.config.how == "left":
            if merged_count != original_count:
                raise ValueError(
                    f"Row count mismatch after LEFT merge: "
                    f"expected {original_count} rows but got {merged_count}. "
                    "A sentiment source likely has duplicate transcript_ids. "
                    "Enable drop_duplicate_transcripts=True."
                )

        # 2. Duplicate merge-key rows
        if key in merged_df.columns:
            dup_count = merged_df.duplicated(subset=[key]).sum()
            if dup_count:
                raise ValueError(
                    f"Merged DataFrame contains {dup_count} duplicate '{key}' rows. "
                    "This indicates a many-to-one join explosion in a sentiment source."
                )

        # 3. Pre-existing columns must not have gained NaNs
        shared_cols = [
            c for c in original_event_df.columns
            if c in merged_df.columns and c != key
        ]
        for col in shared_cols:
            original_nulls = int(original_event_df[col].isna().sum())
            merged_nulls = int(merged_df[col].isna().sum())
            if merged_nulls > original_nulls:
                logger.warning(
                    "validate_merge_integrity: column '%s' gained %d new NaN "
                    "values after merging (before=%d, after=%d). "
                    "Investigate join key alignment.",
                    col,
                    merged_nulls - original_nulls,
                    original_nulls,
                    merged_nulls,
                )

        logger.debug(
            "validate_merge_integrity passed — %d rows, 0 duplicates.",
            merged_count,
        )

    # ------------------------------------------------------------------

    def summarize_merge_results(
        self,
        result: SentimentMergeResult,
    ) -> pd.DataFrame:
        """
        Produce a tabular summary DataFrame from a SentimentMergeResult.

        Returns a single-column DataFrame indexed by metric name, useful
        for logging to parquet or printing to console.

        Parameters
        ----------
        result:
            Completed SentimentMergeResult from merge_sentiment_data().

        Returns
        -------
        pd.DataFrame
            Rows: metric names.  Column: ``value``.
        """
        rows: dict[str, object] = {
            "total_events": result.total_events,
            "columns_added": len(result.columns_added),
            "finbert_coverage_pct": round(result.overall_finbert_coverage * 100, 2),
            "lm_coverage_pct": round(result.overall_lm_coverage * 100, 2),
            "finbert_matched": result.finbert_report.matched,
            "finbert_unmatched": result.finbert_report.unmatched,
            "finbert_duplicates_dropped": result.finbert_report.duplicates_dropped,
            "lm_matched": result.lm_report.matched,
            "lm_unmatched": result.lm_report.unmatched,
            "lm_duplicates_dropped": result.lm_report.duplicates_dropped,
            "warnings_issued": len(result.warnings_issued),
        }
        for col, rate in result.finbert_report.null_rates.items():
            rows[f"null_rate_{col}"] = round(rate, 4)

        df = pd.DataFrame.from_dict(rows, orient="index", columns=["value"])
        logger.info("Merge summary:\n%s", df.to_string())
        return df

    # ------------------------------------------------------------------

    def build_analysis_dataset(
        self,
        event_df: pd.DataFrame,
        finbert_df: pd.DataFrame,
        lm_df: pd.DataFrame,
        merge_key: Optional[str] = None,
        extra_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        High-level convenience method that runs the full merge pipeline and
        returns only the columns relevant for downstream statistical analysis.

        The output contains all event-finance columns plus the canonical
        sentiment columns.  Optionally include additional columns via
        *extra_cols*.

        Parameters
        ----------
        event_df:
            Financial event frame.
        finbert_df:
            FinBERT sentiment outputs.
        lm_df:
            LM dictionary outputs.
        merge_key:
            Join key.
        extra_cols:
            Additional columns from the event frame to retain.  Pass None
            to retain all columns.

        Returns
        -------
        pd.DataFrame
            Analysis-ready merged dataset.
        """
        key = merge_key or self.config.merge_key
        result = self.merge_sentiment_data(
            event_df=event_df,
            finbert_df=finbert_df,
            lm_df=lm_df,
            merge_key=key,
        )
        df = result.merged_df

        if extra_cols is not None:
            keep = [key] + [c for c in extra_cols if c in df.columns and c != key]
            keep += [c for c in _REQUIRED_OUTPUT_COLS if c in df.columns]
            # Preserve column order; deduplicate while keeping order
            seen: set[str] = set()
            ordered: list[str] = []
            for c in keep:
                if c not in seen:
                    ordered.append(c)
                    seen.add(c)
            df = df[ordered]

        logger.info(
            "build_analysis_dataset — %d rows, %d columns",
            len(df),
            len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_fill_policies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fill_missing_finbert and fill_missing_lm policies from config.
        """
        if self.config.fill_missing_finbert:
            numeric_finbert = [
                "finbert_sentiment_score",
                "finbert_positive_score",
                "finbert_negative_score",
                "finbert_neutral_score",
            ]
            for col in numeric_finbert:
                if col in df.columns:
                    n_filled = int(df[col].isna().sum())
                    if n_filled:
                        df[col] = df[col].fillna(0.0)
                        logger.debug(
                            "fill_missing_finbert: filled %d NaN in '%s' with 0.0",
                            n_filled, col,
                        )
            if "finbert_sentiment_label" in df.columns:
                n_filled = int(df["finbert_sentiment_label"].isna().sum())
                if n_filled:
                    df["finbert_sentiment_label"] = (
                        df["finbert_sentiment_label"].fillna("neutral")
                    )
                    logger.debug(
                        "fill_missing_finbert: filled %d NaN labels with 'neutral'",
                        n_filled,
                    )

        if self.config.fill_missing_lm:
            lm_cols = ["lm_tone_score", "lm_positive_count", "lm_negative_count"]
            for col in lm_cols:
                if col in df.columns:
                    n_filled = int(df[col].isna().sum())
                    if n_filled:
                        df[col] = df[col].fillna(0.0)
                        logger.debug(
                            "fill_missing_lm: filled %d NaN in '%s' with 0.0",
                            n_filled, col,
                        )
        return df

    @staticmethod
    def _coverage_rate(df: pd.DataFrame, col: str) -> float:
        """Return fraction of non-NaN values in *col*, or 0.0 if absent."""
        if col not in df.columns or len(df) == 0:
            return 0.0
        return float(df[col].notna().mean())


# ---------------------------------------------------------------------------
# Self-test / demo block
# ---------------------------------------------------------------------------


def _make_demo_event_df(n: int = 5) -> pd.DataFrame:
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"][:n]
    rows = []
    for t in tickers:
        rows.append(
            {
                "transcript_id": f"{t}_Q1_2025",
                "ticker": t,
                "aligned_event_date": pd.Timestamp("2025-01-30"),
                "return_1d": round(float(np.random.default_rng(1).normal(0.005, 0.02)), 5),
                "abnormal_return_1d": round(float(np.random.default_rng(2).normal(0.002, 0.015)), 5),
                "car_3d": round(float(np.random.default_rng(3).normal(0.006, 0.025)), 5),
                "car_5d": round(float(np.random.default_rng(4).normal(0.010, 0.030)), 5),
                "rolling_volatility_20d": 0.22,
                "volume": 20_000_000,
            }
        )
    return pd.DataFrame(rows)


def _make_demo_finbert_df(tids: list[str], missing: int = 0) -> pd.DataFrame:
    """FinBERT frame with optional missing transcript_ids."""
    rng = np.random.default_rng(10)
    available = tids[: len(tids) - missing]
    rows = []
    for tid in available:
        pos = float(rng.random())
        neg = float(rng.random() * (1 - pos))
        neu = 1.0 - pos - neg
        rows.append(
            {
                "transcript_id": tid,
                "finbert_sentiment_score": round(pos - neg, 4),
                "finbert_sentiment_label": "positive" if pos > neg else "negative",
                "finbert_positive_score": round(pos, 4),
                "finbert_negative_score": round(neg, 4),
                "finbert_neutral_score": round(neu, 4),
            }
        )
    return pd.DataFrame(rows)


def _make_demo_lm_df(tids: list[str], use_alt_names: bool = False) -> pd.DataFrame:
    """LM frame — optionally with non-canonical column names to test normalisation."""
    rng = np.random.default_rng(20)
    rows = []
    for tid in tids:
        rows.append(
            {
                "transcript_id": tid,
                "tone_score" if use_alt_names else "lm_tone_score": round(
                    float(rng.normal(0.0, 0.3)), 4
                ),
                "positive_count" if use_alt_names else "lm_positive_count": int(
                    rng.integers(10, 80)
                ),
                "negative_count" if use_alt_names else "lm_negative_count": int(
                    rng.integers(5, 40)
                ),
            }
        )
    return pd.DataFrame(rows)


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
    logger.info("SELF-TEST / DEMO — sentiment_event_merger.py")
    logger.info("=" * 60)

    event_df = _make_demo_event_df(n=5)
    tids = event_df["transcript_id"].tolist()

    # ── Test 1: Full merge — all data present ──────────────────────────
    logger.info("\n--- Test 1: Full merge (all sentiment present) ---")
    finbert_df = _make_demo_finbert_df(tids, missing=0)
    lm_df = _make_demo_lm_df(tids)

    merger = SentimentEventMerger()
    result = merger.merge_sentiment_data(
        event_df=event_df,
        finbert_df=finbert_df,
        lm_df=lm_df,
    )
    result.log_summary()

    assert len(result.merged_df) == len(event_df), "Row count mismatch!"
    assert result.overall_finbert_coverage == 1.0, "Expected 100% FinBERT coverage"
    assert result.overall_lm_coverage == 1.0, "Expected 100% LM coverage"
    for col in _REQUIRED_OUTPUT_COLS:
        assert col in result.merged_df.columns, f"Missing output column: {col}"
    print("\n[Test 1] All assertions passed ✓")
    print(result.merged_df[["transcript_id"] + list(_REQUIRED_OUTPUT_COLS)].to_string())

    # ── Test 2: Partial FinBERT coverage (1 event missing) ────────────
    logger.info("\n--- Test 2: Partial FinBERT coverage ---")
    finbert_partial = _make_demo_finbert_df(tids, missing=2)
    result2 = merger.merge_sentiment_data(
        event_df=event_df,
        finbert_df=finbert_partial,
        lm_df=lm_df,
    )
    assert len(result2.merged_df) == len(event_df), "Row count must be preserved"
    assert result2.finbert_report.unmatched == 2, "Expected 2 unmatched"
    print(f"\n[Test 2] FinBERT coverage={result2.overall_finbert_coverage:.0%}  unmatched={result2.finbert_report.unmatched} ✓")

    # ── Test 3: Empty sentiment sources ──────────────────────────────
    logger.info("\n--- Test 3: Empty sentiment sources ---")
    result3 = merger.merge_sentiment_data(
        event_df=event_df,
        finbert_df=pd.DataFrame(),
        lm_df=pd.DataFrame(),
    )
    assert len(result3.merged_df) == len(event_df), "Rows must survive empty merge"
    assert result3.overall_finbert_coverage == 0.0
    print(f"\n[Test 3] Empty sources handled gracefully, rows={len(result3.merged_df)} ✓")

    # ── Test 4: Non-canonical LM column names ─────────────────────────
    logger.info("\n--- Test 4: Non-canonical column name normalisation ---")
    lm_alt = _make_demo_lm_df(tids, use_alt_names=True)
    result4 = merger.merge_sentiment_data(
        event_df=event_df,
        finbert_df=finbert_df,
        lm_df=lm_alt,
    )
    assert "lm_tone_score" in result4.merged_df.columns, "Column normalisation failed"
    print("\n[Test 4] Non-canonical LM column names normalised ✓")

    # ── Test 5: Duplicate transcript_ids in FinBERT source ────────────
    logger.info("\n--- Test 5: Duplicate transcript_id handling ---")
    finbert_dup = pd.concat([finbert_df, finbert_df.head(2)], ignore_index=True)
    result5 = merger.merge_sentiment_data(
        event_df=event_df,
        finbert_df=finbert_dup,
        lm_df=lm_df,
    )
    assert result5.finbert_report.duplicates_dropped == 2
    assert len(result5.merged_df) == len(event_df), "Dedup must prevent row explosion"
    print(f"\n[Test 5] Duplicates dropped={result5.finbert_report.duplicates_dropped} ✓")

    # ── Test 6: validate_merge_integrity raises on row explosion ──────
    logger.info("\n--- Test 6: Integrity check on simulated row explosion ---")
    inflated = pd.concat([event_df, event_df.head(1)], ignore_index=True)
    try:
        merger.validate_merge_integrity(
            original_event_df=event_df,
            merged_df=inflated,
            merge_key="transcript_id",
        )
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        print(f"\n[Test 6] Correctly raised ValueError: {exc} ✓")

    # ── Test 7: merge() orchestrator entry point ──────────────────────
    logger.info("\n--- Test 7: merge() entry point (EventStudyEngine interface) ---")
    merged_df = merger.merge(
        event_df=event_df,
        finbert_df=finbert_df,
        lm_df=lm_df,
    )
    assert isinstance(merged_df, pd.DataFrame)
    assert len(merged_df) == len(event_df)
    print(f"\n[Test 7] merge() returned DataFrame with {len(merged_df)} rows ✓")

    # ── Test 8: summarize_merge_results() ────────────────────────────
    logger.info("\n--- Test 8: summarize_merge_results() ---")
    summary_df = merger.summarize_merge_results(result)
    assert not summary_df.empty
    assert "finbert_coverage_pct" in summary_df.index
    print("\n[Test 8] Summary:\n", summary_df.to_string())

    # ── Test 9: fill policies ─────────────────────────────────────────
    logger.info("\n--- Test 9: fill_missing_finbert + fill_missing_lm ---")
    fill_config = SentimentMergeConfig(
        fill_missing_finbert=True,
        fill_missing_lm=True,
    )
    fill_merger = SentimentEventMerger(config=fill_config)
    result9 = fill_merger.merge_sentiment_data(
        event_df=event_df,
        finbert_df=finbert_partial,
        lm_df=lm_df,
    )
    assert result9.merged_df["finbert_sentiment_score"].isna().sum() == 0, \
        "fill_missing_finbert should eliminate NaNs"
    print("\n[Test 9] Fill policies applied — no NaN FinBERT scores ✓")

    # ── Test 10: build_analysis_dataset() ────────────────────────────
    logger.info("\n--- Test 10: build_analysis_dataset() ---")
    analysis_df = merger.build_analysis_dataset(
        event_df=event_df,
        finbert_df=finbert_df,
        lm_df=lm_df,
        extra_cols=["ticker", "car_3d", "car_5d"],
    )
    assert "ticker" in analysis_df.columns
    assert "finbert_sentiment_score" in analysis_df.columns
    print(f"\n[Test 10] build_analysis_dataset — cols={list(analysis_df.columns)} ✓")

    # ── Test 11: SentimentMergeConfig validation ──────────────────────
    logger.info("\n--- Test 11: SentimentMergeConfig validation ---")
    try:
        SentimentMergeConfig(how="bad_value")
        assert False, "Should have raised"
    except ValueError:
        pass
    try:
        SentimentMergeConfig(merge_key="")
        assert False, "Should have raised"
    except ValueError:
        pass
    print("\n[Test 11] Config validation ✓")

    print("\n" + "=" * 60)
    print("  SELF-TEST PASSED ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _run_demo()
