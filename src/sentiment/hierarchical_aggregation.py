"""
src/sentiment/hierarchical_aggregation.py
==========================================
Production-grade hierarchical sentiment aggregation for earnings-call NLP.

Position in pipeline
--------------------
    aggregation.py  ──►  hierarchical_aggregation.py  ──►  event_study / regression

Design philosophy
-----------------
This module **wraps** ``aggregation.py`` rather than duplicating its math.
All weighted-averaging logic lives in ``_aggregate_chunk_group``; this module
adds the hierarchy, drift analytics, diagnostics, and multi-DataFrame result
container that downstream event-study and regression layers need.

Aggregation layers
------------------
  1. **Transcript**   — weighted aggregate of all chunks in a call
  2. **Section**      — prepared_remarks / qa / custom sections
  3. **Speaker**      — per-named speaker across all sections
  4. **Drift**        — temporal trend analytics (scipy OLS, rolling stats,
                         momentum, acceleration, half-delta, exec-vs-analyst)

Output container (``HierarchicalAggregationResult``)
----------------------------------------------------
  ┌──────────────────────────┬────────────────────────────────────────────┐
  │ transcript_df            │ one row per transcript                     │
  │ section_df               │ one row per (transcript, section)          │
  │ speaker_df               │ one row per (transcript, speaker)          │
  │ drift_df                 │ one row per transcript — drift analytics   │
  │ diagnostics_df           │ one row per transcript — quality report    │
  └──────────────────────────┴────────────────────────────────────────────┘

Integration
-----------
* Imports ``WeightingStrategy``, ``AggregationConfig``, ``SentimentAggregator``,
  ``SentimentRecord``, and private helpers directly from ``aggregation.py``.
* Does **not** re-implement weighted averaging or probability normalisation.
* Uses ``scipy.stats.linregress`` for OLS drift / p-values / R².
* Uses ``numpy`` for vectorised rolling stats and IQR.

Author : Earnings Call Sentiment Analyzer — DAY 8
Python : 3.11+
"""

from __future__ import annotations

import json
import logging
import math
import sys
import warnings as _py_warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Import from existing aggregation.py — no duplication of math
# ---------------------------------------------------------------------------
try:
    from aggregation import (          # sibling module (src/sentiment/)
        AggregationConfig,
        AggregationError,
        SentimentAggregator,
        SentimentRecord,
        TranscriptSentiment,
        WeightingStrategy,
        _aggregate_chunk_group,
        _drift_slope,
        _MIN_CONFIDENCE,
        _normalise_probs,
        _POSITIVE_THRESHOLD,
        _NEGATIVE_THRESHOLD,
        _PROB_SUM_TOLERANCE,
        _safe_iqr,
        _safe_std,
        _score_to_label,
        _weighted_mean,
        VALID_LABELS,
    )
except ImportError:
    from src.sentiment.aggregation import (  # type: ignore[no-redef]
        AggregationConfig,
        AggregationError,
        SentimentAggregator,
        SentimentRecord,
        TranscriptSentiment,
        WeightingStrategy,
        _aggregate_chunk_group,
        _drift_slope,
        _MIN_CONFIDENCE,
        _normalise_probs,
        _POSITIVE_THRESHOLD,
        _NEGATIVE_THRESHOLD,
        _PROB_SUM_TOLERANCE,
        _safe_iqr,
        _safe_std,
        _score_to_label,
        _weighted_mean,
        VALID_LABELS,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MANAGEMENT_ROLES: frozenset[str] = frozenset({
    "ceo", "cfo", "coo", "cto", "president", "svp", "evp", "vp",
    "chief executive officer", "chief financial officer",
    "chief operating officer", "chief technology officer",
    "chairman", "managing director", "director",
})
_ANALYST_ROLES: frozenset[str] = frozenset({
    "analyst", "equity analyst", "research analyst",
    "sell-side analyst", "buy-side analyst",
})
_OPERATOR_ROLES: frozenset[str] = frozenset({"operator"})

_SPEAKER_COL_ALIASES: list[str] = ["dominant_speaker", "speaker", "speaker_name"]
_ROLE_COL_ALIASES:    list[str] = ["speaker_role", "role"]

_DEFAULT_ROLLING_WINDOW: int = 5
_MIN_DRIFT_CHUNKS: int = 3
_MIN_GROUP_CHUNKS: int = 2


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class HierarchicalAggregationConfig:
    """
    Configuration for ``HierarchicalAggregator``.

    Wraps ``AggregationConfig`` and adds hierarchy-specific settings.

    Attributes
    ----------
    aggregation : AggregationConfig
        Core weighting and threshold config (delegated to SentimentAggregator).
    rolling_window : int
        Window size for rolling mean/std on ordered chunk scores.
    min_drift_chunks : int
        Minimum chunks required to run an OLS drift regression.
    min_section_chunks : int
        Minimum chunks to emit a section aggregate row.
    min_speaker_chunks : int
        Minimum chunks to emit a speaker aggregate row.
    compute_drift : bool
        When False, skip drift analytics (faster for large batches).
    compute_speaker_analytics : bool
        When False, skip speaker-level aggregation.
    compute_section_analytics : bool
        When False, skip section-level aggregation.
    momentum_window : int
        Number of trailing chunks for the "recent slope" used in momentum
        calculation.  Must be >= 2.
    include_rolling_series : bool
        When True, store rolling mean/std/cumulative as JSON strings in
        the drift DataFrame (useful for visualisation; disable for large
        batch exports).
    """

    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    rolling_window: int = _DEFAULT_ROLLING_WINDOW
    min_drift_chunks: int = _MIN_DRIFT_CHUNKS
    min_section_chunks: int = _MIN_GROUP_CHUNKS
    min_speaker_chunks: int = _MIN_GROUP_CHUNKS
    compute_drift: bool = True
    compute_speaker_analytics: bool = True
    compute_section_analytics: bool = True
    momentum_window: int = 5
    include_rolling_series: bool = True

    def __post_init__(self) -> None:
        if self.rolling_window < 1:
            raise ValueError("rolling_window must be >= 1.")
        if self.momentum_window < 2:
            raise ValueError("momentum_window must be >= 2.")
        if self.min_drift_chunks < 2:
            raise ValueError("min_drift_chunks must be >= 2.")


# ===========================================================================
# Output dataclasses
# ===========================================================================


@dataclass
class HierarchicalSentimentRecord:
    """
    Transcript-level hierarchical sentiment record.

    Extends ``SentimentRecord`` with half-delta and positional scores.
    """

    transcript_id: str
    sentiment_score: float
    sentiment_label: str
    positive_probability: float
    neutral_probability: float
    negative_probability: float
    confidence_mean: float
    confidence_std: float
    chunk_count: int
    total_tokens: int
    sentiment_std: float
    sentiment_median: float
    sentiment_iqr: float
    sentiment_min: float
    sentiment_max: float
    strategy_used: str
    # Positional / half scores
    opening_score: float
    closing_score: float
    first_half_score: float
    second_half_score: float
    half_delta: float
    # Diagnostics
    has_warnings: bool = False
    warning_messages: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_sentiment_record(
        cls,
        rec: SentimentRecord,
        ordered_scores: "np.ndarray",
    ) -> "HierarchicalSentimentRecord":
        n = len(ordered_scores)
        q  = max(1, n // 4)
        first_q  = float(np.mean(ordered_scores[:q]))
        last_q   = float(np.mean(ordered_scores[-q:]))
        first_h  = float(np.mean(ordered_scores[: n // 2])) if n >= 2 else float(np.mean(ordered_scores))
        second_h = float(np.mean(ordered_scores[n // 2 :])) if n >= 2 else float(np.mean(ordered_scores))
        return cls(
            transcript_id=rec.transcript_id,
            sentiment_score=rec.sentiment_score,
            sentiment_label=rec.sentiment_label,
            positive_probability=rec.positive_probability,
            neutral_probability=rec.neutral_probability,
            negative_probability=rec.negative_probability,
            confidence_mean=rec.confidence_mean,
            confidence_std=rec.confidence_std,
            chunk_count=rec.chunk_count,
            total_tokens=rec.total_tokens,
            sentiment_std=rec.sentiment_std,
            sentiment_median=rec.sentiment_median,
            sentiment_iqr=rec.sentiment_iqr,
            sentiment_min=rec.sentiment_min,
            sentiment_max=rec.sentiment_max,
            strategy_used=rec.strategy_used,
            opening_score=round(first_q, 8),
            closing_score=round(last_q, 8),
            first_half_score=round(first_h, 8),
            second_half_score=round(second_h, 8),
            half_delta=round(second_h - first_h, 8),
            has_warnings=rec.has_warnings,
            warning_messages="; ".join(rec.warning_messages),
        )


@dataclass
class SectionSentimentRecord:
    """Aggregated sentiment for a single (transcript, section_type) pair."""

    transcript_id: str
    section_type: str
    sentiment_score: float
    sentiment_label: str
    positive_probability: float
    neutral_probability: float
    negative_probability: float
    confidence_mean: float
    chunk_count: int
    total_tokens: int
    sentiment_std: float
    sentiment_median: float
    drift_slope: float
    strategy_used: str
    section_rank: int = 0
    has_warnings: bool = False
    warning_messages: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerSentimentRecord:
    """Aggregated sentiment for a single (transcript, speaker) pair."""

    transcript_id: str
    speaker: str
    speaker_role: str
    speaker_type: str
    sentiment_score: float
    sentiment_label: str
    positive_probability: float
    neutral_probability: float
    negative_probability: float
    confidence_mean: float
    chunk_count: int
    total_tokens: int
    sentiment_std: float
    dominance_ratio: float
    drift_slope: float
    strategy_used: str
    speaker_rank: int = 0
    has_warnings: bool = False
    warning_messages: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriftMetrics:
    """
    Rich temporal drift analytics for a single transcript.

    OLS regression metrics come from ``scipy.stats.linregress``.
    """

    transcript_id: str
    chunk_count: int

    # OLS regression
    drift_slope: float
    drift_intercept: float
    drift_r_squared: float
    drift_p_value: float
    drift_std_error: float

    # Momentum
    recent_slope: float
    trend_momentum: float

    # Acceleration
    first_half_slope: float
    second_half_slope: float
    sentiment_acceleration: float

    # Positional scores
    opening_score: float
    closing_score: float
    first_half_score: float
    second_half_score: float
    half_delta: float

    # Cross-group deltas
    prepared_vs_qa_delta: Optional[float]
    exec_vs_analyst_delta: Optional[float]

    # Rolling series (JSON strings for Parquet compat)
    rolling_mean_json: str = "[]"
    rolling_std_json: str  = "[]"
    cumulative_json: str   = "[]"

    # Peak / trough
    peak_chunk_order: int   = 0
    trough_chunk_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def rolling_mean(self) -> list[float]:
        return json.loads(self.rolling_mean_json)

    @property
    def rolling_std(self) -> list[float]:
        return json.loads(self.rolling_std_json)

    @property
    def cumulative(self) -> list[float]:
        return json.loads(self.cumulative_json)


@dataclass
class HierarchyDiagnostics:
    """Data-quality report for a single transcript."""

    transcript_id: str
    total_chunks: int
    unique_sections: int
    unique_speakers: int
    missing_section_count: int
    missing_speaker_count: int
    duplicate_chunk_count: int
    out_of_order_chunk_count: int
    prob_sum_violation_count: int
    nan_score_count: int
    nan_confidence_count: int
    low_confidence_count: int
    single_section_transcript: bool
    all_neutral: bool
    warning_count: int
    warnings: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicate_chunk_count == 0
            and self.prob_sum_violation_count == 0
            and self.nan_score_count == 0
        )


# ===========================================================================
# Result container
# ===========================================================================


@dataclass
class HierarchicalAggregationResult:
    """
    Multi-DataFrame container returned by ``HierarchicalAggregator.aggregate()``.

    Attributes
    ----------
    transcript_df   : one row per transcript
    section_df      : one row per (transcript, section_type)
    speaker_df      : one row per (transcript, speaker)
    drift_df        : one row per transcript (DriftMetrics)
    diagnostics_df  : one row per transcript (HierarchyDiagnostics)
    config          : configuration snapshot
    warnings        : batch-level warnings
    """

    transcript_df:  pd.DataFrame
    section_df:     pd.DataFrame
    speaker_df:     pd.DataFrame
    drift_df:       pd.DataFrame
    diagnostics_df: pd.DataFrame
    config: HierarchicalAggregationConfig = field(
        default_factory=HierarchicalAggregationConfig
    )
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def transcript_ids(self) -> list[str]:
        if self.transcript_df.empty or "transcript_id" not in self.transcript_df.columns:
            return []
        return self.transcript_df["transcript_id"].tolist()

    def get_transcript(self, transcript_id: str) -> Optional[pd.Series]:
        if self.transcript_df.empty:
            return None
        mask = self.transcript_df["transcript_id"] == transcript_id
        rows = self.transcript_df[mask]
        return rows.iloc[0] if len(rows) else None

    def get_sections(self, transcript_id: str) -> pd.DataFrame:
        if self.section_df.empty:
            return pd.DataFrame()
        return self.section_df[self.section_df["transcript_id"] == transcript_id].copy()

    def get_speakers(self, transcript_id: str) -> pd.DataFrame:
        if self.speaker_df.empty:
            return pd.DataFrame()
        return self.speaker_df[self.speaker_df["transcript_id"] == transcript_id].copy()

    def get_drift(self, transcript_id: str) -> Optional[pd.Series]:
        if self.drift_df.empty or "transcript_id" not in self.drift_df.columns:
            return None
        mask = self.drift_df["transcript_id"] == transcript_id
        rows = self.drift_df[mask]
        return rows.iloc[0] if len(rows) else None

    def get_diagnostics(self, transcript_id: str) -> Optional[pd.Series]:
        if self.diagnostics_df.empty:
            return None
        mask = self.diagnostics_df["transcript_id"] == transcript_id
        rows = self.diagnostics_df[mask]
        return rows.iloc[0] if len(rows) else None

    # ------------------------------------------------------------------
    # Utility analytics
    # ------------------------------------------------------------------

    def summary_statistics(self) -> dict[str, Any]:
        """Batch-level summary for pipeline health monitoring."""
        if self.transcript_df.empty:
            return {"transcript_count": 0}
        df = self.transcript_df
        label_counts = (
            df["sentiment_label"].value_counts().to_dict()
            if "sentiment_label" in df.columns else {}
        )
        summary: dict[str, Any] = {
            "transcript_count":     len(df),
            "label_distribution":   label_counts,
            "sentiment_score_mean": round(float(df["sentiment_score"].mean()), 6),
            "sentiment_score_std":  round(float(df["sentiment_score"].std()), 6),
            "sentiment_score_min":  round(float(df["sentiment_score"].min()), 6),
            "sentiment_score_max":  round(float(df["sentiment_score"].max()), 6),
            "avg_chunk_count":      round(float(df["chunk_count"].mean()), 2),
            "total_chunks":         int(df["chunk_count"].sum()),
            "section_rows":         len(self.section_df),
            "speaker_rows":         len(self.speaker_df),
        }
        if not self.drift_df.empty and "drift_slope" in self.drift_df.columns:
            summary["avg_drift_slope"] = round(
                float(self.drift_df["drift_slope"].mean()), 6
            )
            summary["avg_trend_momentum"] = round(
                float(self.drift_df["trend_momentum"].mean()), 6
            )
        if not self.diagnostics_df.empty:
            clean = (
                self.diagnostics_df["duplicate_chunk_count"].eq(0)
                & self.diagnostics_df["nan_score_count"].eq(0)
            )
            summary["clean_transcript_rate"] = round(float(clean.mean()), 4)
        summary["warning_count"] = len(self.warnings)
        return summary

    def speaker_rankings(
        self,
        transcript_id: Optional[str] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Return speakers ranked by sentiment_score (descending)."""
        if self.speaker_df.empty:
            return pd.DataFrame()
        df = (
            self.speaker_df[self.speaker_df["transcript_id"] == transcript_id]
            if transcript_id else self.speaker_df
        )
        if df.empty:
            return df
        return (
            df.sort_values("sentiment_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def section_rankings(
        self,
        transcript_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return sections ranked by sentiment_score (descending)."""
        if self.section_df.empty:
            return pd.DataFrame()
        df = (
            self.section_df[self.section_df["transcript_id"] == transcript_id]
            if transcript_id else self.section_df
        )
        return df.sort_values("sentiment_score", ascending=False).reset_index(drop=True)

    def top_drift_events(
        self,
        n: int = 5,
        by: str = "abs_slope",
    ) -> pd.DataFrame:
        """
        Return transcripts with the most extreme drift.

        Parameters
        ----------
        n  : number of transcripts to return
        by : "abs_slope" | "slope" | "half_delta" | "acceleration"
        """
        if self.drift_df.empty:
            return pd.DataFrame()
        df = self.drift_df.copy()
        col_map = {
            "abs_slope":    "drift_slope",
            "slope":        "drift_slope",
            "half_delta":   "half_delta",
            "acceleration": "sentiment_acceleration",
        }
        sort_col = col_map.get(by, "drift_slope")
        if sort_col not in df.columns:
            return df.head(n).reset_index(drop=True)
        if by == "abs_slope":
            df["_sort_key"] = df[sort_col].abs()
        else:
            df["_sort_key"] = df[sort_col]
        return (
            df.sort_values("_sort_key", ascending=False)
            .drop(columns=["_sort_key"])
            .head(n)
            .reset_index(drop=True)
        )

    def prepared_vs_qa_summary(self) -> pd.DataFrame:
        """
        DataFrame with prepared_remarks and qa scores side-by-side per transcript.
        """
        if self.section_df.empty:
            return pd.DataFrame()
        pivot = (
            self.section_df[
                self.section_df["section_type"].isin(["prepared_remarks", "qa"])
            ]
            .pivot_table(
                index="transcript_id",
                columns="section_type",
                values="sentiment_score",
                aggfunc="first",
            )
            .reset_index()
        )
        pivot.columns.name = None
        if "prepared_remarks" in pivot.columns and "qa" in pivot.columns:
            pivot["prep_vs_qa_delta"] = pivot["prepared_remarks"] - pivot["qa"]
        return pivot

    def exec_vs_analyst_summary(self) -> pd.DataFrame:
        """
        DataFrame with management and analyst mean scores per transcript.
        """
        if self.speaker_df.empty:
            return pd.DataFrame()
        df = self.speaker_df.copy()
        mgmt    = df[df["speaker_type"] == "management"].groupby("transcript_id")["sentiment_score"].mean()
        analyst = df[df["speaker_type"] == "analyst"].groupby("transcript_id")["sentiment_score"].mean()
        result  = pd.DataFrame({"management_score": mgmt, "analyst_score": analyst})
        if not result.empty:
            result["exec_vs_analyst_delta"] = result["management_score"] - result["analyst_score"]
        return result.reset_index()

    def volatility_summary(self) -> pd.DataFrame:
        """
        Return transcript-level volatility metrics sorted by sentiment_std.
        """
        if self.transcript_df.empty:
            return pd.DataFrame()
        cols = ["transcript_id", "sentiment_std", "sentiment_iqr",
                "sentiment_min", "sentiment_max", "chunk_count"]
        avail = [c for c in cols if c in self.transcript_df.columns]
        return (
            self.transcript_df[avail]
            .sort_values("sentiment_std", ascending=False)
            .reset_index(drop=True)
        )

    def speaker_dominance_report(self) -> pd.DataFrame:
        """
        Return speakers sorted by dominance_ratio (highest contribution first).
        """
        if self.speaker_df.empty:
            return pd.DataFrame()
        cols = ["transcript_id", "speaker", "speaker_type",
                "dominance_ratio", "sentiment_score", "chunk_count"]
        avail = [c for c in cols if c in self.speaker_df.columns]
        return (
            self.speaker_df[avail]
            .sort_values("dominance_ratio", ascending=False)
            .reset_index(drop=True)
        )

    def drift_summary(self) -> pd.DataFrame:
        """
        Compact drift summary suitable for event-study merge.
        """
        if self.drift_df.empty:
            return pd.DataFrame()
        key_cols = [
            "transcript_id", "drift_slope", "drift_r_squared", "drift_p_value",
            "trend_momentum", "sentiment_acceleration", "half_delta",
            "prepared_vs_qa_delta", "exec_vs_analyst_delta",
            "opening_score", "closing_score",
        ]
        avail = [c for c in key_cols if c in self.drift_df.columns]
        return self.drift_df[avail].copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save_parquet(
        self,
        directory: str | Path,
        prefix: str = "hierarchical",
    ) -> dict[str, Path]:
        """
        Save all five DataFrames to Parquet in *directory*.

        Returns a dict mapping DataFrame name -> saved Path.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name, df in self._named_frames():
            if not df.empty:
                p = d / f"{prefix}_{name}.parquet"
                df.to_parquet(p, index=False)
                logger.info("Saved %s (%d rows) -> %s", name, len(df), p)
                paths[name] = p
        return paths

    def save_csv(
        self,
        directory: str | Path,
        prefix: str = "hierarchical",
    ) -> dict[str, Path]:
        """Save all five DataFrames to CSV in *directory*."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name, df in self._named_frames():
            if not df.empty:
                p = d / f"{prefix}_{name}.csv"
                df.to_csv(p, index=False)
                logger.info("Saved %s CSV (%d rows) -> %s", name, len(df), p)
                paths[name] = p
        return paths

    def save_jsonl(
        self,
        path: str | Path,
        include: Optional[list[str]] = None,
    ) -> Path:
        """
        Save a combined JSONL where each line is one transcript's full record.

        Parameters
        ----------
        include : list of frame names to embed per transcript.
                  Defaults to ["transcript", "section", "speaker", "drift"].
        """
        include = include or ["transcript", "section", "speaker", "drift"]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for tid in self.transcript_ids():
            obj: dict[str, Any] = {"transcript_id": tid}
            if "transcript" in include:
                row = self.get_transcript(tid)
                obj["transcript"] = row.to_dict() if row is not None else {}
            if "section" in include:
                obj["sections"] = self.get_sections(tid).to_dict("records")
            if "speaker" in include:
                obj["speakers"] = self.get_speakers(tid).to_dict("records")
            if "drift" in include:
                d = self.get_drift(tid)
                obj["drift"] = d.to_dict() if d is not None else {}
            lines.append(json.dumps(obj, default=str))
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved JSONL (%d transcripts) -> %s", len(lines), p)
        return p

    def to_dataframe(self, level: str = "transcript") -> pd.DataFrame:
        """
        Return a single DataFrame by level name.

        Parameters
        ----------
        level : "transcript" | "section" | "speaker" | "drift" | "diagnostics"
        """
        mapping = {
            "transcript":  self.transcript_df,
            "section":     self.section_df,
            "speaker":     self.speaker_df,
            "drift":       self.drift_df,
            "diagnostics": self.diagnostics_df,
        }
        if level not in mapping:
            raise ValueError(
                f"level must be one of {list(mapping.keys())}, got '{level}'."
            )
        return mapping[level]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _named_frames(self) -> list[tuple[str, pd.DataFrame]]:
        return [
            ("transcript",  self.transcript_df),
            ("section",     self.section_df),
            ("speaker",     self.speaker_df),
            ("drift",       self.drift_df),
            ("diagnostics", self.diagnostics_df),
        ]


# ===========================================================================
# Validation
# ===========================================================================


class HierarchyValidator:
    """
    Validates chunk-level input and hierarchical output quality.

    All methods are static so the class can be used without instantiation
    by ``sentiment_validation.py``.
    """

    @staticmethod
    def validate_input(df: pd.DataFrame) -> list[str]:
        """Validate chunk-level input DataFrame. Returns list of error strings."""
        errors: list[str] = []
        if df is None or df.empty:
            errors.append("Input DataFrame is empty.")
            return errors

        required = [
            "transcript_id", "chunk_order",
            "positive_prob", "neutral_prob", "negative_prob",
            "sentiment_score", "confidence",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
            return errors

        # NaN checks
        for col in ["sentiment_score", "confidence",
                    "positive_prob", "neutral_prob", "negative_prob"]:
            n = int(df[col].isna().sum())
            if n:
                errors.append(f"Column '{col}' has {n} NaN values.")

        # Probability bounds
        for col in ["positive_prob", "neutral_prob", "negative_prob"]:
            oob = int(((df[col] < 0) | (df[col] > 1)).sum())
            if oob:
                errors.append(f"Column '{col}' has {oob} values outside [0, 1].")

        # Probability sum per row
        prob_sum = df["positive_prob"] + df["neutral_prob"] + df["negative_prob"]
        bad = int((abs(prob_sum - 1.0) > _PROB_SUM_TOLERANCE).sum())
        if bad:
            errors.append(
                f"{bad} rows have probability sums deviating from 1.0 "
                f"by > {_PROB_SUM_TOLERANCE}."
            )

        # chunk_order sorting and duplicates per transcript
        for tid, grp in df.groupby("transcript_id", sort=False):
            orders = grp["chunk_order"].tolist()
            if orders != sorted(orders):
                errors.append(f"Transcript '{tid}': chunk_order is not sorted ascending.")
            if "chunk_id" in df.columns:
                dup = int(grp.duplicated(subset=["chunk_id"]).sum())
                if dup:
                    errors.append(f"Transcript '{tid}': {dup} duplicate chunk_id values.")

        # Confidence bounds
        conf_oob = int(((df["confidence"] < 0) | (df["confidence"] > 1)).sum())
        if conf_oob:
            errors.append(f"confidence has {conf_oob} values outside [0, 1].")

        # Label validation
        if "predicted_label" in df.columns:
            invalid = ~df["predicted_label"].isin(VALID_LABELS)
            n_inv = int(invalid.sum())
            if n_inv:
                bad_vals = df.loc[invalid, "predicted_label"].unique().tolist()[:5]
                errors.append(f"{n_inv} rows have invalid predicted_label: {bad_vals}.")

        return errors

    @staticmethod
    def validate_transcript_df(df: pd.DataFrame) -> list[str]:
        """Validate transcript-level output DataFrame."""
        errors: list[str] = []
        if df.empty:
            return errors
        required = [
            "transcript_id", "sentiment_score", "sentiment_label",
            "positive_probability", "neutral_probability", "negative_probability",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"transcript_df missing columns: {missing}")
            return errors
        dup = int(df.duplicated(subset=["transcript_id"]).sum())
        if dup:
            errors.append(f"transcript_df has {dup} duplicate transcript_id rows.")
        prob_sum = (
            df["positive_probability"]
            + df["neutral_probability"]
            + df["negative_probability"]
        )
        bad = int((abs(prob_sum - 1.0) > 1e-3).sum())
        if bad:
            errors.append(f"{bad} transcript rows have malformed probability sums.")
        oob = int(((df["sentiment_score"].abs() > 1.0 + 1e-6)).sum())
        if oob:
            errors.append(f"{oob} transcript rows have sentiment_score outside [-1, 1].")
        if "sentiment_label" in df.columns:
            invalid = ~df["sentiment_label"].isin(VALID_LABELS)
            if int(invalid.sum()):
                errors.append("transcript_df contains invalid sentiment_label values.")
        return errors

    @staticmethod
    def validate_section_df(df: pd.DataFrame) -> list[str]:
        """Validate section-level output DataFrame."""
        errors: list[str] = []
        if df.empty:
            return errors
        required = ["transcript_id", "section_type", "sentiment_score", "chunk_count"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"section_df missing columns: {missing}")
            return errors
        dup = int(df.duplicated(subset=["transcript_id", "section_type"]).sum())
        if dup:
            errors.append(f"section_df has {dup} duplicate (transcript, section) pairs.")
        return errors

    @staticmethod
    def validate_speaker_df(df: pd.DataFrame) -> list[str]:
        """Validate speaker-level output DataFrame."""
        errors: list[str] = []
        if df.empty:
            return errors
        required = ["transcript_id", "speaker", "sentiment_score", "chunk_count"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"speaker_df missing columns: {missing}")
            return errors
        dup = int(df.duplicated(subset=["transcript_id", "speaker"]).sum())
        if dup:
            errors.append(f"speaker_df has {dup} duplicate (transcript, speaker) pairs.")
        return errors

    @staticmethod
    def validate_drift_df(df: pd.DataFrame) -> list[str]:
        """Validate drift metrics DataFrame."""
        errors: list[str] = []
        if df.empty:
            return errors
        if "transcript_id" not in df.columns:
            errors.append("drift_df missing 'transcript_id' column.")
            return errors
        dup = int(df.duplicated(subset=["transcript_id"]).sum())
        if dup:
            errors.append(f"drift_df has {dup} duplicate transcript_id rows.")
        if "drift_r_squared" in df.columns:
            oob = int(((df["drift_r_squared"].dropna() < 0) | (df["drift_r_squared"].dropna() > 1)).sum())
            if oob:
                errors.append(f"drift_df has {oob} R-squared values outside [0, 1].")
        return errors

    def validate_result(self, result: HierarchicalAggregationResult) -> list[str]:
        """Run all output validations and return combined error list."""
        errors: list[str] = []
        errors.extend(self.validate_transcript_df(result.transcript_df))
        errors.extend(self.validate_section_df(result.section_df))
        errors.extend(self.validate_speaker_df(result.speaker_df))
        errors.extend(self.validate_drift_df(result.drift_df))
        return errors


# ===========================================================================
# Internal helper functions
# ===========================================================================


def _ols_regression(
    scores: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    OLS regression of scores on sequential index using scipy.

    Returns
    -------
    slope, intercept, r_squared, p_value, std_error
    """
    n = len(scores)
    if n < 2:
        mu = float(np.mean(scores)) if n else 0.0
        return 0.0, mu, 0.0, float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    with _py_warnings.catch_warnings():
        _py_warnings.simplefilter("ignore")
        res = sp_stats.linregress(x, scores)
    r2 = max(0.0, min(1.0, float(res.rvalue ** 2)))
    return float(res.slope), float(res.intercept), r2, float(res.pvalue), float(res.stderr)


def _rolling_stats(
    scores: np.ndarray,
    window: int,
) -> tuple[list[float], list[float]]:
    """Rolling mean and std with min-periods=1."""
    n = len(scores)
    means: list[float] = []
    stds:  list[float] = []
    for i in range(n):
        lo  = max(0, i - window + 1)
        buf = scores[lo: i + 1]
        means.append(float(np.mean(buf)))
        stds.append(float(np.std(buf, ddof=0)) if len(buf) > 1 else 0.0)
    return means, stds


def _infer_speaker_type(role: str) -> str:
    """Return "management" | "analyst" | "operator" | "unknown"."""
    r = role.strip().lower()
    if r in _OPERATOR_ROLES or "operator" in r:
        return "operator"
    if any(k in r for k in _MANAGEMENT_ROLES):
        return "management"
    if any(k in r for k in _ANALYST_ROLES):
        return "analyst"
    return "unknown"


def _resolve_role_col(df: pd.DataFrame) -> Optional[str]:
    for col in _ROLE_COL_ALIASES:
        if col in df.columns:
            return col
    return None


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Light normalisation of chunk input DataFrame."""
    df = df.copy()

    # Column aliases
    for target, aliases in [
        ("positive_prob",    ["positive_probability"]),
        ("neutral_prob",     ["neutral_probability"]),
        ("negative_prob",    ["negative_probability"]),
        ("dominant_speaker", ["speaker", "speaker_name"]),
    ]:
        if target not in df.columns:
            for a in aliases:
                if a in df.columns:
                    df[target] = df[a]
                    break

    # Defaults for optional columns
    n = len(df)
    if "section_type"     not in df.columns: df["section_type"]     = "unknown"
    if "dominant_speaker" not in df.columns: df["dominant_speaker"] = "unknown"
    if "speaker_role"     not in df.columns: df["speaker_role"]     = "unknown"
    if "token_count"      not in df.columns: df["token_count"]      = 1
    if "chunk_order"      not in df.columns: df["chunk_order"]      = range(n)
    if "chunk_id"         not in df.columns: df["chunk_id"]         = [f"chunk_{i}" for i in range(n)]
    if "predicted_label"  not in df.columns: df["predicted_label"]  = "neutral"

    # Coerce numeric
    for col in ["positive_prob", "neutral_prob", "negative_prob",
                "sentiment_score", "confidence", "token_count", "chunk_order"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Clamp
    for col in ["positive_prob", "neutral_prob", "negative_prob", "confidence"]:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)
    df["token_count"] = df["token_count"].clip(lower=1)

    # String normalisation
    for col in ["transcript_id", "section_type", "dominant_speaker",
                "speaker_role", "predicted_label"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip()
                .replace({"nan": "unknown", "": "unknown", "None": "unknown"})
            )
    return df


def _build_diagnostics(
    transcript_id: str,
    df: pd.DataFrame,
) -> HierarchyDiagnostics:
    """Build HierarchyDiagnostics for one transcript's chunk DataFrame."""
    n        = len(df)
    dup_cnt  = int(df.duplicated(subset=["chunk_id"]).sum()) if "chunk_id" in df.columns else 0
    orders   = df["chunk_order"].tolist()
    oos_cnt  = sum(1 for a, b in zip(orders, orders[1:]) if b < a)
    prob_sum = df["positive_prob"] + df["neutral_prob"] + df["negative_prob"]
    prob_viol= int((abs(prob_sum - 1.0) > _PROB_SUM_TOLERANCE).sum())
    nan_sc   = int(df["sentiment_score"].isna().sum())
    nan_cf   = int(df["confidence"].isna().sum())
    low_cf   = int((df["confidence"] < 0.4).sum())
    miss_sec = int((df["section_type"]     == "unknown").sum()) if "section_type"     in df.columns else 0
    miss_spk = int((df["dominant_speaker"] == "unknown").sum())
    u_secs   = int(df["section_type"].nunique())     if "section_type"     in df.columns else 0
    u_spks   = int(df["dominant_speaker"].nunique())
    all_neu  = bool(
        (df["predicted_label"] == "neutral").all()
        if "predicted_label" in df.columns else False
    )
    wrns: list[str] = []
    if dup_cnt:  wrns.append(f"{dup_cnt} duplicate chunk_ids.")
    if prob_viol: wrns.append(f"{prob_viol} probability sum violations.")
    if nan_sc:   wrns.append(f"{nan_sc} NaN sentiment scores.")
    if all_neu:  wrns.append("All chunks labelled neutral.")
    return HierarchyDiagnostics(
        transcript_id=transcript_id,
        total_chunks=n,
        unique_sections=u_secs,
        unique_speakers=u_spks,
        missing_section_count=miss_sec,
        missing_speaker_count=miss_spk,
        duplicate_chunk_count=dup_cnt,
        out_of_order_chunk_count=oos_cnt,
        prob_sum_violation_count=prob_viol,
        nan_score_count=nan_sc,
        nan_confidence_count=nan_cf,
        low_confidence_count=low_cf,
        single_section_transcript=(u_secs <= 1),
        all_neutral=all_neu,
        warning_count=len(wrns),
        warnings="|".join(wrns),
    )


def _compute_drift(
    transcript_id: str,
    df: pd.DataFrame,
    section_scores: dict[str, float],
    speaker_type_scores: dict[str, float],
    cfg: HierarchicalAggregationConfig,
) -> DriftMetrics:
    """Compute full DriftMetrics for one transcript."""
    scores = df["sentiment_score"].fillna(0.0).values.astype(float)
    n      = len(scores)

    if n == 0:
        return _empty_drift(transcript_id)

    # OLS full
    slope, intercept, r2, p_val, std_err = _ols_regression(scores)

    # Half-slopes
    mid     = n // 2
    s1_sl   = _ols_regression(scores[:mid])[0]   if mid >= 2      else 0.0
    s2_sl   = _ols_regression(scores[mid:])[0]   if (n - mid) >= 2 else 0.0
    accel   = s2_sl - s1_sl

    # Momentum
    mw           = min(cfg.momentum_window, n)
    recent_slope = _ols_regression(scores[-mw:])[0] if mw >= 2 else 0.0
    momentum     = recent_slope - slope

    # Positional
    q        = max(1, n // 4)
    opening  = float(np.mean(scores[:q]))
    closing  = float(np.mean(scores[-q:]))
    first_h  = float(np.mean(scores[:mid])) if mid >= 1 else float(np.mean(scores))
    second_h = float(np.mean(scores[mid:])) if (n - mid) >= 1 else float(np.mean(scores))

    # Cross-group deltas
    prep = section_scores.get("prepared_remarks")
    qa   = section_scores.get("qa")
    pvq  = round(prep - qa, 8) if (prep is not None and qa is not None) else None

    mgmt    = speaker_type_scores.get("management")
    analyst = speaker_type_scores.get("analyst")
    eva     = round(mgmt - analyst, 8) if (mgmt is not None and analyst is not None) else None

    # Rolling series
    rm_list = rs_list = cum_list = []
    if cfg.include_rolling_series:
        rm_list, rs_list = _rolling_stats(scores, cfg.rolling_window)
        cum_list = list(np.cumsum(scores).round(8))

    peak_idx   = int(np.argmax(scores))
    trough_idx = int(np.argmin(scores))
    chunk_orders = df["chunk_order"].values
    peak_order   = int(chunk_orders[peak_idx])
    trough_order = int(chunk_orders[trough_idx])

    return DriftMetrics(
        transcript_id=transcript_id,
        chunk_count=n,
        drift_slope=round(slope, 8),
        drift_intercept=round(intercept, 8),
        drift_r_squared=round(r2, 8),
        drift_p_value=round(p_val, 8) if not math.isnan(p_val) else float("nan"),
        drift_std_error=round(std_err, 8) if not math.isnan(std_err) else float("nan"),
        recent_slope=round(recent_slope, 8),
        trend_momentum=round(momentum, 8),
        first_half_slope=round(s1_sl, 8),
        second_half_slope=round(s2_sl, 8),
        sentiment_acceleration=round(accel, 8),
        opening_score=round(opening, 8),
        closing_score=round(closing, 8),
        first_half_score=round(first_h, 8),
        second_half_score=round(second_h, 8),
        half_delta=round(second_h - first_h, 8),
        prepared_vs_qa_delta=pvq,
        exec_vs_analyst_delta=eva,
        rolling_mean_json=json.dumps([round(v, 6) for v in rm_list]),
        rolling_std_json=json.dumps([round(v, 6) for v in rs_list]),
        cumulative_json=json.dumps([round(v, 6) for v in cum_list]),
        peak_chunk_order=peak_order,
        trough_chunk_order=trough_order,
    )


def _empty_drift(transcript_id: str) -> DriftMetrics:
    return DriftMetrics(
        transcript_id=transcript_id, chunk_count=0,
        drift_slope=0.0, drift_intercept=0.0, drift_r_squared=0.0,
        drift_p_value=float("nan"), drift_std_error=float("nan"),
        recent_slope=0.0, trend_momentum=0.0,
        first_half_slope=0.0, second_half_slope=0.0, sentiment_acceleration=0.0,
        opening_score=0.0, closing_score=0.0,
        first_half_score=0.0, second_half_score=0.0, half_delta=0.0,
        prepared_vs_qa_delta=None, exec_vs_analyst_delta=None,
    )


# ===========================================================================
# HierarchicalAggregator
# ===========================================================================


class HierarchicalAggregator:
    """
    Hierarchical sentiment aggregation engine.

    Wraps ``SentimentAggregator`` for base weighted-averaging and adds
    section, speaker, drift, and diagnostic layers on top.

    Parameters
    ----------
    config : HierarchicalAggregationConfig, optional

    Examples
    --------
    ::

        agg = HierarchicalAggregator()
        result = agg.aggregate(chunk_scores_df)
        result.save_csv("data/processed/sentiment/")
        print(result.summary_statistics())
    """

    def __init__(
        self,
        config: Optional[HierarchicalAggregationConfig] = None,
    ) -> None:
        self.config    = config or HierarchicalAggregationConfig()
        self._base_agg = SentimentAggregator(config=self.config.aggregation)
        self._validator = HierarchyValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        chunk_df: pd.DataFrame,
        validate_input: bool = True,
    ) -> HierarchicalAggregationResult:
        """
        Run the full hierarchical aggregation pipeline.

        Parameters
        ----------
        chunk_df : pd.DataFrame
            Chunk-level FinBERT output.  Required columns:
            transcript_id, chunk_order, sentiment_score, confidence,
            positive_prob, neutral_prob, negative_prob.
            Optional: section_type, dominant_speaker, speaker_role,
            token_count, chunk_id, predicted_label.
        validate_input : bool

        Returns
        -------
        HierarchicalAggregationResult
        """
        global_warnings: list[str] = []

        if chunk_df is None or chunk_df.empty:
            logger.warning("Empty chunk_df — returning empty result.")
            return self._empty_result(["Empty chunk_df."])

        if validate_input:
            for e in self._validator.validate_input(chunk_df):
                logger.warning("Input validation: %s", e)
                global_warnings.append(f"INPUT: {e}")

        chunk_df = _normalise_df(chunk_df)

        # Base aggregation — delegates weighting to aggregation.py
        base_result = self._base_agg.aggregate(chunk_df, validate_input=False)

        transcript_records: list[HierarchicalSentimentRecord] = []
        section_records:    list[SectionSentimentRecord]       = []
        speaker_records:    list[SpeakerSentimentRecord]        = []
        drift_records:      list[DriftMetrics]                  = []
        diag_records:       list[HierarchyDiagnostics]          = []

        for tid, ts in base_result.transcripts.items():
            grp = (
                chunk_df[chunk_df["transcript_id"] == tid]
                .sort_values("chunk_order")
                .reset_index(drop=True)
            )
            if grp.empty:
                continue

            ordered_scores = grp["sentiment_score"].values.astype(float)

            # Transcript record
            t_rec = HierarchicalSentimentRecord.from_sentiment_record(
                ts.transcript_aggregate, ordered_scores
            )
            transcript_records.append(t_rec)

            # Diagnostics
            diag = _build_diagnostics(tid, grp)
            diag_records.append(diag)
            if diag.warning_count:
                global_warnings.extend(
                    [w for w in diag.warnings.split("|") if w]
                )

            # Section records
            section_scores: dict[str, float] = {}
            if self.config.compute_section_analytics:
                sec_recs, section_scores = self._build_section_records(
                    tid, grp, ts, global_warnings
                )
                section_records.extend(sec_recs)

            # Speaker records
            speaker_type_scores: dict[str, float] = {}
            if self.config.compute_speaker_analytics:
                spk_recs, speaker_type_scores = self._build_speaker_records(
                    tid, grp, ts, global_warnings
                )
                speaker_records.extend(spk_recs)

            # Drift
            if self.config.compute_drift and len(grp) >= self.config.min_drift_chunks:
                drift = _compute_drift(
                    tid, grp, section_scores, speaker_type_scores, self.config
                )
                drift_records.append(drift)

        result = HierarchicalAggregationResult(
            transcript_df  = self._to_df(transcript_records),
            section_df     = self._to_df(section_records),
            speaker_df     = self._to_df(speaker_records),
            drift_df       = self._to_df(drift_records),
            diagnostics_df = self._to_df(diag_records),
            config         = self.config,
            warnings       = global_warnings,
        )

        for e in self._validator.validate_result(result):
            logger.error("Output validation: %s", e)
            result.warnings.append(f"OUTPUT: {e}")

        logger.info(
            "HierarchicalAggregator complete — %d transcripts, "
            "%d section rows, %d speaker rows, %d drift rows, %d warnings.",
            len(transcript_records), len(section_records),
            len(speaker_records), len(drift_records), len(global_warnings),
        )
        return result

    # ------------------------------------------------------------------
    # Section aggregation
    # ------------------------------------------------------------------

    def _build_section_records(
        self,
        tid: str,
        grp: pd.DataFrame,
        ts: TranscriptSentiment,
        global_warnings: list[str],
    ) -> tuple[list[SectionSentimentRecord], dict[str, float]]:
        records: list[SectionSentimentRecord] = []
        section_scores: dict[str, float] = {}

        if "section_type" not in grp.columns:
            return records, section_scores

        for section, sec_df in grp.groupby("section_type", sort=True):
            section = str(section)
            if section in ("", "unknown", "nan") or len(sec_df) < self.config.min_section_chunks:
                continue

            base = ts.section_aggregates.get(section)
            if base is None:
                try:
                    base = _aggregate_chunk_group(
                        sec_df.to_dict("records"), section, "section",
                        tid, self.config.aggregation,
                    )
                except AggregationError as exc:
                    global_warnings.append(f"[{tid}] Section '{section}': {exc}")
                    continue

            ordered = sec_df.sort_values("chunk_order")["sentiment_score"].values.astype(float)
            ds = _drift_slope(ordered.tolist()) if len(ordered) >= 2 else 0.0
            section_scores[section] = base.sentiment_score

            records.append(SectionSentimentRecord(
                transcript_id=tid, section_type=section,
                sentiment_score=base.sentiment_score,
                sentiment_label=base.sentiment_label,
                positive_probability=base.positive_probability,
                neutral_probability=base.neutral_probability,
                negative_probability=base.negative_probability,
                confidence_mean=base.confidence_mean,
                chunk_count=base.chunk_count,
                total_tokens=base.total_tokens,
                sentiment_std=base.sentiment_std,
                sentiment_median=base.sentiment_median,
                drift_slope=round(ds, 8),
                strategy_used=base.strategy_used,
                has_warnings=base.has_warnings,
                warning_messages="; ".join(base.warning_messages),
            ))

        if records:
            sorted_r  = sorted(records, key=lambda r: r.sentiment_score, reverse=True)
            rank_map  = {r.section_type: i + 1 for i, r in enumerate(sorted_r)}
            for r in records:
                r.section_rank = rank_map[r.section_type]

        return records, section_scores

    # ------------------------------------------------------------------
    # Speaker aggregation
    # ------------------------------------------------------------------

    def _build_speaker_records(
        self,
        tid: str,
        grp: pd.DataFrame,
        ts: TranscriptSentiment,
        global_warnings: list[str],
    ) -> tuple[list[SpeakerSentimentRecord], dict[str, float]]:
        records: list[SpeakerSentimentRecord] = []
        spk_type_acc: dict[str, list[float]] = {}

        role_col     = _resolve_role_col(grp) or "speaker_role"
        total_tokens = max(1, int(grp["token_count"].sum()))

        for speaker, spk_df in grp.groupby("dominant_speaker", sort=True):
            speaker = str(speaker)
            if speaker in ("", "unknown", "nan") or len(spk_df) < self.config.min_speaker_chunks:
                continue

            base = ts.speaker_aggregates.get(speaker)
            role = (
                str(spk_df[role_col].mode().iloc[0])
                if role_col in spk_df.columns and len(spk_df) else "unknown"
            )

            if base is None:
                try:
                    base = _aggregate_chunk_group(
                        spk_df.to_dict("records"), speaker, "speaker",
                        tid, self.config.aggregation,
                    )
                except AggregationError as exc:
                    global_warnings.append(f"[{tid}] Speaker '{speaker}': {exc}")
                    continue

            spk_type  = _infer_speaker_type(role)
            ordered   = spk_df.sort_values("chunk_order")["sentiment_score"].values.astype(float)
            ds        = _drift_slope(ordered.tolist()) if len(ordered) >= 2 else 0.0
            dom_ratio = round(base.total_tokens / total_tokens, 6)

            records.append(SpeakerSentimentRecord(
                transcript_id=tid, speaker=speaker,
                speaker_role=role, speaker_type=spk_type,
                sentiment_score=base.sentiment_score,
                sentiment_label=base.sentiment_label,
                positive_probability=base.positive_probability,
                neutral_probability=base.neutral_probability,
                negative_probability=base.negative_probability,
                confidence_mean=base.confidence_mean,
                chunk_count=base.chunk_count,
                total_tokens=base.total_tokens,
                sentiment_std=base.sentiment_std,
                dominance_ratio=dom_ratio,
                drift_slope=round(ds, 8),
                strategy_used=base.strategy_used,
                has_warnings=base.has_warnings,
                warning_messages="; ".join(base.warning_messages),
            ))

            spk_type_acc.setdefault(spk_type, []).append(base.sentiment_score)

        if records:
            sorted_r = sorted(records, key=lambda r: r.sentiment_score, reverse=True)
            rank_map = {r.speaker: i + 1 for i, r in enumerate(sorted_r)}
            for r in records:
                r.speaker_rank = rank_map[r.speaker]

        type_means = {t: float(np.mean(v)) for t, v in spk_type_acc.items()}
        return records, type_means

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_df(records: list[Any]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in records]).reset_index(drop=True)

    @staticmethod
    def _empty_result(warnings: list[str]) -> HierarchicalAggregationResult:
        return HierarchicalAggregationResult(
            transcript_df=pd.DataFrame(), section_df=pd.DataFrame(),
            speaker_df=pd.DataFrame(), drift_df=pd.DataFrame(),
            diagnostics_df=pd.DataFrame(), warnings=warnings,
        )


# ===========================================================================
# Public factory
# ===========================================================================


def create_hierarchical_aggregator(
    strategy: str | WeightingStrategy = WeightingStrategy.HYBRID,
    rolling_window: int = _DEFAULT_ROLLING_WINDOW,
    compute_drift: bool = True,
    include_rolling_series: bool = True,
    **config_kwargs: Any,
) -> HierarchicalAggregator:
    """
    Convenience factory for ``HierarchicalAggregator``.

    Parameters
    ----------
    strategy : weighting strategy ("uniform" | "confidence" | "token" | "hybrid")
    rolling_window : rolling window for drift smoothing
    compute_drift : enable drift analytics
    include_rolling_series : embed rolling JSON in drift_df
    **config_kwargs : forwarded to HierarchicalAggregationConfig
    """
    if isinstance(strategy, str):
        strategy = WeightingStrategy(strategy.lower())
    cfg = HierarchicalAggregationConfig(
        aggregation=AggregationConfig(strategy=strategy),
        rolling_window=rolling_window,
        compute_drift=compute_drift,
        include_rolling_series=include_rolling_series,
        **config_kwargs,
    )
    return HierarchicalAggregator(config=cfg)


# ===========================================================================
# Backward-compatible hierarchy API
# ===========================================================================
#
# The project test suite and earlier pipeline stages use a tree-shaped API
# (HierarchyLevel, AggregationNode, HierarchicalBatchResult).  The production
# implementation above exports analytics DataFrames.  The compatibility layer
# below keeps both contracts available from the same correctly placed module.


class HierarchyLevel(str, Enum):
    TRANSCRIPT = "transcript"
    SECTION = "section"
    SPEAKER = "speaker"
    SPEAKER_IN_SECTION = "speaker_in_section"
    CHUNK = "chunk"
    SENTENCE = "sentence"

    @property
    def depth(self) -> int:
        order = {
            HierarchyLevel.TRANSCRIPT: 0,
            HierarchyLevel.SECTION: 1,
            HierarchyLevel.SPEAKER: 1,
            HierarchyLevel.SPEAKER_IN_SECTION: 2,
            HierarchyLevel.CHUNK: 3,
            HierarchyLevel.SENTENCE: 4,
        }
        return order[self]


class SpeakerType(str, Enum):
    MANAGEMENT = "management"
    ANALYST = "analyst"
    OPERATOR = "operator"
    UNKNOWN = "unknown"


def _infer_speaker_type(role: str) -> SpeakerType:  # type: ignore[override]
    value = str(role or "").strip().lower()
    if value in _OPERATOR_ROLES or "operator" in value:
        return SpeakerType.OPERATOR
    if value in _ANALYST_ROLES or "analyst" in value:
        return SpeakerType.ANALYST
    if value in _MANAGEMENT_ROLES or any(token in value for token in ("chief", "ceo", "cfo", "coo", "cto")):
        return SpeakerType.MANAGEMENT
    return SpeakerType.UNKNOWN


def _ols_slope(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    return float(np.polyfit(x, np.asarray(vals, dtype=float), 1)[0])


def _safe_median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    return float(np.median(vals)) if vals else 0.0


@dataclass
class HierarchicalAggregationConfig:  # type: ignore[no-redef]
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    use_confidence_weight: bool = True
    use_token_weight: bool = True
    compute_cross_section_speakers: bool = True
    include_sentence_nodes: bool = False
    positive_threshold: float = _POSITIVE_THRESHOLD
    negative_threshold: float = _NEGATIVE_THRESHOLD
    rolling_window: int = _DEFAULT_ROLLING_WINDOW
    compute_drift: bool = True
    include_rolling_series: bool = True
    min_drift_chunks: int = _MIN_DRIFT_CHUNKS
    min_section_chunks: int = 1
    min_speaker_chunks: int = 1

    def __post_init__(self) -> None:
        if self.positive_threshold <= self.negative_threshold:
            raise ValueError("positive_threshold must be > negative_threshold.")
        strategy = WeightingStrategy.HYBRID
        if self.use_confidence_weight and not self.use_token_weight:
            strategy = WeightingStrategy.CONFIDENCE
        elif self.use_token_weight and not self.use_confidence_weight:
            strategy = WeightingStrategy.TOKEN
        elif not self.use_confidence_weight and not self.use_token_weight:
            strategy = WeightingStrategy.UNIFORM
        self.aggregation.strategy = strategy
        self.aggregation.positive_threshold = self.positive_threshold
        self.aggregation.negative_threshold = self.negative_threshold
        self.aggregation.min_section_chunks = self.min_section_chunks
        self.aggregation.min_speaker_chunks = self.min_speaker_chunks


@dataclass
class AggregationNode:
    node_id: str
    transcript_id: str
    hierarchy_level: HierarchyLevel
    hierarchy_key: str
    positive_probability: float
    neutral_probability: float
    negative_probability: float
    sentiment_score: float
    sentiment_label: str
    observation_count: int
    parent_node_id: str | None = None
    children_node_ids: list[str] = field(default_factory=list)
    speaker_type: SpeakerType = SpeakerType.UNKNOWN
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hierarchy_level"] = self.hierarchy_level.value
        data["speaker_type"] = self.speaker_type.value
        return data


@dataclass
class HierarchicalSentimentResult:
    transcript_id: str
    root_node_id: str
    nodes: dict[str, AggregationNode]

    @property
    def root(self) -> AggregationNode:
        return self.nodes[self.root_node_id]

    def get_node(self, node_id: str) -> AggregationNode | None:
        return self.nodes.get(node_id)

    def nodes_at_level(self, level: HierarchyLevel) -> list[AggregationNode]:
        return [n for n in self.nodes.values() if n.hierarchy_level == level]

    def transcript_score(self) -> float:
        return float(self.root.sentiment_score)

    def section_score(self, section: str) -> float | None:
        for node in self.nodes_at_level(HierarchyLevel.SECTION):
            if node.hierarchy_key == section:
                return float(node.sentiment_score)
        return None

    def speaker_score(self, speaker: str) -> float | None:
        for node in self.nodes_at_level(HierarchyLevel.SPEAKER):
            if node.hierarchy_key == speaker:
                return float(node.sentiment_score)
        return None

    def prepared_vs_qa_delta(self) -> float | None:
        prep = self.section_score("prepared_remarks")
        qa = self.section_score("qa")
        return None if prep is None or qa is None else float(prep - qa)

    def management_sentiment(self) -> float | None:
        vals = [n.sentiment_score for n in self.nodes_at_level(HierarchyLevel.SPEAKER)
                if n.speaker_type == SpeakerType.MANAGEMENT]
        return float(np.mean(vals)) if vals else None

    def analyst_sentiment(self) -> float | None:
        vals = [n.sentiment_score for n in self.nodes_at_level(HierarchyLevel.SPEAKER)
                if n.speaker_type == SpeakerType.ANALYST]
        return float(np.mean(vals)) if vals else None

    def optimism_decay(self) -> float | None:
        chunks = sorted(self.nodes_at_level(HierarchyLevel.CHUNK), key=lambda n: n.hierarchy_key)
        if len(chunks) < 2:
            return None
        mid = len(chunks) // 2
        return float(np.mean([n.sentiment_score for n in chunks[:mid]]) -
                     np.mean([n.sentiment_score for n in chunks[mid:]]))

    def analytics_summary(self) -> dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "sentiment_score": self.root.sentiment_score,
            "sentiment_label": self.root.sentiment_label,
            "prepared_remarks_score": self.section_score("prepared_remarks"),
            "qa_score": self.section_score("qa"),
            "prepared_vs_qa_delta": self.prepared_vs_qa_delta(),
            "management_sentiment": self.management_sentiment(),
            "analyst_sentiment": self.analyst_sentiment(),
            "optimism_decay": self.optimism_decay(),
            "node_count": len(self.nodes),
        }

    def iter_level_dfs(self) -> Any:
        yield self.root
        visited = {self.root_node_id}
        queue = list(self.root.children_node_ids)
        while queue:
            node_id = queue.pop(0)
            if node_id in visited or node_id not in self.nodes:
                continue
            visited.add(node_id)
            node = self.nodes[node_id]
            yield node
            queue.extend(node.children_node_ids)

    def children_of(self, node_id: str) -> list[AggregationNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[cid] for cid in node.children_node_ids if cid in self.nodes]

    def ancestors_of(self, node_id: str) -> list[AggregationNode]:
        ancestors: list[AggregationNode] = []
        node = self.nodes.get(node_id)
        while node is not None and node.parent_node_id:
            parent = self.nodes.get(node.parent_node_id)
            if parent is None:
                break
            ancestors.append(parent)
            node = parent
        return ancestors

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([node.to_dict() for node in self.nodes.values()])

    def save_parquet(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(out, index=False)
        return out

    def save_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(out, index=False)
        return out


@dataclass
class HierarchicalBatchResult:
    results: dict[str, HierarchicalSentimentResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def total_transcripts(self) -> int:
        return len(self.results)

    def get(self, transcript_id: str) -> HierarchicalSentimentResult | None:
        return self.results.get(transcript_id)

    def to_full_dataframe(self) -> pd.DataFrame:
        frames = [r.to_dataframe() for r in self.results.values()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def to_level_dataframe(self, level: HierarchyLevel) -> pd.DataFrame:
        df = self.to_full_dataframe()
        if df.empty:
            return df
        return df[df["hierarchy_level"] == level.value].reset_index(drop=True)

    def to_analytics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.analytics_summary() for r in self.results.values()])

    def save_parquet(self, path: str | Path, level: HierarchyLevel | None = None) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_level_dataframe(level) if level else self.to_full_dataframe()
        df.to_parquet(out, index=False)
        return out

    def save_csv(self, path: str | Path, level: HierarchyLevel | None = None) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_level_dataframe(level) if level else self.to_full_dataframe()
        df.to_csv(out, index=False)
        return out

    def save_jsonl(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for result in self.results.values():
                fh.write(json.dumps(result.analytics_summary()) + "\n")
        return out

    def save_analytics_parquet(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.to_analytics_dataframe().to_parquet(out, index=False)
        return out


def _compat_record_to_node(
    record: SentimentRecord,
    *,
    node_id: str,
    level: HierarchyLevel,
    hierarchy_key: str,
    parent_node_id: str | None = None,
    speaker_type: SpeakerType = SpeakerType.UNKNOWN,
) -> AggregationNode:
    return AggregationNode(
        node_id=node_id,
        transcript_id=record.transcript_id,
        hierarchy_level=level,
        hierarchy_key=hierarchy_key,
        positive_probability=record.positive_probability,
        neutral_probability=record.neutral_probability,
        negative_probability=record.negative_probability,
        sentiment_score=record.sentiment_score,
        sentiment_label=record.sentiment_label,
        observation_count=record.chunk_count,
        parent_node_id=parent_node_id,
        speaker_type=speaker_type,
        token_count=record.total_tokens,
    )


def _compat_sentence_row(row: dict[str, Any]) -> dict[str, Any]:
    score = float(row.get("sentiment_score", 0.0))
    if {"positive_prob", "neutral_prob", "negative_prob"}.issubset(row):
        pos = float(row.get("positive_prob", 0.0))
        neu = float(row.get("neutral_prob", 0.0))
        neg = float(row.get("negative_prob", 0.0))
    else:
        pos = max(score, 0.0)
        neg = max(-score, 0.0)
        neu = max(0.0, 1.0 - pos - neg)
    pos, neu, neg = _normalise_probs(pos, neu, neg)
    row = dict(row)
    row["positive_prob"] = pos
    row["neutral_prob"] = neu
    row["negative_prob"] = neg
    row["confidence"] = float(row.get("confidence", max(pos, neu, neg)))
    row["token_count"] = int(row.get("token_count", max(len(str(row.get("sentence_text", "")).split()), 1)))
    return row


class HierarchicalValidator:  # type: ignore[no-redef]
    @staticmethod
    def validate_chunk_df(df: pd.DataFrame) -> list[str]:
        if df.empty:
            return ["chunk_df is empty"]
        required = ["transcript_id", "chunk_order", "sentiment_score",
                    "positive_prob", "neutral_prob", "negative_prob", "confidence"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return [f"chunk_df missing columns: {missing}"]
        errors: list[str] = []
        sums = df[["positive_prob", "neutral_prob", "negative_prob"]].sum(axis=1)
        if ((sums - 1.0).abs() > _PROB_SUM_TOLERANCE).any():
            errors.append("chunk_df probability sums deviate from 1.0")
        for tid, grp in df.groupby("transcript_id", sort=False):
            orders = grp["chunk_order"].tolist()
            if orders != sorted(orders):
                errors.append(f"transcript '{tid}': chunk_order is not sorted ascending.")
            if grp.duplicated(subset=["chunk_order"]).any():
                errors.append(f"transcript '{tid}': duplicate chunk_order values.")
        return errors

    @staticmethod
    def validate_sentence_df(df: pd.DataFrame) -> list[str]:
        if df.empty:
            return []
        required = ["transcript_id", "sentence_order", "sentiment_score"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return [f"sentence_df missing columns: {missing}"]
        errors: list[str] = []
        for tid, grp in df.groupby("transcript_id", sort=False):
            orders = grp["sentence_order"].tolist()
            if orders != sorted(orders):
                errors.append(f"sentence transcript '{tid}': sentence_order not sorted.")
        return errors

    @staticmethod
    def validate_result(result: HierarchicalSentimentResult) -> list[str]:
        if result is None or not result.nodes:
            return ["HierarchicalSentimentResult has no nodes."]
        errors: list[str] = []
        for node in result.nodes.values():
            prob_sum = node.positive_probability + node.neutral_probability + node.negative_probability
            if abs(prob_sum - 1.0) > 1e-3:
                errors.append(f"node '{node.node_id}' probability sum is {prob_sum:.6f}")
            if node.parent_node_id and node.parent_node_id not in result.nodes:
                errors.append(f"node '{node.node_id}' missing parent '{node.parent_node_id}'")
            for child in node.children_node_ids:
                if child not in result.nodes:
                    errors.append(f"node '{node.node_id}' missing child '{child}'")
        return errors


class HierarchicalAggregator:  # type: ignore[no-redef]
    def __init__(self, config: HierarchicalAggregationConfig | None = None) -> None:
        self.config = config or HierarchicalAggregationConfig()

    def aggregate(
        self,
        chunk_df: pd.DataFrame,
        sentence_df: pd.DataFrame | None = None,
        validate_input: bool = True,
    ) -> HierarchicalBatchResult:
        if chunk_df.empty:
            return HierarchicalBatchResult()
        if validate_input:
            errors = HierarchicalValidator.validate_chunk_df(chunk_df)
            if errors:
                raise HierarchyValidationError("; ".join(errors))

        results: dict[str, HierarchicalSentimentResult] = {}
        for tid, group in chunk_df.groupby("transcript_id", sort=True):
            tid = str(tid)
            rows = group.sort_values("chunk_order").to_dict("records")
            nodes: dict[str, AggregationNode] = {}

            root_record = _aggregate_chunk_group(rows, tid, "transcript", tid, self.config.aggregation)
            root = _compat_record_to_node(
                root_record,
                node_id=tid,
                level=HierarchyLevel.TRANSCRIPT,
                hierarchy_key=tid,
            )
            nodes[root.node_id] = root

            if "section_type" in group.columns:
                for section, sec_df in group.groupby("section_type", sort=True):
                    sec_rows = sec_df.sort_values("chunk_order").to_dict("records")
                    rec = _aggregate_chunk_group(sec_rows, str(section), "section", tid, self.config.aggregation)
                    node_id = f"{tid}||section||{section}"
                    nodes[node_id] = _compat_record_to_node(
                        rec,
                        node_id=node_id,
                        level=HierarchyLevel.SECTION,
                        hierarchy_key=str(section),
                        parent_node_id=root.node_id,
                    )
                    root.children_node_ids.append(node_id)

            speaker_col = next((c for c in _SPEAKER_COL_ALIASES if c in group.columns), None)
            role_col = _resolve_role_col(group)
            if speaker_col:
                for speaker, sp_df in group.groupby(speaker_col, sort=True):
                    sp_rows = sp_df.sort_values("chunk_order").to_dict("records")
                    rec = _aggregate_chunk_group(sp_rows, str(speaker), "speaker", tid, self.config.aggregation)
                    role = str(sp_df[role_col].iloc[0]) if role_col and role_col in sp_df.columns else str(speaker)
                    node_id = f"{tid}||speaker||{speaker}"
                    nodes[node_id] = _compat_record_to_node(
                        rec,
                        node_id=node_id,
                        level=HierarchyLevel.SPEAKER,
                        hierarchy_key=str(speaker),
                        parent_node_id=root.node_id,
                        speaker_type=_infer_speaker_type(role),
                    )
                    root.children_node_ids.append(node_id)

            for _, row in group.sort_values("chunk_order").iterrows():
                chunk_key = int(row.get("chunk_order", 0))
                node_id = f"{tid}||chunk||{chunk_key}"
                pos, neu, neg = _normalise_probs(
                    float(row.get("positive_prob", 0.0)),
                    float(row.get("neutral_prob", 0.0)),
                    float(row.get("negative_prob", 0.0)),
                )
                score = pos - neg
                label = _score_to_label(score, self.config.positive_threshold, self.config.negative_threshold)
                parent_id = root.node_id
                section = row.get("section_type")
                if pd.notna(section):
                    sec_id = f"{tid}||section||{section}"
                    if sec_id in nodes:
                        parent_id = sec_id
                node = AggregationNode(
                    node_id=node_id,
                    transcript_id=tid,
                    hierarchy_level=HierarchyLevel.CHUNK,
                    hierarchy_key=str(chunk_key),
                    positive_probability=pos,
                    neutral_probability=neu,
                    negative_probability=neg,
                    sentiment_score=score,
                    sentiment_label=label,
                    observation_count=1,
                    parent_node_id=parent_id,
                    token_count=int(row.get("token_count", 0)),
                )
                nodes[node_id] = node
                nodes[parent_id].children_node_ids.append(node_id)

            if self.config.include_sentence_nodes and sentence_df is not None and not sentence_df.empty:
                sent_subset = sentence_df[sentence_df["transcript_id"].astype(str) == tid].copy()
                if not sent_subset.empty:
                    for _, srow in sent_subset.sort_values("sentence_order").iterrows():
                        row = _compat_sentence_row(srow.to_dict())
                        order = int(row.get("sentence_order", 0))
                        pos, neu, neg = _normalise_probs(
                            float(row["positive_prob"]), float(row["neutral_prob"]), float(row["negative_prob"])
                        )
                        score = pos - neg
                        node_id = f"{tid}||sentence||{order}"
                        node = AggregationNode(
                            node_id=node_id,
                            transcript_id=tid,
                            hierarchy_level=HierarchyLevel.SENTENCE,
                            hierarchy_key=str(order),
                            positive_probability=pos,
                            neutral_probability=neu,
                            negative_probability=neg,
                            sentiment_score=score,
                            sentiment_label=_score_to_label(score, self.config.positive_threshold, self.config.negative_threshold),
                            observation_count=1,
                            parent_node_id=root.node_id,
                            token_count=int(row.get("token_count", 1)),
                        )
                        nodes[node_id] = node
                        root.children_node_ids.append(node_id)

            results[tid] = HierarchicalSentimentResult(
                transcript_id=tid,
                root_node_id=root.node_id,
                nodes=nodes,
            )

        return HierarchicalBatchResult(results=results)


def create_hierarchical_aggregator(  # type: ignore[no-redef]
    strategy: str | WeightingStrategy = WeightingStrategy.HYBRID,
    rolling_window: int = _DEFAULT_ROLLING_WINDOW,
    compute_drift: bool = True,
    include_rolling_series: bool = True,
    include_sentence_nodes: bool = False,
    **config_kwargs: Any,
) -> HierarchicalAggregator:
    if isinstance(strategy, str):
        strategy = WeightingStrategy(strategy.lower())
    cfg = HierarchicalAggregationConfig(
        aggregation=AggregationConfig(strategy=strategy),
        rolling_window=rolling_window,
        compute_drift=compute_drift,
        include_rolling_series=include_rolling_series,
        include_sentence_nodes=include_sentence_nodes,
        **config_kwargs,
    )
    return HierarchicalAggregator(config=cfg)


# ===========================================================================
# Self-test / demo
# ===========================================================================


def _make_demo_df(
    transcript_ids: list[str] | None = None,
    chunks_per: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    if transcript_ids is None:
        transcript_ids = ["AAPL_Q1_2025", "NVDA_Q2_2025", "MSFT_Q3_2025"]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    spk_map: dict[str, list[tuple[str, str]]] = {
        "AAPL_Q1_2025": [("Tim Cook", "CEO"), ("Luca Maestri", "CFO"), ("Analyst", "Analyst")],
        "NVDA_Q2_2025": [("Jensen Huang", "CEO"), ("Colette Kress", "CFO"), ("Analyst", "Analyst")],
    }
    default_spk = [("Speaker A", "CEO"), ("Speaker B", "CFO"), ("Analyst", "Analyst")]
    for tid in transcript_ids:
        bias = 0.30 if "AAPL" in tid else (-0.20 if "NVDA" in tid else 0.05)
        spks = spk_map.get(tid, default_spk)
        sections = (
            ["prepared_remarks"] * (chunks_per // 2)
            + ["qa"] * (chunks_per - chunks_per // 2)
        )
        for order, section in enumerate(sections):
            in_qa = section == "qa"
            spk, role = (
                spks[2] if in_qa and order % 3 == 0
                else spks[order % 2]
            )
            pos = float(np.clip(0.40 + bias + rng.normal(0, 0.11), 0.02, 0.96))
            neg = float(np.clip(0.25 - bias + rng.normal(0, 0.08), 0.02, 0.96))
            neu = max(0.02, 1.0 - pos - neg)
            t   = pos + neu + neg
            pos, neu, neg = pos / t, neu / t, neg / t
            score = pos - neg
            conf  = float(max(pos, neu, neg))
            tc    = int(rng.integers(60, 450))
            label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
            rows.append({
                "chunk_id":         f"chunk_{tid}_{order:04d}",
                "transcript_id":    tid,
                "chunk_order":      order,
                "section_type":     section,
                "dominant_speaker": spk,
                "speaker_role":     role,
                "positive_prob":    round(pos,   8),
                "neutral_prob":     round(neu,   8),
                "negative_prob":    round(neg,   8),
                "sentiment_score":  round(score, 8),
                "confidence":       round(conf,  8),
                "predicted_label":  label,
                "token_count":      tc,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(name)s | %(message)s",
    )
    SEP  = "=" * 72
    SEP2 = "-" * 72

    print(f"\n{SEP}")
    print("  hierarchical_aggregation.py  —  Self-Test / Demo")
    print(SEP)

    TIDS = ["AAPL_Q1_2025", "NVDA_Q2_2025", "MSFT_Q3_2025"]
    df   = _make_demo_df(TIDS, chunks_per=14)
    print(f"\n✓  Synthetic chunk DataFrame: {df.shape}")

    # ── Validation ────────────────────────────────────────────────────
    v = HierarchyValidator()
    errs = v.validate_input(df)
    print(f"── Input Validation: {'✓ PASS' if not errs else 'FAIL — ' + str(errs)}")

    # ── Aggregate ─────────────────────────────────────────────────────
    agg    = create_hierarchical_aggregator(rolling_window=4, include_rolling_series=True)
    result = agg.aggregate(df)
    out_e  = v.validate_result(result)
    print(f"── Output Validation: {'✓ PASS' if not out_e else 'FAIL — ' + str(out_e)}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n── Summary Statistics {SEP2[21:]}")
    for k, v_ in result.summary_statistics().items():
        print(f"  {k:40s}: {v_}")

    # ── Transcript ────────────────────────────────────────────────────
    print(f"\n── Transcript Sentiment {SEP2[23:]}")
    for _, r in result.transcript_df.iterrows():
        print(
            f"  {r['transcript_id']:20s}  "
            f"score={r['sentiment_score']:+.4f}  ({r['sentiment_label']:8s})  "
            f"half_delta={r['half_delta']:+.4f}  chunks={r['chunk_count']}"
        )

    # ── Sections ──────────────────────────────────────────────────────
    print(f"\n── Section Sentiment {SEP2[20:]}")
    sec_cols = ["transcript_id","section_type","sentiment_score","chunk_count","drift_slope","section_rank"]
    avail = [c for c in sec_cols if c in result.section_df.columns]
    print(result.section_df[avail].to_string(index=False))

    print(f"\n── Prepared vs Q&A {SEP2[18:]}")
    pvq = result.prepared_vs_qa_summary()
    if not pvq.empty:
        print(pvq.to_string(index=False))

    # ── Speakers ──────────────────────────────────────────────────────
    print(f"\n── Speaker Rankings (AAPL_Q1_2025) {SEP2[35:]}")
    sr = result.speaker_rankings("AAPL_Q1_2025")
    if not sr.empty:
        spk_cols = ["speaker","speaker_type","sentiment_score","chunk_count","dominance_ratio","speaker_rank"]
        avail_s  = [c for c in spk_cols if c in sr.columns]
        print(sr[avail_s].to_string(index=False))

    print(f"\n── Exec vs Analyst {SEP2[18:]}")
    eva = result.exec_vs_analyst_summary()
    if not eva.empty:
        print(eva.to_string(index=False))

    print(f"\n── Speaker Dominance Report {SEP2[27:]}")
    dom = result.speaker_dominance_report()
    if not dom.empty:
        print(dom.head(6).to_string(index=False))

    # ── Drift ─────────────────────────────────────────────────────────
    print(f"\n── Drift Analytics {SEP2[18:]}")
    d_cols = ["transcript_id","drift_slope","drift_r_squared","drift_p_value",
              "trend_momentum","sentiment_acceleration","half_delta",
              "prepared_vs_qa_delta","exec_vs_analyst_delta"]
    avail_d = [c for c in d_cols if c in result.drift_df.columns]
    print(result.drift_df[avail_d].round(5).to_string(index=False))

    print(f"\n── Top Drift Events (abs_slope) {SEP2[31:]}")
    top = result.top_drift_events(n=3, by="abs_slope")
    if not top.empty:
        top_c = [c for c in ["transcript_id","drift_slope","drift_r_squared","trend_momentum"] if c in top.columns]
        print(top[top_c].round(5).to_string(index=False))

    # ── Diagnostics ──────────────────────────────────────────────────
    print(f"\n── Diagnostics {SEP2[14:]}")
    diag_c = ["transcript_id","total_chunks","unique_sections","unique_speakers",
              "duplicate_chunk_count","prob_sum_violation_count","nan_score_count","all_neutral"]
    avail_dia = [c for c in diag_c if c in result.diagnostics_df.columns]
    print(result.diagnostics_df[avail_dia].to_string(index=False))

    # ── Edge cases ────────────────────────────────────────────────────
    print(f"\n── Edge-Case Tests {SEP2[18:]}")
    agg2 = create_hierarchical_aggregator()

    r_empty = agg2.aggregate(pd.DataFrame())
    print(f"  Empty input        → transcripts: {len(r_empty.transcript_df)}")

    single = df[df["transcript_id"] == "AAPL_Q1_2025"].head(1).copy()
    r_single = agg2.aggregate(single, validate_input=False)
    ts_row = r_single.get_transcript("AAPL_Q1_2025")
    print(f"  Single-chunk       → score: {ts_row['sentiment_score']:+.4f}" if ts_row is not None else "  Single-chunk → None")

    neutral_df = df[df["transcript_id"] == "AAPL_Q1_2025"].copy()
    neutral_df["positive_prob"]   = 0.3333
    neutral_df["neutral_prob"]    = 0.3334
    neutral_df["negative_prob"]   = 0.3333
    neutral_df["sentiment_score"] = 0.0
    neutral_df["predicted_label"] = "neutral"
    r_neu = agg2.aggregate(neutral_df, validate_input=False)
    ts_n  = r_neu.get_transcript("AAPL_Q1_2025")
    print(f"  All-neutral        → label: {ts_n['sentiment_label']}" if ts_n is not None else "  All-neutral → None")

    no_sec = df.drop(columns=["section_type"])
    r_ns = agg2.aggregate(no_sec, validate_input=False)
    print(f"  No section col     → transcripts: {len(r_ns.transcript_df)}")

    bad_p = df[df["transcript_id"] == "AAPL_Q1_2025"].copy()
    bad_p["positive_prob"] = 2.0
    bad_p["neutral_prob"]  = 2.0
    bad_p["negative_prob"] = 2.0
    r_bad  = agg2.aggregate(bad_p, validate_input=False)
    ts_b   = r_bad.get_transcript("AAPL_Q1_2025")
    if ts_b is not None:
        ps = ts_b["positive_probability"] + ts_b["neutral_probability"] + ts_b["negative_probability"]
        print(f"  Malformed probs    → normalised sum: {ps:.8f}")

    # ── CSV export ────────────────────────────────────────────────────
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = result.save_csv(tmp, prefix="demo")
        print(f"\n── CSV Export: {len(paths)} files")
        for name, p in paths.items():
            size = Path(p).stat().st_size
            print(f"  {name:15s}: {Path(p).name}  ({size:,} bytes)")

    print(f"\n{SEP}")
    print("  Self-test complete — all systems nominal.")
    print(f"{SEP}\n")
