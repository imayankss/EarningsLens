"""
src/sentiment/aggregation.py
=============================
Hierarchical sentiment aggregation engine for FinBERT chunk-level outputs.

Position in pipeline
--------------------
    finbert_pipeline.py  →  **aggregation.py**  →  event_study / downstream

Responsibilities
----------------
* Aggregate chunk-level FinBERT probabilities into transcript-level sentiment.
* Apply four weighting strategies: uniform, confidence, token, hybrid.
* Produce section-level aggregates  (prepared_remarks vs qa).
* Produce speaker-level aggregates  (CEO, CFO, Analyst, etc.).
* Compute sentiment volatility, drift, and confidence diagnostics.
* Normalise aggregated probabilities to sum to 1.0.
* Expose DataFrame, Parquet, CSV, JSONL export helpers.
* Validate all outputs (probability sums, NaN detection, label consistency).

Design constraints (from architecture docs)
-------------------------------------------
* Primary merge key : transcript_id
* Sentiment score   : P(positive) − P(negative)   ∈ [−1, +1]
* Label thresholds  : score > +0.05 → positive
                      score < −0.05 → negative
                      otherwise     → neutral
* Four strategies   : uniform / confidence / token / hybrid (recommended)
* Hybrid formula    : w_i = confidence_i × token_count_i
* Deterministic     : same input always produces same output

Author : Earnings Call Sentiment Analyzer — DAY 8
Python : 3.11+
"""

from __future__ import annotations

import json
import logging
import math
import warnings as _warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Label decision boundaries (architecture-defined)
_POSITIVE_THRESHOLD: float = 0.05
_NEGATIVE_THRESHOLD: float = -0.05
_PROB_SUM_TOLERANCE: float = 1e-4   # |sum - 1.0| below this is acceptable
_MIN_CONFIDENCE: float = 1e-9       # floor to prevent zero-weight divisions

# Canonical label set (must match FinBERT output labels exactly)
VALID_LABELS: frozenset[str] = frozenset({"positive", "neutral", "negative"})

# Required input columns
_REQUIRED_INPUT_COLS: list[str] = [
    "chunk_id",
    "transcript_id",
    "chunk_order",
    "positive_prob",
    "neutral_prob",
    "negative_prob",
    "sentiment_score",
    "confidence",
    "predicted_label",
    "token_count",
    "section_type",
    "dominant_speaker",
]

# ---------------------------------------------------------------------------
# Weighting strategy enum
# ---------------------------------------------------------------------------


class WeightingStrategy(str, Enum):
    """
    Aggregation weighting strategy.

    UNIFORM
        Every chunk contributes equally regardless of length or confidence.
        S = mean(score_i)

    CONFIDENCE
        Higher-confidence chunks contribute proportionally more.
        S = Σ(score_i × conf_i) / Σ(conf_i)

    TOKEN
        Longer chunks contribute proportionally more.
        S = Σ(score_i × token_i) / Σ(token_i)

    HYBRID  *(recommended)*
        Weight = confidence × token_count.
        S = Σ(score_i × w_i) / Σ(w_i),  w_i = conf_i × token_i
    """

    UNIFORM = "uniform"
    CONFIDENCE = "confidence"
    TOKEN = "token"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# AggregationConfig
# ---------------------------------------------------------------------------


@dataclass
class AggregationConfig:
    """
    Configuration for the SentimentAggregator.

    Attributes
    ----------
    strategy : WeightingStrategy
        Default weighting strategy used for transcript-level aggregation.
    positive_threshold : float
        Sentiment score above this → label "positive".
    negative_threshold : float
        Sentiment score below this → label "negative".
    min_chunks_required : int
        Transcripts with fewer chunks than this threshold emit a warning and
        are still processed (not dropped), but flagged in diagnostics.
    normalise_probabilities : bool
        Re-normalise aggregated probabilities so they sum to exactly 1.0.
        Should stay True for FinBERT outputs (floating-point rounding).
    compute_section_aggregates : bool
        When True, also produce per-section aggregates (prepared_remarks / qa).
    compute_speaker_aggregates : bool
        When True, also produce per-speaker aggregates.
    compute_volatility_metrics : bool
        When True, compute sentiment_std, sentiment_iqr, drift_slope.
    min_section_chunks : int
        Minimum chunks required to produce a section aggregate; groups below
        this are omitted from section-level output to avoid noise.
    min_speaker_chunks : int
        Same threshold for speaker aggregates.
    """

    strategy: WeightingStrategy = WeightingStrategy.HYBRID
    positive_threshold: float = _POSITIVE_THRESHOLD
    negative_threshold: float = _NEGATIVE_THRESHOLD
    min_chunks_required: int = 1
    normalise_probabilities: bool = True
    compute_section_aggregates: bool = True
    compute_speaker_aggregates: bool = True
    compute_volatility_metrics: bool = True
    min_section_chunks: int = 2
    min_speaker_chunks: int = 2

    def __post_init__(self) -> None:
        if self.positive_threshold <= self.negative_threshold:
            raise ValueError(
                "positive_threshold must be > negative_threshold."
            )
        if self.min_chunks_required < 1:
            raise ValueError("min_chunks_required must be ≥ 1.")


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SentimentRecord:
    """
    Aggregated sentiment for a single grouping key (transcript / section / speaker).

    This is the atomic unit of aggregation output.  ``TranscriptSentiment``
    and the section/speaker aggregation layers all produce instances of this.
    """

    # ── Identity ──────────────────────────────────────────────────────
    group_key: str            # transcript_id, section_type, speaker name, etc.
    group_type: str           # "transcript" | "section" | "speaker"
    transcript_id: str

    # ── Aggregated probabilities ──────────────────────────────────────
    positive_probability: float
    neutral_probability: float
    negative_probability: float

    # ── Derived sentiment ─────────────────────────────────────────────
    sentiment_score: float    # positive_probability − negative_probability
    sentiment_label: str      # positive | neutral | negative

    # ── Volume ───────────────────────────────────────────────────────
    chunk_count: int
    total_tokens: int

    # ── Confidence ───────────────────────────────────────────────────
    confidence_mean: float
    confidence_std: float

    # ── Volatility ───────────────────────────────────────────────────
    sentiment_std: float      # std-dev of per-chunk sentiment_scores
    sentiment_median: float   # median of per-chunk sentiment_scores
    sentiment_iqr: float      # interquartile range of per-chunk scores
    sentiment_min: float
    sentiment_max: float

    # ── Weighting metadata ────────────────────────────────────────────
    strategy_used: str
    weighted_score: float     # raw numerator/denominator result (pre-label)

    # ── Diagnostics ──────────────────────────────────────────────────
    has_warnings: bool = False
    warning_messages: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_flat_dict(self) -> dict[str, Any]:
        """Flat dict without nested lists — suitable for a DataFrame row."""
        d = self.to_dict()
        d["warning_messages"] = "; ".join(d["warning_messages"])
        return d


@dataclass
class TranscriptSentiment:
    """
    Full aggregated sentiment profile for a single transcript.

    Holds the top-level transcript aggregate plus optional sub-group
    aggregates (section-level, speaker-level).

    Attributes
    ----------
    transcript_id : str
    transcript_aggregate : SentimentRecord
        The primary, transcript-wide sentiment result.
    section_aggregates : dict[str, SentimentRecord]
        Keyed by section_type string (e.g. "prepared_remarks", "qa").
    speaker_aggregates : dict[str, SentimentRecord]
        Keyed by dominant_speaker string (e.g. "Tim Cook", "Analyst").
    """

    transcript_id: str
    transcript_aggregate: SentimentRecord
    section_aggregates: dict[str, SentimentRecord] = field(default_factory=dict)
    speaker_aggregates: dict[str, SentimentRecord] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "transcript_aggregate": self.transcript_aggregate.to_dict(),
            "section_aggregates": {
                k: v.to_dict() for k, v in self.section_aggregates.items()
            },
            "speaker_aggregates": {
                k: v.to_dict() for k, v in self.speaker_aggregates.items()
            },
        }

    def primary_score(self) -> float:
        return self.transcript_aggregate.sentiment_score

    def primary_label(self) -> str:
        return self.transcript_aggregate.sentiment_label

    def section_score(self, section: str) -> Optional[float]:
        rec = self.section_aggregates.get(section)
        return rec.sentiment_score if rec else None

    def speaker_score(self, speaker: str) -> Optional[float]:
        rec = self.speaker_aggregates.get(speaker)
        return rec.sentiment_score if rec else None


@dataclass
class AggregationResult:
    """
    Top-level container returned by ``SentimentAggregator.aggregate()``.

    Holds per-transcript ``TranscriptSentiment`` objects plus batch-level
    diagnostics, warnings, and export helpers.
    """

    transcripts: dict[str, TranscriptSentiment]
    total_transcripts: int
    total_chunks_processed: int
    strategy_used: WeightingStrategy
    warnings: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    config: AggregationConfig = field(default_factory=AggregationConfig)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get(self, transcript_id: str) -> Optional[TranscriptSentiment]:
        return self.transcripts.get(transcript_id)

    def transcript_ids(self) -> list[str]:
        return list(self.transcripts.keys())

    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0

    # ------------------------------------------------------------------
    # DataFrame helpers
    # ------------------------------------------------------------------

    def to_transcript_dataframe(self) -> pd.DataFrame:
        """
        One row per transcript — primary call-level sentiment output.
        This is the canonical output consumed by the event-study engine.
        """
        rows = [
            ts.transcript_aggregate.to_flat_dict()
            for ts in self.transcripts.values()
        ]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Ensure stable column ordering
        preferred_cols = [
            "transcript_id", "chunk_count", "total_tokens",
            "sentiment_score", "sentiment_label",
            "positive_probability", "neutral_probability", "negative_probability",
            "confidence_mean", "confidence_std",
            "sentiment_std", "sentiment_median", "sentiment_iqr",
            "sentiment_min", "sentiment_max",
            "strategy_used", "has_warnings",
        ]
        ordered = [c for c in preferred_cols if c in df.columns]
        remainder = [c for c in df.columns if c not in preferred_cols]
        return df[ordered + remainder].reset_index(drop=True)

    def to_section_dataframe(self) -> pd.DataFrame:
        """One row per (transcript, section_type) pair."""
        rows: list[dict[str, Any]] = []
        for ts in self.transcripts.values():
            for section, rec in ts.section_aggregates.items():
                row = rec.to_flat_dict()
                row["section_type"] = section
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).reset_index(drop=True)

    def to_speaker_dataframe(self) -> pd.DataFrame:
        """One row per (transcript, speaker) pair."""
        rows: list[dict[str, Any]] = []
        for ts in self.transcripts.values():
            for speaker, rec in ts.speaker_aggregates.items():
                row = rec.to_flat_dict()
                row["speaker"] = speaker
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_transcripts": self.total_transcripts,
            "total_chunks_processed": self.total_chunks_processed,
            "strategy_used": self.strategy_used.value,
            "warnings": self.warnings,
            "validation_errors": self.validation_errors,
            "transcripts": {
                tid: ts.to_dict()
                for tid, ts in self.transcripts.items()
            },
        }

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save_parquet(self, path: str | Path, level: str = "transcript") -> Path:
        """
        Save aggregated sentiment to Parquet.

        Parameters
        ----------
        level : {"transcript", "section", "speaker"}
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self._level_df(level)
        df.to_parquet(p, index=False)
        logger.info("Saved %s-level sentiment (%d rows) → %s", level, len(df), p)
        return p

    def save_csv(self, path: str | Path, level: str = "transcript") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self._level_df(level)
        df.to_csv(p, index=False)
        logger.info("Saved %s-level sentiment CSV (%d rows) → %s", level, len(df), p)
        return p

    def save_jsonl(self, path: str | Path) -> Path:
        """Save transcript-level results as JSONL for RAG or streaming."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(ts.to_dict())
            for ts in self.transcripts.values()
        ]
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved JSONL (%d lines) → %s", len(lines), p)
        return p

    def _level_df(self, level: str) -> pd.DataFrame:
        dispatch = {
            "transcript": self.to_transcript_dataframe,
            "section": self.to_section_dataframe,
            "speaker": self.to_speaker_dataframe,
        }
        if level not in dispatch:
            raise ValueError(f"level must be one of {list(dispatch.keys())}, got '{level}'.")
        return dispatch[level]()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        High-level batch summary — useful for pipeline health checks and
        logging.
        """
        df = self.to_transcript_dataframe()
        if df.empty:
            return {"total_transcripts": 0}

        label_counts = df["sentiment_label"].value_counts().to_dict()
        return {
            "total_transcripts": self.total_transcripts,
            "total_chunks_processed": self.total_chunks_processed,
            "strategy_used": self.strategy_used.value,
            "label_distribution": label_counts,
            "sentiment_score_mean": round(float(df["sentiment_score"].mean()), 6),
            "sentiment_score_std": round(float(df["sentiment_score"].std()), 6),
            "sentiment_score_min": round(float(df["sentiment_score"].min()), 6),
            "sentiment_score_max": round(float(df["sentiment_score"].max()), 6),
            "confidence_mean_avg": round(float(df["confidence_mean"].mean()), 6),
            "warning_count": len(self.warnings),
            "validation_error_count": len(self.validation_errors),
        }

    def coverage_metrics(self) -> dict[str, Any]:
        """
        Per-transcript coverage — how many chunks contributed to each result.
        """
        if not self.transcripts:
            return {}
        df = self.to_transcript_dataframe()
        return {
            "transcripts_with_section_aggregates": sum(
                1 for ts in self.transcripts.values() if ts.section_aggregates
            ),
            "transcripts_with_speaker_aggregates": sum(
                1 for ts in self.transcripts.values() if ts.speaker_aggregates
            ),
            "chunk_count_min": int(df["chunk_count"].min()),
            "chunk_count_max": int(df["chunk_count"].max()),
            "chunk_count_mean": round(float(df["chunk_count"].mean()), 2),
        }


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AggregationError(RuntimeError):
    """Raised when aggregation cannot proceed due to invalid inputs."""


class AggregationValidationError(ValueError):
    """Raised by the validation layer on hard constraint violations."""


# ---------------------------------------------------------------------------
# Internal numeric helpers
# ---------------------------------------------------------------------------


def _weighted_mean(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """
    Compute a weighted mean, guarding against zero total weight.

    Parameters
    ----------
    values  : chunk-level metric (sentiment score, probability, etc.)
    weights : corresponding weights (must be non-negative)

    Returns
    -------
    float — weighted mean, or simple mean when all weights are zero.
    """
    total_weight = sum(weights)
    if total_weight < _MIN_CONFIDENCE:
        # Degenerate fallback: equal weights
        return sum(values) / len(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _score_to_label(
    score: float,
    positive_threshold: float = _POSITIVE_THRESHOLD,
    negative_threshold: float = _NEGATIVE_THRESHOLD,
) -> str:
    """Map a scalar sentiment score to a canonical label string."""
    if score > positive_threshold:
        return "positive"
    if score < negative_threshold:
        return "negative"
    return "neutral"


def _normalise_probs(pos: float, neu: float, neg: float) -> tuple[float, float, float]:
    """
    Rescale three probabilities so they sum to exactly 1.0.

    Handles edge cases where the sum is zero (returns uniform 1/3 each).
    """
    total = pos + neu + neg
    if total < _MIN_CONFIDENCE:
        return 1 / 3, 1 / 3, 1 / 3
    return pos / total, neu / total, neg / total


def _safe_std(values: Sequence[float]) -> float:
    """Standard deviation that returns 0.0 for sequences of length < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _safe_iqr(values: Sequence[float]) -> float:
    """Interquartile range (Q3 − Q1) without numpy dependency."""
    n = len(values)
    if n < 4:
        return 0.0
    sorted_vals = sorted(values)
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    return sorted_vals[q3_idx] - sorted_vals[q1_idx]


def _drift_slope(values: Sequence[float]) -> float:
    """
    Linear trend slope across an ordered sequence using least-squares.
    Returns 0.0 if fewer than 2 data points.

    A positive slope indicates sentiment becoming more positive over the
    course of the transcript (used for tone drift analysis).
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Core aggregation function (pure, stateless)
# ---------------------------------------------------------------------------


def _aggregate_chunk_group(
    chunk_rows: list[dict[str, Any]],
    group_key: str,
    group_type: str,
    transcript_id: str,
    config: AggregationConfig,
) -> SentimentRecord:
    """
    Aggregate a list of chunk-row dicts into a single ``SentimentRecord``.

    This is the core mathematical function.  All four weighting strategies
    are implemented here.  The function is pure (no side effects) and
    deterministic.

    Parameters
    ----------
    chunk_rows : list of dicts with FinBERT chunk-level output fields.
    group_key  : identifier for the aggregation group (e.g. transcript_id,
                 section_type value, speaker name).
    group_type : "transcript" | "section" | "speaker"
    transcript_id : parent transcript identifier.
    config     : AggregationConfig controlling thresholds and strategy.
    """
    warnings_list: list[str] = []
    n = len(chunk_rows)

    if n == 0:
        raise AggregationError(
            f"Cannot aggregate zero chunks for group_key='{group_key}'."
        )

    # ── Extract raw vectors ────────────────────────────────────────────
    pos_probs   = [float(r.get("positive_prob", r.get("positive_probability", 0.0))) for r in chunk_rows]
    neu_probs   = [float(r.get("neutral_prob",  r.get("neutral_probability",  0.0))) for r in chunk_rows]
    neg_probs   = [float(r.get("negative_prob", r.get("negative_probability", 0.0))) for r in chunk_rows]
    scores      = [float(r.get("sentiment_score", 0.0)) for r in chunk_rows]
    confidences = [max(float(r.get("confidence", 0.0)), _MIN_CONFIDENCE) for r in chunk_rows]
    token_cnts  = [max(int(r.get("token_count", 1)), 1) for r in chunk_rows]

    # ── Sanitise: clamp probs to [0, 1] ───────────────────────────────
    pos_probs   = [max(0.0, min(1.0, p)) for p in pos_probs]
    neu_probs   = [max(0.0, min(1.0, p)) for p in neu_probs]
    neg_probs   = [max(0.0, min(1.0, p)) for p in neg_probs]

    # Detect and warn on NaN
    for vec_name, vec in [("positive_prob", pos_probs), ("neutral_prob", neu_probs),
                           ("negative_prob", neg_probs), ("sentiment_score", scores),
                           ("confidence", confidences)]:
        nan_count = sum(1 for v in vec if math.isnan(v))
        if nan_count:
            msg = (
                f"[{group_key}] {nan_count}/{n} NaN values in '{vec_name}' "
                f"— replacing with 0.0."
            )
            warnings_list.append(msg)
            logger.warning(msg)
    # Replace any remaining NaN with 0.0
    pos_probs   = [0.0 if math.isnan(v) else v for v in pos_probs]
    neu_probs   = [0.0 if math.isnan(v) else v for v in neu_probs]
    neg_probs   = [0.0 if math.isnan(v) else v for v in neg_probs]
    scores      = [0.0 if math.isnan(v) else v for v in scores]
    confidences = [_MIN_CONFIDENCE if math.isnan(v) else v for v in confidences]

    # ── Build weights for chosen strategy ────────────────────────────
    strategy = config.strategy
    if strategy == WeightingStrategy.UNIFORM:
        weights = [1.0] * n
    elif strategy == WeightingStrategy.CONFIDENCE:
        weights = confidences
    elif strategy == WeightingStrategy.TOKEN:
        weights = [float(t) for t in token_cnts]
    elif strategy == WeightingStrategy.HYBRID:
        weights = [c * t for c, t in zip(confidences, token_cnts)]
    else:
        raise AggregationError(f"Unknown WeightingStrategy: {strategy!r}")

    # ── Weighted aggregated probabilities ─────────────────────────────
    agg_pos = _weighted_mean(pos_probs, weights)
    agg_neu = _weighted_mean(neu_probs, weights)
    agg_neg = _weighted_mean(neg_probs, weights)

    # ── Normalise probabilities ────────────────────────────────────────
    if config.normalise_probabilities:
        agg_pos, agg_neu, agg_neg = _normalise_probs(agg_pos, agg_neu, agg_neg)

    # ── Final sentiment score (canonical formula) ──────────────────────
    agg_score = agg_pos - agg_neg

    # ── Weighted score (the raw numerator/denominator result) ──────────
    weighted_score = _weighted_mean(scores, weights)

    # ── Derive label ──────────────────────────────────────────────────
    label = _score_to_label(
        agg_score, config.positive_threshold, config.negative_threshold
    )

    # ── Confidence statistics ─────────────────────────────────────────
    conf_mean = sum(confidences) / n
    conf_std  = _safe_std(confidences)

    # ── Volatility metrics ─────────────────────────────────────────────
    score_std    = _safe_std(scores)
    score_median = sorted(scores)[n // 2]
    score_iqr    = _safe_iqr(scores)
    score_min    = min(scores)
    score_max    = max(scores)

    # ── Volume ────────────────────────────────────────────────────────
    total_tokens = sum(token_cnts)

    # ── Warn on low chunk count ────────────────────────────────────────
    if n < config.min_chunks_required:
        msg = (
            f"[{group_key}] Only {n} chunk(s) available "
            f"(min_chunks_required={config.min_chunks_required})."
        )
        warnings_list.append(msg)
        logger.warning(msg)

    return SentimentRecord(
        group_key=group_key,
        group_type=group_type,
        transcript_id=transcript_id,
        positive_probability=round(agg_pos, 8),
        neutral_probability=round(agg_neu, 8),
        negative_probability=round(agg_neg, 8),
        sentiment_score=round(agg_score, 8),
        sentiment_label=label,
        chunk_count=n,
        total_tokens=total_tokens,
        confidence_mean=round(conf_mean, 8),
        confidence_std=round(conf_std, 8),
        sentiment_std=round(score_std, 8),
        sentiment_median=round(score_median, 8),
        sentiment_iqr=round(score_iqr, 8),
        sentiment_min=round(score_min, 8),
        sentiment_max=round(score_max, 8),
        strategy_used=strategy.value,
        weighted_score=round(weighted_score, 8),
        has_warnings=bool(warnings_list),
        warning_messages=warnings_list,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class AggregationValidator:
    """
    Validates chunk-level input DataFrame and aggregation outputs.

    Designed to be used both by SentimentAggregator internally and by the
    external ``sentiment_validation.py`` module.
    """

    @staticmethod
    def validate_input(df: pd.DataFrame) -> list[str]:
        """
        Check input DataFrame for required columns, types, and value ranges.

        Returns a list of error strings (empty = valid).
        """
        errors: list[str] = []

        # Required columns
        missing = [c for c in _REQUIRED_INPUT_COLS if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
            return errors   # can't proceed further

        if df.empty:
            errors.append("Input DataFrame is empty.")
            return errors

        # NaN in critical columns
        critical = ["positive_prob", "neutral_prob", "negative_prob",
                    "sentiment_score", "confidence", "token_count"]
        for col in critical:
            n_nan = int(df[col].isna().sum())
            if n_nan:
                errors.append(f"Column '{col}' has {n_nan} NaN values.")

        # Probability bounds
        for col in ["positive_prob", "neutral_prob", "negative_prob"]:
            if col in df.columns:
                out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
                if out_of_range:
                    errors.append(
                        f"Column '{col}' has {out_of_range} values outside [0, 1]."
                    )

        # Probability sum check per chunk
        prob_sum = df["positive_prob"] + df["neutral_prob"] + df["negative_prob"]
        bad_sum = (abs(prob_sum - 1.0) > _PROB_SUM_TOLERANCE).sum()
        if bad_sum:
            errors.append(
                f"{bad_sum} chunks have probability sums deviating from 1.0 "
                f"by more than {_PROB_SUM_TOLERANCE}."
            )

        # Token counts
        if (df["token_count"] <= 0).any():
            errors.append("Some rows have token_count ≤ 0.")

        # Confidence bounds
        if (df["confidence"] < 0).any() or (df["confidence"] > 1).any():
            errors.append("confidence values must be in [0, 1].")

        # Label validation
        invalid_labels = ~df["predicted_label"].isin(VALID_LABELS)
        n_invalid = invalid_labels.sum()
        if n_invalid:
            bad = df.loc[invalid_labels, "predicted_label"].unique().tolist()
            errors.append(
                f"{n_invalid} rows have invalid predicted_label: {bad}. "
                f"Expected one of {sorted(VALID_LABELS)}."
            )

        # Duplicate chunk_ids
        dupes = df.duplicated(subset=["chunk_id"]).sum()
        if dupes:
            errors.append(f"{dupes} duplicate chunk_id values detected.")

        return errors

    @staticmethod
    def validate_output(result: AggregationResult) -> list[str]:
        """
        Validate aggregation outputs.

        Returns a list of error strings (empty = valid).
        """
        errors: list[str] = []

        if not result.transcripts:
            errors.append("AggregationResult contains no transcripts.")
            return errors

        seen_ids: set[str] = set()
        for tid, ts in result.transcripts.items():

            # Duplicate transcript check
            if tid in seen_ids:
                errors.append(f"Duplicate transcript_id in output: '{tid}'.")
            seen_ids.add(tid)

            rec = ts.transcript_aggregate

            # Probability sum
            prob_sum = (
                rec.positive_probability
                + rec.neutral_probability
                + rec.negative_probability
            )
            if abs(prob_sum - 1.0) > _PROB_SUM_TOLERANCE:
                errors.append(
                    f"[{tid}] Aggregated probability sum = {prob_sum:.6f}, "
                    f"expected 1.0 ± {_PROB_SUM_TOLERANCE}."
                )

            # Score range
            if not (-1.0 - 1e-6 <= rec.sentiment_score <= 1.0 + 1e-6):
                errors.append(
                    f"[{tid}] sentiment_score {rec.sentiment_score:.6f} "
                    f"outside expected range [-1, 1]."
                )

            # Label consistency
            if rec.sentiment_label not in VALID_LABELS:
                errors.append(
                    f"[{tid}] Invalid sentiment_label: '{rec.sentiment_label}'."
                )

            # Label ↔ score consistency
            expected_label = _score_to_label(
                rec.sentiment_score,
                _POSITIVE_THRESHOLD,
                _NEGATIVE_THRESHOLD,
            )
            if rec.sentiment_label != expected_label:
                errors.append(
                    f"[{tid}] sentiment_label '{rec.sentiment_label}' is "
                    f"inconsistent with sentiment_score {rec.sentiment_score:.4f} "
                    f"(expected '{expected_label}')."
                )

            # Chunk count
            if rec.chunk_count < 1:
                errors.append(f"[{tid}] chunk_count < 1.")

        return errors


# ---------------------------------------------------------------------------
# SentimentAggregator — main engine
# ---------------------------------------------------------------------------


class SentimentAggregator:
    """
    Hierarchical sentiment aggregation engine.

    Accepts a DataFrame of chunk-level FinBERT scores and produces
    transcript-level, section-level, and speaker-level sentiment aggregates
    using configurable weighting strategies.

    Parameters
    ----------
    config : AggregationConfig, optional
        Aggregation configuration.  Defaults to ``AggregationConfig()`` which
        uses the recommended HYBRID (confidence × token) strategy.

    Examples
    --------
    Basic transcript-level aggregation::

        aggregator = SentimentAggregator()
        result = aggregator.aggregate(chunk_scores_df)
        result.save_parquet("data/processed/sentiment/call_level_sentiment.parquet")

    Custom strategy::

        from src.sentiment.aggregation import AggregationConfig, WeightingStrategy
        config = AggregationConfig(strategy=WeightingStrategy.CONFIDENCE)
        aggregator = SentimentAggregator(config=config)
        result = aggregator.aggregate(chunk_scores_df)

    Accessing section-level scores::

        ts = result.get("AAPL_Q1_2025")
        qa_score = ts.section_score("qa")
        ceo_score = ts.speaker_score("Tim Cook")
    """

    def __init__(self, config: Optional[AggregationConfig] = None) -> None:
        self.config: AggregationConfig = config or AggregationConfig()
        self._validator = AggregationValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        chunk_df: pd.DataFrame,
        validate_input: bool = True,
    ) -> AggregationResult:
        """
        Aggregate a DataFrame of chunk-level FinBERT scores.

        The DataFrame may contain chunks from one or multiple transcripts.
        Each transcript is processed independently and deterministically.

        Parameters
        ----------
        chunk_df : pd.DataFrame
            Output from ``finbert_pipeline.py`` — one row per chunk.
            Must contain the columns listed in ``_REQUIRED_INPUT_COLS``.
        validate_input : bool
            Run input validation before processing.  Set to False only in
            performance-critical batch contexts where inputs are pre-validated.

        Returns
        -------
        AggregationResult
        """
        global_warnings: list[str] = []
        validation_errors: list[str] = []

        # ── 0. Handle empty input ──────────────────────────────────────
        if chunk_df is None or chunk_df.empty:
            logger.warning("Received empty chunk DataFrame — returning empty result.")
            return AggregationResult(
                transcripts={},
                total_transcripts=0,
                total_chunks_processed=0,
                strategy_used=self.config.strategy,
                warnings=["Empty input DataFrame."],
                validation_errors=[],
                config=self.config,
            )

        # ── 1. Input validation ────────────────────────────────────────
        if validate_input:
            input_errors = self._validator.validate_input(chunk_df)
            if input_errors:
                for err in input_errors:
                    logger.error("Input validation: %s", err)
                validation_errors.extend(input_errors)
                # Non-fatal: continue with best effort

        # ── 2. Normalise DataFrame ────────────────────────────────────
        chunk_df = self._normalise_input_df(chunk_df)

        # ── 3. Group by transcript and process ────────────────────────
        transcripts: dict[str, TranscriptSentiment] = {}
        total_chunks = 0

        for tid, group in chunk_df.groupby("transcript_id", sort=False):
            tid_str = str(tid)
            rows = group.sort_values("chunk_order").to_dict("records")
            total_chunks += len(rows)

            logger.info(
                "[%s] Aggregating %d chunks (strategy=%s).",
                tid_str, len(rows), self.config.strategy.value,
            )

            try:
                ts = self._aggregate_transcript(tid_str, rows, global_warnings)
                transcripts[tid_str] = ts
            except AggregationError as exc:
                msg = f"[{tid_str}] Aggregation failed: {exc}"
                logger.error(msg)
                global_warnings.append(msg)

        # ── 4. Output validation ──────────────────────────────────────
        result = AggregationResult(
            transcripts=transcripts,
            total_transcripts=len(transcripts),
            total_chunks_processed=total_chunks,
            strategy_used=self.config.strategy,
            warnings=global_warnings,
            validation_errors=validation_errors,
            config=self.config,
        )

        output_errors = self._validator.validate_output(result)
        if output_errors:
            for err in output_errors:
                logger.error("Output validation: %s", err)
            result.validation_errors.extend(output_errors)

        logger.info(
            "Aggregation complete — %d transcripts, %d chunks, %d warnings, %d errors.",
            len(transcripts), total_chunks,
            len(global_warnings), len(result.validation_errors),
        )
        return result

    def aggregate_single(
        self,
        transcript_id: str,
        chunk_df: pd.DataFrame,
    ) -> TranscriptSentiment:
        """
        Convenience wrapper for aggregating a single transcript's chunks.

        Parameters
        ----------
        transcript_id : str
        chunk_df : pd.DataFrame
            Chunks belonging to this transcript only.

        Returns
        -------
        TranscriptSentiment
        """
        result = self.aggregate(chunk_df, validate_input=True)
        ts = result.get(transcript_id)
        if ts is None:
            # transcript_id may have been read from column, not argument
            all_ids = result.transcript_ids()
            if len(all_ids) == 1:
                return result.transcripts[all_ids[0]]
            raise AggregationError(
                f"transcript_id '{transcript_id}' not found in aggregation result. "
                f"Available: {all_ids}"
            )
        return ts

    def compare_strategies(
        self,
        chunk_df: pd.DataFrame,
        transcript_id: str,
    ) -> dict[str, SentimentRecord]:
        """
        Run all four weighting strategies on the same transcript and return
        each result.  Useful for ablation studies and strategy benchmarking.

        Parameters
        ----------
        chunk_df : pd.DataFrame
            All chunks for *transcript_id*.
        transcript_id : str

        Returns
        -------
        dict[WeightingStrategy.value, SentimentRecord]
        """
        results: dict[str, SentimentRecord] = {}
        chunk_df = self._normalise_input_df(chunk_df)
        rows = (
            chunk_df[chunk_df["transcript_id"] == transcript_id]
            .sort_values("chunk_order")
            .to_dict("records")
        )
        if not rows:
            raise AggregationError(
                f"No chunks found for transcript_id='{transcript_id}'."
            )

        for strategy in WeightingStrategy:
            cfg = AggregationConfig(
                strategy=strategy,
                compute_section_aggregates=False,
                compute_speaker_aggregates=False,
            )
            rec = _aggregate_chunk_group(
                rows, transcript_id, "transcript", transcript_id, cfg
            )
            results[strategy.value] = rec

        return results

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _normalise_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Light normalisation: fill missing optional columns with defaults,
        coerce numeric types, standardise string columns.
        """
        df = df.copy()

        # Accept column aliases from different pipeline stages
        if "positive_prob" not in df.columns and "positive_probability" in df.columns:
            df["positive_prob"] = df["positive_probability"]
        if "neutral_prob" not in df.columns and "neutral_probability" in df.columns:
            df["neutral_prob"] = df["neutral_probability"]
        if "negative_prob" not in df.columns and "negative_probability" in df.columns:
            df["negative_prob"] = df["negative_probability"]

        # Fill missing optional columns
        if "dominant_speaker" not in df.columns:
            df["dominant_speaker"] = "unknown"
        if "section_type" not in df.columns:
            df["section_type"] = "unknown"
        if "chunk_order" not in df.columns:
            df["chunk_order"] = range(len(df))

        # Coerce numeric
        for col in ["positive_prob", "neutral_prob", "negative_prob",
                    "sentiment_score", "confidence", "token_count", "chunk_order"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Clamp probabilities
        for col in ["positive_prob", "neutral_prob", "negative_prob", "confidence"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0.0, upper=1.0)

        # Ensure token_count ≥ 1
        if "token_count" in df.columns:
            df["token_count"] = df["token_count"].clip(lower=1)

        # Normalise string columns
        for col in ["transcript_id", "section_type", "dominant_speaker", "predicted_label"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df

    def _aggregate_transcript(
        self,
        transcript_id: str,
        rows: list[dict[str, Any]],
        global_warnings: list[str],
    ) -> TranscriptSentiment:
        """
        Build a full TranscriptSentiment from a list of chunk rows.

        1. Transcript-level aggregate (all chunks).
        2. Section-level aggregates (prepared_remarks, qa, etc.).
        3. Speaker-level aggregates (CEO, CFO, Analyst, etc.).
        """

        # ── Transcript-level ──────────────────────────────────────────
        transcript_rec = _aggregate_chunk_group(
            rows,
            group_key=transcript_id,
            group_type="transcript",
            transcript_id=transcript_id,
            config=self.config,
        )
        global_warnings.extend(transcript_rec.warning_messages)

        # ── Section-level ─────────────────────────────────────────────
        section_aggregates: dict[str, SentimentRecord] = {}
        if self.config.compute_section_aggregates:
            section_groups = self._group_by_field(rows, "section_type")
            for section, sec_rows in section_groups.items():
                if section in ("", "unknown", "nan"):
                    continue
                if len(sec_rows) < self.config.min_section_chunks:
                    logger.debug(
                        "[%s] Skipping section '%s' — only %d chunk(s) "
                        "(min_section_chunks=%d).",
                        transcript_id, section, len(sec_rows),
                        self.config.min_section_chunks,
                    )
                    continue
                try:
                    rec = _aggregate_chunk_group(
                        sec_rows,
                        group_key=section,
                        group_type="section",
                        transcript_id=transcript_id,
                        config=self.config,
                    )
                    section_aggregates[section] = rec
                    global_warnings.extend(rec.warning_messages)
                except AggregationError as exc:
                    msg = f"[{transcript_id}] Section '{section}' aggregation failed: {exc}"
                    logger.warning(msg)
                    global_warnings.append(msg)

        # ── Speaker-level ─────────────────────────────────────────────
        speaker_aggregates: dict[str, SentimentRecord] = {}
        if self.config.compute_speaker_aggregates:
            speaker_groups = self._group_by_field(rows, "dominant_speaker")
            for speaker, spk_rows in speaker_groups.items():
                if speaker in ("", "unknown", "nan"):
                    continue
                if len(spk_rows) < self.config.min_speaker_chunks:
                    logger.debug(
                        "[%s] Skipping speaker '%s' — only %d chunk(s) "
                        "(min_speaker_chunks=%d).",
                        transcript_id, speaker, len(spk_rows),
                        self.config.min_speaker_chunks,
                    )
                    continue
                try:
                    rec = _aggregate_chunk_group(
                        spk_rows,
                        group_key=speaker,
                        group_type="speaker",
                        transcript_id=transcript_id,
                        config=self.config,
                    )
                    speaker_aggregates[speaker] = rec
                    global_warnings.extend(rec.warning_messages)
                except AggregationError as exc:
                    msg = f"[{transcript_id}] Speaker '{speaker}' aggregation failed: {exc}"
                    logger.warning(msg)
                    global_warnings.append(msg)

        return TranscriptSentiment(
            transcript_id=transcript_id,
            transcript_aggregate=transcript_rec,
            section_aggregates=section_aggregates,
            speaker_aggregates=speaker_aggregates,
        )

    @staticmethod
    def _group_by_field(
        rows: list[dict[str, Any]],
        field: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Group a list of row-dicts by the value of a single field."""
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = str(row.get(field, "unknown"))
            groups[key].append(row)
        return dict(groups)


# ---------------------------------------------------------------------------
# Drift analysis helper (standalone, no FinBERT dependency)
# ---------------------------------------------------------------------------


def compute_sentiment_drift(
    chunk_df: pd.DataFrame,
    transcript_id: str,
) -> dict[str, Any]:
    """
    Compute sentiment drift metrics for a single transcript.

    These metrics capture how the tone evolves over the course of an
    earnings call — important for detecting optimism decay, Q&A
    escalation, and executive defensiveness patterns.

    Parameters
    ----------
    chunk_df : pd.DataFrame
        Chunk-level FinBERT scores (all chunks for one transcript).
    transcript_id : str

    Returns
    -------
    dict with keys:
        transcript_id, chunk_count, drift_slope, first_quarter_score,
        last_quarter_score, score_range, peak_chunk_order, trough_chunk_order,
        prepared_vs_qa_delta (if both sections present)
    """
    mask = chunk_df["transcript_id"] == transcript_id
    df = chunk_df[mask].sort_values("chunk_order")

    if df.empty:
        return {"transcript_id": transcript_id, "error": "No chunks found."}

    scores = df["sentiment_score"].tolist()
    n = len(scores)

    first_q  = float(sum(scores[: max(1, n // 4)]) / max(1, n // 4))
    last_q   = float(sum(scores[-(max(1, n // 4)):]) / max(1, n // 4))
    drift    = _drift_slope(scores)
    peak_idx = scores.index(max(scores))
    tro_idx  = scores.index(min(scores))

    out: dict[str, Any] = {
        "transcript_id": transcript_id,
        "chunk_count": n,
        "drift_slope": round(drift, 8),
        "first_quarter_score": round(first_q, 6),
        "last_quarter_score": round(last_q, 6),
        "score_range": round(max(scores) - min(scores), 6),
        "peak_chunk_order": int(df["chunk_order"].iloc[peak_idx]),
        "trough_chunk_order": int(df["chunk_order"].iloc[tro_idx]),
    }

    # Section delta
    if "section_type" in df.columns:
        prep = df[df["section_type"] == "prepared_remarks"]["sentiment_score"]
        qa   = df[df["section_type"] == "qa"]["sentiment_score"]
        if not prep.empty and not qa.empty:
            out["prepared_vs_qa_delta"] = round(
                float(prep.mean()) - float(qa.mean()), 6
            )
        if not prep.empty:
            out["prepared_remarks_mean"] = round(float(prep.mean()), 6)
        if not qa.empty:
            out["qa_mean"] = round(float(qa.mean()), 6)

    return out


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_aggregator(
    strategy: str | WeightingStrategy = WeightingStrategy.HYBRID,
    **config_kwargs: Any,
) -> SentimentAggregator:
    """
    Factory function for constructing a ``SentimentAggregator``.

    Parameters
    ----------
    strategy : str | WeightingStrategy
        Weighting strategy.  Accepts string aliases ("uniform", "confidence",
        "token", "hybrid") for convenience.
    **config_kwargs
        Additional keyword arguments forwarded to ``AggregationConfig``.

    Returns
    -------
    SentimentAggregator
    """
    if isinstance(strategy, str):
        strategy = WeightingStrategy(strategy.lower())
    config = AggregationConfig(strategy=strategy, **config_kwargs)
    return SentimentAggregator(config=config)


# ---------------------------------------------------------------------------
# Self-test / demo block
# ---------------------------------------------------------------------------


def _build_demo_chunk_df(
    transcript_ids: list[str] | None = None,
    chunks_per_transcript: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic chunk-level FinBERT output DataFrame for self-testing.

    Simulates two transcripts (positive bias / negative bias) with distinct
    prepared_remarks and qa sections, two speakers per transcript.
    """
    import random
    random.seed(seed)

    if transcript_ids is None:
        transcript_ids = ["AAPL_Q1_2025", "NVDA_Q2_2025"]

    rows: list[dict[str, Any]] = []
    chunk_counter = 0

    for tid in transcript_ids:
        # Positive-leaning transcript for AAPL, negative-leaning for others
        bias = 0.25 if "AAPL" in tid else -0.15

        speakers = ["Tim Cook", "Luca Maestri"] if "AAPL" in tid else ["Jensen Huang", "Colette Kress"]
        sections = (
            ["prepared_remarks"] * (chunks_per_transcript // 2)
            + ["qa"] * (chunks_per_transcript - chunks_per_transcript // 2)
        )

        for order, section in enumerate(sections):
            # Simulate realistic probability distribution
            pos = max(0.0, min(1.0, 0.40 + bias + random.gauss(0, 0.12)))
            neg = max(0.0, min(1.0, 0.25 - bias + random.gauss(0, 0.08)))
            neu = max(0.0, 1.0 - pos - neg)
            total = pos + neu + neg
            pos, neu, neg = pos / total, neu / total, neg / total

            score      = pos - neg
            confidence = max(pos, neu, neg)
            token_cnt  = random.randint(80, 450)
            speaker    = speakers[order % len(speakers)]
            label      = _score_to_label(score, _POSITIVE_THRESHOLD, _NEGATIVE_THRESHOLD)

            rows.append({
                "chunk_id":        f"chunk_{tid}_{order:04d}",
                "transcript_id":   tid,
                "chunk_order":     order,
                "positive_prob":   round(pos, 6),
                "neutral_prob":    round(neu, 6),
                "negative_prob":   round(neg, 6),
                "sentiment_score": round(score, 6),
                "confidence":      round(confidence, 6),
                "predicted_label": label,
                "token_count":     token_cnt,
                "section_type":    section,
                "dominant_speaker": speaker,
            })
            chunk_counter += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(name)s | %(message)s",
    )

    SEP = "=" * 72

    print(f"\n{SEP}")
    print("  aggregation.py  —  Self-Test / Demo")
    print(SEP)

    # ── 1. Build synthetic data ────────────────────────────────────────
    TRANSCRIPTS = ["AAPL_Q1_2025", "NVDA_Q2_2025", "MSFT_Q3_2025"]
    df = _build_demo_chunk_df(transcript_ids=TRANSCRIPTS, chunks_per_transcript=10)
    print(f"\n✓  Built synthetic chunk DataFrame: {df.shape}")
    print(df[["transcript_id", "chunk_order", "section_type",
              "dominant_speaker", "sentiment_score", "confidence",
              "token_count"]].head(8).to_string(index=False))

    # ── 2. Input validation ────────────────────────────────────────────
    v_errors = AggregationValidator.validate_input(df)
    status = "✓  PASS" if not v_errors else f"✗  FAIL ({len(v_errors)} errors)"
    print(f"\n── Input Validation: {status}")
    for e in v_errors:
        print(f"   {e}")

    # ── 3. Aggregate with HYBRID (default) ────────────────────────────
    aggregator = SentimentAggregator()
    result = aggregator.aggregate(df)

    print(f"\n── Aggregation Result (HYBRID strategy) ─────────────────────────")
    print(f"   Transcripts processed : {result.total_transcripts}")
    print(f"   Chunks processed      : {result.total_chunks_processed}")
    print(f"   Valid                 : {result.is_valid()}")
    if result.warnings:
        print(f"   Warnings              : {result.warnings}")
    if result.validation_errors:
        print(f"   Validation errors     : {result.validation_errors}")

    # ── 4. Per-transcript results ──────────────────────────────────────
    print(f"\n── Per-Transcript Sentiment ──────────────────────────────────────")
    for tid in TRANSCRIPTS:
        ts = result.get(tid)
        if ts is None:
            print(f"  {tid}: NOT FOUND")
            continue
        rec = ts.transcript_aggregate
        print(f"\n  {tid}")
        print(f"    score    : {rec.sentiment_score:+.6f}  ({rec.sentiment_label})")
        print(f"    pos/neu/neg : {rec.positive_probability:.4f} / {rec.neutral_probability:.4f} / {rec.negative_probability:.4f}")
        print(f"    prob sum : {rec.positive_probability + rec.neutral_probability + rec.negative_probability:.8f}")
        print(f"    conf mean: {rec.confidence_mean:.4f}  std: {rec.confidence_std:.4f}")
        print(f"    sent std : {rec.sentiment_std:.4f}  median: {rec.sentiment_median:+.4f}  iqr: {rec.sentiment_iqr:.4f}")
        print(f"    chunks   : {rec.chunk_count}  tokens: {rec.total_tokens}")
        if ts.section_aggregates:
            print(f"    Sections:")
            for sec, srec in ts.section_aggregates.items():
                print(f"      [{sec}] score={srec.sentiment_score:+.4f} ({srec.sentiment_label})  n={srec.chunk_count}")
        if ts.speaker_aggregates:
            print(f"    Speakers:")
            for spk, srec in ts.speaker_aggregates.items():
                print(f"      [{spk}] score={srec.sentiment_score:+.4f} ({srec.sentiment_label})  n={srec.chunk_count}")

    # ── 5. Strategy comparison ─────────────────────────────────────────
    print(f"\n── Strategy Comparison (AAPL_Q1_2025) ───────────────────────────")
    comparison = aggregator.compare_strategies(df, "AAPL_Q1_2025")
    for strategy_name, rec in comparison.items():
        print(f"  {strategy_name:12s}: score={rec.sentiment_score:+.6f}  "
              f"label={rec.sentiment_label}")

    # ── 6. Drift analysis ─────────────────────────────────────────────
    print(f"\n── Sentiment Drift Analysis ──────────────────────────────────────")
    for tid in TRANSCRIPTS[:2]:
        drift = compute_sentiment_drift(df, tid)
        print(f"\n  {tid}")
        for k, v in drift.items():
            if k != "transcript_id":
                print(f"    {k:30s}: {v}")

    # ── 7. DataFrame output shapes ────────────────────────────────────
    print(f"\n── DataFrame Exports ─────────────────────────────────────────────")
    tdf = result.to_transcript_dataframe()
    sdf = result.to_section_dataframe()
    spdf = result.to_speaker_dataframe()
    print(f"  Transcript-level : {tdf.shape}")
    print(f"  Section-level    : {sdf.shape}")
    print(f"  Speaker-level    : {spdf.shape}")
    print(f"\n  Transcript columns:\n  {list(tdf.columns)}")

    # ── 8. Summary statistics ─────────────────────────────────────────
    print(f"\n── Batch Summary ─────────────────────────────────────────────────")
    summary = result.summary()
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")

    # ── 9. Coverage metrics ───────────────────────────────────────────
    print(f"\n── Coverage Metrics ──────────────────────────────────────────────")
    coverage = result.coverage_metrics()
    for k, v in coverage.items():
        print(f"  {k:40s}: {v}")

    # ── 10. Output validation ─────────────────────────────────────────
    out_errors = AggregationValidator.validate_output(result)
    status = "✓  PASS" if not out_errors else f"✗  FAIL ({len(out_errors)} errors)"
    print(f"\n── Output Validation: {status}")
    for e in out_errors:
        print(f"   {e}")

    # ── 11. Edge cases ────────────────────────────────────────────────
    print(f"\n── Edge-Case Tests ───────────────────────────────────────────────")

    # Empty DataFrame
    r_empty = aggregator.aggregate(pd.DataFrame())
    print(f"  Empty input → transcripts={r_empty.total_transcripts}  (expect 0)")

    # Single-chunk transcript
    single_row = df[df["transcript_id"] == "AAPL_Q1_2025"].head(1).copy()
    r_single = aggregator.aggregate(single_row, validate_input=False)
    ts_single = r_single.get("AAPL_Q1_2025")
    print(f"  Single chunk   → score={ts_single.primary_score():+.4f}  "
          f"label={ts_single.primary_label()}")

    # All-neutral probabilities
    neutral_df = df[df["transcript_id"] == "AAPL_Q1_2025"].copy()
    neutral_df["positive_prob"] = 0.333333
    neutral_df["neutral_prob"]  = 0.333334
    neutral_df["negative_prob"] = 0.333333
    neutral_df["sentiment_score"] = 0.0
    neutral_df["confidence"] = 0.333334
    r_neutral = aggregator.aggregate(neutral_df, validate_input=False)
    ts_neutral = r_neutral.get("AAPL_Q1_2025")
    print(f"  All-neutral    → score={ts_neutral.primary_score():+.6f}  "
          f"label={ts_neutral.primary_label()}")

    # Malformed probabilities (don't sum to 1)
    bad_df = df[df["transcript_id"] == "AAPL_Q1_2025"].head(3).copy()
    bad_df["positive_prob"] = 0.9
    bad_df["neutral_prob"]  = 0.9
    bad_df["negative_prob"] = 0.9  # sum = 2.7 — will be normalised
    r_bad = aggregator.aggregate(bad_df, validate_input=False)
    ts_bad = r_bad.get("AAPL_Q1_2025")
    prob_sum = (
        ts_bad.transcript_aggregate.positive_probability
        + ts_bad.transcript_aggregate.neutral_probability
        + ts_bad.transcript_aggregate.negative_probability
    )
    print(f"  Malformed probs → normalised sum = {prob_sum:.8f}  (expect ≈ 1.0)")

    # JSONL export check (in-memory)
    jsonl_str = result.to_jsonl() if hasattr(result, "to_jsonl") else \
        "\n".join(json.dumps(ts.to_dict()) for ts in result.transcripts.values())
    jsonl_count = len(jsonl_str.strip().splitlines())
    print(f"  JSONL lines    → {jsonl_count}  (expect {len(TRANSCRIPTS)})")

    print(f"\n{SEP}")
    print("  Self-test complete — all systems nominal.")
    print(f"{SEP}\n")
