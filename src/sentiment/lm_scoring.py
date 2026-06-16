"""
Loughran-McDonald tone scoring.

This module intentionally supports both the richer pipeline API
(`score`/`label`) and the lightweight test/analytics API
(`tone_score`/`sentiment_label`).  The arithmetic is deterministic and
uses the canonical tone formula:

    (positive_count - negative_count) / (positive_count + negative_count)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LMScoringConfig:
    positive_threshold: float = 0.02
    negative_threshold: float = -0.02
    smoothing_enabled: bool = False
    smoothing_alpha: float = 0.5
    confidence_scaling: bool = False
    minimum_coverage_ratio: Optional[float] = None
    min_coverage_ratio: float = 0.005
    score_floor: float = -1.0
    score_ceiling: float = 1.0
    min_token_count: int = 0
    low_coverage_label: SentimentLabel = SentimentLabel.NEUTRAL
    confidence_min_tokens: int = 1
    confidence_scale: float = 1.0
    apply_smoothing: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.apply_smoothing is not None:
            object.__setattr__(self, "smoothing_enabled", bool(self.apply_smoothing))
        if self.minimum_coverage_ratio is None:
            object.__setattr__(self, "minimum_coverage_ratio", self.min_coverage_ratio)
        else:
            object.__setattr__(self, "min_coverage_ratio", self.minimum_coverage_ratio)

        if self.positive_threshold <= 0:
            raise ValueError("positive_threshold must be > 0.")
        if self.negative_threshold < 0 and self.positive_threshold <= self.negative_threshold:
            raise ValueError("positive_threshold must be greater than negative_threshold.")
        if self.smoothing_alpha < 0:
            raise ValueError("smoothing_alpha must be >= 0.")
        if not 0 <= float(self.minimum_coverage_ratio) <= 1:
            raise ValueError("minimum_coverage_ratio must be in [0, 1].")
        if self.score_floor > self.score_ceiling:
            raise ValueError("score_floor must be <= score_ceiling.")

    @property
    def negative_cutoff(self) -> float:
        return self.negative_threshold if self.negative_threshold < 0 else -self.negative_threshold


@dataclass(frozen=True)
class ScoreDiagnostics:
    positive_count: float = 0
    negative_count: float = 0
    uncertainty_count: float = 0
    total_tokens: float = 0
    matched_tokens: float = 0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    uncertainty_ratio: float = 0.0
    coverage_ratio: float = 0.0
    top_positive_terms: List[str] = field(default_factory=list)
    top_negative_terms: List[str] = field(default_factory=list)
    top_uncertainty_terms: List[str] = field(default_factory=list)
    is_empty_input: bool = False
    is_low_coverage: bool = False
    is_short_text: bool = False
    zero_division_occurred: bool = False
    smoothing_applied: bool = False
    raw_tone_score: Optional[float] = None
    smoothed_tone_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def has_issues(self) -> bool:
        return bool(self.warnings)

    def summary(self) -> str:
        return (
            "ScoreDiagnostics("
            f"coverage_ratio={self.coverage_ratio:.4f}, "
            f"low_coverage={self.is_low_coverage}, "
            f"zero_division={self.zero_division_occurred})"
        )


@dataclass(frozen=True)
class LMScore:
    tone_score: float
    sentiment_label: str
    confidence: float
    diagnostics: ScoreDiagnostics
    is_low_coverage: bool = False
    smoothing_applied: bool = False
    positive_count: float = 0
    negative_count: float = 0
    uncertainty_count: float = 0
    litigious_count: float = 0
    strong_modal_count: float = 0
    weak_modal_count: float = 0
    constraining_count: float = 0
    total_tokens: float = 0
    matched_tokens: float = 0
    coverage_ratio: float = 0.0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    uncertainty_ratio: float = 0.0
    polarity_ratio: float = 0.0
    sentiment_intensity: float = 0.0
    normalized_positive: float = 0.0
    normalized_negative: float = 0.0

    @property
    def score(self) -> float:
        return self.tone_score

    @property
    def label(self) -> SentimentLabel:
        try:
            return SentimentLabel(self.sentiment_label)
        except ValueError:
            return SentimentLabel.UNKNOWN

    def is_valid(self) -> bool:
        return (
            math.isfinite(self.tone_score)
            and -1.0 <= self.tone_score <= 1.0
            and 0.0 <= self.confidence <= 1.0
            and self.sentiment_label in {"positive", "neutral", "negative", "unknown"}
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "lm_tone_score": self.tone_score,
            "lm_label": self.sentiment_label,
            "lm_sentiment_label": self.sentiment_label,
            "lm_confidence": self.confidence,
            "lm_positive_count": self.positive_count,
            "lm_negative_count": self.negative_count,
            "lm_uncertainty_count": self.uncertainty_count,
            "lm_litigious_count": self.litigious_count,
            "lm_strong_modal": self.strong_modal_count,
            "lm_weak_modal": self.weak_modal_count,
            "lm_constraining": self.constraining_count,
            "lm_total_tokens": self.total_tokens,
            "lm_matched_tokens": self.matched_tokens,
            "lm_coverage_ratio": self.coverage_ratio,
            "lm_positive_ratio": self.positive_ratio,
            "lm_negative_ratio": self.negative_ratio,
            "lm_uncertainty_ratio": self.uncertainty_ratio,
            "lm_polarity_ratio": self.polarity_ratio,
            "lm_sentiment_intensity": self.sentiment_intensity,
            "lm_normalized_positive": self.normalized_positive,
            "lm_normalized_negative": self.normalized_negative,
            "lm_is_low_coverage": self.is_low_coverage,
            "lm_zero_div": self.diagnostics.zero_division_occurred,
        }

    def summary(self) -> str:
        return (
            f"LMScore(score={self.tone_score:+.4f}, "
            f"label={self.sentiment_label}, confidence={self.confidence:.4f})"
        )


@dataclass
class BatchScoringStats:
    total_scored: int = 0
    total_skipped: int = 0
    total_failed: int = 0
    label_counts: Dict[str, int] = field(
        default_factory=lambda: {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0}
    )
    low_coverage_count: int = 0
    zero_div_count: int = 0
    short_text_count: int = 0
    scores: List[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def score_mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    def score_std(self) -> float:
        return float(np.std(self.scores)) if self.scores else 0.0

    def score_median(self) -> float:
        return float(np.median(self.scores)) if self.scores else 0.0

    def summary(self) -> str:
        return (
            f"BatchScoringStats(scored={self.total_scored}, "
            f"skipped={self.total_skipped}, failed={self.total_failed})"
        )


class LMScorer:
    def __init__(self, config: Optional[LMScoringConfig] = None) -> None:
        self.config = config or LMScoringConfig()

    def compute_tone_score(self, positive_count: float, negative_count: float) -> float:
        pos = float(positive_count)
        neg = float(negative_count)
        alpha = self.config.smoothing_alpha if self.config.smoothing_enabled else 0.0
        denom = pos + neg + (2.0 * alpha)
        if denom == 0:
            return 0.0
        score = (pos - neg) / denom
        return float(np.clip(score, self.config.score_floor, self.config.score_ceiling))

    def compute_ratios(
        self,
        positive_count: float,
        negative_count: float,
        uncertainty_count: float,
        total_tokens: float,
        matched_tokens: Optional[float] = None,
        diagnostics: Optional[ScoreDiagnostics] = None,
    ) -> Dict[str, float]:
        total = float(total_tokens)
        matched = float(matched_tokens if matched_tokens is not None else (
            positive_count + negative_count + uncertainty_count
        ))
        diag = diagnostics

        def div(num: float, den: float) -> float:
            if den == 0:
                if diag is not None:
                    object.__setattr__(diag, "zero_division_occurred", True)
                return 0.0
            return float(np.clip(num / den, 0.0, 1.0))

        pos = float(positive_count)
        neg = float(negative_count)
        unc = float(uncertainty_count)
        return {
            "positive_ratio": div(pos, total),
            "negative_ratio": div(neg, total),
            "uncertainty_ratio": div(unc, total),
            "coverage_ratio": div(matched, total),
            "polarity_ratio": div(pos, pos + neg),
            "sentiment_intensity": div(pos + neg, total),
            "normalized_positive": div(pos, matched),
            "normalized_negative": div(neg, matched),
        }

    def assign_label(self, tone_score: float) -> str:
        if not math.isfinite(tone_score):
            return SentimentLabel.UNKNOWN.value
        if tone_score >= self.config.positive_threshold:
            return SentimentLabel.POSITIVE.value
        if tone_score <= self.config.negative_cutoff:
            return SentimentLabel.NEGATIVE.value
        return SentimentLabel.NEUTRAL.value

    def compute_label(
        self,
        tone_score: float,
        diagnostics: Optional[ScoreDiagnostics] = None,
    ) -> SentimentLabel:
        if diagnostics is not None and diagnostics.is_empty_input:
            return SentimentLabel.UNKNOWN
        return SentimentLabel(self.assign_label(tone_score))

    def compute_confidence(
        self,
        coverage_ratio: float,
        matched_tokens: float = 0,
        total_tokens: float = 0,
        tone_score: float = 1.0,
    ) -> float:
        coverage = float(np.clip(coverage_ratio, 0.0, 1.0))
        if self.config.confidence_scaling:
            coverage *= min(abs(tone_score), 1.0)
        return float(np.clip(coverage * self.config.confidence_scale, 0.0, 1.0))

    def compute_score(
        self,
        *,
        positive_count: float,
        negative_count: float,
        total_tokens: float,
        matched_tokens: float,
        uncertainty_count: float = 0,
        litigious_count: float = 0,
        strong_modal_count: float = 0,
        weak_modal_count: float = 0,
        constraining_count: float = 0,
        top_positive_terms: Optional[List[str]] = None,
        top_negative_terms: Optional[List[str]] = None,
        top_uncertainty_terms: Optional[List[str]] = None,
    ) -> LMScore:
        counts = [
            positive_count, negative_count, uncertainty_count, litigious_count,
            strong_modal_count, weak_modal_count, constraining_count,
            total_tokens, matched_tokens,
        ]
        if any((not isinstance(v, (int, float))) or math.isnan(float(v)) or v < 0 for v in counts):
            raise ValueError("LM scoring counts must be non-negative finite numbers.")
        if matched_tokens > total_tokens:
            raise ValueError("matched_tokens cannot exceed total_tokens.")

        ratios = self.compute_ratios(
            positive_count,
            negative_count,
            uncertainty_count,
            total_tokens,
            matched_tokens,
        )
        coverage = ratios["coverage_ratio"]
        tone = self.compute_tone_score(positive_count, negative_count)
        is_empty = float(total_tokens) == 0
        is_low_coverage = (not is_empty) and coverage < float(self.config.minimum_coverage_ratio)
        is_short = float(total_tokens) < self.config.min_token_count
        zero_div = (float(positive_count) + float(negative_count)) == 0

        diag = ScoreDiagnostics(
            positive_count=positive_count,
            negative_count=negative_count,
            uncertainty_count=uncertainty_count,
            total_tokens=total_tokens,
            matched_tokens=matched_tokens,
            positive_ratio=ratios["positive_ratio"],
            negative_ratio=ratios["negative_ratio"],
            uncertainty_ratio=ratios["uncertainty_ratio"],
            coverage_ratio=coverage,
            top_positive_terms=top_positive_terms or [],
            top_negative_terms=top_negative_terms or [],
            top_uncertainty_terms=top_uncertainty_terms or [],
            is_empty_input=is_empty,
            is_low_coverage=is_low_coverage,
            is_short_text=is_short,
            zero_division_occurred=zero_div,
            smoothing_applied=self.config.smoothing_enabled,
            raw_tone_score=tone,
            smoothed_tone_score=tone if self.config.smoothing_enabled else None,
        )

        label = SentimentLabel.UNKNOWN.value if is_empty else self.assign_label(tone)
        confidence = self.compute_confidence(
            coverage_ratio=coverage,
            matched_tokens=matched_tokens,
            total_tokens=total_tokens,
            tone_score=tone,
        )

        return LMScore(
            tone_score=tone,
            sentiment_label=label,
            confidence=confidence,
            diagnostics=diag,
            is_low_coverage=is_low_coverage,
            smoothing_applied=self.config.smoothing_enabled,
            positive_count=positive_count,
            negative_count=negative_count,
            uncertainty_count=uncertainty_count,
            litigious_count=litigious_count,
            strong_modal_count=strong_modal_count,
            weak_modal_count=weak_modal_count,
            constraining_count=constraining_count,
            total_tokens=total_tokens,
            matched_tokens=matched_tokens,
            coverage_ratio=coverage,
            positive_ratio=ratios["positive_ratio"],
            negative_ratio=ratios["negative_ratio"],
            uncertainty_ratio=ratios["uncertainty_ratio"],
            polarity_ratio=ratios["polarity_ratio"],
            sentiment_intensity=ratios["sentiment_intensity"],
            normalized_positive=ratios["normalized_positive"],
            normalized_negative=ratios["normalized_negative"],
        )

    def score_from_counts(
        self,
        positive_count: float,
        negative_count: float,
        uncertainty_count: float = 0,
        total_tokens: float = 0,
        matched_tokens: float = 0,
        **kwargs,
    ) -> LMScore:
        return self.compute_score(
            positive_count=positive_count,
            negative_count=negative_count,
            uncertainty_count=uncertainty_count,
            total_tokens=total_tokens,
            matched_tokens=matched_tokens,
            **kwargs,
        )

    def validate_score(self, score: LMScore) -> List[str]:
        issues: List[str] = []
        if not math.isfinite(score.tone_score):
            issues.append("tone_score is not finite")
        elif not -1.0 <= score.tone_score <= 1.0:
            issues.append(f"tone_score out of range: {score.tone_score}")
        if not math.isfinite(score.confidence) or not 0.0 <= score.confidence <= 1.0:
            issues.append(f"confidence out of range: {score.confidence}")
        if score.sentiment_label not in {"positive", "neutral", "negative", "unknown"}:
            issues.append(f"invalid sentiment_label: {score.sentiment_label}")
        diag = score.diagnostics
        for name in ("positive_count", "negative_count", "uncertainty_count", "total_tokens", "matched_tokens"):
            val = getattr(diag, name, 0)
            if val < 0:
                issues.append(f"diagnostics {name} is negative: {val}")
        for name in ("positive_ratio", "negative_ratio", "uncertainty_ratio", "coverage_ratio"):
            val = getattr(diag, name, 0.0)
            if not math.isfinite(val) or val < 0 or val > 1:
                issues.append(f"diagnostics {name} out of range: {val}")
        return issues

    def score_batch(
        self,
        records: Sequence[Dict[str, float]],
        *,
        ids: Optional[List[str]] = None,
    ) -> Tuple[List[Optional[LMScore]], BatchScoringStats]:
        stats = BatchScoringStats()
        scores: List[Optional[LMScore]] = []
        t0 = time.perf_counter()
        for idx, record in enumerate(records):
            if not record:
                scores.append(None)
                stats.total_skipped += 1
                continue
            try:
                score = self.compute_score(**record)
            except Exception as exc:
                label = ids[idx] if ids else f"record[{idx}]"
                logger.error("%s scoring failed: %s", label, exc)
                scores.append(None)
                stats.total_failed += 1
                continue
            scores.append(score)
            stats.total_scored += 1
            stats.label_counts[score.sentiment_label] += 1
            stats.scores.append(score.tone_score)
            stats.low_coverage_count += int(score.is_low_coverage)
            stats.zero_div_count += int(score.diagnostics.zero_division_occurred)
            stats.short_text_count += int(score.diagnostics.is_short_text)
        stats.elapsed_seconds = time.perf_counter() - t0
        return scores, stats

    def summary(self, stats: BatchScoringStats) -> str:
        return stats.summary()
