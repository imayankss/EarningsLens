"""
tests/test_lm_scoring.py
========================
Comprehensive unit test suite for lm_scoring.py.

Coverage
--------
1.  LMScoringConfig — construction, defaults, validation
2.  LMScore — field presence, immutability, serialisation
3.  ScoreDiagnostics — ratio consistency, field integrity
4.  LMScorer.compute_tone_score — exact formula validation
5.  LMScorer.compute_ratios — ratio arithmetic
6.  LMScorer.assign_label — threshold boundary behaviour
7.  LMScorer.compute_confidence — bounds and scaling
8.  LMScorer.score_from_counts — integration of all components
9.  Smoothing behaviour — alpha semantics, zero-count stability
10. Edge cases — zeros, huge counts, sparse matches, NaN safety
11. Coverage handling — minimum_coverage_ratio gating
12. Determinism — identical inputs → identical outputs
13. Numeric stability — large counts, near-boundary scores
14. Validation helpers — invalid inputs rejected

Test design principles
----------------------
* No external I/O, no network calls, no heavy model loading.
* All expected values are derived analytically from the formula:
      LM Tone = (Positive − Negative) / (Positive + Negative)
  With Laplace-style smoothing (alpha):
      LM Tone = (Positive − Negative) / (Positive + Negative + 2α)
* Floating-point comparisons use math.isclose (rel_tol=1e-9) or
  pytest.approx with explicit tolerances.
* Every fixture is a pure function — no shared mutable state.
"""

from __future__ import annotations

import math
import sys
from dataclasses import fields as dc_fields
from typing import Any, Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
# Adjust the import path to match your project layout if needed.
# ---------------------------------------------------------------------------
try:
    from src.sentiment.lm_scoring import (
        LMScoringConfig,
        LMScore,
        ScoreDiagnostics,
        LMScorer,
    )
except ImportError:
    from lm_scoring import (   # type: ignore[no-redef]
        LMScoringConfig,
        LMScore,
        ScoreDiagnostics,
        LMScorer,
    )


# ===========================================================================
# ── Helpers ─────────────────────────────────────────────────────────────────
# ===========================================================================

def _tone(pos: float, neg: float, alpha: float = 0.0) -> float:
    """
    Reference implementation of the LM tone formula used to derive
    expected values in assertions.

    With smoothing (alpha > 0):
        tone = (pos − neg) / (pos + neg + 2α)
    Without smoothing:
        tone = (pos − neg) / (pos + neg)   if pos + neg > 0
        tone = 0.0                         otherwise
    """
    denom = pos + neg + 2 * alpha
    if denom == 0:
        return 0.0
    return (pos - neg) / denom


def _make_scorer(
    positive_threshold: float = 0.02,
    negative_threshold: float = -0.02,
    smoothing_enabled: bool = True,
    smoothing_alpha: float = 0.5,
    confidence_scaling: bool = False,
    minimum_coverage_ratio: float = 0.005,
) -> LMScorer:
    """Factory for LMScorer with explicit config, avoiding default-drift."""
    cfg = LMScoringConfig(
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        smoothing_enabled=smoothing_enabled,
        smoothing_alpha=smoothing_alpha,
        confidence_scaling=confidence_scaling,
        minimum_coverage_ratio=minimum_coverage_ratio,
    )
    return LMScorer(cfg)


def _score(
    pos: int,
    neg: int,
    unc: int = 0,
    total: int = 100,
    matched: int = 20,
    **kwargs,
) -> LMScore:
    """Convenience wrapper around LMScorer.score_from_counts."""
    scorer = _make_scorer(**kwargs)
    return scorer.score_from_counts(
        positive_count=pos,
        negative_count=neg,
        uncertainty_count=unc,
        total_tokens=total,
        matched_tokens=matched,
    )


# ===========================================================================
# ── 1. LMScoringConfig ───────────────────────────────────────────────────────
# ===========================================================================

class TestLMScoringConfig:

    def test_default_construction(self):
        cfg = LMScoringConfig()
        assert cfg.positive_threshold > 0
        assert cfg.negative_threshold < 0
        assert cfg.smoothing_alpha >= 0
        assert 0.0 <= cfg.minimum_coverage_ratio <= 1.0

    def test_positive_threshold_above_zero(self):
        cfg = LMScoringConfig(positive_threshold=0.02)
        assert cfg.positive_threshold == pytest.approx(0.02)

    def test_negative_threshold_below_zero(self):
        cfg = LMScoringConfig(negative_threshold=-0.02)
        assert cfg.negative_threshold == pytest.approx(-0.02)

    def test_threshold_asymmetry_allowed(self):
        """positive and negative thresholds may differ in magnitude."""
        cfg = LMScoringConfig(positive_threshold=0.05, negative_threshold=-0.10)
        assert cfg.positive_threshold == pytest.approx(0.05)
        assert cfg.negative_threshold == pytest.approx(-0.10)

    def test_smoothing_alpha_zero_allowed(self):
        cfg = LMScoringConfig(smoothing_enabled=False, smoothing_alpha=0.0)
        assert cfg.smoothing_alpha == pytest.approx(0.0)

    def test_smoothing_alpha_one(self):
        cfg = LMScoringConfig(smoothing_alpha=1.0)
        assert cfg.smoothing_alpha == pytest.approx(1.0)

    def test_immutability(self):
        cfg = LMScoringConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.positive_threshold = 0.99  # type: ignore[misc]

    def test_minimum_coverage_ratio_range(self):
        cfg = LMScoringConfig(minimum_coverage_ratio=0.01)
        assert 0.0 <= cfg.minimum_coverage_ratio <= 1.0

    def test_score_floor_ceiling_defaults(self):
        cfg = LMScoringConfig()
        assert cfg.score_floor == pytest.approx(-1.0)
        assert cfg.score_ceiling == pytest.approx(1.0)

    def test_invalid_positive_threshold_raises(self):
        """positive_threshold must be > negative_threshold."""
        with pytest.raises((ValueError, AssertionError)):
            LMScoringConfig(positive_threshold=-0.10, negative_threshold=0.10)

    def test_invalid_negative_smoothing_alpha_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            LMScoringConfig(smoothing_alpha=-1.0)

    def test_invalid_coverage_ratio_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            LMScoringConfig(minimum_coverage_ratio=1.5)


# ===========================================================================
# ── 2. ScoreDiagnostics ──────────────────────────────────────────────────────
# ===========================================================================

class TestScoreDiagnostics:

    def _make(self, **kwargs) -> ScoreDiagnostics:
        defaults = dict(
            positive_count=10,
            negative_count=3,
            uncertainty_count=2,
            total_tokens=200,
            matched_tokens=25,
            positive_ratio=0.05,
            negative_ratio=0.015,
            uncertainty_ratio=0.01,
            coverage_ratio=0.125,
        )
        defaults.update(kwargs)
        return ScoreDiagnostics(**defaults)

    def test_construction_succeeds(self):
        diag = self._make()
        assert diag.positive_count == 10
        assert diag.negative_count == 3

    def test_immutability(self):
        diag = self._make()
        with pytest.raises((AttributeError, TypeError)):
            diag.positive_count = 99  # type: ignore[misc]

    def test_ratio_fields_present(self):
        diag = self._make()
        assert hasattr(diag, "positive_ratio")
        assert hasattr(diag, "negative_ratio")
        assert hasattr(diag, "uncertainty_ratio")
        assert hasattr(diag, "coverage_ratio")

    def test_coverage_ratio_in_unit_interval(self):
        diag = self._make(coverage_ratio=0.125)
        assert 0.0 <= diag.coverage_ratio <= 1.0

    def test_non_negative_counts(self):
        diag = self._make(positive_count=0, negative_count=0, uncertainty_count=0)
        assert diag.positive_count >= 0
        assert diag.negative_count >= 0
        assert diag.uncertainty_count >= 0


# ===========================================================================
# ── 3. LMScore ───────────────────────────────────────────────────────────────
# ===========================================================================

class TestLMScore:

    def _make(self, tone: float = 0.10, label: str = "positive") -> LMScore:
        diag = ScoreDiagnostics(
            positive_count=10,
            negative_count=2,
            uncertainty_count=1,
            total_tokens=100,
            matched_tokens=15,
            positive_ratio=0.10,
            negative_ratio=0.02,
            uncertainty_ratio=0.01,
            coverage_ratio=0.15,
        )
        return LMScore(
            tone_score=tone,
            sentiment_label=label,
            confidence=0.80,
            diagnostics=diag,
            is_low_coverage=False,
            smoothing_applied=True,
        )

    def test_required_fields_present(self):
        score = self._make()
        assert hasattr(score, "tone_score")
        assert hasattr(score, "sentiment_label")
        assert hasattr(score, "confidence")
        assert hasattr(score, "diagnostics")
        assert hasattr(score, "is_low_coverage")
        assert hasattr(score, "smoothing_applied")

    def test_tone_score_in_range(self):
        score = self._make(tone=0.30)
        assert -1.0 <= score.tone_score <= 1.0

    def test_valid_label_values(self):
        for label in ("positive", "neutral", "negative"):
            score = self._make(label=label)
            assert score.sentiment_label == label

    def test_confidence_in_unit_interval(self):
        score = self._make()
        assert 0.0 <= score.confidence <= 1.0

    def test_immutability(self):
        score = self._make()
        with pytest.raises((AttributeError, TypeError)):
            score.tone_score = 0.99  # type: ignore[misc]

    def test_diagnostics_type(self):
        score = self._make()
        assert isinstance(score.diagnostics, ScoreDiagnostics)

    def test_bool_flags_are_bool(self):
        score = self._make()
        assert isinstance(score.is_low_coverage, bool)
        assert isinstance(score.smoothing_applied, bool)


# ===========================================================================
# ── 4. Core Tone Score Formula ───────────────────────────────────────────────
# ===========================================================================

class TestToneScoreFormula:
    """Exact numeric validation of LM tone formula with and without smoothing."""

    ALPHA = 0.5

    # ------------------------------------------------------------------ #
    # 4a. Without smoothing                                               #
    # ------------------------------------------------------------------ #

    def test_positive_heavy_no_smoothing(self):
        """pos=10, neg=2, alpha=0 → (10-2)/(10+2) = 8/12 ≈ 0.666667"""
        scorer = _make_scorer(smoothing_enabled=False, smoothing_alpha=0.0)
        result = scorer.compute_tone_score(positive_count=10, negative_count=2)
        expected = _tone(10, 2, alpha=0.0)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"Expected {expected:.10f}, got {result:.10f}"
        )

    def test_negative_heavy_no_smoothing(self):
        """pos=2, neg=10, alpha=0 → (2-10)/12 = -8/12 ≈ -0.666667"""
        scorer = _make_scorer(smoothing_enabled=False, smoothing_alpha=0.0)
        result = scorer.compute_tone_score(positive_count=2, negative_count=10)
        expected = _tone(2, 10, alpha=0.0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_balanced_no_smoothing(self):
        """pos=5, neg=5, alpha=0 → 0/10 = 0.0"""
        scorer = _make_scorer(smoothing_enabled=False, smoothing_alpha=0.0)
        result = scorer.compute_tone_score(positive_count=5, negative_count=5)
        assert math.isclose(result, 0.0, abs_tol=1e-12)

    def test_all_positive_no_smoothing(self):
        """pos=N, neg=0 → N/N = 1.0 for any N > 0"""
        scorer = _make_scorer(smoothing_enabled=False, smoothing_alpha=0.0)
        result = scorer.compute_tone_score(positive_count=20, negative_count=0)
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_all_negative_no_smoothing(self):
        """pos=0, neg=N → -N/N = -1.0 for any N > 0"""
        scorer = _make_scorer(smoothing_enabled=False, smoothing_alpha=0.0)
        result = scorer.compute_tone_score(positive_count=0, negative_count=20)
        assert math.isclose(result, -1.0, rel_tol=1e-9)

    # ------------------------------------------------------------------ #
    # 4b. With Laplace smoothing (alpha=0.5)                              #
    # ------------------------------------------------------------------ #

    def test_positive_heavy_with_smoothing(self):
        """pos=10, neg=2, alpha=0.5 → (10-2)/(10+2+1) = 8/13 ≈ 0.615385"""
        scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=0.5)
        result = scorer.compute_tone_score(positive_count=10, negative_count=2)
        expected = _tone(10, 2, alpha=0.5)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"Expected {expected:.10f}, got {result:.10f}"
        )

    def test_negative_heavy_with_smoothing(self):
        """pos=2, neg=10, alpha=0.5 → -8/13 ≈ -0.615385"""
        scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=0.5)
        result = scorer.compute_tone_score(positive_count=2, negative_count=10)
        expected = _tone(2, 10, alpha=0.5)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_balanced_with_smoothing(self):
        """pos=5, neg=5, alpha=0.5 → 0/(10+1) = 0.0"""
        scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=0.5)
        result = scorer.compute_tone_score(positive_count=5, negative_count=5)
        assert math.isclose(result, 0.0, abs_tol=1e-12)

    def test_zero_counts_with_smoothing(self):
        """pos=0, neg=0, alpha=0.5 → 0/(0+0+1) = 0.0"""
        scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=0.5)
        result = scorer.compute_tone_score(positive_count=0, negative_count=0)
        assert math.isclose(result, 0.0, abs_tol=1e-12)

    def test_smoothing_alpha_one(self):
        """pos=6, neg=2, alpha=1.0 → (6-2)/(6+2+2) = 4/10 = 0.4"""
        scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=1.0)
        result = scorer.compute_tone_score(positive_count=6, negative_count=2)
        expected = _tone(6, 2, alpha=1.0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_smoothing_shrinks_magnitude_vs_no_smoothing(self):
        """Smoothing always reduces |score| compared to raw formula."""
        pos, neg = 8, 2
        raw_score = _tone(pos, neg, alpha=0.0)
        smoothed_score = _tone(pos, neg, alpha=0.5)
        assert abs(smoothed_score) < abs(raw_score)

    # ------------------------------------------------------------------ #
    # 4c. Symmetry                                                        #
    # ------------------------------------------------------------------ #

    def test_antisymmetry(self):
        """score(pos, neg) == -score(neg, pos) for any alpha."""
        scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=0.5)
        s1 = scorer.compute_tone_score(positive_count=7, negative_count=3)
        s2 = scorer.compute_tone_score(positive_count=3, negative_count=7)
        assert math.isclose(s1, -s2, rel_tol=1e-9)

    def test_result_in_closed_interval(self):
        """Score must always lie in [-1, 1]."""
        scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=0.5)
        for pos, neg in [(0, 0), (1, 0), (0, 1), (100, 1), (1, 100), (50, 50)]:
            s = scorer.compute_tone_score(positive_count=pos, negative_count=neg)
            assert -1.0 <= s <= 1.0, f"Score {s} out of range for pos={pos} neg={neg}"

    def test_finite_output(self):
        scorer = _make_scorer()
        for pos, neg in [(0, 0), (1000, 0), (0, 1000)]:
            s = scorer.compute_tone_score(positive_count=pos, negative_count=neg)
            assert math.isfinite(s)

    def test_not_nan(self):
        scorer = _make_scorer()
        for pos, neg in [(0, 0), (5, 5), (10, 3)]:
            s = scorer.compute_tone_score(positive_count=pos, negative_count=neg)
            assert not math.isnan(s)


# ===========================================================================
# ── 5. Ratio Computation ─────────────────────────────────────────────────────
# ===========================================================================

class TestRatioComputation:

    def test_positive_ratio(self):
        """positive_ratio = positive_count / total_tokens."""
        scorer = _make_scorer()
        ratios = scorer.compute_ratios(
            positive_count=10,
            negative_count=5,
            uncertainty_count=3,
            total_tokens=100,
        )
        assert math.isclose(ratios["positive_ratio"], 0.10, rel_tol=1e-9)

    def test_negative_ratio(self):
        scorer = _make_scorer()
        ratios = scorer.compute_ratios(
            positive_count=10,
            negative_count=5,
            uncertainty_count=3,
            total_tokens=100,
        )
        assert math.isclose(ratios["negative_ratio"], 0.05, rel_tol=1e-9)

    def test_uncertainty_ratio(self):
        scorer = _make_scorer()
        ratios = scorer.compute_ratios(
            positive_count=10,
            negative_count=5,
            uncertainty_count=3,
            total_tokens=100,
        )
        assert math.isclose(ratios["uncertainty_ratio"], 0.03, rel_tol=1e-9)

    def test_coverage_ratio(self):
        """coverage_ratio = matched_tokens / total_tokens."""
        scorer = _make_scorer()
        ratios = scorer.compute_ratios(
            positive_count=10,
            negative_count=5,
            uncertainty_count=3,
            total_tokens=200,
            matched_tokens=40,
        )
        assert math.isclose(ratios["coverage_ratio"], 0.20, rel_tol=1e-9)

    def test_ratios_sum_leq_one(self):
        """pos + neg + unc ratios must be ≤ 1 (they are disjoint subsets)."""
        scorer = _make_scorer()
        ratios = scorer.compute_ratios(
            positive_count=10,
            negative_count=5,
            uncertainty_count=3,
            total_tokens=100,
        )
        total_cat = (
            ratios["positive_ratio"]
            + ratios["negative_ratio"]
            + ratios["uncertainty_ratio"]
        )
        assert total_cat <= 1.0 + 1e-9

    def test_zero_total_tokens_safe(self):
        """Division by zero is handled; ratios default to 0.0."""
        scorer = _make_scorer()
        ratios = scorer.compute_ratios(
            positive_count=0,
            negative_count=0,
            uncertainty_count=0,
            total_tokens=0,
        )
        for key in ("positive_ratio", "negative_ratio", "uncertainty_ratio"):
            assert math.isfinite(ratios[key])
            assert not math.isnan(ratios[key])

    def test_all_ratios_in_unit_interval(self):
        scorer = _make_scorer()
        for pos, neg, unc, tot in [
            (0, 0, 0, 1),
            (10, 5, 3, 100),
            (50, 50, 0, 100),
            (1, 0, 0, 1),
        ]:
            ratios = scorer.compute_ratios(pos, neg, unc, tot)
            for key in ("positive_ratio", "negative_ratio", "uncertainty_ratio"):
                assert 0.0 <= ratios[key] <= 1.0, (
                    f"{key}={ratios[key]} out of [0,1]"
                )


# ===========================================================================
# ── 6. Label Assignment ──────────────────────────────────────────────────────
# ===========================================================================

class TestLabelAssignment:
    """Tests threshold boundary behaviour for sentiment label assignment."""

    def test_clearly_positive_score(self):
        scorer = _make_scorer(positive_threshold=0.02)
        assert scorer.assign_label(0.10) == "positive"

    def test_clearly_negative_score(self):
        scorer = _make_scorer(negative_threshold=-0.02)
        assert scorer.assign_label(-0.10) == "negative"

    def test_zero_score_is_neutral(self):
        scorer = _make_scorer(positive_threshold=0.02, negative_threshold=-0.02)
        assert scorer.assign_label(0.0) == "neutral"

    def test_exactly_positive_threshold_is_positive(self):
        scorer = _make_scorer(positive_threshold=0.02)
        assert scorer.assign_label(0.02) == "positive"

    def test_just_below_positive_threshold_is_neutral(self):
        scorer = _make_scorer(positive_threshold=0.02)
        label = scorer.assign_label(0.019999)
        assert label == "neutral"

    def test_exactly_negative_threshold_is_negative(self):
        scorer = _make_scorer(negative_threshold=-0.02)
        assert scorer.assign_label(-0.02) == "negative"

    def test_just_above_negative_threshold_is_neutral(self):
        scorer = _make_scorer(negative_threshold=-0.02)
        label = scorer.assign_label(-0.019999)
        assert label == "neutral"

    def test_plus_one_is_positive(self):
        scorer = _make_scorer()
        assert scorer.assign_label(1.0) == "positive"

    def test_minus_one_is_negative(self):
        scorer = _make_scorer()
        assert scorer.assign_label(-1.0) == "negative"

    def test_asymmetric_thresholds(self):
        scorer = _make_scorer(positive_threshold=0.10, negative_threshold=-0.05)
        assert scorer.assign_label(0.05) == "neutral"   # between thresholds
        assert scorer.assign_label(0.10) == "positive"
        assert scorer.assign_label(-0.05) == "negative"
        assert scorer.assign_label(-0.04) == "neutral"

    def test_only_three_labels_produced(self):
        scorer = _make_scorer()
        scores = [-1.0, -0.5, -0.02, -0.01, 0.0, 0.01, 0.02, 0.5, 1.0]
        labels = {scorer.assign_label(s) for s in scores}
        assert labels.issubset({"positive", "neutral", "negative"})


# ===========================================================================
# ── 7. Confidence Computation ────────────────────────────────────────────────
# ===========================================================================

class TestConfidenceComputation:

    def test_confidence_in_unit_interval(self):
        scorer = _make_scorer()
        for matched, total in [(0, 100), (5, 100), (50, 100), (100, 100)]:
            cov = matched / max(total, 1)
            conf = scorer.compute_confidence(
                coverage_ratio=cov,
                matched_tokens=matched,
                total_tokens=total,
            )
            assert 0.0 <= conf <= 1.0, (
                f"Confidence {conf} out of [0,1] for matched={matched}"
            )

    def test_higher_coverage_not_lower_confidence(self):
        """Confidence should be non-decreasing in coverage."""
        scorer = _make_scorer()
        prev_conf = 0.0
        for matched in range(0, 101, 10):
            cov = matched / 100
            conf = scorer.compute_confidence(
                coverage_ratio=cov,
                matched_tokens=matched,
                total_tokens=100,
            )
            assert conf >= prev_conf - 1e-9, (
                f"Confidence decreased: matched={matched}, conf={conf}"
            )
            prev_conf = conf

    def test_zero_coverage_confidence(self):
        scorer = _make_scorer()
        conf = scorer.compute_confidence(
            coverage_ratio=0.0,
            matched_tokens=0,
            total_tokens=100,
        )
        assert math.isfinite(conf)
        assert 0.0 <= conf <= 1.0

    def test_full_coverage_confidence(self):
        scorer = _make_scorer()
        conf = scorer.compute_confidence(
            coverage_ratio=1.0,
            matched_tokens=100,
            total_tokens=100,
        )
        assert math.isfinite(conf)
        assert 0.0 <= conf <= 1.0

    def test_confidence_is_finite_for_all_edge_cases(self):
        scorer = _make_scorer()
        for matched, total in [(0, 0), (0, 1), (1, 1), (1000, 1000)]:
            cov = matched / max(total, 1)
            conf = scorer.compute_confidence(
                coverage_ratio=cov,
                matched_tokens=matched,
                total_tokens=total,
            )
            assert math.isfinite(conf), (
                f"Non-finite confidence for matched={matched} total={total}"
            )


# ===========================================================================
# ── 8. score_from_counts Integration ─────────────────────────────────────────
# ===========================================================================

class TestScoreFromCounts:
    """Integration tests that exercise the full scoring pipeline."""

    def test_positive_heavy_transcript(self):
        """pos=30, neg=5: tone should be clearly positive."""
        result = _score(pos=30, neg=5, unc=3, total=300, matched=50)
        assert result.sentiment_label == "positive"
        assert result.tone_score > 0.02

    def test_negative_heavy_transcript(self):
        """pos=5, neg=30: tone should be clearly negative."""
        result = _score(pos=5, neg=30, unc=3, total=300, matched=50)
        assert result.sentiment_label == "negative"
        assert result.tone_score < -0.02

    def test_neutral_balanced_transcript(self):
        """pos=10, neg=10: tone should be zero / neutral."""
        result = _score(pos=10, neg=10, unc=2, total=200, matched=30)
        assert result.sentiment_label == "neutral"
        assert math.isclose(result.tone_score, 0.0, abs_tol=0.02)

    def test_zero_match_transcript(self):
        """pos=0, neg=0: no dictionary hits; score should be 0.0 and not NaN."""
        result = _score(pos=0, neg=0, unc=0, total=100, matched=0)
        assert math.isfinite(result.tone_score)
        assert not math.isnan(result.tone_score)
        assert result.tone_score == pytest.approx(0.0, abs=1e-9)

    def test_extremely_short_transcript(self):
        """total=5 tokens; scoring must still produce a finite result."""
        result = _score(pos=1, neg=0, unc=0, total=5, matched=1)
        assert math.isfinite(result.tone_score)
        assert -1.0 <= result.tone_score <= 1.0

    def test_low_coverage_flagged(self):
        """matched=1, total=1000 → coverage=0.001 < minimum."""
        result = _score(
            pos=1, neg=0, unc=0, total=1000, matched=1,
            minimum_coverage_ratio=0.01,
        )
        assert result.is_low_coverage is True

    def test_sufficient_coverage_not_flagged(self):
        result = _score(
            pos=5, neg=2, unc=1, total=100, matched=20,
            minimum_coverage_ratio=0.01,
        )
        assert result.is_low_coverage is False

    def test_smoothing_flag_reflects_config(self):
        """smoothing_applied should mirror config.smoothing_enabled."""
        r_on  = _score(pos=5, neg=2, smoothing_enabled=True)
        r_off = _score(pos=5, neg=2, smoothing_enabled=False)
        assert r_on.smoothing_applied is True
        assert r_off.smoothing_applied is False

    def test_diagnostics_counts_preserved(self):
        result = _score(pos=12, neg=4, unc=6, total=250, matched=30)
        diag = result.diagnostics
        assert diag.positive_count == 12
        assert diag.negative_count == 4
        assert diag.uncertainty_count == 6
        assert diag.total_tokens == 250
        assert diag.matched_tokens == 30

    def test_diagnostics_coverage_ratio_consistent(self):
        """Stored coverage_ratio must match matched/total."""
        result = _score(pos=5, neg=2, unc=1, total=100, matched=20)
        diag = result.diagnostics
        expected_cov = 20 / 100
        assert math.isclose(diag.coverage_ratio, expected_cov, rel_tol=1e-9)

    def test_exact_tone_score_positive_heavy(self):
        """pos=20, neg=4, alpha=0.5 → (20-4)/(20+4+1) = 16/25 = 0.64"""
        result = _score(
            pos=20, neg=4, unc=0, total=200, matched=30,
            smoothing_enabled=True, smoothing_alpha=0.5,
        )
        expected = _tone(20, 4, alpha=0.5)
        assert math.isclose(result.tone_score, expected, rel_tol=1e-9), (
            f"Expected {expected:.10f}, got {result.tone_score:.10f}"
        )

    def test_exact_tone_score_negative_heavy(self):
        """pos=3, neg=15, alpha=0.5 → (3-15)/(3+15+1) = -12/19 ≈ -0.631579"""
        result = _score(
            pos=3, neg=15, unc=0, total=150, matched=25,
            smoothing_enabled=True, smoothing_alpha=0.5,
        )
        expected = _tone(3, 15, alpha=0.5)
        assert math.isclose(result.tone_score, expected, rel_tol=1e-9)

    def test_return_type_is_lm_score(self):
        result = _score(pos=5, neg=2)
        assert isinstance(result, LMScore)


# ===========================================================================
# ── 9. Smoothing Behaviour ────────────────────────────────────────────────────
# ===========================================================================

class TestSmoothingBehaviour:

    def test_zero_zero_without_smoothing_is_safe(self):
        """pos=0, neg=0, no smoothing: must not raise ZeroDivisionError."""
        scorer = _make_scorer(smoothing_enabled=False, smoothing_alpha=0.0)
        result = scorer.score_from_counts(
            positive_count=0, negative_count=0,
            uncertainty_count=0, total_tokens=50, matched_tokens=0,
        )
        assert math.isfinite(result.tone_score)
        assert not math.isnan(result.tone_score)

    def test_smoothing_moves_score_toward_zero(self):
        """Smoothing always reduces absolute tone score magnitude."""
        pos, neg = 10, 0
        scorer_on  = _make_scorer(smoothing_enabled=True,  smoothing_alpha=0.5)
        scorer_off = _make_scorer(smoothing_enabled=False, smoothing_alpha=0.0)
        score_on  = scorer_on.compute_tone_score(positive_count=pos, negative_count=neg)
        score_off = scorer_off.compute_tone_score(positive_count=pos, negative_count=neg)
        assert abs(score_on) < abs(score_off)

    def test_larger_alpha_shrinks_score_more(self):
        """alpha=1.0 should produce a score closer to 0 than alpha=0.1."""
        pos, neg = 6, 2
        scorer_small = _make_scorer(smoothing_enabled=True, smoothing_alpha=0.1)
        scorer_large = _make_scorer(smoothing_enabled=True, smoothing_alpha=1.0)
        s_small = scorer_small.compute_tone_score(positive_count=pos, negative_count=neg)
        s_large = scorer_large.compute_tone_score(positive_count=pos, negative_count=neg)
        assert abs(s_large) < abs(s_small)

    def test_smoothing_symmetry_preserved(self):
        """Swapping pos/neg with smoothing should still negate the score."""
        scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=0.5)
        s1 = scorer.compute_tone_score(positive_count=8, negative_count=2)
        s2 = scorer.compute_tone_score(positive_count=2, negative_count=8)
        assert math.isclose(s1, -s2, rel_tol=1e-9)

    def test_alpha_zero_with_smoothing_on_matches_no_smoothing(self):
        """smoothing_enabled=True, alpha=0.0 should behave like no smoothing."""
        scorer_a = _make_scorer(smoothing_enabled=True,  smoothing_alpha=0.0)
        scorer_b = _make_scorer(smoothing_enabled=False, smoothing_alpha=0.0)
        for pos, neg in [(5, 3), (10, 0), (0, 10)]:
            sa = scorer_a.compute_tone_score(positive_count=pos, negative_count=neg)
            sb = scorer_b.compute_tone_score(positive_count=pos, negative_count=neg)
            assert math.isclose(sa, sb, rel_tol=1e-9)


# ===========================================================================
# ── 10. Edge Cases ────────────────────────────────────────────────────────────
# ===========================================================================

class TestEdgeCases:

    def test_extremely_large_counts(self):
        """Large counts must not overflow or produce non-finite results."""
        result = _score(pos=10_000_000, neg=1_000_000, total=50_000_000, matched=11_000_000)
        assert math.isfinite(result.tone_score)
        assert -1.0 <= result.tone_score <= 1.0

    def test_very_sparse_matches(self):
        """Only 1 matched token from 10 000: coverage is tiny but well-defined."""
        result = _score(
            pos=1, neg=0, unc=0, total=10_000, matched=1,
            minimum_coverage_ratio=0.01,
        )
        assert math.isfinite(result.tone_score)
        assert result.is_low_coverage is True

    def test_high_uncertainty_ratio(self):
        """Transcripts dominated by uncertainty words: tone should still compute."""
        result = _score(pos=2, neg=1, unc=80, total=200, matched=90)
        assert math.isfinite(result.tone_score)
        assert result.diagnostics.uncertainty_count == 80

    def test_all_tokens_matched(self):
        """matched_tokens == total_tokens → coverage = 1.0, not flagged."""
        result = _score(
            pos=10, neg=5, unc=3, total=50, matched=50,
            minimum_coverage_ratio=0.01,
        )
        assert math.isclose(result.diagnostics.coverage_ratio, 1.0, rel_tol=1e-9)
        assert result.is_low_coverage is False

    def test_single_token_transcript(self):
        """total=1, matched=1, pos=1: minimal document."""
        result = _score(pos=1, neg=0, unc=0, total=1, matched=1)
        assert math.isfinite(result.tone_score)
        assert result.tone_score > 0.0

    def test_positive_equals_negative_large(self):
        """pos=50, neg=50 → tone = 0 regardless of alpha."""
        for alpha in (0.0, 0.5, 1.0, 2.0):
            scorer = _make_scorer(smoothing_enabled=True, smoothing_alpha=alpha)
            s = scorer.compute_tone_score(positive_count=50, negative_count=50)
            assert math.isclose(s, 0.0, abs_tol=1e-12), (
                f"alpha={alpha}: expected 0.0, got {s}"
            )

    def test_score_clips_to_ceiling(self):
        """Score must never exceed 1.0 even for extreme inputs."""
        result = _score(pos=1_000_000, neg=0, total=1_000_000, matched=1_000_000)
        assert result.tone_score <= 1.0

    def test_score_clips_to_floor(self):
        """Score must never go below -1.0 even for extreme inputs."""
        result = _score(pos=0, neg=1_000_000, total=1_000_000, matched=1_000_000)
        assert result.tone_score >= -1.0


# ===========================================================================
# ── 11. Coverage Handling ─────────────────────────────────────────────────────
# ===========================================================================

class TestCoverageHandling:

    def test_below_minimum_coverage_returns_nan_or_zero(self):
        """
        When coverage < minimum_coverage_ratio the scorer may return a
        NaN or 0.0 tone (implementation choice), but must never raise and
        must flag is_low_coverage.
        """
        result = _score(
            pos=1, neg=0, unc=0,
            total=10_000, matched=1,
            minimum_coverage_ratio=0.05,
        )
        assert result.is_low_coverage is True
        # Score must be finite or NaN — but never inf
        assert not math.isinf(result.tone_score)

    def test_exactly_at_minimum_coverage_not_flagged(self):
        """matched/total == min_coverage exactly: boundary should NOT be flagged."""
        result = _score(
            pos=2, neg=1, unc=0,
            total=100, matched=5,
            minimum_coverage_ratio=0.05,
        )
        assert result.is_low_coverage is False

    def test_coverage_ratio_stored_in_diagnostics(self):
        result = _score(pos=3, neg=2, unc=0, total=200, matched=50)
        expected_cov = 50 / 200
        assert math.isclose(
            result.diagnostics.coverage_ratio, expected_cov, rel_tol=1e-9
        )

    def test_zero_total_tokens_does_not_raise(self):
        scorer = _make_scorer()
        result = scorer.score_from_counts(
            positive_count=0, negative_count=0,
            uncertainty_count=0, total_tokens=0, matched_tokens=0,
        )
        assert isinstance(result, LMScore)
        assert math.isfinite(result.tone_score) or math.isnan(result.tone_score)


# ===========================================================================
# ── 12. Determinism ──────────────────────────────────────────────────────────
# ===========================================================================

class TestDeterminism:

    def test_same_inputs_same_score(self):
        """Repeated calls with identical arguments must return identical scores."""
        kwargs = dict(pos=12, neg=4, unc=3, total=300, matched=45)
        r1 = _score(**kwargs)
        r2 = _score(**kwargs)
        assert r1.tone_score == r2.tone_score
        assert r1.sentiment_label == r2.sentiment_label
        assert r1.confidence == r2.confidence

    def test_order_independence_of_scorer_instantiation(self):
        """Two separately instantiated scorers with same config must agree."""
        cfg = LMScoringConfig(
            positive_threshold=0.02,
            negative_threshold=-0.02,
            smoothing_enabled=True,
            smoothing_alpha=0.5,
        )
        scorer_a = LMScorer(cfg)
        scorer_b = LMScorer(cfg)
        s_a = scorer_a.score_from_counts(8, 3, 1, 200, 20)
        s_b = scorer_b.score_from_counts(8, 3, 1, 200, 20)
        assert s_a.tone_score == s_b.tone_score

    def test_determinism_across_many_calls(self):
        """100 repeated calls must all return the same tone score."""
        scorer = _make_scorer()
        scores = [
            scorer.score_from_counts(7, 2, 1, 150, 30).tone_score
            for _ in range(100)
        ]
        assert len(set(scores)) == 1, "Non-deterministic scores detected"

    def test_config_independence(self):
        """Different smoothing_alpha values produce different scores."""
        s1 = _score(pos=10, neg=3, smoothing_alpha=0.5)
        s2 = _score(pos=10, neg=3, smoothing_alpha=2.0)
        assert s1.tone_score != s2.tone_score


# ===========================================================================
# ── 13. Numeric Stability ─────────────────────────────────────────────────────
# ===========================================================================

class TestNumericStability:

    def test_near_threshold_boundary_positive(self):
        """Score just above positive_threshold should be labelled positive."""
        scorer = _make_scorer(
            positive_threshold=0.02,
            smoothing_enabled=False,
            smoothing_alpha=0.0,
        )
        # Find exact pos, neg that gives tone ≈ 0.021
        # tone = (pos-neg)/(pos+neg) = 0.021 → e.g. pos=521, neg=479
        # 42/2000 = 0.021 → pos=1021, neg=979
        tone = scorer.compute_tone_score(positive_count=1021, negative_count=979)
        expected = _tone(1021, 979, alpha=0.0)
        assert math.isclose(tone, expected, rel_tol=1e-9)
        assert tone > 0.02
        assert scorer.assign_label(tone) == "positive"

    def test_near_threshold_boundary_negative(self):
        scorer = _make_scorer(
            negative_threshold=-0.02,
            smoothing_enabled=False,
            smoothing_alpha=0.0,
        )
        tone = scorer.compute_tone_score(positive_count=979, negative_count=1021)
        assert tone < -0.02
        assert scorer.assign_label(tone) == "negative"

    def test_integer_overflow_safe(self):
        """Counts exceeding sys.maxsize / 2 should not cause overflow."""
        large = 10 ** 15
        scorer = _make_scorer()
        tone = scorer.compute_tone_score(
            positive_count=large, negative_count=large // 2
        )
        assert math.isfinite(tone)
        assert -1.0 <= tone <= 1.0

    def test_float_input_counts(self):
        """If counts arrive as floats, scoring must not crash."""
        scorer = _make_scorer()
        tone = scorer.compute_tone_score(
            positive_count=10.0, negative_count=3.0  # type: ignore[arg-type]
        )
        assert math.isfinite(tone)

    def test_no_nan_propagation_from_ratios(self):
        """NaN in intermediate ratios must not leak into the final score."""
        result = _score(pos=0, neg=0, unc=0, total=0, matched=0)
        assert not math.isnan(result.tone_score)
        assert not math.isinf(result.tone_score)


# ===========================================================================
# ── 14. Validation Helper ─────────────────────────────────────────────────────
# ===========================================================================

class TestValidateScore:

    def _valid_score(self) -> LMScore:
        return _score(pos=10, neg=3, unc=2, total=200, matched=30)

    def test_valid_score_has_no_issues(self):
        scorer = _make_scorer()
        issues = scorer.validate_score(self._valid_score())
        assert issues == []

    def test_out_of_range_tone_detected(self):
        """Manually constructed LMScore with tone > 1.0 must be flagged."""
        diag = ScoreDiagnostics(
            positive_count=10, negative_count=0,
            uncertainty_count=0, total_tokens=100, matched_tokens=10,
            positive_ratio=0.10, negative_ratio=0.0,
            uncertainty_ratio=0.0, coverage_ratio=0.10,
        )
        bad_score = LMScore(
            tone_score=1.5,
            sentiment_label="positive",
            confidence=0.9,
            diagnostics=diag,
            is_low_coverage=False,
            smoothing_applied=True,
        )
        scorer = _make_scorer()
        issues = scorer.validate_score(bad_score)
        assert len(issues) > 0
        assert any("range" in msg.lower() or "1.5" in msg or "tone" in msg.lower()
                   for msg in issues)

    def test_negative_confidence_detected(self):
        diag = ScoreDiagnostics(
            positive_count=5, negative_count=1,
            uncertainty_count=0, total_tokens=50, matched_tokens=8,
            positive_ratio=0.10, negative_ratio=0.02,
            uncertainty_ratio=0.0, coverage_ratio=0.16,
        )
        bad_score = LMScore(
            tone_score=0.30,
            sentiment_label="positive",
            confidence=-0.1,
            diagnostics=diag,
            is_low_coverage=False,
            smoothing_applied=True,
        )
        scorer = _make_scorer()
        issues = scorer.validate_score(bad_score)
        assert any("confidence" in msg.lower() for msg in issues)

    def test_invalid_label_detected(self):
        diag = ScoreDiagnostics(
            positive_count=5, negative_count=1,
            uncertainty_count=0, total_tokens=50, matched_tokens=8,
            positive_ratio=0.10, negative_ratio=0.02,
            uncertainty_ratio=0.0, coverage_ratio=0.16,
        )
        bad_score = LMScore(
            tone_score=0.30,
            sentiment_label="bullish",   # invalid
            confidence=0.7,
            diagnostics=diag,
            is_low_coverage=False,
            smoothing_applied=True,
        )
        scorer = _make_scorer()
        issues = scorer.validate_score(bad_score)
        assert any("label" in msg.lower() or "bullish" in msg for msg in issues)

    def test_nan_tone_detected(self):
        diag = ScoreDiagnostics(
            positive_count=0, negative_count=0,
            uncertainty_count=0, total_tokens=100, matched_tokens=0,
            positive_ratio=0.0, negative_ratio=0.0,
            uncertainty_ratio=0.0, coverage_ratio=0.0,
        )
        nan_score = LMScore(
            tone_score=float("nan"),
            sentiment_label="neutral",
            confidence=0.0,
            diagnostics=diag,
            is_low_coverage=True,
            smoothing_applied=True,
        )
        scorer = _make_scorer()
        issues = scorer.validate_score(nan_score)
        assert any("nan" in msg.lower() or "finite" in msg.lower() for msg in issues)

    def test_negative_count_in_diagnostics_detected(self):
        diag = ScoreDiagnostics(
            positive_count=-1,            # invalid
            negative_count=2,
            uncertainty_count=0,
            total_tokens=100,
            matched_tokens=5,
            positive_ratio=-0.01,         # invalid
            negative_ratio=0.02,
            uncertainty_ratio=0.0,
            coverage_ratio=0.05,
        )
        bad_score = LMScore(
            tone_score=0.10,
            sentiment_label="positive",
            confidence=0.5,
            diagnostics=diag,
            is_low_coverage=False,
            smoothing_applied=True,
        )
        scorer = _make_scorer()
        issues = scorer.validate_score(bad_score)
        assert len(issues) > 0


# ===========================================================================
# ── 15. Parametrised Batch Scenarios ─────────────────────────────────────────
# ===========================================================================

@pytest.mark.parametrize("pos,neg,alpha,expected_tone", [
    # (pos, neg, alpha, expected_tone)
    (0,   0,   0.5,  0.0),                                 # zero counts
    (10,  0,   0.5,  _tone(10,  0,  0.5)),                 # all positive
    (0,   10,  0.5,  _tone(0,   10, 0.5)),                 # all negative
    (5,   5,   0.5,  0.0),                                  # exactly balanced
    (10,  2,   0.5,  _tone(10,  2,  0.5)),                 # positive heavy
    (2,   10,  0.5,  _tone(2,   10, 0.5)),                 # negative heavy
    (100, 99,  0.5,  _tone(100, 99, 0.5)),                 # near-balanced large
    (1,   0,   0.0,  1.0),                                  # single positive, no smoothing
    (0,   1,   0.0, -1.0),                                  # single negative, no smoothing
])
def test_parametrised_tone_formula(pos, neg, alpha, expected_tone):
    smoothing = alpha > 0
    scorer = _make_scorer(smoothing_enabled=smoothing, smoothing_alpha=alpha)
    result = scorer.compute_tone_score(positive_count=pos, negative_count=neg)
    assert math.isclose(result, expected_tone, abs_tol=1e-9), (
        f"pos={pos} neg={neg} alpha={alpha}: "
        f"expected={expected_tone:.10f} got={result:.10f}"
    )


@pytest.mark.parametrize("pos,neg,threshold,expected_label", [
    (10,  2,  0.02,  "positive"),
    (2,   10, 0.02,  "negative"),
    (5,   5,  0.02,  "neutral"),
    (1,   0,  0.02,  "positive"),
    (0,   0,  0.02,  "neutral"),
])
def test_parametrised_label_assignment(pos, neg, threshold, expected_label):
    scorer = _make_scorer(
        positive_threshold=threshold,
        negative_threshold=-threshold,
        smoothing_enabled=True,
        smoothing_alpha=0.5,
    )
    result = scorer.score_from_counts(
        positive_count=pos, negative_count=neg,
        uncertainty_count=0, total_tokens=100, matched_tokens=max(pos + neg, 1),
    )
    assert result.sentiment_label == expected_label, (
        f"pos={pos} neg={neg}: expected '{expected_label}' "
        f"got '{result.sentiment_label}' (score={result.tone_score:.6f})"
    )


# ===========================================================================
# ── Standalone execution
# ===========================================================================

if __name__ == "__main__":
    import subprocess
    print("Running test_lm_scoring.py via pytest …\n")
    ret = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    if ret.returncode == 0:
        print("\n✅  All tests passed.")
    else:
        print(f"\n❌  Tests failed (exit code {ret.returncode}).")
    sys.exit(ret.returncode)
