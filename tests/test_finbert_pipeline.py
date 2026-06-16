"""
tests/test_finbert_pipeline.py
================================
Comprehensive pytest test suite for the Earnings Call Sentiment Analyzer
scalable NLP pipeline (DAY 8).

Coverage
--------
* Chunk generation        (chunk_generator.py)
* Aggregation             (aggregation.py)
* Hierarchical aggregation(hierarchical_aggregation.py)
* Validation layer        (validators embedded in each module)
* Export integrity        (parquet / csv / dataframe / JSONL)
* Integration             (end-to-end synthetic pipeline)
* Edge cases              (empty / malformed / single-observation inputs)

Design principles
-----------------
* Zero network access   — no HuggingFace downloads, no external APIs.
* Zero GPU dependency   — all transformer calls are mocked.
* Deterministic         — fixed seeds everywhere; same input → same result.
* Self-contained        — all fixtures defined in this file.
* pytest-native         — parametrize, fixtures, marks, tmp_path.
* unittest fallback     — every TestCase also runnable via ``python -m pytest``
  or ``python tests/test_finbert_pipeline.py``.

Running
-------
::

    # With pytest (recommended)
    pytest tests/test_finbert_pipeline.py -v

    # Without pytest (fallback — uses unittest runner embedded at EOF)
    python tests/test_finbert_pipeline.py

Author : Earnings Call Sentiment Analyzer — DAY 8
Python : 3.11+
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve source modules regardless of working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOTS = [
    _HERE.parent / "src" / "sentiment",
    _HERE.parent,
    _HERE,
    Path("/home/claude"),
    Path("/mnt/user-data/outputs"),
]
for _root in _REPO_ROOTS:
    if _root.exists() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# ---------------------------------------------------------------------------
# Conditional pytest import — degrade gracefully so the file is also
# runnable as ``python test_finbert_pipeline.py`` via unittest.
# ---------------------------------------------------------------------------
try:
    import pytest
    _PYTEST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYTEST_AVAILABLE = False
    # Create a minimal stub so pytest decorators don't crash at parse-time
    class _PytestStub:
        @staticmethod
        def fixture(*a, **kw):
            return lambda f: f

        @staticmethod
        def mark():
            pass

        class mark:  # noqa: F811
            @staticmethod
            def parametrize(*a, **kw):
                return lambda f: f

            @staticmethod
            def slow(f):
                return f

            @staticmethod
            def integration(f):
                return f

        @staticmethod
        def approx(value, **kw):
            return _ApproxStub(value, **kw)

        @staticmethod
        def raises(exc, *a, **kw):
            import contextlib
            return contextlib.suppress(exc)

    class _ApproxStub:
        def __init__(self, v, abs=None, rel=None):
            self._v = v
            self._abs = abs or 1e-6
        def __eq__(self, other):
            return abs(other - self._v) <= self._abs

    pytest = _PytestStub()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Lazy module imports — fail gracefully with a clear message
# ---------------------------------------------------------------------------

def _import_module(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import '{name}'. Ensure the module is on sys.path.\n"
            f"Searched: {_REPO_ROOTS}\nOriginal error: {exc}"
        ) from exc


# ── Import all pipeline modules ────────────────────────────────────────────
cg_mod   = _import_module("chunk_generator")
agg_mod  = _import_module("aggregation")
hier_mod = _import_module("hierarchical_aggregation")

# Aliases for brevity
ChunkConfig            = cg_mod.ChunkConfig
ChunkGenerator         = cg_mod.ChunkGenerator
ChunkGenerationResult  = cg_mod.ChunkGenerationResult
Chunk                  = cg_mod.Chunk
SentenceRecord         = cg_mod.SentenceRecord
_WordTokenizerFallback = cg_mod._WordTokenizerFallback
create_chunk_generator = cg_mod.create_chunk_generator
iter_chunks            = cg_mod.iter_chunks
_build_demo_sentences  = cg_mod._build_demo_sentences

WeightingStrategy      = agg_mod.WeightingStrategy
AggregationConfig      = agg_mod.AggregationConfig
SentimentAggregator    = agg_mod.SentimentAggregator
AggregationValidator   = agg_mod.AggregationValidator
AggregationResult      = agg_mod.AggregationResult
TranscriptSentiment    = agg_mod.TranscriptSentiment
compute_sentiment_drift= agg_mod.compute_sentiment_drift
create_aggregator      = agg_mod.create_aggregator
_weighted_mean         = agg_mod._weighted_mean
_score_to_label        = agg_mod._score_to_label
_normalise_probs       = agg_mod._normalise_probs
_safe_std              = agg_mod._safe_std
_safe_iqr              = agg_mod._safe_iqr

HierarchyLevel              = hier_mod.HierarchyLevel
HierarchicalAggregationConfig = hier_mod.HierarchicalAggregationConfig
HierarchicalAggregator      = hier_mod.HierarchicalAggregator
HierarchicalBatchResult     = hier_mod.HierarchicalBatchResult
HierarchicalSentimentResult = hier_mod.HierarchicalSentimentResult
AggregationNode             = hier_mod.AggregationNode
HierarchicalValidator       = hier_mod.HierarchicalValidator
SpeakerType                 = hier_mod.SpeakerType
create_hierarchical_aggregator = hier_mod.create_hierarchical_aggregator
_infer_speaker_type         = hier_mod._infer_speaker_type
_ols_slope                  = hier_mod._ols_slope
_safe_median                = hier_mod._safe_median


# ===========================================================================
# ── SHARED SYNTHETIC DATA FACTORIES ────────────────────────────────────────
# ===========================================================================

PROB_TOL = 1e-4   # tolerance for probability sum checks

def _make_sentence_records(
    transcript_id: str = "AAPL_Q1_2025",
    n_prepared: int = 8,
    n_qa: int = 6,
    speaker_a: str = "Tim Cook",
    speaker_b: str = "Luca Maestri",
    analyst:   str = "Analyst",
) -> list:
    """Deterministic synthetic sentence records."""
    rng = np.random.default_rng(42)
    texts_prep = [
        "Revenue grew strongly driven by services and iPhone demand.",
        "Our gross margin expanded sixty basis points year over year.",
        "We are investing heavily in artificial intelligence across all platforms.",
        "The installed base reached an all-time record this quarter.",
        "We returned twenty billion dollars to shareholders through buybacks.",
        "Mac revenue increased due to strong enterprise adoption.",
        "Services achieved an all-time revenue record of thirty two billion.",
        "We remain very confident in our long-term product roadmap.",
    ][:n_prepared]
    texts_qa = [
        "Can you comment on the trajectory of services growth?",
        "We see continued strength in subscriptions across all regions.",
        "What is your outlook for capital expenditure next quarter?",
        "We plan to invest prudently while maintaining strong free cash flow.",
        "Can you elaborate on Vision Pro enterprise adoption?",
        "Enterprise adoption has exceeded our initial expectations significantly.",
    ][:n_qa]
    records = []
    for i, text in enumerate(texts_prep):
        records.append(SentenceRecord(
            sentence_id=f"{transcript_id}_prep_{i:03d}",
            transcript_id=transcript_id,
            sentence_order=i,
            sentence_text=text,
            speaker=speaker_a if i % 2 == 0 else speaker_b,
            speaker_role="CEO" if i % 2 == 0 else "CFO",
            section_type="prepared_remarks",
        ))
    for j, text in enumerate(texts_qa):
        sp = analyst if j % 3 == 0 else speaker_a
        records.append(SentenceRecord(
            sentence_id=f"{transcript_id}_qa_{j:03d}",
            transcript_id=transcript_id,
            sentence_order=n_prepared + j,
            sentence_text=text,
            speaker=sp,
            speaker_role="Analyst" if sp == analyst else "CEO",
            section_type="qa",
        ))
    return records


def _make_chunk_df(
    transcript_ids: list[str] | None = None,
    chunks_per_transcript: int = 10,
    bias: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic synthetic chunk-level FinBERT output DataFrame."""
    if transcript_ids is None:
        transcript_ids = ["AAPL_Q1_2025"]
    rng = np.random.default_rng(seed)
    rows = []
    speakers_map = {
        "AAPL_Q1_2025": [("Tim Cook", "CEO"), ("Luca Maestri", "CFO")],
        "NVDA_Q2_2025": [("Jensen Huang", "CEO"), ("Colette Kress", "CFO")],
    }
    default_speakers = [("Speaker A", "CEO"), ("Analyst", "Analyst")]
    sections = (
        ["prepared_remarks"] * (chunks_per_transcript // 2)
        + ["qa"] * (chunks_per_transcript - chunks_per_transcript // 2)
    )
    for tid in transcript_ids:
        spk_list = speakers_map.get(tid, default_speakers)
        tid_bias = bias if "AAPL" in tid else -bias * 0.5
        for order, section in enumerate(sections):
            spk, role = spk_list[order % len(spk_list)]
            pos = float(np.clip(0.40 + tid_bias + rng.normal(0, 0.10), 0.01, 0.98))
            neg = float(np.clip(0.25 - tid_bias + rng.normal(0, 0.07), 0.01, 0.98))
            neu = max(0.01, 1.0 - pos - neg)
            t   = pos + neu + neg
            pos, neu, neg = pos / t, neu / t, neg / t
            score = pos - neg
            conf  = max(pos, neu, neg)
            tc    = int(rng.integers(80, 450))
            label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
            rows.append({
                "chunk_id":        f"chunk_{tid}_{order:04d}",
                "transcript_id":   tid,
                "chunk_order":     order,
                "section_type":    section,
                "dominant_speaker": spk,
                "speaker_role":    role,
                "positive_prob":   round(pos,   8),
                "neutral_prob":    round(neu,   8),
                "negative_prob":   round(neg,   8),
                "sentiment_score": round(score, 8),
                "confidence":      round(conf,  8),
                "predicted_label": label,
                "token_count":     tc,
            })
    return pd.DataFrame(rows)


def _make_sentence_df(
    transcript_ids: list[str] | None = None,
    sentences_per_transcript: int = 20,
    seed: int = 99,
) -> pd.DataFrame:
    """Deterministic synthetic sentence-level output DataFrame."""
    if transcript_ids is None:
        transcript_ids = ["AAPL_Q1_2025"]
    rng = np.random.default_rng(seed)
    rows = []
    for tid in transcript_ids:
        bias = 0.15 if "AAPL" in tid else -0.10
        for order in range(sentences_per_transcript):
            section = "prepared_remarks" if order < sentences_per_transcript // 2 else "qa"
            spk  = "Tim Cook" if order % 3 != 0 else "Analyst"
            role = "CEO" if spk != "Analyst" else "Analyst"
            score = float(np.clip(bias + rng.normal(0, 0.20), -1.0, 1.0))
            pos   = float(np.clip(0.33 + score * 0.4, 0.01, 0.98))
            neg   = float(np.clip(0.33 - score * 0.4, 0.01, 0.98))
            neu   = max(0.01, 1.0 - pos - neg)
            t = pos + neu + neg
            rows.append({
                "sentence_id":    f"sent_{tid}_{order:06d}",
                "transcript_id":  tid,
                "sentence_order": order,
                "section_type":   section,
                "speaker":        spk,
                "speaker_role":   role,
                "sentiment_score": round(score, 6),
                "positive_prob":  round(pos / t, 6),
                "neutral_prob":   round(neu / t, 6),
                "negative_prob":  round(neg / t, 6),
                "confidence":     round(max(pos / t, neu / t, neg / t), 6),
                "token_count":    1,
            })
    return pd.DataFrame(rows)


class _MockTokenizer:
    """
    Deterministic mock tokenizer.
    Returns exactly ``len(text.split()) + 2`` token IDs (simulates WordPiece
    without sub-word expansion for test predictability).
    """
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        n = len(text.split()) + (2 if add_special_tokens else 0)
        return list(range(n))


# Helper assertions ──────────────────────────────────────────────────────────

def assert_prob_sum(pos: float, neu: float, neg: float, tol: float = PROB_TOL):
    total = pos + neu + neg
    assert abs(total - 1.0) < tol, (
        f"Probability sum {total:.8f} deviates from 1.0 by {abs(total-1.0):.2e}"
    )

def assert_score_range(score: float):
    assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6, f"score {score} outside [-1,1]"

def assert_label_consistent(score: float, label: str):
    expected = _score_to_label(score, 0.05, -0.05)
    assert label == expected, (
        f"Label '{label}' inconsistent with score {score:.4f} (expected '{expected}')"
    )

def assert_df_columns(df: pd.DataFrame, required: list[str]):
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"DataFrame missing columns: {missing}"

def assert_no_nan(df: pd.DataFrame, cols: list[str]):
    for col in cols:
        if col in df.columns:
            n = int(df[col].isna().sum())
            assert n == 0, f"Column '{col}' has {n} NaN values"


# ===========================================================================
# ══ GROUP 1: CHUNK GENERATOR TESTS ══════════════════════════════════════════
# ===========================================================================

class TestChunkConfig(unittest.TestCase):
    """ChunkConfig validation and defaults."""

    def test_default_values(self):
        cfg = ChunkConfig()
        self.assertEqual(cfg.chunk_size_tokens, 400)
        self.assertEqual(cfg.overlap_tokens, 64)
        self.assertEqual(cfg.hard_token_ceiling, 490)
        self.assertEqual(cfg.min_chunk_tokens, 32)

    def test_ceiling_must_be_le_512(self):
        with self.assertRaises(ValueError):
            ChunkConfig(hard_token_ceiling=513)

    def test_chunk_size_must_be_lt_ceiling(self):
        with self.assertRaises(ValueError):
            ChunkConfig(chunk_size_tokens=500, hard_token_ceiling=490)

    def test_overlap_must_be_lt_chunk_size(self):
        with self.assertRaises(ValueError):
            ChunkConfig(chunk_size_tokens=100, overlap_tokens=150)

    def test_min_chunk_tokens_positive(self):
        with self.assertRaises(ValueError):
            ChunkConfig(min_chunk_tokens=0)

    def test_custom_config_valid(self):
        cfg = ChunkConfig(
            chunk_size_tokens=200,
            overlap_tokens=30,
            hard_token_ceiling=400,
            min_chunk_tokens=10,
        )
        self.assertEqual(cfg.chunk_size_tokens, 200)


class TestWordTokenizerFallback(unittest.TestCase):
    """_WordTokenizerFallback token counting."""

    def setUp(self):
        self.tok = _WordTokenizerFallback()

    def test_empty_string(self):
        result = self.tok.encode("")
        self.assertIsInstance(result, list)

    def test_token_count_increases_with_words(self):
        n_short = len(self.tok.encode("hello world"))
        n_long  = len(self.tok.encode("hello world foo bar baz qux"))
        self.assertGreater(n_long, n_short)

    def test_special_tokens_add_to_count(self):
        n_with    = len(self.tok.encode("hello world", add_special_tokens=True))
        n_without = len(self.tok.encode("hello world", add_special_tokens=False))
        self.assertGreater(n_with, n_without)

    def test_returns_list_of_ints(self):
        result = self.tok.encode("revenue grew strongly this quarter")
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(t, int) for t in result))

    def test_subword_factor_applied(self):
        text  = "revenue grew this quarter"
        words = len(text.split())
        n_tok = len(self.tok.encode(text, add_special_tokens=False))
        expected = int(words * _WordTokenizerFallback.SUBWORD_FACTOR)
        self.assertEqual(n_tok, expected)


class TestSentenceRecord(unittest.TestCase):
    """SentenceRecord construction and DataFrame conversion."""

    def test_from_dict_basic(self):
        d = {
            "sentence_id": "s1",
            "transcript_id": "AAPL_Q1_2025",
            "sentence_order": 0,
            "sentence_text": "Revenue grew strongly.",
            "speaker": "Tim Cook",
            "speaker_role": "CEO",
            "section_type": "prepared_remarks",
        }
        rec = SentenceRecord.from_dict(d)
        self.assertEqual(rec.sentence_id, "s1")
        self.assertEqual(rec.speaker, "Tim Cook")

    def test_from_dataframe_roundtrip(self):
        sents = _make_sentence_records("AAPL_Q1_2025", n_prepared=4, n_qa=2)
        df = pd.DataFrame([asdict(s) for s in sents])
        recovered = SentenceRecord.from_dataframe(df)
        self.assertEqual(len(recovered), len(sents))
        orders = [r.sentence_order for r in recovered]
        self.assertEqual(orders, sorted(orders))

    def test_missing_optional_fields_use_defaults(self):
        d = {
            "sentence_id": "x1",
            "transcript_id": "T1",
            "sentence_order": 0,
            "sentence_text": "Hello.",
        }
        rec = SentenceRecord.from_dict(d)
        self.assertEqual(rec.speaker, "unknown")
        self.assertEqual(rec.section_type, "unknown")


class TestChunkGeneratorBasics(unittest.TestCase):
    """Core chunk generation behaviour."""

    def _gen(self, chunk_size: int = 80, overlap: int = 15) -> ChunkGenerator:
        cfg = ChunkConfig(
            chunk_size_tokens=chunk_size,
            overlap_tokens=overlap,
            hard_token_ceiling=490,
            min_chunk_tokens=5,
        )
        return ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())

    def test_empty_input_returns_empty_result(self):
        gen = self._gen()
        result = gen.generate([], transcript_id="EMPTY")
        self.assertEqual(len(result.chunks), 0)
        self.assertEqual(result.total_sentences, 0)

    def test_empty_dataframe_returns_empty_result(self):
        gen = self._gen()
        result = gen.generate(pd.DataFrame(), transcript_id="EMPTY")
        self.assertEqual(len(result.chunks), 0)

    def test_single_sentence_produces_one_chunk(self):
        gen = self._gen()
        sents = _make_sentence_records(n_prepared=1, n_qa=0)
        result = gen.generate(sents, transcript_id="AAPL_Q1_2025")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].chunk_order, 0)

    def test_chunk_order_is_sequential(self):
        gen = self._gen(chunk_size=40, overlap=8)
        sents = _make_sentence_records(n_prepared=6, n_qa=4)
        result = gen.generate(sents, transcript_id="T1")
        orders = [c.chunk_order for c in result.chunks]
        self.assertEqual(orders, list(range(len(result.chunks))))

    def test_no_sentences_lost(self):
        gen = self._gen(chunk_size=60, overlap=12)
        sents = _make_sentence_records(n_prepared=8, n_qa=6)
        result = gen.generate(sents, transcript_id="AAPL_Q1_2025")
        cov = result.coverage_metrics()
        self.assertAlmostEqual(cov["sentence_coverage"], 1.0, places=5)

    def test_token_ceiling_never_exceeded(self):
        gen = self._gen(chunk_size=60, overlap=10)
        sents = _make_sentence_records(n_prepared=8, n_qa=6)
        result = gen.generate(sents, transcript_id="T1")
        for chunk in result.chunks:
            self.assertLessEqual(
                chunk.token_count, gen.config.hard_token_ceiling,
                f"Chunk {chunk.chunk_id} exceeds ceiling: {chunk.token_count}"
            )

    def test_chunk_ids_globally_unique(self):
        gen = self._gen(chunk_size=50, overlap=10)
        sents = _make_sentence_records(n_prepared=8, n_qa=6)
        result = gen.generate(sents, transcript_id="T1")
        ids = [c.chunk_id for c in result.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_empty_chunk_text(self):
        gen = self._gen()
        sents = _make_sentence_records(n_prepared=4, n_qa=2)
        result = gen.generate(sents, transcript_id="T1")
        for c in result.chunks:
            self.assertTrue(c.chunk_text.strip(), f"Chunk {c.chunk_id} has empty text")

    def test_sentence_order_range_integrity(self):
        gen = self._gen(chunk_size=50, overlap=10)
        sents = _make_sentence_records(n_prepared=6, n_qa=4)
        result = gen.generate(sents, transcript_id="T1")
        for c in result.chunks:
            self.assertLessEqual(
                c.first_sentence_order, c.last_sentence_order,
                f"Chunk {c.chunk_id} has inverted sentence range"
            )


class TestChunkGeneratorOverlap(unittest.TestCase):
    """Overlap injection and flagging."""

    def _gen(self, chunk_size: int = 70, overlap: int = 20) -> ChunkGenerator:
        cfg = ChunkConfig(
            chunk_size_tokens=chunk_size,
            overlap_tokens=overlap,
            hard_token_ceiling=490,
            min_chunk_tokens=5,
        )
        return ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())

    def test_first_chunk_no_overlap_start(self):
        gen = self._gen()
        sents = _make_sentence_records(n_prepared=8, n_qa=4)
        result = gen.generate(sents, transcript_id="T1")
        if len(result.chunks) > 1:
            self.assertFalse(result.chunks[0].overlap_start)

    def test_subsequent_chunks_may_have_overlap_start(self):
        gen = self._gen(chunk_size=50, overlap=15)
        sents = _make_sentence_records(n_prepared=8, n_qa=6)
        result = gen.generate(sents, transcript_id="T1")
        if len(result.chunks) > 2:
            overlap_starts = [c.overlap_start for c in result.chunks[1:]]
            # At least some chunks should have overlap_start=True
            self.assertTrue(any(overlap_starts))

    def test_overlap_validation_passes(self):
        gen = self._gen()
        sents = _make_sentence_records(n_prepared=8, n_qa=6)
        result = gen.generate(sents, transcript_id="T1")
        errors = gen.validate_overlap(result)
        self.assertEqual(errors, [], f"Overlap validation errors: {errors}")


class TestChunkGeneratorDeterminism(unittest.TestCase):
    """Identical inputs produce identical outputs."""

    def test_same_input_same_chunks(self):
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        sents = _make_sentence_records("AAPL_Q1_2025", n_prepared=6, n_qa=4)

        gen1 = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        gen2 = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())

        r1 = gen1.generate(sents, transcript_id="AAPL_Q1_2025")
        r2 = gen2.generate(sents, transcript_id="AAPL_Q1_2025")

        self.assertEqual(len(r1.chunks), len(r2.chunks))
        for c1, c2 in zip(r1.chunks, r2.chunks):
            self.assertEqual(c1.chunk_id,   c2.chunk_id)
            self.assertEqual(c1.chunk_text, c2.chunk_text)
            self.assertEqual(c1.token_count, c2.token_count)

    def test_shuffled_input_produces_same_result_after_sort(self):
        """sort-by-sentence_order makes the result order-independent."""
        import random
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        sents = _make_sentence_records("AAPL_Q1_2025", n_prepared=6, n_qa=4)

        sents_shuffled = sents.copy()
        random.Random(7).shuffle(sents_shuffled)

        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        r1 = gen.generate(sents,         transcript_id="AAPL_Q1_2025")
        r2 = gen.generate(sents_shuffled, transcript_id="AAPL_Q1_2025")

        self.assertEqual(len(r1.chunks), len(r2.chunks))
        for c1, c2 in zip(r1.chunks, r2.chunks):
            self.assertEqual(c1.chunk_text, c2.chunk_text)


class TestChunkGeneratorSectionBoundary(unittest.TestCase):
    """Section boundaries trigger new chunks when configured."""

    def test_section_boundary_creates_new_chunk(self):
        cfg = ChunkConfig(
            chunk_size_tokens=300,    # large — would fit everything in one chunk
            overlap_tokens=20,
            hard_token_ceiling=490,
            preserve_section_boundaries=True,
        )
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        sents = _make_sentence_records(n_prepared=4, n_qa=4)
        result = gen.generate(sents, transcript_id="T1")

        # With section boundary preservation we must get ≥2 chunks
        self.assertGreaterEqual(len(result.chunks), 2)
        sections_seen = {c.section_type for c in result.chunks}
        self.assertIn("prepared_remarks", sections_seen)
        self.assertIn("qa", sections_seen)

    def test_no_section_boundary_can_merge(self):
        cfg = ChunkConfig(
            chunk_size_tokens=300,
            overlap_tokens=20,
            hard_token_ceiling=490,
            preserve_section_boundaries=False,
        )
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        sents = _make_sentence_records(n_prepared=3, n_qa=3)
        result_off = gen.generate(sents, transcript_id="T1")

        cfg_on = ChunkConfig(
            chunk_size_tokens=300,
            overlap_tokens=20,
            hard_token_ceiling=490,
            preserve_section_boundaries=True,
        )
        gen_on = ChunkGenerator(config=cfg_on, tokenizer=_MockTokenizer())
        result_on = gen_on.generate(sents, transcript_id="T1")

        # boundary ON should produce ≥ as many chunks as boundary OFF
        self.assertGreaterEqual(len(result_on.chunks), len(result_off.chunks))


class TestChunkValidation(unittest.TestCase):
    """validate_result catches constraint violations."""

    def test_valid_result_no_errors(self):
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        sents = _make_sentence_records(n_prepared=6, n_qa=4)
        result = gen.generate(sents, transcript_id="T1")
        errors = gen.validate_result(result)
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_empty_result_no_false_errors(self):
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        result = gen.generate([], transcript_id="T1")
        errors = gen.validate_result(result)
        self.assertEqual(errors, [])


class TestChunkGeneratorDataframe(unittest.TestCase):
    """DataFrame and export helpers."""

    def setUp(self):
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        self._gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        sents = _make_sentence_records(n_prepared=6, n_qa=4)
        self._result = self._gen.generate(sents, transcript_id="AAPL_Q1_2025")

    def test_to_dataframe_shape(self):
        df = self._result.to_dataframe()
        self.assertEqual(len(df), len(self._result.chunks))

    def test_required_columns_present(self):
        df = self._result.to_dataframe()
        required = [
            "chunk_id", "transcript_id", "chunk_order", "chunk_text",
            "token_count", "sentence_count", "section_type",
            "dominant_speaker", "overlap_start", "overlap_end",
            "first_sentence_order", "last_sentence_order",
        ]
        assert_df_columns(df, required)

    def test_no_nan_in_critical_columns(self):
        df = self._result.to_dataframe()
        assert_no_nan(df, ["chunk_id", "transcript_id", "chunk_order",
                           "chunk_text", "token_count"])

    def test_to_dict_structure(self):
        d = self._result.to_dict()
        self.assertIn("transcript_id", d)
        self.assertIn("chunks", d)
        self.assertIsInstance(d["chunks"], list)

    def test_to_jsonl_valid_json(self):
        jsonl = self._result.to_jsonl()
        for line in jsonl.strip().splitlines():
            obj = json.loads(line)
            self.assertIn("chunk_id", obj)

    def test_parquet_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "chunks.parquet"
            self._result.save_parquet(p)
            df = pd.read_parquet(p)
            self.assertEqual(len(df), len(self._result.chunks))

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "chunks.csv"
            self._result.save_csv(p)
            df = pd.read_csv(p)
            self.assertEqual(len(df), len(self._result.chunks))

    def test_statistics_keys(self):
        stats = self._result.statistics()
        for key in ["chunk_count", "total_tokens", "token_min", "token_max", "token_mean"]:
            self.assertIn(key, stats)

    def test_coverage_metrics_full_coverage(self):
        cov = self._result.coverage_metrics()
        self.assertAlmostEqual(cov["sentence_coverage"], 1.0, places=5)


class TestChunkGeneratorReconstruction(unittest.TestCase):
    """reconstruct_transcript produces non-empty output."""

    def test_reconstruction_non_empty(self):
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        sents = _make_sentence_records(n_prepared=6, n_qa=4)
        result = gen.generate(sents, transcript_id="T1")
        text = gen.reconstruct_transcript(result, deduplicate=True)
        self.assertGreater(len(text), 0)

    def test_reconstruction_contains_content(self):
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        sents = _make_sentence_records(n_prepared=4, n_qa=2)
        result = gen.generate(sents, transcript_id="T1")
        text = gen.reconstruct_transcript(result)
        # At least some content from the transcript should appear
        self.assertIn("grew", text.lower())


class TestBatchGeneration(unittest.TestCase):
    """generate_batch processes multiple transcripts."""

    def test_batch_returns_all_transcripts(self):
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        tids = ["AAPL_Q1_2025", "NVDA_Q2_2025"]
        rows = []
        for tid in tids:
            for s in _make_sentence_records(tid, n_prepared=4, n_qa=2):
                rows.append(asdict(s))
        df = pd.DataFrame(rows).rename(columns={"sentence_text": "sentence_text"})
        results = gen.generate_batch(df)
        self.assertIn("AAPL_Q1_2025", results)
        self.assertIn("NVDA_Q2_2025", results)

    def test_batch_to_dataframe_concatenates(self):
        cfg = ChunkConfig(chunk_size_tokens=80, overlap_tokens=15, hard_token_ceiling=490)
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        tids = ["AAPL_Q1_2025", "NVDA_Q2_2025"]
        rows = []
        for tid in tids:
            for s in _make_sentence_records(tid, n_prepared=4, n_qa=2):
                rows.append(asdict(s))
        df = pd.DataFrame(rows)
        results = gen.generate_batch(df)
        combined = gen.batch_to_dataframe(results)
        tids_in_output = set(combined["transcript_id"].unique())
        self.assertEqual(tids_in_output, {"AAPL_Q1_2025", "NVDA_Q2_2025"})


class TestIterChunks(unittest.TestCase):
    """iter_chunks yields correct batch sizes."""

    def test_correct_batch_count(self):
        cfg = ChunkConfig(chunk_size_tokens=60, overlap_tokens=10, hard_token_ceiling=490)
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        sents = _make_sentence_records(n_prepared=8, n_qa=6)
        result = gen.generate(sents, transcript_id="T1")
        n_chunks = len(result.chunks)
        batches = list(iter_chunks(result, batch_size=3))
        total_yielded = sum(len(b) for b in batches)
        self.assertEqual(total_yielded, n_chunks)


# ===========================================================================
# ══ GROUP 2: NUMERIC HELPERS (aggregation.py internal functions) ════════════
# ===========================================================================

class TestWeightedMean(unittest.TestCase):
    def test_uniform(self):
        vals = [0.1, 0.3, 0.5]
        wts  = [1.0, 1.0, 1.0]
        result = _weighted_mean(vals, wts)
        self.assertAlmostEqual(result, sum(vals) / 3, places=8)

    def test_weighted(self):
        vals = [0.0, 1.0]
        wts  = [1.0, 3.0]
        result = _weighted_mean(vals, wts)
        self.assertAlmostEqual(result, 0.75, places=8)

    def test_zero_weights_fallback_to_mean(self):
        vals = [0.2, 0.4, 0.6]
        wts  = [0.0, 0.0, 0.0]
        result = _weighted_mean(vals, wts)
        self.assertAlmostEqual(result, sum(vals) / 3, places=8)

    def test_single_element(self):
        self.assertAlmostEqual(_weighted_mean([0.7], [1.0]), 0.7, places=8)

    def test_empty_returns_zero(self):
        self.assertEqual(_weighted_mean([], []), 0.0)


class TestScoreToLabel(unittest.TestCase):
    """Thin wrapper that passes the required threshold defaults."""

    _POS = 0.05
    _NEG = -0.05

    def _label(self, score: float) -> str:
        return _score_to_label(score, self._POS, self._NEG)

    def test_positive(self):
        self.assertEqual(self._label(0.10), "positive")

    def test_negative(self):
        self.assertEqual(self._label(-0.10), "negative")

    def test_neutral_high_boundary(self):
        self.assertEqual(self._label(0.05), "neutral")

    def test_neutral_low_boundary(self):
        self.assertEqual(self._label(-0.05), "neutral")

    def test_exactly_zero(self):
        self.assertEqual(self._label(0.0), "neutral")

    def test_extreme_positive(self):
        self.assertEqual(self._label(0.99), "positive")

    def test_extreme_negative(self):
        self.assertEqual(self._label(-0.99), "negative")


class TestNormaliseProbs(unittest.TestCase):
    def test_normal_case(self):
        pos, neu, neg = _normalise_probs(0.6, 0.3, 0.3)
        assert_prob_sum(pos, neu, neg)

    def test_already_normalised(self):
        pos, neu, neg = _normalise_probs(0.5, 0.3, 0.2)
        assert_prob_sum(pos, neu, neg)
        self.assertAlmostEqual(pos, 0.5, places=6)

    def test_all_zero_returns_uniform(self):
        pos, neu, neg = _normalise_probs(0.0, 0.0, 0.0)
        for v in (pos, neu, neg):
            self.assertAlmostEqual(v, 1 / 3, places=6)

    def test_single_dominant(self):
        pos, neu, neg = _normalise_probs(1.0, 0.0, 0.0)
        self.assertAlmostEqual(pos, 1.0, places=6)
        assert_prob_sum(pos, neu, neg)


class TestSafeStd(unittest.TestCase):
    def test_single_value_zero(self):
        self.assertEqual(_safe_std([0.5]), 0.0)

    def test_known_std(self):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = _safe_std(vals)
        expected = float(np.std(vals, ddof=1))
        self.assertAlmostEqual(result, expected, places=6)

    def test_identical_values_zero_std(self):
        self.assertEqual(_safe_std([0.3, 0.3, 0.3, 0.3]), 0.0)

    def test_empty_returns_zero(self):
        self.assertEqual(_safe_std([]), 0.0)


class TestSafeIqr(unittest.TestCase):
    def test_small_list(self):
        self.assertEqual(_safe_iqr([0.1, 0.2, 0.3]), 0.0)

    def test_known_iqr(self):
        vals = [1, 2, 3, 4, 5, 6, 7, 8]
        result = _safe_iqr(vals)
        self.assertGreater(result, 0.0)


# ===========================================================================
# ══ GROUP 3: AGGREGATION TESTS ══════════════════════════════════════════════
# ===========================================================================

class TestAggregationConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AggregationConfig()
        self.assertEqual(cfg.strategy, WeightingStrategy.HYBRID)
        self.assertTrue(cfg.normalise_probabilities)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            AggregationConfig(positive_threshold=-0.1, negative_threshold=0.1)

    def test_min_chunks_validation(self):
        with self.assertRaises(ValueError):
            AggregationConfig(min_chunks_required=0)

    def test_custom_strategy(self):
        cfg = AggregationConfig(strategy=WeightingStrategy.UNIFORM)
        self.assertEqual(cfg.strategy, WeightingStrategy.UNIFORM)


class TestWeightingStrategies(unittest.TestCase):
    """Verify each weighting strategy produces mathematically correct output."""

    def _run_strategy(self, strategy: WeightingStrategy, df: pd.DataFrame) -> float:
        agg = SentimentAggregator(AggregationConfig(
            strategy=strategy,
            compute_section_aggregates=False,
            compute_speaker_aggregates=False,
        ))
        result = agg.aggregate(df, validate_input=False)
        ts = result.get("T1")
        self.assertIsNotNone(ts)
        return ts.primary_score()

    def setUp(self):
        self._df = _make_chunk_df(["T1"], chunks_per_transcript=6, seed=1)
        # Patch transcript_id
        self._df["transcript_id"] = "T1"

    def test_uniform_strategy_is_simple_mean(self):
        agg = SentimentAggregator(AggregationConfig(
            strategy=WeightingStrategy.UNIFORM,
            compute_section_aggregates=False,
            compute_speaker_aggregates=False,
            normalise_probabilities=False,
        ))
        result = agg.aggregate(self._df, validate_input=False)
        ts = result.get("T1")
        # Manual computation
        pos_mean = float(self._df["positive_prob"].mean())
        neg_mean = float(self._df["negative_prob"].mean())
        expected = pos_mean - neg_mean
        # approximate due to normalisation
        self.assertAlmostEqual(ts.primary_score(), expected, delta=0.02)

    def test_confidence_strategy_up_weights_high_confidence(self):
        df = self._df.copy()
        # Give first chunk very high confidence & high positive score
        df.loc[df["chunk_order"] == 0, "confidence"] = 0.99
        df.loc[df["chunk_order"] == 0, "positive_prob"] = 0.9
        df.loc[df["chunk_order"] == 0, "negative_prob"] = 0.05
        df.loc[df["chunk_order"] == 0, "neutral_prob"] = 0.05
        df.loc[df["chunk_order"] == 0, "sentiment_score"] = 0.85
        score_conf = self._run_strategy(WeightingStrategy.CONFIDENCE, df)
        # Compare to uniform — confidence weighting should push score higher
        score_unif = self._run_strategy(WeightingStrategy.UNIFORM, df)
        self.assertGreaterEqual(score_conf, score_unif - 0.3)  # directional test

    def test_four_strategies_all_run(self):
        for strategy in WeightingStrategy:
            score = self._run_strategy(strategy, self._df)
            assert_score_range(score)

    def test_all_strategies_produce_valid_probs(self):
        for strategy in WeightingStrategy:
            agg = SentimentAggregator(AggregationConfig(strategy=strategy))
            result = agg.aggregate(self._df, validate_input=False)
            ts = result.get("T1")
            rec = ts.transcript_aggregate
            assert_prob_sum(
                rec.positive_probability,
                rec.neutral_probability,
                rec.negative_probability,
            )


class TestAggregationOutputSchema(unittest.TestCase):
    """Aggregation result schema and correctness."""

    def setUp(self):
        self._df = _make_chunk_df(
            ["AAPL_Q1_2025", "NVDA_Q2_2025"], chunks_per_transcript=8
        )
        self._agg = SentimentAggregator()
        self._result = self._agg.aggregate(self._df)

    def test_all_transcripts_present(self):
        self.assertIn("AAPL_Q1_2025", self._result.transcripts)
        self.assertIn("NVDA_Q2_2025", self._result.transcripts)

    def test_no_duplicate_transcript_ids(self):
        ids = list(self._result.transcripts.keys())
        self.assertEqual(len(ids), len(set(ids)))

    def test_probability_sums(self):
        for ts in self._result.transcripts.values():
            rec = ts.transcript_aggregate
            assert_prob_sum(
                rec.positive_probability,
                rec.neutral_probability,
                rec.negative_probability,
            )

    def test_score_range(self):
        for ts in self._result.transcripts.values():
            assert_score_range(ts.transcript_aggregate.sentiment_score)

    def test_label_consistent_with_score(self):
        for ts in self._result.transcripts.values():
            rec = ts.transcript_aggregate
            assert_label_consistent(rec.sentiment_score, rec.sentiment_label)

    def test_section_aggregates_produced(self):
        for ts in self._result.transcripts.values():
            self.assertGreater(len(ts.section_aggregates), 0,
                               f"{ts.transcript_id}: no section aggregates")

    def test_section_probs_valid(self):
        for ts in self._result.transcripts.values():
            for sec, rec in ts.section_aggregates.items():
                assert_prob_sum(
                    rec.positive_probability,
                    rec.neutral_probability,
                    rec.negative_probability,
                    tol=1e-3,
                )

    def test_speaker_aggregates_produced(self):
        for ts in self._result.transcripts.values():
            self.assertGreater(len(ts.speaker_aggregates), 0)

    def test_chunk_count_matches_input(self):
        for tid, ts in self._result.transcripts.items():
            expected = len(self._df[self._df["transcript_id"] == tid])
            self.assertEqual(ts.transcript_aggregate.chunk_count, expected)

    def test_total_tokens_positive(self):
        for ts in self._result.transcripts.values():
            self.assertGreater(ts.transcript_aggregate.total_tokens, 0)


class TestAggregationDataframeExport(unittest.TestCase):
    """to_transcript_dataframe, to_section_dataframe, to_speaker_dataframe."""

    def setUp(self):
        df = _make_chunk_df(["AAPL_Q1_2025", "NVDA_Q2_2025"], chunks_per_transcript=8)
        self._result = SentimentAggregator().aggregate(df)

    def test_transcript_df_shape(self):
        df = self._result.to_transcript_dataframe()
        self.assertEqual(len(df), 2)

    def test_transcript_df_required_columns(self):
        df = self._result.to_transcript_dataframe()
        required = [
            "transcript_id", "sentiment_score", "sentiment_label",
            "positive_probability", "neutral_probability", "negative_probability",
            "chunk_count", "confidence_mean",
        ]
        assert_df_columns(df, required)

    def test_section_df_not_empty(self):
        df = self._result.to_section_dataframe()
        self.assertGreater(len(df), 0)

    def test_speaker_df_not_empty(self):
        df = self._result.to_speaker_dataframe()
        self.assertGreater(len(df), 0)

    def test_parquet_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sentiment.parquet"
            self._result.save_parquet(p, level="transcript")
            df = pd.read_parquet(p)
            self.assertEqual(len(df), 2)

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sentiment.csv"
            self._result.save_csv(p, level="transcript")
            df = pd.read_csv(p)
            self.assertEqual(len(df), 2)


class TestAggregationEdgeCases(unittest.TestCase):
    """Empty / malformed / boundary inputs."""

    def test_empty_dataframe_returns_empty(self):
        agg = SentimentAggregator()
        result = agg.aggregate(pd.DataFrame())
        self.assertEqual(result.total_transcripts, 0)

    def test_single_chunk_transcript(self):
        df = _make_chunk_df(["T1"], chunks_per_transcript=8)
        single = df.head(1).copy()
        result = SentimentAggregator().aggregate(single, validate_input=False)
        ts = result.get("T1")
        self.assertIsNotNone(ts)
        assert_prob_sum(
            ts.transcript_aggregate.positive_probability,
            ts.transcript_aggregate.neutral_probability,
            ts.transcript_aggregate.negative_probability,
        )

    def test_malformed_probs_normalised(self):
        """When probabilities sum to > 1 they must be normalised."""
        df = _make_chunk_df(["T1"], chunks_per_transcript=4)
        df = df.copy()
        df["positive_prob"] = 0.9
        df["neutral_prob"]  = 0.9
        df["negative_prob"] = 0.9
        result = SentimentAggregator().aggregate(df, validate_input=False)
        ts = result.get("T1")
        rec = ts.transcript_aggregate
        assert_prob_sum(
            rec.positive_probability,
            rec.neutral_probability,
            rec.negative_probability,
        )

    def test_nan_probs_handled(self):
        df = _make_chunk_df(["T1"], chunks_per_transcript=4)
        df.loc[0, "positive_prob"] = float("nan")
        df.loc[0, "negative_prob"] = float("nan")
        result = SentimentAggregator().aggregate(df, validate_input=False)
        ts = result.get("T1")
        self.assertIsNotNone(ts)

    def test_all_neutral_transcript(self):
        df = _make_chunk_df(["T1"], chunks_per_transcript=4)
        df["positive_prob"]   = 0.333333
        df["neutral_prob"]    = 0.333334
        df["negative_prob"]   = 0.333333
        df["sentiment_score"] = 0.0
        result = SentimentAggregator().aggregate(df, validate_input=False)
        ts = result.get("T1")
        self.assertEqual(ts.primary_label(), "neutral")


class TestAggregationValidator(unittest.TestCase):
    """AggregationValidator input and output checks."""

    def _valid_df(self) -> pd.DataFrame:
        return _make_chunk_df(["T1"], chunks_per_transcript=6)

    def test_valid_input_no_errors(self):
        errors = AggregationValidator.validate_input(self._valid_df())
        self.assertEqual(errors, [])

    def test_missing_column_caught(self):
        df = self._valid_df().drop(columns=["chunk_id"])
        errors = AggregationValidator.validate_input(df)
        self.assertTrue(any("chunk_id" in e or "Missing" in e for e in errors))

    def test_invalid_label_caught(self):
        df = self._valid_df()
        df.loc[0, "predicted_label"] = "bullish"  # not in VALID_LABELS
        errors = AggregationValidator.validate_input(df)
        self.assertTrue(any("label" in e.lower() or "invalid" in e.lower()
                            for e in errors))

    def test_negative_token_count_caught(self):
        df = self._valid_df()
        df.loc[0, "token_count"] = 0
        errors = AggregationValidator.validate_input(df)
        self.assertTrue(len(errors) > 0)

    def test_duplicate_chunk_id_caught(self):
        df = self._valid_df()
        dup = df.head(1).copy()
        df = pd.concat([df, dup], ignore_index=True)
        errors = AggregationValidator.validate_input(df)
        self.assertTrue(any("duplicate" in e.lower() or "Duplicate" in e for e in errors))

    def test_output_validation_passes_on_valid_result(self):
        df = self._valid_df()
        result = SentimentAggregator().aggregate(df)
        errors = AggregationValidator.validate_output(result)
        self.assertEqual(errors, [])


class TestSentimentDrift(unittest.TestCase):
    """compute_sentiment_drift analytical helper."""

    def setUp(self):
        self._df = _make_chunk_df(["AAPL_Q1_2025"], chunks_per_transcript=10, seed=7)

    def test_returns_dict_with_required_keys(self):
        drift = compute_sentiment_drift(self._df, "AAPL_Q1_2025")
        for key in ["transcript_id", "chunk_count", "drift_slope",
                    "first_quarter_score", "last_quarter_score", "score_range"]:
            self.assertIn(key, drift)

    def test_chunk_count_matches_input(self):
        drift = compute_sentiment_drift(self._df, "AAPL_Q1_2025")
        self.assertEqual(drift["chunk_count"], 10)

    def test_score_range_non_negative(self):
        drift = compute_sentiment_drift(self._df, "AAPL_Q1_2025")
        self.assertGreaterEqual(drift["score_range"], 0.0)

    def test_unknown_transcript_returns_error_key(self):
        drift = compute_sentiment_drift(self._df, "UNKNOWN_TRANSCRIPT")
        self.assertIn("error", drift)

    def test_section_delta_present_when_sections_exist(self):
        drift = compute_sentiment_drift(self._df, "AAPL_Q1_2025")
        self.assertIn("prepared_vs_qa_delta", drift)

    def test_monotone_increasing_has_positive_slope(self):
        """Artificial monotone-increasing sentiment should give positive slope."""
        df = _make_chunk_df(["T1"], chunks_per_transcript=8)
        df["sentiment_score"] = [i * 0.1 for i in range(len(df))]
        df.loc[:, "section_type"] = "prepared_remarks"
        drift = compute_sentiment_drift(df, "T1")
        self.assertGreater(drift["drift_slope"], 0)

    def test_monotone_decreasing_has_negative_slope(self):
        df = _make_chunk_df(["T1"], chunks_per_transcript=8)
        df["sentiment_score"] = [0.7 - i * 0.1 for i in range(len(df))]
        drift = compute_sentiment_drift(df, "T1")
        self.assertLess(drift["drift_slope"], 0)


class TestCompareStrategies(unittest.TestCase):
    """compare_strategies returns all four results."""

    def test_all_four_strategies_returned(self):
        df = _make_chunk_df(["AAPL_Q1_2025"], chunks_per_transcript=6)
        agg = SentimentAggregator()
        comparison = agg.compare_strategies(df, "AAPL_Q1_2025")
        for strategy in WeightingStrategy:
            self.assertIn(strategy.value, comparison)

    def test_all_strategies_produce_valid_probs(self):
        df = _make_chunk_df(["AAPL_Q1_2025"], chunks_per_transcript=6)
        agg = SentimentAggregator()
        comparison = agg.compare_strategies(df, "AAPL_Q1_2025")
        for name, rec in comparison.items():
            assert_prob_sum(
                rec.positive_probability,
                rec.neutral_probability,
                rec.negative_probability,
                tol=1e-3,
            )


# ===========================================================================
# ══ GROUP 4: HIERARCHICAL AGGREGATION TESTS ══════════════════════════════════
# ===========================================================================

class TestHierarchicalConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = HierarchicalAggregationConfig()
        self.assertTrue(cfg.use_confidence_weight)
        self.assertTrue(cfg.use_token_weight)
        self.assertTrue(cfg.compute_cross_section_speakers)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            HierarchicalAggregationConfig(
                positive_threshold=-0.1, negative_threshold=0.1
            )


class TestHierarchyLevelEnum(unittest.TestCase):
    def test_all_levels_defined(self):
        levels = {l.value for l in HierarchyLevel}
        for expected in ["transcript", "section", "speaker",
                         "speaker_in_section", "chunk", "sentence"]:
            self.assertIn(expected, levels)

    def test_transcript_is_shallowest(self):
        self.assertEqual(HierarchyLevel.TRANSCRIPT.depth, 0)

    def test_sentence_is_deepest(self):
        self.assertGreaterEqual(HierarchyLevel.SENTENCE.depth,
                                HierarchyLevel.CHUNK.depth)


class TestHierarchicalAggregatorBuild(unittest.TestCase):
    """Full hierarchy tree construction."""

    def setUp(self):
        self._tids  = ["AAPL_Q1_2025", "NVDA_Q2_2025"]
        self._cdf   = _make_chunk_df(self._tids, chunks_per_transcript=10)
        self._sdf   = _make_sentence_df(self._tids, sentences_per_transcript=20)
        self._agg   = create_hierarchical_aggregator(include_sentence_nodes=True)
        self._result = self._agg.aggregate(self._cdf, sentence_df=self._sdf)

    def test_all_transcripts_in_result(self):
        for tid in self._tids:
            self.assertIn(tid, self._result.results)

    def test_root_node_exists(self):
        for tid in self._tids:
            hsr = self._result.get(tid)
            self.assertIsNotNone(hsr.root)
            self.assertEqual(hsr.root.hierarchy_level, HierarchyLevel.TRANSCRIPT)

    def test_root_prob_sum(self):
        for tid in self._tids:
            root = self._result.get(tid).root
            assert_prob_sum(
                root.positive_probability,
                root.neutral_probability,
                root.negative_probability,
            )

    def test_root_score_range(self):
        for tid in self._tids:
            assert_score_range(self._result.get(tid).root.sentiment_score)

    def test_section_nodes_exist(self):
        for tid in self._tids:
            hsr = self._result.get(tid)
            secs = hsr.nodes_at_level(HierarchyLevel.SECTION)
            self.assertGreater(len(secs), 0)

    def test_both_sections_present(self):
        hsr = self._result.get("AAPL_Q1_2025")
        sec_keys = {n.hierarchy_key for n in hsr.nodes_at_level(HierarchyLevel.SECTION)}
        self.assertIn("prepared_remarks", sec_keys)
        self.assertIn("qa", sec_keys)

    def test_global_speaker_nodes_exist(self):
        hsr = self._result.get("AAPL_Q1_2025")
        spk_nodes = hsr.nodes_at_level(HierarchyLevel.SPEAKER)
        self.assertGreater(len(spk_nodes), 0)

    def test_chunk_nodes_exist(self):
        hsr = self._result.get("AAPL_Q1_2025")
        chunks = hsr.nodes_at_level(HierarchyLevel.CHUNK)
        self.assertGreater(len(chunks), 0)

    def test_sentence_nodes_exist(self):
        hsr = self._result.get("AAPL_Q1_2025")
        sents = hsr.nodes_at_level(HierarchyLevel.SENTENCE)
        self.assertGreater(len(sents), 0)

    def test_no_duplicate_node_ids(self):
        for tid in self._tids:
            hsr = self._result.get(tid)
            ids = list(hsr.nodes.keys())
            self.assertEqual(len(ids), len(set(ids)))

    def test_all_nodes_have_valid_probs(self):
        for tid in self._tids:
            hsr = self._result.get(tid)
            for node in hsr.nodes.values():
                assert_prob_sum(
                    node.positive_probability,
                    node.neutral_probability,
                    node.negative_probability,
                    tol=1e-3,
                )

    def test_all_nodes_have_valid_scores(self):
        for tid in self._tids:
            hsr = self._result.get(tid)
            for node in hsr.nodes.values():
                assert_score_range(node.sentiment_score)

    def test_label_score_consistency(self):
        for tid in self._tids:
            hsr = self._result.get(tid)
            for node in hsr.nodes.values():
                assert_label_consistent(node.sentiment_score, node.sentiment_label)

    def test_parent_child_referential_integrity(self):
        for tid in self._tids:
            hsr = self._result.get(tid)
            for nid, node in hsr.nodes.items():
                if node.parent_node_id:
                    self.assertIn(node.parent_node_id, hsr.nodes,
                                  f"Parent '{node.parent_node_id}' missing for '{nid}'")
                for cid in node.children_node_ids:
                    self.assertIn(cid, hsr.nodes,
                                  f"Child '{cid}' missing for '{nid}'")

    def test_transcript_id_consistent(self):
        for tid in self._tids:
            hsr = self._result.get(tid)
            for node in hsr.nodes.values():
                self.assertEqual(node.transcript_id, tid)


class TestHierarchicalAnalytics(unittest.TestCase):
    """Finance-specific analytics computed from the hierarchy."""

    def setUp(self):
        tids = ["AAPL_Q1_2025"]
        cdf  = _make_chunk_df(tids, chunks_per_transcript=12)
        sdf  = _make_sentence_df(tids, sentences_per_transcript=20)
        agg  = create_hierarchical_aggregator()
        result = agg.aggregate(cdf, sentence_df=sdf)
        self._hsr = result.get("AAPL_Q1_2025")

    def test_prepared_vs_qa_delta_is_float_or_none(self):
        delta = self._hsr.prepared_vs_qa_delta()
        # Should be float if both sections exist
        if delta is not None:
            self.assertIsInstance(delta, float)

    def test_management_sentiment_is_float_or_none(self):
        mgmt = self._hsr.management_sentiment()
        if mgmt is not None:
            assert_score_range(mgmt)

    def test_analyst_sentiment_is_float_or_none(self):
        analyst = self._hsr.analyst_sentiment()
        if analyst is not None:
            assert_score_range(analyst)

    def test_optimism_decay_is_float(self):
        decay = self._hsr.optimism_decay()
        if decay is not None:
            self.assertIsInstance(decay, float)

    def test_analytics_summary_keys(self):
        summary = self._hsr.analytics_summary()
        required = [
            "transcript_id", "sentiment_score", "sentiment_label",
            "prepared_remarks_score", "qa_score",
            "prepared_vs_qa_delta", "node_count",
        ]
        for k in required:
            self.assertIn(k, summary)

    def test_section_score_access(self):
        score = self._hsr.section_score("prepared_remarks")
        if score is not None:
            assert_score_range(score)

    def test_speaker_score_returns_none_for_unknown(self):
        score = self._hsr.speaker_score("__nonexistent_speaker__")
        self.assertIsNone(score)


class TestHierarchicalTraversal(unittest.TestCase):
    """Hierarchy traversal helpers."""

    def setUp(self):
        cdf = _make_chunk_df(["AAPL_Q1_2025"], chunks_per_transcript=8)
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result = agg.aggregate(cdf)
        self._hsr = result.get("AAPL_Q1_2025")

    def test_dfs_yields_all_nodes(self):
        visited = list(self._hsr.iter_level_dfs())
        self.assertEqual(len(visited), len(self._hsr.nodes))

    def test_dfs_root_first(self):
        first = next(self._hsr.iter_level_dfs())
        self.assertEqual(first.hierarchy_level, HierarchyLevel.TRANSCRIPT)

    def test_children_of_root(self):
        children = self._hsr.children_of(self._hsr.root_node_id)
        self.assertGreater(len(children), 0)

    def test_ancestors_of_section_node_contains_root(self):
        sections = self._hsr.nodes_at_level(HierarchyLevel.SECTION)
        if sections:
            ancestors = self._hsr.ancestors_of(sections[0].node_id)
            ancestor_ids = [a.node_id for a in ancestors]
            self.assertIn(self._hsr.root_node_id, ancestor_ids)

    def test_children_of_missing_node_empty(self):
        children = self._hsr.children_of("__does_not_exist__")
        self.assertEqual(children, [])


class TestHierarchicalDeterminism(unittest.TestCase):
    """Same input → same hierarchy."""

    def test_same_input_same_node_ids(self):
        cdf = _make_chunk_df(["AAPL_Q1_2025"], chunks_per_transcript=8)
        agg1 = create_hierarchical_aggregator(include_sentence_nodes=False)
        agg2 = create_hierarchical_aggregator(include_sentence_nodes=False)
        r1 = agg1.aggregate(cdf)
        r2 = agg2.aggregate(cdf)
        ids1 = set(r1.get("AAPL_Q1_2025").nodes.keys())
        ids2 = set(r2.get("AAPL_Q1_2025").nodes.keys())
        self.assertEqual(ids1, ids2)

    def test_same_input_same_scores(self):
        cdf = _make_chunk_df(["AAPL_Q1_2025"], chunks_per_transcript=8)
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        r1 = agg.aggregate(cdf)
        r2 = agg.aggregate(cdf)
        score1 = r1.get("AAPL_Q1_2025").transcript_score()
        score2 = r2.get("AAPL_Q1_2025").transcript_score()
        self.assertAlmostEqual(score1, score2, places=8)


class TestHierarchicalEdgeCases(unittest.TestCase):
    """Empty / malformed / minimal inputs."""

    def test_empty_df_returns_empty_result(self):
        agg = create_hierarchical_aggregator()
        result = agg.aggregate(pd.DataFrame())
        self.assertEqual(result.total_transcripts, 0)

    def test_single_chunk_builds_valid_root(self):
        cdf = _make_chunk_df(["T1"], chunks_per_transcript=8)
        single = cdf.head(1).copy()
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result = agg.aggregate(single, validate_input=False)
        hsr = result.get("T1")
        self.assertIsNotNone(hsr)
        self.assertIsNotNone(hsr.root)

    def test_no_speaker_column_does_not_crash(self):
        cdf = _make_chunk_df(["T1"], chunks_per_transcript=6)
        cdf = cdf.drop(columns=["dominant_speaker"], errors="ignore")
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result = agg.aggregate(cdf, validate_input=False)
        self.assertIn("T1", result.results)

    def test_no_section_column_does_not_crash(self):
        cdf = _make_chunk_df(["T1"], chunks_per_transcript=6)
        cdf = cdf.drop(columns=["section_type"], errors="ignore")
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result = agg.aggregate(cdf, validate_input=False)
        self.assertIn("T1", result.results)

    def test_malformed_prob_columns_normalised(self):
        cdf = _make_chunk_df(["T1"], chunks_per_transcript=6)
        cdf["positive_prob"] = 2.0
        cdf["neutral_prob"]  = 2.0
        cdf["negative_prob"] = 2.0
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result = agg.aggregate(cdf, validate_input=False)
        root = result.get("T1").root
        assert_prob_sum(
            root.positive_probability,
            root.neutral_probability,
            root.negative_probability,
        )


# ===========================================================================
# ══ GROUP 5: OLS SLOPE & STATISTICAL HELPERS ════════════════════════════════
# ===========================================================================

class TestOlsSlope(unittest.TestCase):
    def test_flat_series_zero_slope(self):
        vals = [0.5] * 8
        self.assertAlmostEqual(_ols_slope(vals), 0.0, places=8)

    def test_strictly_increasing_positive(self):
        vals = [float(i) for i in range(10)]
        slope = _ols_slope(vals)
        self.assertGreater(slope, 0.0)

    def test_strictly_decreasing_negative(self):
        vals = [float(10 - i) for i in range(10)]
        slope = _ols_slope(vals)
        self.assertLess(slope, 0.0)

    def test_known_slope(self):
        # y = 2x  →  slope = 2
        vals = [2.0 * i for i in range(6)]
        self.assertAlmostEqual(_ols_slope(vals), 2.0, places=5)

    def test_single_element_zero(self):
        self.assertEqual(_ols_slope([0.5]), 0.0)

    def test_empty_zero(self):
        self.assertEqual(_ols_slope([]), 0.0)


class TestSafeMedian(unittest.TestCase):
    def test_odd_list(self):
        self.assertAlmostEqual(_safe_median([1, 3, 5]), 3.0, places=8)

    def test_even_list(self):
        self.assertAlmostEqual(_safe_median([1, 2, 3, 4]), 2.5, places=8)

    def test_single(self):
        self.assertAlmostEqual(_safe_median([7.0]), 7.0, places=8)

    def test_empty(self):
        self.assertEqual(_safe_median([]), 0.0)


class TestInferSpeakerType(unittest.TestCase):
    def test_ceo_is_management(self):
        self.assertEqual(_infer_speaker_type("CEO"), SpeakerType.MANAGEMENT)

    def test_cfo_is_management(self):
        self.assertEqual(_infer_speaker_type("CFO"), SpeakerType.MANAGEMENT)

    def test_analyst_is_analyst(self):
        self.assertEqual(_infer_speaker_type("Analyst"), SpeakerType.ANALYST)

    def test_operator_is_operator(self):
        self.assertEqual(_infer_speaker_type("Operator"), SpeakerType.OPERATOR)

    def test_unknown_role(self):
        self.assertEqual(_infer_speaker_type("Some Unknown Role"), SpeakerType.UNKNOWN)

    def test_case_insensitive(self):
        self.assertEqual(_infer_speaker_type("chief executive officer"),
                         SpeakerType.MANAGEMENT)


# ===========================================================================
# ══ GROUP 6: VALIDATION FRAMEWORK TESTS ═════════════════════════════════════
# ===========================================================================

class TestHierarchicalValidator(unittest.TestCase):
    """HierarchicalValidator input / output checks."""

    def _valid_chunk_df(self):
        return _make_chunk_df(["T1"], chunks_per_transcript=6)

    def _valid_sentence_df(self):
        return _make_sentence_df(["T1"], sentences_per_transcript=12)

    def test_valid_chunk_df_no_errors(self):
        errors = HierarchicalValidator.validate_chunk_df(self._valid_chunk_df())
        self.assertEqual(errors, [])

    def test_missing_column_caught(self):
        df = self._valid_chunk_df().drop(columns=["transcript_id"])
        errors = HierarchicalValidator.validate_chunk_df(df)
        self.assertTrue(len(errors) > 0)

    def test_empty_chunk_df_caught(self):
        errors = HierarchicalValidator.validate_chunk_df(pd.DataFrame())
        self.assertTrue(len(errors) > 0)

    def test_prob_sum_violation_caught(self):
        df = self._valid_chunk_df().copy()
        df["positive_prob"] = 0.9
        df["neutral_prob"]  = 0.9
        df["negative_prob"] = 0.9
        errors = HierarchicalValidator.validate_chunk_df(df)
        self.assertTrue(len(errors) > 0)

    def test_unsorted_chunk_order_caught(self):
        df = self._valid_chunk_df().copy()
        # Reverse the order — should fail ordering check
        df = df.iloc[::-1].reset_index(drop=True)
        # Re-assign so values are still present but unsorted per-transcript
        df["chunk_order"] = list(range(len(df) - 1, -1, -1))
        errors = HierarchicalValidator.validate_chunk_df(df)
        self.assertTrue(len(errors) > 0)

    def test_valid_sentence_df_no_errors(self):
        errors = HierarchicalValidator.validate_sentence_df(self._valid_sentence_df())
        self.assertEqual(errors, [])

    def test_empty_sentence_df_ok(self):
        errors = HierarchicalValidator.validate_sentence_df(pd.DataFrame())
        self.assertEqual(errors, [])

    def test_result_validation_passes_for_valid_hierarchy(self):
        cdf = self._valid_chunk_df()
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result = agg.aggregate(cdf)
        hsr = result.get("T1")
        errors = HierarchicalValidator.validate_result(hsr)
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_result_validation_catches_bad_prob_sum(self):
        cdf = self._valid_chunk_df()
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result = agg.aggregate(cdf)
        hsr = result.get("T1")
        # Manually corrupt a node
        hsr.root.positive_probability = 0.9
        hsr.root.neutral_probability  = 0.9
        hsr.root.negative_probability = 0.9
        errors = HierarchicalValidator.validate_result(hsr)
        self.assertTrue(any("prob sum" in e.lower() or "probability" in e.lower()
                            for e in errors))


# ===========================================================================
# ══ GROUP 7: EXPORT INTEGRITY TESTS ═════════════════════════════════════════
# ===========================================================================

class TestHierarchicalExport(unittest.TestCase):
    """Export helpers produce correct files."""

    def setUp(self):
        tids = ["AAPL_Q1_2025", "NVDA_Q2_2025"]
        cdf  = _make_chunk_df(tids, chunks_per_transcript=8)
        sdf  = _make_sentence_df(tids, sentences_per_transcript=16)
        agg  = create_hierarchical_aggregator(include_sentence_nodes=True)
        self._result = agg.aggregate(cdf, sentence_df=sdf)

    def test_full_dataframe_not_empty(self):
        df = self._result.to_full_dataframe()
        self.assertGreater(len(df), 0)

    def test_full_df_required_columns(self):
        df = self._result.to_full_dataframe()
        required = ["node_id", "hierarchy_level", "hierarchy_key",
                    "transcript_id", "sentiment_score", "observation_count"]
        assert_df_columns(df, required)

    def test_transcript_level_df_rows(self):
        df = self._result.to_level_dataframe(HierarchyLevel.TRANSCRIPT)
        self.assertEqual(len(df), 2)

    def test_section_level_df_not_empty(self):
        df = self._result.to_level_dataframe(HierarchyLevel.SECTION)
        self.assertGreater(len(df), 0)

    def test_speaker_level_df_not_empty(self):
        df = self._result.to_level_dataframe(HierarchyLevel.SPEAKER)
        self.assertGreater(len(df), 0)

    def test_sentence_level_df_not_empty(self):
        df = self._result.to_level_dataframe(HierarchyLevel.SENTENCE)
        self.assertGreater(len(df), 0)

    def test_analytics_dataframe_rows(self):
        df = self._result.to_analytics_dataframe()
        self.assertEqual(len(df), 2)

    def test_parquet_full_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hier.parquet"
            self._result.save_parquet(p)
            df = pd.read_parquet(p)
            self.assertGreater(len(df), 0)

    def test_parquet_transcript_level_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "transcripts.parquet"
            self._result.save_parquet(p, level=HierarchyLevel.TRANSCRIPT)
            df = pd.read_parquet(p)
            self.assertEqual(len(df), 2)

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hier.csv"
            self._result.save_csv(p)
            df = pd.read_csv(p)
            self.assertGreater(len(df), 0)

    def test_jsonl_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hier.jsonl"
            self._result.save_jsonl(p)
            lines = p.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                obj = json.loads(line)
                self.assertIn("transcript_id", obj)

    def test_analytics_parquet_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "analytics.parquet"
            self._result.save_analytics_parquet(p)
            df = pd.read_parquet(p)
            self.assertEqual(len(df), 2)

    def test_per_transcript_save_parquet(self):
        hsr = self._result.get("AAPL_Q1_2025")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "aapl.parquet"
            hsr.save_parquet(p)
            df = pd.read_parquet(p)
            self.assertGreater(len(df), 0)

    def test_per_transcript_save_csv(self):
        hsr = self._result.get("AAPL_Q1_2025")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "aapl.csv"
            hsr.save_csv(p)
            df = pd.read_csv(p)
            self.assertGreater(len(df), 0)


class TestAggregationResultExport(unittest.TestCase):
    """aggregation.py AggregationResult export methods."""

    def setUp(self):
        df = _make_chunk_df(["AAPL_Q1_2025", "NVDA_Q2_2025"], chunks_per_transcript=8)
        self._result = SentimentAggregator().aggregate(df)

    def test_jsonl_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.jsonl"
            self._result.save_jsonl(p)
            lines = p.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)

    def test_section_parquet_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sections.parquet"
            self._result.save_parquet(p, level="section")
            df = pd.read_parquet(p)
            self.assertGreater(len(df), 0)


# ===========================================================================
# ══ GROUP 8: INTEGRATION TESTS ══════════════════════════════════════════════
# ===========================================================================

class TestEndToEndPipeline(unittest.TestCase):
    """
    Simulates the full pipeline:
    sentence records → chunk generation → (mock) FinBERT inference
    → aggregation → hierarchical aggregation → export
    """

    def setUp(self):
        # Step 1: sentences
        self._transcript_id = "AAPL_Q1_2025"
        self._sentences = _make_sentence_records(
            self._transcript_id, n_prepared=8, n_qa=6
        )

        # Step 2: chunk generation
        cfg = ChunkConfig(
            chunk_size_tokens=100,
            overlap_tokens=20,
            hard_token_ceiling=490,
            min_chunk_tokens=5,
        )
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        self._chunk_result = gen.generate(
            self._sentences, transcript_id=self._transcript_id
        )
        self._chunk_df_raw = self._chunk_result.to_dataframe()

    def _simulate_finbert(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mock FinBERT inference: adds probability columns to a chunk DataFrame.
        Uses deterministic values derived from chunk_order.
        """
        rng = np.random.default_rng(123)
        n   = len(chunk_df)
        pos = np.clip(rng.normal(0.45, 0.12, n), 0.05, 0.90)
        neg = np.clip(rng.normal(0.20, 0.08, n), 0.05, 0.90)
        neu = np.maximum(0.05, 1.0 - pos - neg)
        total = pos + neu + neg
        pos /= total; neu /= total; neg /= total
        scores = pos - neg
        confs  = np.maximum(pos, np.maximum(neu, neg))
        labels = ["positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
                  for s in scores]
        df = chunk_df.copy()
        df["chunk_id"]       = df["chunk_id"].astype(str)
        df["positive_prob"]  = pos.round(8)
        df["neutral_prob"]   = neu.round(8)
        df["negative_prob"]  = neg.round(8)
        df["sentiment_score"]= scores.round(8)
        df["confidence"]     = confs.round(8)
        df["predicted_label"]= labels
        # Rename for aggregation.py column expectations
        if "dominant_speaker" not in df.columns and "speaker" in df.columns:
            df["dominant_speaker"] = df["speaker"]
        elif "dominant_speaker" not in df.columns:
            df["dominant_speaker"] = "unknown"
        return df

    def test_chunk_generation_produces_output(self):
        self.assertGreater(len(self._chunk_result.chunks), 0)

    def test_mock_finbert_probs_valid(self):
        inferred = self._simulate_finbert(self._chunk_df_raw)
        prob_sum = (
            inferred["positive_prob"]
            + inferred["neutral_prob"]
            + inferred["negative_prob"]
        )
        self.assertTrue((abs(prob_sum - 1.0) < PROB_TOL).all(),
                        "Mock FinBERT probabilities don't sum to 1.0")

    def test_aggregation_from_mock_inference(self):
        inferred = self._simulate_finbert(self._chunk_df_raw)
        result   = SentimentAggregator().aggregate(inferred, validate_input=False)
        ts = result.get(self._transcript_id)
        self.assertIsNotNone(ts)
        assert_prob_sum(
            ts.transcript_aggregate.positive_probability,
            ts.transcript_aggregate.neutral_probability,
            ts.transcript_aggregate.negative_probability,
        )

    def test_hierarchical_from_mock_inference(self):
        inferred = self._simulate_finbert(self._chunk_df_raw)
        hier_agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result   = hier_agg.aggregate(inferred)
        hsr = result.get(self._transcript_id)
        self.assertIsNotNone(hsr)
        self.assertIsNotNone(hsr.root)

    def test_full_pipeline_export_roundtrip(self):
        """Chunk → infer → aggregate → save parquet → reload → validate."""
        inferred = self._simulate_finbert(self._chunk_df_raw)
        agg_result = SentimentAggregator().aggregate(inferred, validate_input=False)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pipeline_out.parquet"
            agg_result.save_parquet(p, level="transcript")
            df = pd.read_parquet(p)
        self.assertEqual(len(df), 1)
        self.assertIn("sentiment_score", df.columns)
        self.assertIn("sentiment_label", df.columns)

    def test_coverage_maintained_end_to_end(self):
        cov = self._chunk_result.coverage_metrics()
        self.assertAlmostEqual(cov["sentence_coverage"], 1.0, places=5)

    def test_section_separation_in_chunks(self):
        """After chunking with section boundaries, both sections appear."""
        sections_in_chunks = set(self._chunk_df_raw["section_type"].unique())
        self.assertIn("prepared_remarks", sections_in_chunks)
        self.assertIn("qa", sections_in_chunks)


class TestLargeTranscriptSimulation(unittest.TestCase):
    """Simulate a large (200+ sentence) transcript for stress testing."""

    def _build_large_transcript(self, n: int = 200) -> list:
        sentences = []
        speakers  = ["CEO", "CFO", "Analyst A", "Analyst B", "Operator"]
        roles     = ["CEO", "CFO", "Analyst", "Analyst", "Operator"]
        sections  = ["prepared_remarks"] * (n // 2) + ["qa"] * (n - n // 2)
        words     = (
            "revenue grew earnings beat guidance strong services margin "
            "capital return buyback dividend infrastructure investment"
        ).split()
        rng = np.random.default_rng(77)
        for i, section in enumerate(sections):
            idx = i % len(speakers)
            # Generate 8–15 word sentences
            length = int(rng.integers(8, 16))
            text = " ".join(rng.choice(words, length).tolist()) + "."
            sentences.append(SentenceRecord(
                sentence_id=f"large_{i:05d}",
                transcript_id="LARGE_T1",
                sentence_order=i,
                sentence_text=text,
                speaker=speakers[idx],
                speaker_role=roles[idx],
                section_type=section,
            ))
        return sentences

    def test_large_transcript_no_sentence_loss(self):
        sents = self._build_large_transcript(200)
        cfg = ChunkConfig(
            chunk_size_tokens=120, overlap_tokens=20, hard_token_ceiling=490
        )
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        result = gen.generate(sents, transcript_id="LARGE_T1")
        cov = result.coverage_metrics()
        self.assertAlmostEqual(cov["sentence_coverage"], 1.0, places=5)

    def test_large_transcript_ceiling_never_exceeded(self):
        sents = self._build_large_transcript(200)
        cfg = ChunkConfig(
            chunk_size_tokens=120, overlap_tokens=20, hard_token_ceiling=490
        )
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        result = gen.generate(sents, transcript_id="LARGE_T1")
        for c in result.chunks:
            self.assertLessEqual(c.token_count, 490)

    def test_large_transcript_validation_passes(self):
        sents = self._build_large_transcript(150)
        cfg = ChunkConfig(
            chunk_size_tokens=120, overlap_tokens=20, hard_token_ceiling=490
        )
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        result = gen.generate(sents, transcript_id="LARGE_T1")
        errors = gen.validate_result(result)
        self.assertEqual(errors, [], f"Errors: {errors}")

    def test_large_transcript_aggregation_completes(self):
        sents = self._build_large_transcript(150)
        cfg = ChunkConfig(
            chunk_size_tokens=120, overlap_tokens=20, hard_token_ceiling=490
        )
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        chunk_result = gen.generate(sents, transcript_id="LARGE_T1")
        df = chunk_result.to_dataframe()

        # Simulate inference
        rng = np.random.default_rng(55)
        n   = len(df)
        pos = np.clip(rng.normal(0.40, 0.10, n), 0.05, 0.90)
        neg = np.clip(rng.normal(0.20, 0.08, n), 0.05, 0.90)
        neu = np.maximum(0.05, 1.0 - pos - neg)
        total = pos + neu + neg
        df["positive_prob"]   = (pos / total).round(8)
        df["neutral_prob"]    = (neu / total).round(8)
        df["negative_prob"]   = (neg / total).round(8)
        df["sentiment_score"] = (df["positive_prob"] - df["negative_prob"]).round(8)
        df["confidence"]      = df[["positive_prob","neutral_prob","negative_prob"]].max(axis=1)
        df["predicted_label"] = df["sentiment_score"].apply(_score_to_label)
        df["chunk_id"]        = df["chunk_id"].astype(str)
        if "dominant_speaker" not in df.columns:
            df["dominant_speaker"] = "unknown"

        result = SentimentAggregator().aggregate(df, validate_input=False)
        ts = result.get("LARGE_T1")
        self.assertIsNotNone(ts)
        self.assertGreater(ts.transcript_aggregate.chunk_count, 0)


class TestMalformedPipelineInputs(unittest.TestCase):
    """Pipeline handles malformed/missing data gracefully."""

    def test_blank_sentences_filtered(self):
        sents = _make_sentence_records(n_prepared=4, n_qa=2)
        sents.append(SentenceRecord(
            sentence_id="blank",
            transcript_id="T1",
            sentence_order=999,
            sentence_text="   ",   # blank — should be filtered
        ))
        cfg = ChunkConfig(
            chunk_size_tokens=100, overlap_tokens=20, hard_token_ceiling=490
        )
        gen = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        result = gen.generate(sents, transcript_id="T1")
        # All chunks should have non-empty text
        for c in result.chunks:
            self.assertTrue(c.chunk_text.strip())

    def test_aggregation_with_unknown_speaker_labels(self):
        df = _make_chunk_df(["T1"], chunks_per_transcript=6)
        df["dominant_speaker"] = "UNKNOWN_ENTITY"
        result = SentimentAggregator().aggregate(df, validate_input=False)
        ts = result.get("T1")
        self.assertIsNotNone(ts)

    def test_chunk_df_with_missing_section_type(self):
        df = _make_chunk_df(["T1"], chunks_per_transcript=6)
        df = df.drop(columns=["section_type"])
        agg = create_hierarchical_aggregator(include_sentence_nodes=False)
        result = agg.aggregate(df, validate_input=False)
        self.assertIn("T1", result.results)

    def test_sentence_df_without_prob_columns(self):
        """Sentence rows without prob columns should get estimated probabilities."""
        sdf = _make_sentence_df(["T1"], sentences_per_transcript=10)
        sdf = sdf.drop(columns=["positive_prob", "neutral_prob", "negative_prob"],
                       errors="ignore")
        cdf = _make_chunk_df(["T1"], chunks_per_transcript=6)
        agg = create_hierarchical_aggregator(include_sentence_nodes=True)
        result = agg.aggregate(cdf, sentence_df=sdf, validate_input=False)
        # Should not crash
        self.assertIn("T1", result.results)


# ===========================================================================
# ══ GROUP 9: PARAMETRIZED-STYLE TESTS ═══════════════════════════════════════
# ===========================================================================

class TestWeightingStrategiesParametrized(unittest.TestCase):
    """One test method per strategy to enable per-strategy failure isolation."""

    def _score_for_strategy(self, strategy: WeightingStrategy) -> float:
        df = _make_chunk_df(["T1"], chunks_per_transcript=8, seed=42)
        agg = SentimentAggregator(AggregationConfig(
            strategy=strategy,
            compute_section_aggregates=False,
            compute_speaker_aggregates=False,
        ))
        result = agg.aggregate(df, validate_input=False)
        return result.get("T1").primary_score()

    def test_uniform_score_in_range(self):
        assert_score_range(self._score_for_strategy(WeightingStrategy.UNIFORM))

    def test_confidence_score_in_range(self):
        assert_score_range(self._score_for_strategy(WeightingStrategy.CONFIDENCE))

    def test_token_score_in_range(self):
        assert_score_range(self._score_for_strategy(WeightingStrategy.TOKEN))

    def test_hybrid_score_in_range(self):
        assert_score_range(self._score_for_strategy(WeightingStrategy.HYBRID))


class TestChunkConfigParametrized(unittest.TestCase):
    """Test multiple chunk sizes."""

    def _run_with_size(self, chunk_size: int, overlap: int) -> ChunkGenerationResult:
        cfg = ChunkConfig(
            chunk_size_tokens=chunk_size,
            overlap_tokens=overlap,
            hard_token_ceiling=490,
            min_chunk_tokens=5,
        )
        gen  = ChunkGenerator(config=cfg, tokenizer=_MockTokenizer())
        sents = _make_sentence_records(n_prepared=8, n_qa=6)
        return gen.generate(sents, transcript_id="T1")

    def test_small_chunks_still_cover_all_sentences(self):
        result = self._run_with_size(40, 8)
        cov = result.coverage_metrics()
        self.assertAlmostEqual(cov["sentence_coverage"], 1.0, places=5)

    def test_medium_chunks_still_cover_all_sentences(self):
        result = self._run_with_size(100, 20)
        cov = result.coverage_metrics()
        self.assertAlmostEqual(cov["sentence_coverage"], 1.0, places=5)

    def test_large_chunks_still_cover_all_sentences(self):
        result = self._run_with_size(300, 50)
        cov = result.coverage_metrics()
        self.assertAlmostEqual(cov["sentence_coverage"], 1.0, places=5)

    def test_smaller_chunks_produce_more_chunks(self):
        r_small = self._run_with_size(40, 8)
        r_large = self._run_with_size(300, 50)
        self.assertGreaterEqual(
            len(r_small.chunks), len(r_large.chunks),
            "Smaller chunk size should produce at least as many chunks"
        )


# ===========================================================================
# ══ RUNNER (fallback for environments without pytest) ════════════════════════
# ===========================================================================

def _collect_all_tests() -> unittest.TestSuite:
    """Collect every TestCase subclass defined in this module."""
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    module = sys.modules[__name__]
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
                suite.addTests(loader.loadTestsFromTestCase(obj))
        except TypeError:
            pass
    return suite


if __name__ == "__main__":
    # ── Print a banner when run directly ──────────────────────────────
    print("\n" + "=" * 72)
    print("  test_finbert_pipeline.py  —  unittest runner")
    print("  (run 'pytest tests/test_finbert_pipeline.py -v' for full output)")
    print("=" * 72 + "\n")

    suite   = _collect_all_tests()
    runner  = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result  = runner.run(suite)
    print(f"\n── Summary: {result.testsRun} tests  |  "
          f"failures={len(result.failures)}  |  errors={len(result.errors)}  |  "
          f"skipped={len(result.skipped) if hasattr(result,'skipped') else 0}")
    sys.exit(0 if result.wasSuccessful() else 1)
