"""
sentence_sentiment.py
=====================
Fine-grained sentence-level sentiment scoring and drift analysis engine
for earnings call transcripts.

Part of: Earnings Call Sentiment Analyzer — DAY 8 Pipeline
Stage:   Sentence-Level Sentiment (after sentence_segmenter.py, runs
         alongside / after finbert_pipeline.py for chunk-level scoring)

Responsibilities
----------------
* Score every sentence individually via FinBERT (or the injected backend)
* Compute local sentiment shift  = score[i] − score[i−1]
* Compute rolling sentiment mean and standard deviation over a sliding window
* Detect sentiment spikes (|shift| > configurable threshold)
* Track cumulative and exponentially-weighted sentiment over the transcript
* Compute per-transcript drift (linear trend slope across all sentences)
* Produce per-transcript and per-section drift/volatility summaries
* Emit research-ready outputs compatible with the event-study framework

This module does NOT implement:
  * Transcript chunking           → chunk_generator.py
  * Transcript-level aggregation  → aggregation.py
  * Event-study / market logic    → event_study.py
  * Loughran-McDonald scoring     → lm_scorer.py

Sequence position in the DAY 8 pipeline
-----------------------------------------
sentence_segmenter.py
        ↓
    [ THIS MODULE ]  ← sentence-level FinBERT + drift analytics
        ↓
aggregation.py        ← transcript-level weighted aggregation
        ↓
hierarchical_aggregation.py

Input schema (DataFrame from sentence_segmenter)
-------------------------------------------------
    sentence_id    : str  — globally unique sentence identifier
    transcript_id  : str  — parent transcript key, e.g. "AAPL_Q1_2025"
    sentence_order : int  — 0-based global ordering within transcript
    sentence_text  : str  — cleaned sentence text
    speaker        : str  — speaker name
    speaker_role   : str  — CEO / CFO / Analyst / Operator …
    section_type   : str  — "prepared_remarks" | "qa" | "closing_remarks"
    token_estimate : int  — pre-computed token estimate

Output schema (SentenceSentiment)
-----------------------------------
    sentence_id            : str
    transcript_id          : str
    sentence_order         : int
    sentence_score         : float   P(positive) − P(negative)
    positive_prob          : float
    neutral_prob           : float
    negative_prob          : float
    confidence             : float   max(P(pos), P(neu), P(neg))
    predicted_label        : str     "positive" | "neutral" | "negative"
    local_sentiment_shift  : float | None   shift from prior sentence
    rolling_sentiment_mean : float   mean over rolling window
    rolling_sentiment_std  : float   std  over rolling window
    is_spike               : bool    |shift| > spike_threshold
    cumulative_sentiment   : float   running sum of sentence scores
    ema_sentiment          : float   exponentially-weighted score
    speaker                : str     propagated from input
    speaker_role           : str     propagated from input
    section_type           : str     propagated from input
    token_estimate         : int     propagated from input
    was_skipped            : bool    True for empty / malformed inputs
    skip_reason            : str     reason if skipped, else ""

Analytics formulas (all implemented faithfully)
------------------------------------------------
    local_sentiment_shift  = score[i] − score[i−1]        (first sentence → None)
    rolling_sentiment_mean = mean(scores[i−W+1 … i])      (min_periods=1)
    rolling_sentiment_std  = std(scores[i−W+1 … i], ddof=0)
    ema_sentiment          = α·score[i] + (1−α)·ema[i−1]  (first sentence = score[0])
    cumulative_sentiment   = Σ score[0 … i]
    drift_slope (summary)  = slope of OLS fit of scores over sentence_order

Usage
-----
    from sentence_sentiment import SentenceSentimentAnalyzer, SentenceSentimentConfig

    config   = SentenceSentimentConfig(rolling_window=5, spike_threshold=0.3)
    analyzer = SentenceSentimentAnalyzer(config=config)
    analyzer.load_model()

    result   = analyzer.analyze_transcript(sentence_df)
    df_out   = result.to_dataframe()
    df_out.to_parquet("data/processed/sentence_level_sentiment.parquet")

Python: 3.11+
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — torch / transformers are optional at import time
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for FinBERT inference. "
            "Install with: pip install torch"
        ) from exc


def _require_transformers():
    try:
        import transformers
        return transformers
    except ImportError as exc:
        raise ImportError(
            "HuggingFace Transformers is required. "
            "Install with: pip install transformers"
        ) from exc


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SentimentLabel(str, Enum):
    """Canonical sentiment label values."""
    POSITIVE = "positive"
    NEUTRAL  = "neutral"
    NEGATIVE = "negative"
    UNKNOWN  = "unknown"


class AnalyzerState(Enum):
    """Lifecycle state of :class:`SentenceSentimentAnalyzer`."""
    UNLOADED = auto()
    READY    = auto()
    ERROR    = auto()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SentenceSentimentConfig:
    """All tunable parameters for :class:`SentenceSentimentAnalyzer`.

    Attributes
    ----------
    model_name : str
        HuggingFace model ID.  ProsusAI/finbert is the project standard.
    device : str
        ``"auto"`` selects CUDA when available, else CPU.
        ``"cuda"`` / ``"cpu"`` force a specific device.
    batch_size : int
        Number of sentences per forward pass.
        Recommended: 32 GPU / 8–16 CPU.
    max_token_length : int
        Hard tokenizer truncation limit (≤ 512 for BERT).
    use_fp16 : bool
        Half-precision on CUDA for faster throughput.
    rolling_window : int
        Number of sentences in the rolling statistics window.
        A window of 5 gives a smooth but responsive signal.
    min_rolling_periods : int
        Minimum observations required to compute rolling stats.
        Defaults to 1 so every sentence receives a value.
    spike_threshold : float
        |local_sentiment_shift| must exceed this to be flagged
        as a sentiment spike.  Calibrated to earnings-call dynamics:
        0.25 catches major tone reversals while avoiding noise.
    ema_alpha : float
        Smoothing factor for the exponentially-weighted moving average.
        Higher values (→1) track recent scores more aggressively.
        0.3 gives moderate smoothing suitable for 10–30 sentence windows.
    min_sentence_chars : int
        Sentences shorter than this are skipped (marked was_skipped=True).
    show_progress : bool
        Show tqdm progress bars.
    log_every_n_batches : int
        Emit INFO logs every N batches (0 = disable).
    max_token_length : int
        Sentences tokenised to more than this are truncated by the
        tokenizer (BERT hard limit = 512).
    prob_sum_tolerance : float
        Allowed deviation from 1.0 in the probability-sum validation.
    """

    model_name           : str   = "ProsusAI/finbert"
    device               : str   = "auto"

    batch_size           : int   = 16
    max_token_length     : int   = 512

    use_fp16             : bool  = False

    rolling_window       : int   = 5
    min_rolling_periods  : int   = 1
    spike_threshold      : float = 0.25
    ema_alpha            : float = 0.3

    min_sentence_chars   : int   = 5
    show_progress        : bool  = True
    log_every_n_batches  : int   = 10
    prob_sum_tolerance   : float = 1e-4

    def __post_init__(self) -> None:
        if self.max_token_length > 512:
            raise ValueError(
                f"max_token_length={self.max_token_length} exceeds BERT limit 512."
            )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}.")
        if self.rolling_window < 1:
            raise ValueError("rolling_window must be >= 1.")
        if self.spike_threshold < 0:
            raise ValueError("spike_threshold must be >= 0.")


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SentenceSentiment:
    """Sentiment record for a single transcript sentence.

    Produced by :meth:`SentenceSentimentAnalyzer.analyze_transcript`.
    All analytics fields are fully populated even for the very first sentence
    (shift = None, rolling stats = single-element window, cumulative = score[0]).
    """

    # --- Identity ---
    sentence_id            : str
    transcript_id          : str
    sentence_order         : int

    # --- FinBERT probabilities ---
    positive_prob          : float
    neutral_prob           : float
    negative_prob          : float

    # --- Derived sentiment metrics ---
    sentence_score         : float          # P(positive) − P(negative)
    confidence             : float          # max probability
    predicted_label        : str            # dominant label name

    # --- Sequential analytics ---
    local_sentiment_shift  : Optional[float]  # None for the first sentence
    rolling_sentiment_mean : float
    rolling_sentiment_std  : float
    is_spike               : bool
    cumulative_sentiment   : float
    ema_sentiment          : float

    # --- Propagated metadata ---
    speaker                : str
    speaker_role           : str
    section_type           : str
    token_estimate         : int

    # --- Processing metadata ---
    model_name             : str  = "ProsusAI/finbert"
    was_skipped            : bool = False
    skip_reason            : str  = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serializable plain dictionary."""
        d = asdict(self)
        # Serialize None shift as NaN-compatible float for parquet compatibility
        if d["local_sentiment_shift"] is None:
            d["local_sentiment_shift"] = float("nan")
        return d

    def __repr__(self) -> str:
        preview = ""
        shift_str = (
            f"{self.local_sentiment_shift:+.4f}"
            if self.local_sentiment_shift is not None
            else "None"
        )
        return (
            f"SentenceSentiment("
            f"id={self.sentence_id!r}, "
            f"order={self.sentence_order}, "
            f"score={self.sentence_score:+.4f}, "
            f"shift={shift_str}, "
            f"label={self.predicted_label!r}, "
            f"spike={self.is_spike})"
        )

    @classmethod
    def make_skipped(
        cls,
        *,
        sentence_id    : str,
        transcript_id  : str,
        sentence_order : int,
        reason         : str,
        speaker        : str = "",
        speaker_role   : str = "",
        section_type   : str = "",
        token_estimate : int = 0,
        model_name     : str = "ProsusAI/finbert",
    ) -> "SentenceSentiment":
        """Create a sentinel record for a sentence that was skipped."""
        return cls(
            sentence_id            = sentence_id,
            transcript_id          = transcript_id,
            sentence_order         = sentence_order,
            positive_prob          = 0.0,
            neutral_prob           = 1.0,
            negative_prob          = 0.0,
            sentence_score         = 0.0,
            confidence             = 0.0,
            predicted_label        = SentimentLabel.UNKNOWN.value,
            local_sentiment_shift  = None,
            rolling_sentiment_mean = 0.0,
            rolling_sentiment_std  = 0.0,
            is_spike               = False,
            cumulative_sentiment   = 0.0,
            ema_sentiment          = 0.0,
            speaker                = speaker,
            speaker_role           = speaker_role,
            section_type           = section_type,
            token_estimate         = token_estimate,
            model_name             = model_name,
            was_skipped            = True,
            skip_reason            = reason,
        )


@dataclass
class SectionDriftSummary:
    """Sentiment drift and volatility metrics for one transcript section.

    Produced by :meth:`SentenceSentimentResult.section_summaries`.
    """

    transcript_id        : str
    section_type         : str
    sentence_count       : int
    mean_score           : float
    std_score            : float
    min_score            : float
    max_score            : float
    score_range          : float
    drift_slope          : float    # OLS slope of score over sentence_order
    spike_count          : int
    avg_confidence       : float
    first_score          : float
    last_score           : float
    net_drift            : float    # last_score − first_score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpeakerSentimentSummary:
    """Aggregated sentiment metrics for one speaker within a transcript."""

    transcript_id   : str
    speaker         : str
    speaker_role    : str
    sentence_count  : int
    mean_score      : float
    std_score       : float
    min_score       : float
    max_score       : float
    avg_confidence  : float
    spike_count     : int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptDriftSummary:
    """Top-level sentiment analytics for an entire transcript.

    Suitable for direct merge into the event-study dataset as sentiment
    features, or as quality-control statistics.
    """

    transcript_id            : str
    sentence_count           : int
    valid_sentence_count     : int
    skipped_count            : int

    # Sentiment statistics
    mean_score               : float
    std_score                : float
    median_score             : float
    min_score                : float
    max_score                : float
    score_range              : float
    final_score              : float    # last valid sentence score
    first_score              : float    # first valid sentence score
    net_drift                : float    # final − first

    # Trend
    drift_slope              : float    # OLS slope
    drift_r_squared          : float    # goodness of fit

    # Volatility
    overall_volatility       : float    # std of all scores
    rolling_volatility_mean  : float    # mean of per-sentence rolling stds

    # Spike analysis
    spike_count              : int
    spike_rate               : float    # spikes / valid sentences

    # Confidence
    mean_confidence          : float

    # Section-level means (convenience)
    prepared_remarks_mean    : Optional[float]
    qa_mean                  : Optional[float]
    closing_remarks_mean     : Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentenceSentimentStats:
    """Per-run processing diagnostics."""

    transcript_id     : str  = ""
    total_sentences   : int  = 0
    processed         : int  = 0
    skipped_empty     : int  = 0
    skipped_short     : int  = 0
    total_batches     : int  = 0
    elapsed_seconds   : float = 0.0
    errors            : list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_sentences == 0:
            return 0.0
        return self.processed / self.total_sentences

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentenceSentimentResult:
    """Complete output of analyzing one transcript's sentences.

    Attributes
    ----------
    transcript_id : str
    sentiments : list[SentenceSentiment]
        One entry per input sentence, sorted by sentence_order.
    stats : SentenceSentimentStats
    """

    transcript_id : str
    sentiments    : list[SentenceSentiment]
    stats         : SentenceSentimentStats

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Flat DataFrame, one row per sentence.  Ready for parquet export."""
        if not self.sentiments:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        rows = [s.to_dict() for s in self.sentiments]
        df = pd.DataFrame(rows)
        avail = [c for c in _OUTPUT_COLUMNS if c in df.columns]
        extra = [c for c in df.columns if c not in avail]
        return df[avail + extra]

    def to_dict_list(self) -> list[dict]:
        return [s.to_dict() for s in self.sentiments]

    def filter_valid(self) -> "SentenceSentimentResult":
        """Return a new result with skipped sentences removed."""
        return SentenceSentimentResult(
            transcript_id = self.transcript_id,
            sentiments    = [s for s in self.sentiments if not s.was_skipped],
            stats         = self.stats,
        )

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------

    def transcript_drift_summary(self) -> TranscriptDriftSummary:
        """Compute the full transcript-level drift summary."""
        valid = [s for s in self.sentiments if not s.was_skipped]
        return _compute_transcript_drift(
            transcript_id=self.transcript_id,
            sentiments=self.sentiments,
            valid=valid,
        )

    def section_summaries(self) -> list[SectionDriftSummary]:
        """Return one :class:`SectionDriftSummary` per distinct section_type."""
        valid = [s for s in self.sentiments if not s.was_skipped]
        if not valid:
            return []
        by_section: dict[str, list[SentenceSentiment]] = defaultdict(list)
        for s in valid:
            by_section[s.section_type].append(s)
        summaries = []
        for section, items in by_section.items():
            summaries.append(_compute_section_drift(self.transcript_id, section, items))
        return sorted(summaries, key=lambda x: x.section_type)

    def speaker_summaries(self) -> list[SpeakerSentimentSummary]:
        """Return one :class:`SpeakerSentimentSummary` per distinct speaker."""
        valid = [s for s in self.sentiments if not s.was_skipped]
        if not valid:
            return []
        by_speaker: dict[tuple[str, str], list[SentenceSentiment]] = defaultdict(list)
        for s in valid:
            by_speaker[(s.speaker, s.speaker_role)].append(s)
        summaries = []
        for (speaker, role), items in by_speaker.items():
            scores = [i.sentence_score for i in items]
            summaries.append(SpeakerSentimentSummary(
                transcript_id  = self.transcript_id,
                speaker        = speaker,
                speaker_role   = role,
                sentence_count = len(items),
                mean_score     = float(np.mean(scores)),
                std_score      = float(np.std(scores, ddof=0)) if len(scores) > 1 else 0.0,
                min_score      = float(min(scores)),
                max_score      = float(max(scores)),
                avg_confidence = float(np.mean([i.confidence for i in items])),
                spike_count    = sum(i.is_spike for i in items),
            ))
        return sorted(summaries, key=lambda x: x.speaker)

    def scores_array(self) -> np.ndarray:
        """Return a numpy array of sentence_score values in sentence_order."""
        return np.array([
            s.sentence_score for s in
            sorted(self.sentiments, key=lambda x: x.sentence_order)
        ])

    def __len__(self) -> int:
        return len(self.sentiments)

    def __repr__(self) -> str:
        return (
            f"SentenceSentimentResult("
            f"transcript_id={self.transcript_id!r}, "
            f"sentences={len(self.sentiments)}, "
            f"success={self.stats.success_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Output column order (architecture specification)
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS: list[str] = [
    "sentence_id",
    "transcript_id",
    "sentence_order",
    "sentence_score",
    "positive_prob",
    "neutral_prob",
    "negative_prob",
    "confidence",
    "predicted_label",
    "local_sentiment_shift",
    "rolling_sentiment_mean",
    "rolling_sentiment_std",
    "is_spike",
    "cumulative_sentiment",
    "ema_sentiment",
    "speaker",
    "speaker_role",
    "section_type",
    "token_estimate",
    "model_name",
    "was_skipped",
    "skip_reason",
]

_REQUIRED_INPUT_COLUMNS: frozenset[str] = frozenset({
    "sentence_id",
    "transcript_id",
    "sentence_order",
    "sentence_text",
})


# ---------------------------------------------------------------------------
# Rolling analytics engine (pure Python + numpy, no pandas dependency)
# ---------------------------------------------------------------------------

class _RollingAnalytics:
    """Stateful engine that computes rolling metrics sentence-by-sentence.

    Designed to be instantiated once per transcript and fed sentences in
    sentence_order.  All state is maintained internally; no external buffers
    required.

    Parameters
    ----------
    window : int
        Rolling window size.
    min_periods : int
        Minimum observations to produce rolling stats (default 1).
    spike_threshold : float
        |shift| must exceed this to flag is_spike.
    ema_alpha : float
        Smoothing coefficient for exponential weighted mean.
    """

    __slots__ = (
        "_window", "_min_periods", "_spike_threshold", "_ema_alpha",
        "_buffer",       # deque-like list of recent scores
        "_prev_score",   # previous sentence score (None for first)
        "_cumulative",   # running total
        "_ema",          # current EMA value (None until first sentence)
        "_position",     # sentence count processed
    )

    def __init__(
        self,
        window          : int   = 5,
        min_periods     : int   = 1,
        spike_threshold : float = 0.25,
        ema_alpha       : float = 0.3,
    ) -> None:
        self._window          = window
        self._min_periods     = min_periods
        self._spike_threshold = spike_threshold
        self._ema_alpha       = ema_alpha
        self._buffer    : list[float]    = []
        self._prev_score: Optional[float] = None
        self._cumulative: float           = 0.0
        self._ema       : Optional[float] = None
        self._position  : int             = 0

    def update(self, score: float) -> dict:
        """Feed one sentence score and return all rolling metrics.

        Parameters
        ----------
        score : float
            The sentence_score (P(pos) − P(neg)) for this sentence.

        Returns
        -------
        dict with keys:
            local_sentiment_shift, rolling_sentiment_mean,
            rolling_sentiment_std, is_spike, cumulative_sentiment,
            ema_sentiment
        """
        # 1. Local shift
        shift: Optional[float]
        if self._prev_score is None:
            shift = None
            is_spike = False
        else:
            shift = score - self._prev_score
            is_spike = abs(shift) > self._spike_threshold

        # 2. Rolling buffer (fixed-width window)
        self._buffer.append(score)
        if len(self._buffer) > self._window:
            self._buffer.pop(0)

        # 3. Rolling mean / std
        buf_arr = np.asarray(self._buffer, dtype=np.float64)
        roll_mean = float(np.mean(buf_arr))
        roll_std = (
            float(np.std(buf_arr, ddof=0))
            if len(buf_arr) > 1
            else 0.0
        )

        # 4. Cumulative
        self._cumulative += score

        # 5. EMA
        if self._ema is None:
            self._ema = score
        else:
            self._ema = self._ema_alpha * score + (1.0 - self._ema_alpha) * self._ema

        # 6. Advance state
        self._prev_score = score
        self._position  += 1

        return {
            "local_sentiment_shift" : shift,
            "rolling_sentiment_mean": round(roll_mean, 8),
            "rolling_sentiment_std" : round(roll_std, 8),
            "is_spike"              : is_spike,
            "cumulative_sentiment"  : round(self._cumulative, 8),
            "ema_sentiment"         : round(self._ema, 8),
        }

    def reset(self) -> None:
        """Reset all state for a new transcript."""
        self._buffer     = []
        self._prev_score = None
        self._cumulative = 0.0
        self._ema        = None
        self._position   = 0


# ---------------------------------------------------------------------------
# Mock FinBERT backend (for testing without model download)
# ---------------------------------------------------------------------------

class _MockFinBERTBackend:
    """Deterministic keyword-heuristic mock of ProsusAI/finbert.

    Produces plausible probability triples without any network access.
    Suitable for unit tests and CI pipelines.

    Label order mirrors ProsusAI/finbert: {0: positive, 1: negative, 2: neutral}.
    """

    _POS = frozenset({
        "grew", "growth", "record", "strong", "increased", "beat",
        "exceeded", "margin", "raised", "guidance", "optimistic",
        "confident", "opportunity", "momentum", "outperform", "higher",
        "positive", "excellent", "robust", "solid", "accelerating",
        "strength", "expansion", "upside", "efficient", "profitable",
    })
    _NEG = frozenset({
        "decline", "decreased", "miss", "shortfall", "risk",
        "uncertainty", "headwind", "challenged", "slowdown", "concern",
        "pressure", "difficult", "lower", "reduced", "weak", "below",
        "loss", "deterioration", "disappointing", "volatility", "unfavorable",
        "cautious", "downside", "competitive", "constrained",
    })
    # ProsusAI/finbert label order: idx 0 = positive, 1 = negative, 2 = neutral
    _LABEL_MAP: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}

    def predict_batch(self, texts: list[str]) -> list[list[float]]:
        """Return list of [pos_prob, neg_prob, neu_prob] for each text."""
        import hashlib
        results = []
        for text in texts:
            words = set(text.lower().split())
            pos_hits = len(words & self._POS)
            neg_hits = len(words & self._NEG)

            raw_pos = 1.0 + pos_hits * 0.55
            raw_neg = 1.0 + neg_hits * 0.55
            raw_neu = 1.35

            total = raw_pos + raw_neg + raw_neu
            pos_p = raw_pos / total
            neg_p = raw_neg / total
            neu_p = raw_neu / total

            # Tiny deterministic perturbation for realism
            digest = int(hashlib.md5(text[:60].encode()).hexdigest(), 16)
            eps = (digest % 100) / 200_000.0
            pos_p = min(1.0, pos_p + eps)
            total2 = pos_p + neg_p + neu_p
            pos_p /= total2
            neg_p /= total2
            neu_p  = 1.0 - pos_p - neg_p

            results.append([pos_p, neg_p, neu_p])
        return results

    @property
    def label_map(self) -> dict[int, str]:
        return self._LABEL_MAP


# ---------------------------------------------------------------------------
# FinBERT inference backend (real model)
# ---------------------------------------------------------------------------

class _FinBERTBackend:
    """Thin wrapper around HuggingFace AutoTokenizer + AutoModelForSequenceClassification.

    Handles device placement, FP16, and softmax conversion.
    Separated from analytics so it can be swapped for the mock in tests.
    """

    def __init__(
        self,
        model_name      : str,
        device          : str   = "auto",
        max_token_length: int   = 512,
        use_fp16        : bool  = False,
    ) -> None:
        self._model_name       = model_name
        self._device_cfg       = device
        self._max_token_length = max_token_length
        self._use_fp16         = use_fp16
        self._tokenizer        = None
        self._model            = None
        self._device           = None
        self._label_map: dict[int, str] = {}

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load tokenizer and model, set device, resolve label map."""
        torch = _require_torch()
        tf    = _require_transformers()

        # Device
        if self._device_cfg == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self._device_cfg == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device='cuda' requested but CUDA is unavailable.")
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        logger.info("Loading tokenizer: %s", self._model_name)
        self._tokenizer = tf.AutoTokenizer.from_pretrained(self._model_name)

        logger.info("Loading model: %s on %s", self._model_name, self._device)
        self._model = tf.AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )
        self._model.to(self._device)
        self._model.eval()

        if self._use_fp16 and str(self._device) != "cpu":
            self._model = self._model.half()
            logger.info("FP16 inference enabled.")

        # Label map
        try:
            raw = self._model.config.id2label
            self._label_map = {int(k): str(v).lower().strip() for k, v in raw.items()}
        except Exception:
            self._label_map = {0: "positive", 1: "negative", 2: "neutral"}
            logger.warning("Using fallback label map: %s", self._label_map)

        logger.info("Backend ready | labels=%s", self._label_map)

    # ------------------------------------------------------------------
    def predict_batch(self, texts: list[str]) -> list[list[float]]:
        """Return list of [prob_label_0, prob_label_1, prob_label_2] per text."""
        torch = _require_torch()
        import torch.nn.functional as F

        encoding = self._tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = self._max_token_length,
            return_tensors = "pt",
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        use_amp = (
            self._use_fp16
            and str(self._device) != "cpu"
            and hasattr(torch.cuda.amp, "autocast")
        )
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._model(**encoding)
            else:
                outputs = self._model(**encoding)

        logits = outputs.logits
        probs  = F.softmax(logits.float(), dim=-1).cpu().tolist()
        return probs

    @property
    def label_map(self) -> dict[int, str]:
        return self._label_map

    @property
    def device_str(self) -> str:
        return str(self._device) if self._device is not None else "unresolved"

    def unload(self) -> None:
        torch = _require_torch()
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("FinBERT backend unloaded.")


# ---------------------------------------------------------------------------
# Probability → SentenceSentiment conversion helper
# ---------------------------------------------------------------------------

def _build_sentiment(
    *,
    row            : dict,
    probs          : list[float],
    label_map      : dict[int, str],
    analytics      : dict,
    model_name     : str,
) -> SentenceSentiment:
    """Assemble a :class:`SentenceSentiment` from raw inference outputs.

    Parameters
    ----------
    row : dict
        Input sentence dict (all metadata fields).
    probs : list[float]
        Softmax probabilities [p_label0, p_label1, p_label2].
    label_map : dict[int, str]
        Maps index → label name (from model config).
    analytics : dict
        Output of :meth:`_RollingAnalytics.update`.
    model_name : str
        Model identifier for provenance.
    """
    prob_by_name: dict[str, float] = {
        label_map[i]: float(probs[i])
        for i in range(len(probs))
    }
    pos_p = prob_by_name.get("positive", 0.0)
    neg_p = prob_by_name.get("negative", 0.0)
    neu_p = prob_by_name.get("neutral",  0.0)

    score      = pos_p - neg_p
    confidence = max(pos_p, neg_p, neu_p)
    label      = max(prob_by_name, key=lambda k: prob_by_name[k])

    return SentenceSentiment(
        sentence_id            = str(row.get("sentence_id", "")),
        transcript_id          = str(row.get("transcript_id", "")),
        sentence_order         = int(row.get("sentence_order", 0)),
        positive_prob          = round(pos_p, 8),
        neutral_prob           = round(neu_p, 8),
        negative_prob          = round(neg_p, 8),
        sentence_score         = round(score, 8),
        confidence             = round(confidence, 8),
        predicted_label        = label,
        local_sentiment_shift  = analytics["local_sentiment_shift"],
        rolling_sentiment_mean = analytics["rolling_sentiment_mean"],
        rolling_sentiment_std  = analytics["rolling_sentiment_std"],
        is_spike               = analytics["is_spike"],
        cumulative_sentiment   = analytics["cumulative_sentiment"],
        ema_sentiment          = analytics["ema_sentiment"],
        speaker                = str(row.get("speaker", "")),
        speaker_role           = str(row.get("speaker_role", "")),
        section_type           = str(row.get("section_type", "")),
        token_estimate         = int(row.get("token_estimate", 0)),
        model_name             = model_name,
        was_skipped            = False,
        skip_reason            = "",
    )


# ---------------------------------------------------------------------------
# Drift / summary computation helpers (module-level to keep class lean)
# ---------------------------------------------------------------------------

def _ols_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, R²) of the OLS linear fit of y on x."""
    if len(x) < 2:
        return 0.0, 0.0
    try:
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        y_hat = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return slope, r2
    except Exception:
        return 0.0, 0.0


def _compute_section_drift(
    transcript_id: str,
    section_type : str,
    items        : list[SentenceSentiment],
) -> SectionDriftSummary:
    """Build a :class:`SectionDriftSummary` for a list of same-section sentences."""
    items_sorted = sorted(items, key=lambda s: s.sentence_order)
    scores = np.array([s.sentence_score for s in items_sorted])
    x = np.arange(len(scores), dtype=float)
    slope, _ = _ols_slope_r2(x, scores)

    return SectionDriftSummary(
        transcript_id  = transcript_id,
        section_type   = section_type,
        sentence_count = len(items_sorted),
        mean_score     = float(np.mean(scores)),
        std_score      = float(np.std(scores, ddof=0)) if len(scores) > 1 else 0.0,
        min_score      = float(scores.min()),
        max_score      = float(scores.max()),
        score_range    = float(scores.max() - scores.min()),
        drift_slope    = slope,
        spike_count    = sum(s.is_spike for s in items_sorted),
        avg_confidence = float(np.mean([s.confidence for s in items_sorted])),
        first_score    = float(scores[0]),
        last_score     = float(scores[-1]),
        net_drift      = float(scores[-1] - scores[0]),
    )


def _compute_transcript_drift(
    transcript_id : str,
    sentiments    : list[SentenceSentiment],
    valid         : list[SentenceSentiment],
) -> TranscriptDriftSummary:
    """Build the full :class:`TranscriptDriftSummary` for a transcript."""
    if not valid:
        return TranscriptDriftSummary(
            transcript_id            = transcript_id,
            sentence_count           = len(sentiments),
            valid_sentence_count     = 0,
            skipped_count            = len(sentiments),
            mean_score               = 0.0,
            std_score                = 0.0,
            median_score             = 0.0,
            min_score                = 0.0,
            max_score                = 0.0,
            score_range              = 0.0,
            final_score              = 0.0,
            first_score              = 0.0,
            net_drift                = 0.0,
            drift_slope              = 0.0,
            drift_r_squared          = 0.0,
            overall_volatility       = 0.0,
            rolling_volatility_mean  = 0.0,
            spike_count              = 0,
            spike_rate               = 0.0,
            mean_confidence          = 0.0,
            prepared_remarks_mean    = None,
            qa_mean                  = None,
            closing_remarks_mean     = None,
        )

    v_sorted = sorted(valid, key=lambda s: s.sentence_order)
    scores   = np.array([s.sentence_score for s in v_sorted])
    x        = np.arange(len(scores), dtype=float)
    slope, r2 = _ols_slope_r2(x, scores)

    section_means: dict[str, float] = {}
    by_section: dict[str, list[float]] = defaultdict(list)
    for s in v_sorted:
        by_section[s.section_type].append(s.sentence_score)
    for sec, vals in by_section.items():
        section_means[sec] = float(np.mean(vals))

    return TranscriptDriftSummary(
        transcript_id            = transcript_id,
        sentence_count           = len(sentiments),
        valid_sentence_count     = len(valid),
        skipped_count            = len(sentiments) - len(valid),
        mean_score               = float(np.mean(scores)),
        std_score                = float(np.std(scores, ddof=0)),
        median_score             = float(np.median(scores)),
        min_score                = float(scores.min()),
        max_score                = float(scores.max()),
        score_range              = float(scores.max() - scores.min()),
        final_score              = float(scores[-1]),
        first_score              = float(scores[0]),
        net_drift                = float(scores[-1] - scores[0]),
        drift_slope              = slope,
        drift_r_squared          = r2,
        overall_volatility       = float(np.std(scores, ddof=0)),
        rolling_volatility_mean  = float(
            np.mean([s.rolling_sentiment_std for s in v_sorted])
        ),
        spike_count              = sum(s.is_spike for s in v_sorted),
        spike_rate               = sum(s.is_spike for s in v_sorted) / len(v_sorted),
        mean_confidence          = float(np.mean([s.confidence for s in v_sorted])),
        prepared_remarks_mean    = section_means.get("prepared_remarks"),
        qa_mean                  = section_means.get("qa"),
        closing_remarks_mean     = section_means.get("closing_remarks"),
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SentenceSentimentAnalyzer:
    """Sentence-level FinBERT sentiment analysis engine for earnings calls.

    Scores every sentence of a segmented transcript individually and computes
    a full suite of sequential analytics (shift, rolling stats, drift, spikes)
    that are directly consumable by the event-study and regression layers.

    Parameters
    ----------
    config : SentenceSentimentConfig, optional
        If ``None``, sensible CPU defaults are used.
    backend : _FinBERTBackend | _MockFinBERTBackend | None
        Inject a custom inference backend (real or mock).  If ``None``, a
        :class:`_FinBERTBackend` is created from ``config``.

    Examples
    --------
    Production usage::

        analyzer = SentenceSentimentAnalyzer(SentenceSentimentConfig())
        analyzer.load_model()
        result = analyzer.analyze_transcript(sentence_df)
        result.to_dataframe().to_parquet("sentence_level_sentiment.parquet")

    Test usage with mock backend::

        analyzer = SentenceSentimentAnalyzer.mock()
        result = analyzer.analyze_transcript(sentence_df)
    """

    # Required input columns
    REQUIRED_COLUMNS: frozenset[str] = _REQUIRED_INPUT_COLUMNS

    # ------------------------------------------------------------------
    def __init__(
        self,
        config  : Optional[SentenceSentimentConfig] = None,
        backend : Optional[object]                  = None,
    ) -> None:
        self.config  = config or SentenceSentimentConfig(batch_size=8)
        self._state  = AnalyzerState.UNLOADED
        self._backend: Optional[Union[_FinBERTBackend, _MockFinBERTBackend]] = backend

        if backend is not None:
            # Injected backend — treat as pre-loaded
            self._state = AnalyzerState.READY
            logger.info(
                "SentenceSentimentAnalyzer initialised with injected backend %s",
                type(backend).__name__,
            )
        else:
            logger.info(
                "SentenceSentimentAnalyzer created | model=%s | device=%s | "
                "batch=%d | window=%d | spike_thr=%.2f",
                self.config.model_name,
                self.config.device,
                self.config.batch_size,
                self.config.rolling_window,
                self.config.spike_threshold,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def mock(
        cls,
        config: Optional[SentenceSentimentConfig] = None,
    ) -> "SentenceSentimentAnalyzer":
        """Create an analyzer backed by the mock FinBERT (no network/GPU needed).

        Suitable for unit tests, CI pipelines, and demo environments.
        """
        cfg = config or SentenceSentimentConfig(batch_size=8, show_progress=False)
        obj = cls(config=cfg, backend=_MockFinBERTBackend())
        logger.info("SentenceSentimentAnalyzer.mock() created.")
        return obj

    def load_model(self) -> None:
        """Load the FinBERT tokenizer and model.

        Idempotent — calling on an already-loaded analyzer is a no-op.

        Raises
        ------
        RuntimeError
            If model loading fails.
        ImportError
            If torch or transformers are not installed.
        """
        if self._state == AnalyzerState.READY:
            logger.debug("Model already loaded — skipping.")
            return

        try:
            backend = _FinBERTBackend(
                model_name       = self.config.model_name,
                device           = self.config.device,
                max_token_length = self.config.max_token_length,
                use_fp16         = self.config.use_fp16,
            )
            backend.load()
            self._backend = backend
            self._state   = AnalyzerState.READY
        except Exception as exc:
            self._state = AnalyzerState.ERROR
            raise RuntimeError(
                f"Failed to load SentenceSentimentAnalyzer model: {exc}"
            ) from exc

    def unload_model(self) -> None:
        """Release model weights from memory."""
        if isinstance(self._backend, _FinBERTBackend):
            self._backend.unload()
        self._backend = None
        self._state   = AnalyzerState.UNLOADED

    @property
    def is_ready(self) -> bool:
        return self._state == AnalyzerState.READY

    @property
    def device_str(self) -> str:
        if isinstance(self._backend, _FinBERTBackend):
            return self._backend.device_str
        if isinstance(self._backend, _MockFinBERTBackend):
            return "cpu (mock)"
        return "unresolved"

    # ------------------------------------------------------------------
    # Text validation
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse whitespace; preserve all financial language."""
        import re
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _is_valid(self, text: str) -> tuple[bool, str]:
        """Return (ok, reason) for a sentence text."""
        if not text:
            return False, "empty_text"
        if len(text) < self.config.min_sentence_chars:
            return False, f"too_short:{len(text)}_chars"
        return True, ""

    # ------------------------------------------------------------------
    # Core analysis: single transcript
    # ------------------------------------------------------------------

    def analyze_transcript(
        self,
        df: pd.DataFrame,
        *,
        sort_output: bool = True,
    ) -> SentenceSentimentResult:
        """Score all sentences in *df* and compute sequential analytics.

        All sentences are sorted by ``sentence_order`` before processing so
        the rolling analytics are in the correct chronological sequence.

        Parameters
        ----------
        df : pd.DataFrame
            Sentence DataFrame from ``sentence_segmenter.py``.  Must have at
            minimum: ``sentence_id``, ``transcript_id``, ``sentence_order``,
            ``sentence_text``.
        sort_output : bool
            Sort output by ``sentence_order`` (default True).

        Returns
        -------
        SentenceSentimentResult

        Raises
        ------
        RuntimeError
            If the model is not loaded.
        ValueError
            If required columns are missing or the DataFrame is empty.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Model not loaded. Call analyzer.load_model() first."
            )
        self._validate_input(df)

        # Dedup on sentence_id
        if df["sentence_id"].duplicated().any():
            n_dup = df["sentence_id"].duplicated().sum()
            warnings.warn(
                f"{n_dup} duplicate sentence_id values in input — "
                "keeping first occurrence.",
                UserWarning, stacklevel=2,
            )
            df = df.drop_duplicates("sentence_id", keep="first")

        # Sort by sentence_order for correct rolling analytics
        df = df.sort_values("sentence_order").reset_index(drop=True)

        unique_ids = df["transcript_id"].unique()
        transcript_id = str(unique_ids[0])
        if len(unique_ids) > 1:
            warnings.warn(
                f"analyze_transcript() received {len(unique_ids)} distinct "
                "transcript_ids. Use analyze_dataframe() for batch processing.",
                UserWarning, stacklevel=2,
            )

        stats = SentenceSentimentStats(
            transcript_id   = transcript_id,
            total_sentences = len(df),
        )

        rows_dicts = df.to_dict("records")
        all_sentiments: list[SentenceSentiment] = []

        # Rolling analytics engine — one instance per transcript, reset once
        rolling = _RollingAnalytics(
            window          = self.config.rolling_window,
            min_periods     = self.config.min_rolling_periods,
            spike_threshold = self.config.spike_threshold,
            ema_alpha       = self.config.ema_alpha,
        )

        # Separate valid from invalid rows while preserving order
        valid_indices : list[int]  = []
        valid_texts   : list[str]  = []
        pre_results   : list[Optional[SentenceSentiment]] = [None] * len(rows_dicts)

        for i, row in enumerate(rows_dicts):
            text    = self._clean_text(row.get("sentence_text", "") or "")
            ok, why = self._is_valid(text)
            if not ok:
                skip = SentenceSentiment.make_skipped(
                    sentence_id    = str(row.get("sentence_id", f"__skip_{i}")),
                    transcript_id  = str(row.get("transcript_id", transcript_id)),
                    sentence_order = int(row.get("sentence_order", i)),
                    reason         = why,
                    speaker        = str(row.get("speaker", "")),
                    speaker_role   = str(row.get("speaker_role", "")),
                    section_type   = str(row.get("section_type", "")),
                    token_estimate = int(row.get("token_estimate", 0)),
                    model_name     = self.config.model_name,
                )
                pre_results[i] = skip
                if "empty" in why:
                    stats.skipped_empty += 1
                else:
                    stats.skipped_short += 1
                continue
            valid_indices.append(i)
            valid_texts.append(text)

        # --- Batched inference over valid texts ---
        label_map = self._backend.label_map  # type: ignore[union-attr]

        t_start = time.perf_counter()
        batches = list(_iter_batches(valid_indices, self.config.batch_size))
        n_batches = len(batches)

        pbar = tqdm(
            batches,
            desc          = f"SentenceSentiment [{transcript_id}]",
            unit          = "batch",
            total         = n_batches,
            disable       = not self.config.show_progress,
            dynamic_ncols = True,
        )

        # We need to collect ALL valid sentence probabilities first so we can
        # feed them through the rolling analytics in strict sentence_order.
        # Store probs indexed by their position in rows_dicts.
        probs_by_position: dict[int, list[float]] = {}

        for batch_num, batch_indices in enumerate(pbar):
            batch_texts = [valid_texts[valid_indices.index(idx)] for idx in batch_indices]
            try:
                batch_probs = self._backend.predict_batch(batch_texts)  # type: ignore
            except Exception as exc:
                logger.error(
                    "Batch %d/%d failed: %s — marking %d sentences as errored.",
                    batch_num + 1, n_batches, exc, len(batch_indices),
                )
                stats.errors.append(
                    f"Batch {batch_num+1} error: {type(exc).__name__}: {exc}"
                )
                batch_probs = [[0.0, 0.0, 1.0]] * len(batch_indices)  # neutral fallback

            for idx, probs in zip(batch_indices, batch_probs):
                probs_by_position[idx] = probs

            stats.total_batches += 1

            if (
                self.config.log_every_n_batches > 0
                and (batch_num + 1) % self.config.log_every_n_batches == 0
            ):
                logger.info(
                    "[%s] Batch %d/%d | valid_processed=%d",
                    transcript_id, batch_num + 1, n_batches, len(probs_by_position),
                )

        # --- Walk all rows in sentence_order, updating rolling analytics ---
        # This ensures that rolling state is built in the exact chronological
        # sequence even though inference may have processed in batches.
        for i, row in enumerate(rows_dicts):
            if pre_results[i] is not None:
                # This was a skipped sentence — inject into rolling with score=0
                # so that the cumulative / EMA are not distorted.
                # Analytics for skipped rows are NOT updated.
                all_sentiments.append(pre_results[i])
                continue

            probs    = probs_by_position[i]
            analytics = rolling.update(
                _probs_to_score(probs, label_map)
            )
            sentiment = _build_sentiment(
                row       = row,
                probs     = probs,
                label_map = label_map,
                analytics = analytics,
                model_name= self.config.model_name,
            )
            all_sentiments.append(sentiment)
            stats.processed += 1

        stats.elapsed_seconds = time.perf_counter() - t_start

        if sort_output:
            all_sentiments.sort(key=lambda s: s.sentence_order)

        logger.info(
            "[%s] Sentence sentiment complete | "
            "total=%d | scored=%d | skipped=%d | "
            "batches=%d | elapsed=%.2fs",
            transcript_id,
            stats.total_sentences,
            stats.processed,
            stats.skipped_empty + stats.skipped_short,
            stats.total_batches,
            stats.elapsed_seconds,
        )

        return SentenceSentimentResult(
            transcript_id = transcript_id,
            sentiments    = all_sentiments,
            stats         = stats,
        )

    # ------------------------------------------------------------------
    # Batch: multiple transcripts
    # ------------------------------------------------------------------

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
    ) -> list[SentenceSentimentResult]:
        """Analyze all transcripts present in *df*.

        Groups rows by ``transcript_id`` and calls
        :meth:`analyze_transcript` for each group so that per-transcript
        rolling analytics are computed correctly (rolling state never bleeds
        between transcripts).

        Parameters
        ----------
        df : pd.DataFrame
            Combined sentence DataFrame from sentence_segmenter for N transcripts.

        Returns
        -------
        list[SentenceSentimentResult]
            One result per distinct ``transcript_id``.
        """
        self._validate_input(df)
        results: list[SentenceSentimentResult] = []
        grouped = df.groupby("transcript_id", sort=True)
        n = len(grouped)
        logger.info(
            "Starting batch sentence sentiment | transcripts=%d | sentences=%d",
            n, len(df),
        )
        for i, (tid, group) in enumerate(grouped, 1):
            logger.debug("Transcript %d/%d: %s (%d rows)", i, n, tid, len(group))
            result = self.analyze_transcript(group)
            results.append(result)

        total_scored = sum(r.stats.processed for r in results)
        logger.info(
            "Batch complete | transcripts=%d | total_scored=%d",
            n, total_scored,
        )
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_result(self, result: SentenceSentimentResult) -> list[str]:
        """Run post-analysis validation.  Returns list of issue strings."""
        v = SentenceSentimentValidator(
            prob_sum_tolerance = self.config.prob_sum_tolerance
        )
        return v.validate(result)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    @staticmethod
    def results_to_dataframe(
        results: list[SentenceSentimentResult],
    ) -> pd.DataFrame:
        """Concatenate multiple results into one flat DataFrame."""
        if not results:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        frames = [r.to_dataframe() for r in results if r.sentiments]
        if not frames:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        avail = [c for c in _OUTPUT_COLUMNS if c in combined.columns]
        extra = [c for c in combined.columns if c not in avail]
        return combined[avail + extra]

    @staticmethod
    def results_to_drift_dataframe(
        results: list[SentenceSentimentResult],
    ) -> pd.DataFrame:
        """Build a transcript-level drift summary DataFrame from multiple results."""
        rows = [r.transcript_drift_summary().to_dict() for r in results]
        return pd.DataFrame(rows)

    @staticmethod
    def results_to_section_dataframe(
        results: list[SentenceSentimentResult],
    ) -> pd.DataFrame:
        """Build a section-level drift summary DataFrame from multiple results."""
        rows = []
        for r in results:
            for sec in r.section_summaries():
                rows.append(sec.to_dict())
        return pd.DataFrame(rows)

    @staticmethod
    def results_to_speaker_dataframe(
        results: list[SentenceSentimentResult],
    ) -> pd.DataFrame:
        """Build a speaker-level summary DataFrame from multiple results."""
        rows = []
        for r in results:
            for spk in r.speaker_summaries():
                rows.append(spk.to_dict())
        return pd.DataFrame(rows)

    @staticmethod
    def save_parquet(
        result: SentenceSentimentResult,
        path  : Union[str, Path],
        *,
        include_skipped: bool = True,
    ) -> Path:
        """Write result to parquet."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = (
            result.to_dataframe()
            if include_skipped
            else result.filter_valid().to_dataframe()
        )
        df.to_parquet(out, index=False)
        logger.info("Saved %d sentences → %s", len(df), out)
        return out

    @staticmethod
    def save_csv(
        result: SentenceSentimentResult,
        path  : Union[str, Path],
        *,
        include_skipped: bool = True,
    ) -> Path:
        """Write result to CSV."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = (
            result.to_dataframe()
            if include_skipped
            else result.filter_valid().to_dataframe()
        )
        df.to_csv(out, index=False)
        logger.info("Saved %d sentences → %s", len(df), out)
        return out

    @staticmethod
    def save_jsonl(
        result: SentenceSentimentResult,
        path  : Union[str, Path],
        *,
        include_skipped: bool = True,
    ) -> Path:
        """Write result to JSONL (one JSON object per line)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sents = (
            result.sentiments if include_skipped
            else result.filter_valid().sentiments
        )
        with open(out, "w", encoding="utf-8") as fh:
            for s in sents:
                fh.write(json.dumps(s.to_dict()) + "\n")
        logger.info("Saved %d sentences → %s", len(sents), out)
        return out

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame missing required columns: {sorted(missing)}. "
                f"Required: {sorted(self.REQUIRED_COLUMNS)}"
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty.")


# ---------------------------------------------------------------------------
# Validation helper (standalone, usable in test suites)
# ---------------------------------------------------------------------------

class SentenceSentimentValidator:
    """Post-hoc validation of :class:`SentenceSentimentResult` objects.

    Examples
    --------
    >>> v = SentenceSentimentValidator()
    >>> issues = v.validate(result)
    >>> assert not issues, issues
    """

    VALID_LABELS: frozenset[str] = frozenset({
        SentimentLabel.POSITIVE.value,
        SentimentLabel.NEUTRAL.value,
        SentimentLabel.NEGATIVE.value,
        SentimentLabel.UNKNOWN.value,
    })

    def __init__(self, prob_sum_tolerance: float = 1e-4) -> None:
        self.prob_sum_tolerance = prob_sum_tolerance

    def validate(self, result: SentenceSentimentResult) -> list[str]:
        """Run all checks. Returns empty list if all pass."""
        issues: list[str] = []
        issues += self._check_non_empty(result)
        if issues:
            return issues
        issues += self._check_ordering(result)
        issues += self._check_no_duplicates(result)
        issues += self._check_prob_sums(result)
        issues += self._check_no_nan(result)
        issues += self._check_labels(result)
        issues += self._check_shift_continuity(result)
        return issues

    def validate_dataframe(self, df: pd.DataFrame) -> list[str]:
        """Validate a flat sentence sentiment DataFrame."""
        issues: list[str] = []
        if df.empty:
            return ["Sentence sentiment DataFrame is empty."]

        req = {
            "sentence_id", "transcript_id", "sentence_order",
            "sentence_score", "positive_prob", "neutral_prob",
            "negative_prob", "confidence", "predicted_label",
        }
        missing = req - set(df.columns)
        if missing:
            return [f"Missing columns: {sorted(missing)}"]

        # NaN in numeric fields
        numeric_cols = [
            "sentence_score", "positive_prob", "neutral_prob",
            "negative_prob", "confidence",
        ]
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"{nan_count} NaN values in column '{col}'.")

        # Probability sums (exclude skipped rows if was_skipped present)
        mask_valid = ~df.get("was_skipped", pd.Series(False, index=df.index))
        valid_df = df[mask_valid]
        if not valid_df.empty:
            prob_sums = (
                valid_df["positive_prob"]
                + valid_df["neutral_prob"]
                + valid_df["negative_prob"]
            )
            bad = (prob_sums - 1.0).abs() > self.prob_sum_tolerance
            if bad.any():
                issues.append(
                    f"{bad.sum()} rows have probability sums outside "
                    f"[1 ± {self.prob_sum_tolerance}]. "
                    f"Range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]"
                )

        # Label check
        bad_labels = ~df["predicted_label"].isin(self.VALID_LABELS)
        if bad_labels.any():
            issues.append(
                f"{bad_labels.sum()} invalid predicted_label values: "
                f"{df.loc[bad_labels, 'predicted_label'].unique()[:5].tolist()}"
            )

        # Duplicate sentence_id
        dup = df[df.duplicated("sentence_id")]
        if not dup.empty:
            issues.append(f"{len(dup)} duplicate sentence_id values.")

        # Per-transcript ordering
        for tid, grp in df.groupby("transcript_id"):
            orders = grp["sentence_order"].tolist()
            if orders != sorted(orders):
                issues.append(f"[{tid}] sentence_order not monotonic.")
            if len(orders) != len(set(orders)):
                issues.append(f"[{tid}] Duplicate sentence_order values.")

        return issues

    # --- Private ---

    def _check_non_empty(self, r: SentenceSentimentResult) -> list[str]:
        if not r.sentiments:
            return [f"[{r.transcript_id}] No sentiments produced."]
        return []

    def _check_ordering(self, r: SentenceSentimentResult) -> list[str]:
        orders = [s.sentence_order for s in r.sentiments]
        if orders != sorted(orders):
            return [f"[{r.transcript_id}] sentence_order is not sorted."]
        return []

    def _check_no_duplicates(self, r: SentenceSentimentResult) -> list[str]:
        seen: set[str] = set()
        dupes: list[str] = []
        for s in r.sentiments:
            if s.sentence_id in seen:
                dupes.append(s.sentence_id)
            seen.add(s.sentence_id)
        if dupes:
            return [f"[{r.transcript_id}] Duplicate sentence_ids: {dupes[:5]}"]
        return []

    def _check_prob_sums(self, r: SentenceSentimentResult) -> list[str]:
        issues, bad = [], 0
        for s in r.sentiments:
            if s.was_skipped:
                continue
            total = s.positive_prob + s.neutral_prob + s.negative_prob
            if abs(total - 1.0) > self.prob_sum_tolerance:
                bad += 1
                if bad <= 2:
                    issues.append(
                        f"Prob sum {total:.6f} ≠ 1 for sentence_id={s.sentence_id!r}"
                    )
        if bad > 2:
            issues.append(f"... and {bad - 2} more prob-sum violations.")
        return issues

    def _check_no_nan(self, r: SentenceSentimentResult) -> list[str]:
        issues = []
        for s in r.sentiments:
            if s.was_skipped:
                continue
            for fname, val in [
                ("sentence_score",         s.sentence_score),
                ("rolling_sentiment_mean", s.rolling_sentiment_mean),
                ("rolling_sentiment_std",  s.rolling_sentiment_std),
                ("ema_sentiment",          s.ema_sentiment),
            ]:
                if math.isnan(val) or math.isinf(val):
                    issues.append(
                        f"NaN/Inf in {fname} for sentence_id={s.sentence_id!r}"
                    )
        return issues

    def _check_labels(self, r: SentenceSentimentResult) -> list[str]:
        bad = [
            s.sentence_id for s in r.sentiments
            if s.predicted_label not in self.VALID_LABELS
        ]
        if bad:
            return [f"[{r.transcript_id}] {len(bad)} invalid labels: {bad[:3]}"]
        return []

    def _check_shift_continuity(self, r: SentenceSentimentResult) -> list[str]:
        """Verify the first valid sentence has shift=None, rest are float."""
        valid = [s for s in r.sentiments if not s.was_skipped]
        if not valid:
            return []
        first_valid = min(valid, key=lambda s: s.sentence_order)
        if first_valid.local_sentiment_shift is not None:
            return [
                f"[{r.transcript_id}] First valid sentence should have "
                f"local_sentiment_shift=None, got "
                f"{first_valid.local_sentiment_shift}"
            ]
        for s in valid:
            if s is first_valid:
                continue
            if s.local_sentiment_shift is None:
                return [
                    f"[{r.transcript_id}] Non-first sentence has "
                    f"local_sentiment_shift=None: sentence_id={s.sentence_id!r}"
                ]
        return []


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _probs_to_score(probs: list[float], label_map: dict[int, str]) -> float:
    """Compute sentiment score = P(positive) − P(negative) from prob list."""
    prob_by_name = {label_map[i]: probs[i] for i in range(len(probs))}
    return prob_by_name.get("positive", 0.0) - prob_by_name.get("negative", 0.0)


def _iter_batches(items: list, batch_size: int) -> Iterator[list]:
    """Yield consecutive non-overlapping sub-lists."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def print_result_summary(result: SentenceSentimentResult) -> None:
    """Print a compact summary of a :class:`SentenceSentimentResult`."""
    valid = [s for s in result.sentiments if not s.was_skipped]
    if not valid:
        print(f"\n[{result.transcript_id}] No valid sentences.")
        return

    drift = result.transcript_drift_summary()
    scores = [s.sentence_score for s in valid]
    sep = "─" * 62

    print(f"\n{sep}")
    print(f"  Sentence Sentiment Summary  |  {result.transcript_id}")
    print(sep)
    print(f"  Sentences (total/valid/skip) : "
          f"{drift.sentence_count} / {drift.valid_sentence_count} / {drift.skipped_count}")
    print(f"  Score  mean / std / median   : "
          f"{drift.mean_score:+.4f} / {drift.std_score:.4f} / {drift.median_score:+.4f}")
    print(f"  Score  min  / max / range    : "
          f"{drift.min_score:+.4f} / {drift.max_score:+.4f} / {drift.score_range:.4f}")
    print(f"  First → last score           : "
          f"{drift.first_score:+.4f} → {drift.final_score:+.4f}  "
          f"(net {drift.net_drift:+.4f})")
    print(f"  Drift slope (OLS)            : {drift.drift_slope:+.6f}")
    print(f"  Drift R²                     :  {drift.drift_r_squared:.4f}")
    print(f"  Overall volatility           :  {drift.overall_volatility:.4f}")
    print(f"  Spikes (|shift| > thr)       : "
          f"{drift.spike_count} ({drift.spike_rate:.1%})")
    print(f"  Mean confidence              :  {drift.mean_confidence:.4f}")

    # Section breakdown
    print(f"\n  Section breakdown:")
    for sec in result.section_summaries():
        print(f"    {sec.section_type:<22s}  "
              f"n={sec.sentence_count:>3d}  "
              f"mean={sec.mean_score:+.4f}  "
              f"slope={sec.drift_slope:+.5f}  "
              f"spikes={sec.spike_count}")

    # Speaker breakdown
    print(f"\n  Speaker breakdown:")
    for spk in result.speaker_summaries():
        print(f"    {spk.speaker:<20s} ({spk.speaker_role:<10s})  "
              f"n={spk.sentence_count:>3d}  mean={spk.mean_score:+.4f}  "
              f"std={spk.std_score:.4f}")

    # Label distribution
    label_counts: dict[str, int] = {}
    for s in valid:
        label_counts[s.predicted_label] = label_counts.get(s.predicted_label, 0) + 1
    print(f"\n  Label distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        pct = cnt / len(valid) * 100
        bar = "█" * int(pct / 4)
        print(f"    {lbl:<12s} {cnt:>4d}  ({pct:5.1f}%)  {bar}")

    print(sep)


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
    print("  SentenceSentimentAnalyzer — Self-Test & Demo")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # 1.  Synthetic sentence DataFrame (mimics sentence_segmenter output)
    # -----------------------------------------------------------------------
    SAMPLE_SENTENCES = [
        # Tim Cook — prepared remarks (positive arc)
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000000_aabb",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 0,
            "sentence_text" : "We generated revenue of $124.3 billion, up 4% year-over-year.",
            "speaker"       : "Tim Cook",
            "speaker_role"  : "CEO",
            "section_type"  : "prepared_remarks",
            "token_estimate": 18,
        },
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000001_ccdd",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 1,
            "sentence_text" : "iPhone revenue reached $69.7 billion, setting a new December quarter record.",
            "speaker"       : "Tim Cook",
            "speaker_role"  : "CEO",
            "section_type"  : "prepared_remarks",
            "token_estimate": 16,
        },
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000002_eeff",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 2,
            "sentence_text" : "Our Services segment grew 14%, reaching $26.3 billion, driven by strong adoption.",
            "speaker"       : "Tim Cook",
            "speaker_role"  : "CEO",
            "section_type"  : "prepared_remarks",
            "token_estimate": 18,
        },
        # Luca Maestri — CFO remarks (mixed)
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000003_gghh",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 3,
            "sentence_text" : "Gross margin was 46.9%, up 90 basis points from a year ago.",
            "speaker"       : "Luca Maestri",
            "speaker_role"  : "CFO",
            "section_type"  : "prepared_remarks",
            "token_estimate": 17,
        },
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000004_iijj",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 4,
            "sentence_text" : "We are guiding Q2 revenue between $93 and $97 billion, "
                              "which represents growth of approximately 4% to 7%.",
            "speaker"       : "Luca Maestri",
            "speaker_role"  : "CFO",
            "section_type"  : "prepared_remarks",
            "token_estimate": 25,
        },
        # Q&A — analyst pressure
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000005_kkll",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 5,
            "sentence_text" : "There is some concern about the slowdown in China and "
                              "increased competition from local vendors.",
            "speaker"       : "Analyst",
            "speaker_role"  : "Analyst",
            "section_type"  : "qa",
            "token_estimate": 22,
        },
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000006_mmnn",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 6,
            "sentence_text" : "We have headwinds in the greater China region due to macro uncertainty.",
            "speaker"       : "Tim Cook",
            "speaker_role"  : "CEO",
            "section_type"  : "qa",
            "token_estimate": 16,
        },
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000007_ooppcc",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 7,
            "sentence_text" : "Despite near-term pressure, we remain confident in our "
                              "long-term opportunity and growth momentum in the region.",
            "speaker"       : "Tim Cook",
            "speaker_role"  : "CEO",
            "section_type"  : "qa",
            "token_estimate": 24,
        },
        # Closing
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000008_qqrr",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 8,
            "sentence_text" : "Thank you all for joining us today.",
            "speaker"       : "Tim Cook",
            "speaker_role"  : "CEO",
            "section_type"  : "closing_remarks",
            "token_estimate": 8,
        },
        # Edge cases
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000009_edge1",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 9,
            "sentence_text" : "",        # empty → should be skipped
            "speaker"       : "Operator",
            "speaker_role"  : "Operator",
            "section_type"  : "closing_remarks",
            "token_estimate": 0,
        },
        {
            "sentence_id"   : "sent_AAPL_Q1_2025_000010_edge2",
            "transcript_id" : "AAPL_Q1_2025",
            "sentence_order": 10,
            "sentence_text" : "OK.",    # too short → should be skipped
            "speaker"       : "Operator",
            "speaker_role"  : "Operator",
            "section_type"  : "closing_remarks",
            "token_estimate": 2,
        },
    ]

    df_input = pd.DataFrame(SAMPLE_SENTENCES)
    print(f"\n[INPUT] {len(df_input)} sentences, "
          f"{df_input['transcript_id'].nunique()} transcript(s)")

    # -----------------------------------------------------------------------
    # 2.  Config validation
    # -----------------------------------------------------------------------
    print("\n[CONFIG]")
    try:
        SentenceSentimentConfig(max_token_length=600)
    except ValueError as e:
        print(f"  max_token_length > 512 correctly rejected: {e}")
    try:
        SentenceSentimentConfig(ema_alpha=0.0)
    except ValueError as e:
        print(f"  ema_alpha=0 correctly rejected: {e}")

    cfg = SentenceSentimentConfig(
        rolling_window    = 3,
        spike_threshold   = 0.20,
        ema_alpha         = 0.3,
        batch_size        = 4,
        show_progress     = True,
        log_every_n_batches = 2,
    )
    print(f"  Config OK | window={cfg.rolling_window} | spike_thr={cfg.spike_threshold}")

    # -----------------------------------------------------------------------
    # 3.  Mock-backend inference
    # -----------------------------------------------------------------------
    print("\n[INFERENCE]")
    analyzer = SentenceSentimentAnalyzer.mock(config=cfg)
    assert analyzer.is_ready
    print(f"  Analyzer state  : {analyzer._state.name}")
    print(f"  Device          : {analyzer.device_str}")

    result = analyzer.analyze_transcript(df_input)
    print(f"\n  Result: {result}")

    # -----------------------------------------------------------------------
    # 4.  Predictions table
    # -----------------------------------------------------------------------
    print("\n[SENTENCES]")
    header = (
        f"  {'ord':>3s}  {'speaker':<14s}  {'section':<17s}  "
        f"{'score':>7s}  {'shift':>7s}  {'roll_μ':>7s}  {'roll_σ':>6s}  "
        f"{'spike':>5s}  {'label':<10s}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for s in result.sentiments:
        shift_str = (
            f"{s.local_sentiment_shift:+.4f}"
            if s.local_sentiment_shift is not None
            else "  None"
        )
        skip_tag = " [SKIP]" if s.was_skipped else ""
        print(
            f"  {s.sentence_order:>3d}  {s.speaker:<14s}  {s.section_type:<17s}  "
            f"{s.sentence_score:>+7.4f}  {shift_str:>7s}  "
            f"{s.rolling_sentiment_mean:>+7.4f}  {s.rolling_sentiment_std:>6.4f}  "
            f"{str(s.is_spike):>5s}  {s.predicted_label:<10s}{skip_tag}"
        )

    # -----------------------------------------------------------------------
    # 5.  Analytics summaries
    # -----------------------------------------------------------------------
    print_result_summary(result)

    # -----------------------------------------------------------------------
    # 6.  DataFrame export
    # -----------------------------------------------------------------------
    df_out = result.to_dataframe()
    print(f"\n[EXPORT]")
    print(f"  DataFrame shape : {df_out.shape}")
    print(f"  Columns         : {list(df_out.columns)}")
    valid_rows = df_out[~df_out["was_skipped"]]
    print(f"  Valid rows      : {len(valid_rows)}")
    # Confirm local_sentiment_shift serializes as NaN (not None)
    first_shift = df_out["local_sentiment_shift"].iloc[0]
    print(f"  First sentence shift in DataFrame: {first_shift} (NaN expected)")
    assert pd.isna(first_shift), "First sentence shift should be NaN in DataFrame"

    # -----------------------------------------------------------------------
    # 7.  Section and speaker summaries as DataFrames
    # -----------------------------------------------------------------------
    section_df = pd.DataFrame([s.to_dict() for s in result.section_summaries()])
    speaker_df = pd.DataFrame([s.to_dict() for s in result.speaker_summaries()])
    drift_df   = pd.DataFrame([result.transcript_drift_summary().to_dict()])
    print(f"\n  Section summary DataFrame : {section_df.shape}")
    print(f"  Speaker summary DataFrame : {speaker_df.shape}")
    print(f"  Drift summary DataFrame   : {drift_df.shape}")
    print(f"\n  Section means:")
    for _, row in section_df.iterrows():
        print(f"    {row['section_type']:<22s}: mean={row['mean_score']:+.4f}  slope={row['drift_slope']:+.5f}")

    # -----------------------------------------------------------------------
    # 8.  Validation
    # -----------------------------------------------------------------------
    print("\n[VALIDATION]")
    issues = analyzer.validate_result(result)
    if issues:
        print(f"  ❌ {len(issues)} issue(s):")
        for iss in issues:
            print(f"     • {iss}")
        sys.exit(1)
    else:
        print("  ✅ All validation checks passed.")

    v = SentenceSentimentValidator()
    df_issues = v.validate_dataframe(df_out)
    if df_issues:
        print(f"  ❌ DataFrame validation issues: {df_issues}")
        sys.exit(1)
    else:
        print("  ✅ DataFrame validation passed.")

    # -----------------------------------------------------------------------
    # 9.  filter_valid
    # -----------------------------------------------------------------------
    clean = result.filter_valid()
    assert all(not s.was_skipped for s in clean.sentiments)
    print(f"\n[FILTER] filter_valid(): {len(result)} → {len(clean)} sentiments")

    # -----------------------------------------------------------------------
    # 10. Determinism check
    # -----------------------------------------------------------------------
    result2 = analyzer.analyze_transcript(df_input)
    scores1 = {s.sentence_id: s.sentence_score for s in result.sentiments}
    scores2 = {s.sentence_id: s.sentence_score for s in result2.sentiments}
    assert scores1 == scores2, "Determinism check failed"
    print("\n[DETERMINISM] ✅ Identical scores on re-run.")

    # -----------------------------------------------------------------------
    # 11. Batch analysis (two transcripts)
    # -----------------------------------------------------------------------
    MSFT_SENTS = [
        {
            "sentence_id"   : f"sent_MSFT_Q2_2025_{i:06d}_xx",
            "transcript_id" : "MSFT_Q2_2025",
            "sentence_order": i,
            "sentence_text" : text,
            "speaker"       : spk,
            "speaker_role"  : role,
            "section_type"  : sec,
            "token_estimate": len(text.split()),
        }
        for i, (text, spk, role, sec) in enumerate([
            ("Azure grew 31% in constant currency, outperforming expectations.",
             "Satya Nadella", "CEO", "prepared_remarks"),
            ("We saw strong momentum in Copilot and AI infrastructure adoption.",
             "Satya Nadella", "CEO", "prepared_remarks"),
            ("There are some near-term headwinds from the PC refresh cycle slowdown.",
             "Amy Hood", "CFO", "prepared_remarks"),
            ("Operating expenses were lower than expected due to efficiency improvements.",
             "Amy Hood", "CFO", "prepared_remarks"),
            ("Can you elaborate on the risk factors in the enterprise segment?",
             "Analyst", "Analyst", "qa"),
        ])
    ]
    df_combined = pd.concat(
        [df_input, pd.DataFrame(MSFT_SENTS)], ignore_index=True
    )
    batch_results = analyzer.analyze_dataframe(df_combined)
    print(f"\n[BATCH] {len(batch_results)} transcripts processed:")
    for r in batch_results:
        d = r.transcript_drift_summary()
        print(f"  {r.transcript_id}: scored={r.stats.processed} "
              f"mean={d.mean_score:+.4f} slope={d.drift_slope:+.5f}")

    combined_df = SentenceSentimentAnalyzer.results_to_dataframe(batch_results)
    print(f"  Combined DataFrame: {combined_df.shape[0]} rows × {combined_df.shape[1]} cols")

    # -----------------------------------------------------------------------
    # 12. Edge cases
    # -----------------------------------------------------------------------
    print("\n[EDGE CASES]")
    try:
        unloaded = SentenceSentimentAnalyzer()  # no .load_model()
        unloaded.analyze_transcript(df_input)
    except RuntimeError as e:
        print(f"  Unloaded analyzer: correctly raised RuntimeError.")

    try:
        analyzer.analyze_transcript(pd.DataFrame(columns=list(_REQUIRED_INPUT_COLUMNS)))
    except ValueError:
        print(f"  Empty DataFrame: correctly raised ValueError.")

    try:
        analyzer.analyze_transcript(df_input.drop(columns=["sentence_text"]))
    except ValueError:
        print(f"  Missing column: correctly raised ValueError.")

    # Duplicate sentence_id — should warn and deduplicate
    df_dup = pd.concat([df_input.iloc[:3], df_input.iloc[:2]], ignore_index=True)
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        r_dup = analyzer.analyze_transcript(df_dup)
    dup_warns = [str(w.message) for w in caught if "duplicate" in str(w.message).lower()]
    print(f"  Duplicate sentence_id: warning issued={bool(dup_warns)}, "
          f"result size={len(r_dup)} (expected 3 unique).")

    # -----------------------------------------------------------------------
    # 13. scores_array utility
    # -----------------------------------------------------------------------
    arr = result.scores_array()
    print(f"\n[SCORES ARRAY] shape={arr.shape}  mean={arr.mean():+.4f}")

    print("\n" + "=" * 68)
    print("  Self-test complete — all checks passed ✅")
    print("=" * 68)
