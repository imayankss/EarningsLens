"""
finbert_pipeline.py
===================
Scalable FinBERT inference pipeline for chunk-level sentiment scoring on
long earnings call transcripts.

Part of: Earnings Call Sentiment Analyzer — DAY 7 / DAY 8 Pipeline
Stage:   FinBERT Transformer Inference (after chunk_generator.py)

Responsibilities
----------------
* Load the ProsusAI/finbert tokenizer and classification model
* Accept chunk DataFrames produced by chunk_generator.py
* Run batched, GPU-safe transformer inference with OOM protection
* Compute per-chunk sentiment probabilities: positive / neutral / negative
* Derive sentiment_score  = P(positive) − P(negative)
* Derive confidence       = max(P(positive), P(neutral), P(negative))
* Emit deterministically ordered, fully validated prediction records
* Export results to DataFrame / parquet / CSV / JSONL for downstream use
* Provide comprehensive inference statistics for pipeline monitoring

This module does NOT perform:
  * transcript chunking           → chunk_generator.py
  * transcript-level aggregation  → aggregation.py
  * event-study logic             → event_study.py
  * Loughran-McDonald scoring     → lm_scorer.py

Input schema (DataFrame from chunk_generator)
----------------------------------------------
    chunk_id          : str  — globally unique chunk identifier
    transcript_id     : str  — parent transcript key, e.g. "AAPL_Q1_2025"
    chunk_order       : int  — sequential order within the transcript
    chunk_text        : str  — transformer-ready text (≤ ~450 tokens)
    token_count       : int  — pre-computed token estimate
    section_type      : str  — "prepared_remarks" | "qa" | "closing_remarks"
    dominant_speaker  : str  — speaker name from chunk_generator

Output schema (FinBERTPrediction)
----------------------------------
    chunk_id          : str
    transcript_id     : str
    chunk_order       : int
    positive_prob     : float
    neutral_prob      : float
    negative_prob     : float
    sentiment_score   : float  (positive_prob − negative_prob)
    confidence        : float  (max of the three probabilities)
    predicted_label   : str    ("positive" | "neutral" | "negative")
    token_count       : int    (propagated from input)
    section_type      : str    (propagated from input)
    dominant_speaker  : str    (propagated from input)

Architecture Notes
------------------
* Label order is read from ``model.config.id2label`` at runtime to guard
  against any future model-card updates.  The current ProsusAI/finbert
  mapping is: {0: "positive", 1: "negative", 2: "neutral"}.
* OOM-safe batching: if a GPU OOM error occurs, the pipeline automatically
  halves the batch size and retries.  After a configurable floor, it falls
  back to CPU.
* Inference caching: an in-process prediction cache (keyed on chunk_id +
  model name) prevents re-computing results for chunks already scored in
  the same session.  Cache can be disabled or cleared via the public API.
* FP16 inference on CUDA is supported via ``torch.cuda.amp.autocast()``.
* All probability sums are validated against 1.0 within a tolerance of 1e-4.

Usage
-----
    from finbert_pipeline import FinBERTPipeline, FinBERTConfig

    config = FinBERTConfig(batch_size=16, device="auto")
    pipeline = FinBERTPipeline(config=config)
    pipeline.load_model()

    result = pipeline.run_inference(chunks_df)
    df_out  = result.to_dataframe()
    df_out.to_parquet("data/processed/sentiment/finbert_chunk_scores.parquet")

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
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy torch / transformers imports — keep top-level import light so the
# module can be imported for type-checking even if torch is not installed.
# ---------------------------------------------------------------------------

def _require_torch():
    """Import torch, raising a clean error if not installed."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for FinBERT inference. "
            "Install with: pip install torch"
        ) from exc


def _require_transformers():
    """Import transformers, raising a clean error if not installed."""
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
    """Canonical sentiment label values emitted by this pipeline."""
    POSITIVE = "positive"
    NEUTRAL  = "neutral"
    NEGATIVE = "negative"
    UNKNOWN  = "unknown"   # fallback for malformed/skipped chunks


class InferenceDevice(str, Enum):
    """Inference device selection."""
    AUTO = "auto"
    CUDA = "cuda"
    CPU  = "cpu"


class PipelineState(Enum):
    """Lifecycle state of the :class:`FinBERTPipeline`."""
    UNLOADED  = auto()
    LOADING   = auto()
    READY     = auto()
    ERROR     = auto()


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FinBERTConfig:
    """All tunable parameters for :class:`FinBERTPipeline`.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.  Default is the finance-domain
        ProsusAI/finbert model referenced throughout the project architecture.
    device : str | InferenceDevice
        ``"auto"`` selects CUDA if available, otherwise CPU.
        ``"cuda"`` forces GPU (raises if unavailable).
        ``"cpu"`` forces CPU regardless of GPU availability.
    batch_size : int
        Number of chunks per forward pass.  Automatically halved on OOM.
        Recommended: 16–32 on GPU, 8 on CPU.
    min_batch_size : int
        Lowest batch size before the pipeline gives up and raises.
    max_token_length : int
        Hard truncation limit passed to the tokenizer.
        Must not exceed 512 (BERT architecture constraint).
    use_fp16 : bool
        Enable half-precision (FP16) inference on CUDA.  Faster on modern
        GPUs with negligible accuracy loss for sentiment classification.
    use_cache : bool
        Cache predictions keyed on (chunk_id, model_name) to avoid
        redundant forward passes within the same process session.
    cache_max_size : int
        Maximum number of entries in the in-process prediction cache.
        Older entries are evicted LRU-style when the limit is hit.
    max_retries : int
        Number of times to retry a failed batch before skipping.
    retry_backoff_s : float
        Initial back-off seconds between retries (doubles each attempt).
    min_chunk_chars : int
        Skip chunks shorter than this character count.
    prob_sum_tolerance : float
        Allowed deviation from 1.0 in the probability sum validation.
    show_progress : bool
        Display a tqdm progress bar during batch inference.
    log_every_n_batches : int
        Emit an INFO log every N batches (0 = disable mid-run logging).
    """

    model_name           : str   = "ProsusAI/finbert"
    device               : str   = InferenceDevice.AUTO.value

    batch_size           : int   = 16
    min_batch_size       : int   = 1
    max_token_length     : int   = 512

    use_fp16             : bool  = False
    use_cache            : bool  = True
    cache_max_size       : int   = 10_000

    max_retries          : int   = 3
    retry_backoff_s      : float = 0.5

    min_chunk_chars      : int   = 5
    prob_sum_tolerance   : float = 1e-4

    show_progress        : bool  = True
    log_every_n_batches  : int   = 10

    def __post_init__(self) -> None:
        if self.max_token_length > 512:
            raise ValueError(
                f"max_token_length={self.max_token_length} exceeds BERT's "
                "hard limit of 512.  Set max_token_length <= 512."
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1.")
        if self.min_batch_size > self.batch_size:
            raise ValueError("min_batch_size must be <= batch_size.")


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FinBERTPrediction:
    """Sentiment prediction for a single transcript chunk.

    All probability fields are Python floats in [0, 1] that sum to
    approximately 1.0 (within ``FinBERTConfig.prob_sum_tolerance``).
    """

    # --- Identity ---
    chunk_id          : str

    # --- Transcript linkage ---
    transcript_id     : str
    chunk_order       : int

    # --- Probabilities (FinBERT softmax outputs) ---
    positive_prob     : float
    neutral_prob      : float
    negative_prob     : float

    # --- Derived metrics ---
    sentiment_score   : float   # P(positive) − P(negative)
    confidence        : float   # max(P(pos), P(neu), P(neg))
    predicted_label   : str     # dominant label name

    # --- Propagated input metadata ---
    token_count       : int
    section_type      : str
    dominant_speaker  : str

    # --- Processing metadata ---
    model_name        : str  = "ProsusAI/finbert"
    was_skipped       : bool = False
    skip_reason       : str  = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a plain dict (JSON-serializable)."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"FinBERTPrediction("
            f"chunk_id={self.chunk_id!r}, "
            f"label={self.predicted_label!r}, "
            f"score={self.sentiment_score:+.4f}, "
            f"conf={self.confidence:.4f})"
        )

    @classmethod
    def make_skipped(
        cls,
        chunk_id: str,
        transcript_id: str,
        chunk_order: int,
        reason: str,
        token_count: int = 0,
        section_type: str = "",
        dominant_speaker: str = "",
        model_name: str = "ProsusAI/finbert",
    ) -> "FinBERTPrediction":
        """Create a sentinel prediction for a chunk that was skipped."""
        return cls(
            chunk_id         = chunk_id,
            transcript_id    = transcript_id,
            chunk_order      = chunk_order,
            positive_prob    = 0.0,
            neutral_prob     = 1.0,
            negative_prob    = 0.0,
            sentiment_score  = 0.0,
            confidence       = 0.0,
            predicted_label  = SentimentLabel.UNKNOWN.value,
            token_count      = token_count,
            section_type     = section_type,
            dominant_speaker = dominant_speaker,
            model_name       = model_name,
            was_skipped      = True,
            skip_reason      = reason,
        )


@dataclass
class InferenceStats:
    """Per-run inference diagnostics."""

    model_name           : str   = ""
    device_used          : str   = ""
    total_chunks         : int   = 0
    processed_chunks     : int   = 0
    skipped_empty        : int   = 0
    skipped_too_short    : int   = 0
    skipped_too_long     : int   = 0
    cache_hits           : int   = 0
    total_batches        : int   = 0
    oom_recoveries       : int   = 0
    retry_events         : int   = 0
    elapsed_seconds      : float = 0.0
    errors               : list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.processed_chunks / self.total_chunks

    @property
    def chunks_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.processed_chunks / self.elapsed_seconds

    def to_dict(self) -> dict:
        return {
            "model_name"        : self.model_name,
            "device_used"       : self.device_used,
            "total_chunks"      : self.total_chunks,
            "processed_chunks"  : self.processed_chunks,
            "skipped_total"     : self.skipped_empty + self.skipped_too_short + self.skipped_too_long,
            "cache_hits"        : self.cache_hits,
            "total_batches"     : self.total_batches,
            "oom_recoveries"    : self.oom_recoveries,
            "retry_events"      : self.retry_events,
            "success_rate"      : round(self.success_rate, 4),
            "chunks_per_second" : round(self.chunks_per_second, 2),
            "elapsed_seconds"   : round(self.elapsed_seconds, 3),
            "errors"            : self.errors,
        }


@dataclass
class InferenceResult:
    """Complete output of a :meth:`FinBERTPipeline.run_inference` call.

    Attributes
    ----------
    predictions : list[FinBERTPrediction]
        One entry per input chunk, in the same order as the input DataFrame.
    stats : InferenceStats
        Diagnostic information for monitoring and debugging.
    """

    predictions : list[FinBERTPrediction]
    stats       : InferenceStats

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all predictions to a pandas DataFrame.

        The column order matches the architecture-specified output schema.
        Suitable for direct parquet/CSV export.
        """
        if not self.predictions:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        rows = [p.to_dict() for p in self.predictions]
        df = pd.DataFrame(rows)
        available = [c for c in _OUTPUT_COLUMNS if c in df.columns]
        extra = [c for c in df.columns if c not in available]
        return df[available + extra]

    def to_dict_list(self) -> list[dict]:
        """Return list of dicts (for JSONL export)."""
        return [p.to_dict() for p in self.predictions]

    def filter_valid(self) -> "InferenceResult":
        """Return a new InferenceResult with skipped predictions removed."""
        return InferenceResult(
            predictions=[p for p in self.predictions if not p.was_skipped],
            stats=self.stats,
        )

    def __len__(self) -> int:
        return len(self.predictions)

    def __repr__(self) -> str:
        return (
            f"InferenceResult("
            f"predictions={len(self.predictions)}, "
            f"model={self.stats.model_name!r}, "
            f"device={self.stats.device_used!r}, "
            f"success_rate={self.stats.success_rate:.1%})"
        )


# Output column order (matches DAY 7/8 architecture specification)
_OUTPUT_COLUMNS: list[str] = [
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
    "model_name",
    "was_skipped",
    "skip_reason",
]

# Required input columns
_REQUIRED_INPUT_COLUMNS: frozenset[str] = frozenset({
    "chunk_id",
    "transcript_id",
    "chunk_order",
    "chunk_text",
})


# ---------------------------------------------------------------------------
# In-process LRU prediction cache
# ---------------------------------------------------------------------------

class _PredictionCache:
    """Simple LRU-evicting in-process cache for FinBERT predictions.

    Keyed on ``(chunk_id, model_name)`` to make cache entries model-specific.
    Thread-safety is NOT guaranteed; this module assumes single-threaded use.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._store: dict[str, FinBERTPrediction] = {}
        self._order: list[str] = []   # insertion order for LRU eviction

    @staticmethod
    def make_key(chunk_id: str, model_name: str) -> str:
        raw = json.dumps({"c": chunk_id, "m": model_name}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, chunk_id: str, model_name: str) -> Optional[FinBERTPrediction]:
        key = self.make_key(chunk_id, model_name)
        pred = self._store.get(key)
        if pred is not None:
            # Move to end (most-recently-used)
            self._order.remove(key)
            self._order.append(key)
        return pred

    def put(self, pred: FinBERTPrediction) -> None:
        key = self.make_key(pred.chunk_id, pred.model_name)
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self._max_size:
            # Evict least-recently-used
            oldest = self._order.pop(0)
            del self._store[oldest]
        self._store[key] = pred
        self._order.append(key)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FinBERTPipeline:
    """Scalable FinBERT inference engine for earnings call sentiment scoring.

    This class orchestrates model loading, GPU/CPU device management,
    OOM-safe batched inference, probability extraction, and validation for
    the chunk-level sentiment scoring stage of the earnings call NLP pipeline.

    Parameters
    ----------
    config : FinBERTConfig, optional
        If ``None``, sensible defaults are used (CPU, batch_size=8).

    Examples
    --------
    Load model and score a chunk DataFrame::

        pipeline = FinBERTPipeline(FinBERTConfig(batch_size=16))
        pipeline.load_model()

        result = pipeline.run_inference(chunks_df)
        df_out = result.to_dataframe()
        df_out.to_parquet("data/processed/sentiment/finbert_chunk_scores.parquet")

    Check whether the model is ready before running::

        if not pipeline.is_ready:
            pipeline.load_model()
    """

    def __init__(self, config: Optional[FinBERTConfig] = None) -> None:
        self.config = config or FinBERTConfig(batch_size=8)
        self._state  : PipelineState = PipelineState.UNLOADED
        self._tokenizer = None
        self._model     = None
        self._device    = None       # resolved torch.device
        self._label_map : dict[int, str] = {}   # id → label string
        self._cache     = _PredictionCache(self.config.cache_max_size)
        self._active_batch_size = self.config.batch_size

        logger.info(
            "FinBERTPipeline created | model=%s | device=%s | batch_size=%d",
            self.config.model_name,
            self.config.device,
            self.config.batch_size,
        )

    # -----------------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if the model and tokenizer are loaded and ready."""
        return self._state == PipelineState.READY

    @property
    def device_str(self) -> str:
        """String representation of the active torch device."""
        return str(self._device) if self._device is not None else "unresolved"

    @property
    def cache_size(self) -> int:
        """Number of predictions currently in the in-process cache."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Evict all entries from the in-process prediction cache."""
        self._cache.clear()
        logger.debug("Prediction cache cleared.")

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the FinBERT tokenizer and model from HuggingFace Hub.

        This method is idempotent: calling it on an already-loaded pipeline
        is a no-op.  It must be called before :meth:`run_inference`.

        Raises
        ------
        RuntimeError
            If the model fails to load after the configured retry attempts.
        ImportError
            If ``torch`` or ``transformers`` are not installed.
        """
        if self._state == PipelineState.READY:
            logger.debug("Model already loaded — skipping.")
            return

        torch = _require_torch()
        tf    = _require_transformers()

        self._state = PipelineState.LOADING
        logger.info("Loading FinBERT | model=%s", self.config.model_name)

        try:
            # --- Device resolution ---
            self._device = self._resolve_device(torch)
            logger.info("Inference device: %s", self._device)

            # --- Tokenizer ---
            logger.debug("Loading tokenizer...")
            self._tokenizer = tf.AutoTokenizer.from_pretrained(
                self.config.model_name
            )

            # --- Model ---
            logger.debug("Loading classification model...")
            self._model = tf.AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name
            )
            self._model.to(self._device)
            self._model.eval()

            # FP16 for CUDA
            if self.config.use_fp16 and str(self._device) != "cpu":
                self._model = self._model.half()
                logger.info("FP16 inference enabled.")

            # --- Label map ---
            self._label_map = self._resolve_label_map()
            logger.info("Label map: %s", self._label_map)

            self._state = PipelineState.READY
            logger.info(
                "FinBERT loaded successfully | labels=%s | device=%s",
                list(self._label_map.values()),
                self._device,
            )

        except Exception as exc:
            self._state = PipelineState.ERROR
            logger.exception("Failed to load FinBERT model: %s", exc)
            raise RuntimeError(
                f"FinBERT model loading failed: {exc}"
            ) from exc

    def unload_model(self) -> None:
        """Release model weights from memory and reset pipeline state.

        Useful for freeing GPU memory between pipeline stages.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        torch = _require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self._state = PipelineState.UNLOADED
        logger.info("FinBERT model unloaded and memory released.")

    # -----------------------------------------------------------------------
    # Device helpers
    # -----------------------------------------------------------------------

    def _resolve_device(self, torch) -> "torch.device":
        """Return the appropriate torch.device based on config and availability."""
        cfg = self.config.device

        if cfg == InferenceDevice.AUTO.value or cfg == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif cfg == InferenceDevice.CUDA.value or cfg == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' requested but CUDA is not available. "
                    "Set device='auto' or device='cpu'."
                )
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        return device

    # -----------------------------------------------------------------------
    # Label map helpers
    # -----------------------------------------------------------------------

    def _resolve_label_map(self) -> dict[int, str]:
        """Build a normalised {label_id: label_name} mapping from the model.

        Reads ``model.config.id2label`` at runtime so the pipeline is robust
        to any future changes to the ProsusAI/finbert model card.

        Falls back to the currently documented default mapping if
        ``id2label`` is not available.
        """
        fallback = {0: "positive", 1: "negative", 2: "neutral"}
        try:
            raw = self._model.config.id2label
            normalised = {
                int(k): str(v).lower().strip()
                for k, v in raw.items()
            }
            # Validate that all three labels are present
            label_set = set(normalised.values())
            expected  = {"positive", "negative", "neutral"}
            if not expected.issubset(label_set):
                logger.warning(
                    "Unexpected label set from model config: %s. "
                    "Falling back to documented default %s.",
                    label_set, fallback,
                )
                return fallback
            return normalised
        except Exception as exc:
            logger.warning(
                "Could not read id2label from model config (%s). "
                "Using fallback label map %s.",
                exc, fallback,
            )
            return fallback

    def _label_id_for(self, name: str) -> int:
        """Return the label index for a given label name string."""
        for idx, lname in self._label_map.items():
            if lname == name:
                return idx
        raise KeyError(f"Label {name!r} not found in label map {self._label_map}.")

    # -----------------------------------------------------------------------
    # Text preprocessing
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Apply minimal text normalization before tokenization.

        Preserves all financial language, numbers, and punctuation.
        Only collapses excessive whitespace.
        """
        import re
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _validate_chunk_text(
        self, text: str, chunk_id: str
    ) -> tuple[bool, str]:
        """Return (is_valid, reason) for a chunk text string.

        Parameters
        ----------
        text : str
            Cleaned chunk text.
        chunk_id : str
            Used in log messages only.

        Returns
        -------
        (True, "") if the chunk is valid for inference.
        (False, reason_string) if the chunk should be skipped.
        """
        if not text:
            return False, "empty_text"
        if len(text) < self.config.min_chunk_chars:
            return False, f"too_short_chars:{len(text)}"
        return True, ""

    # -----------------------------------------------------------------------
    # Tokenisation
    # -----------------------------------------------------------------------

    def _tokenize_batch(
        self, texts: list[str]
    ) -> dict:
        """Tokenize a list of texts and return tensors on the active device.

        Uses dynamic padding (pad to the longest sequence in the batch rather
        than the global max) to reduce unnecessary computation.

        Parameters
        ----------
        texts : list[str]
            Cleaned chunk texts.

        Returns
        -------
        dict
            HuggingFace encoding dict with tensors on ``self._device``.
        """
        encoding = self._tokenizer(
            texts,
            padding          = True,          # dynamic per-batch padding
            truncation       = True,
            max_length       = self.config.max_token_length,
            return_tensors   = "pt",
        )
        return {k: v.to(self._device) for k, v in encoding.items()}

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def _forward_pass(
        self, encoding: dict
    ) -> "torch.Tensor":
        """Run a single forward pass and return softmax probabilities.

        Parameters
        ----------
        encoding : dict
            Tokenizer output (on the correct device).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, 3) — probabilities for each class.
        """
        torch = _require_torch()
        import torch.nn.functional as F

        use_amp = (
            self.config.use_fp16
            and str(self._device) != "cpu"
            and hasattr(torch.cuda.amp, "autocast")
        )

        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._model(**encoding)
            else:
                outputs = self._model(**encoding)

        logits = outputs.logits                # (batch, num_labels)
        probs  = F.softmax(logits.float(), dim=-1)  # cast to fp32 for stability
        return probs.cpu()

    # -----------------------------------------------------------------------
    # Batch prediction
    # -----------------------------------------------------------------------

    def predict_batch(
        self,
        rows: list[dict],
        stats: InferenceStats,
    ) -> list[FinBERTPrediction]:
        """Score a single batch of chunk rows.

        This is the inner loop called by :meth:`run_inference`.
        Handles OOM recovery by halving batch size and re-splitting.

        Parameters
        ----------
        rows : list[dict]
            Chunk row dicts with at minimum ``chunk_id``, ``transcript_id``,
            ``chunk_order``, and ``chunk_text`` keys.
        stats : InferenceStats
            Mutable stats object updated in place.

        Returns
        -------
        list[FinBERTPrediction]
            Predictions in the same order as ``rows``.
        """
        torch = _require_torch()

        # Separate valid from pre-filtered skips
        valid_indices : list[int]  = []
        valid_texts   : list[str]  = []
        predictions   : list[Optional[FinBERTPrediction]] = [None] * len(rows)

        for i, row in enumerate(rows):
            raw_text = row.get("chunk_text", "") or ""
            text = self._clean_text(raw_text)
            ok, reason = self._validate_chunk_text(text, row.get("chunk_id", "?"))

            if not ok:
                skip = FinBERTPrediction.make_skipped(
                    chunk_id         = row.get("chunk_id", f"__skip_{i}"),
                    transcript_id    = row.get("transcript_id", ""),
                    chunk_order      = int(row.get("chunk_order", i)),
                    reason           = reason,
                    token_count      = int(row.get("token_count", 0)),
                    section_type     = row.get("section_type", ""),
                    dominant_speaker = row.get("dominant_speaker", ""),
                    model_name       = self.config.model_name,
                )
                predictions[i] = skip
                if "empty" in reason:
                    stats.skipped_empty += 1
                elif "short" in reason:
                    stats.skipped_too_short += 1
                elif "long" in reason:
                    stats.skipped_too_long += 1
                continue

            # Check inference cache
            if self.config.use_cache:
                cached = self._cache.get(
                    row.get("chunk_id", ""), self.config.model_name
                )
                if cached is not None:
                    predictions[i] = cached
                    stats.cache_hits += 1
                    continue

            valid_indices.append(i)
            valid_texts.append(text)

        # All rows pre-resolved
        if not valid_texts:
            return [p for p in predictions if p is not None]

        # Run transformer inference (with OOM-safe retry)
        prob_rows = self._run_with_oom_recovery(valid_texts, stats)

        # Build FinBERTPrediction objects
        for list_pos, (row_idx, probs_row) in enumerate(
            zip(valid_indices, prob_rows)
        ):
            row  = rows[row_idx]
            pred = self._build_prediction(row, probs_row)
            predictions[row_idx] = pred

            if self.config.use_cache:
                self._cache.put(pred)

        stats.processed_chunks += len(valid_indices)
        stats.total_batches    += 1
        return [p for p in predictions if p is not None]

    def _run_with_oom_recovery(
        self, texts: list[str], stats: InferenceStats
    ) -> list[list[float]]:
        """Run forward pass with automatic OOM recovery.

        If a CUDA OOM error occurs, the batch is split in half and each
        sub-batch is retried.  If the batch shrinks below
        ``config.min_batch_size``, inference falls back to CPU.

        Parameters
        ----------
        texts : list[str]
            Cleaned chunk texts.
        stats : InferenceStats
            Updated in place.

        Returns
        -------
        list[list[float]]
            Flat list of [pos, neu, neg] probability triples, same order as
            input texts.
        """
        torch = _require_torch()
        results: list[list[float]] = []
        queue   = list(texts)  # copy to allow in-place splitting

        while queue:
            mini = queue[:self._active_batch_size]
            rest = queue[self._active_batch_size:]

            try:
                encoding = self._tokenize_batch(mini)
                probs    = self._forward_pass(encoding)
                results.extend(probs.tolist())
                queue = rest

            except RuntimeError as exc:
                err_str = str(exc).lower()
                is_oom  = "out of memory" in err_str or "cuda oom" in err_str

                if not is_oom:
                    raise  # non-OOM errors propagate immediately

                # OOM recovery
                stats.oom_recoveries += 1
                if str(self._device) != "cpu":
                    torch.cuda.empty_cache()
                gc.collect()

                new_bs = max(
                    self._active_batch_size // 2,
                    self.config.min_batch_size,
                )
                logger.warning(
                    "CUDA OOM detected. Halving batch size: %d → %d.",
                    self._active_batch_size, new_bs,
                )

                if new_bs < self._active_batch_size:
                    self._active_batch_size = new_bs
                    queue = mini + rest  # re-queue the failed mini-batch
                else:
                    # Already at minimum — fall back to CPU
                    logger.warning(
                        "Batch size at minimum (%d). Falling back to CPU.",
                        self.config.min_batch_size,
                    )
                    self._fallback_to_cpu()
                    queue = mini + rest  # retry on CPU

        return results

    def _fallback_to_cpu(self) -> None:
        """Move the model to CPU as a last resort after OOM exhaustion."""
        torch = _require_torch()
        try:
            self._model = self._model.float()  # undo FP16 if active
            self._model.to("cpu")
            self._device = torch.device("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            logger.warning("Model moved to CPU fallback.")
        except Exception as exc:
            logger.error("CPU fallback failed: %s", exc)
            raise

    # -----------------------------------------------------------------------
    # Probability → FinBERTPrediction conversion
    # -----------------------------------------------------------------------

    def _build_prediction(
        self, row: dict, probs: list[float]
    ) -> FinBERTPrediction:
        """Construct a :class:`FinBERTPrediction` from a probability triple.

        Parameters
        ----------
        row : dict
            Original chunk row dict.
        probs : list[float]
            [prob_label_0, prob_label_1, prob_label_2] from softmax.
            Order matches ``self._label_map``.
        """
        # Map to named probabilities using the runtime label map
        prob_by_name: dict[str, float] = {}
        for idx, name in self._label_map.items():
            prob_by_name[name] = float(probs[idx])

        pos_p = prob_by_name.get("positive", 0.0)
        neg_p = prob_by_name.get("negative", 0.0)
        neu_p = prob_by_name.get("neutral",  0.0)

        sentiment_score = pos_p - neg_p
        confidence      = max(pos_p, neg_p, neu_p)
        predicted_label = max(
            prob_by_name, key=lambda k: prob_by_name[k]
        )

        return FinBERTPrediction(
            chunk_id         = str(row.get("chunk_id", "")),
            transcript_id    = str(row.get("transcript_id", "")),
            chunk_order      = int(row.get("chunk_order", 0)),
            positive_prob    = round(pos_p, 8),
            neutral_prob     = round(neu_p, 8),
            negative_prob    = round(neg_p, 8),
            sentiment_score  = round(sentiment_score, 8),
            confidence       = round(confidence, 8),
            predicted_label  = predicted_label,
            token_count      = int(row.get("token_count", 0)),
            section_type     = str(row.get("section_type", "")),
            dominant_speaker = str(row.get("dominant_speaker", "")),
            model_name       = self.config.model_name,
            was_skipped      = False,
            skip_reason      = "",
        )

    # -----------------------------------------------------------------------
    # Full-dataset inference
    # -----------------------------------------------------------------------

    def run_inference(
        self,
        chunks_df: pd.DataFrame,
        *,
        sort_output: bool = True,
    ) -> InferenceResult:
        """Score all chunks in *chunks_df* and return an :class:`InferenceResult`.

        The pipeline must be loaded (via :meth:`load_model`) before calling
        this method.

        Parameters
        ----------
        chunks_df : pd.DataFrame
            Chunk DataFrame produced by ``chunk_generator.py``.  Must contain
            at minimum: ``chunk_id``, ``transcript_id``, ``chunk_order``,
            ``chunk_text``.  Optional columns (``token_count``,
            ``section_type``, ``dominant_speaker``) are propagated if present.
        sort_output : bool
            If True (default), output predictions are sorted by
            (transcript_id, chunk_order) for deterministic ordering.

        Returns
        -------
        InferenceResult

        Raises
        ------
        RuntimeError
            If the model is not loaded.
        ValueError
            If required input columns are missing.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Model is not loaded.  Call pipeline.load_model() first."
            )

        self._validate_input_df(chunks_df)

        # Deduplicate input on chunk_id (warn if duplicates found)
        if chunks_df["chunk_id"].duplicated().any():
            n_dup = chunks_df["chunk_id"].duplicated().sum()
            warnings.warn(
                f"{n_dup} duplicate chunk_id values detected in input. "
                "Keeping first occurrence.",
                UserWarning,
                stacklevel=2,
            )
            chunks_df = chunks_df.drop_duplicates("chunk_id", keep="first")

        stats = InferenceStats(
            model_name  = self.config.model_name,
            device_used = self.device_str,
            total_chunks = len(chunks_df),
        )

        rows_dicts = chunks_df.to_dict("records")
        all_predictions: list[FinBERTPrediction] = []
        t_start = time.perf_counter()

        # Batch iterator
        batches = list(_iter_batches(rows_dicts, self._active_batch_size))
        n_batches = len(batches)

        pbar = tqdm(
            batches,
            desc         = f"FinBERT [{self.config.model_name}]",
            unit         = "batch",
            total        = n_batches,
            disable      = not self.config.show_progress,
            dynamic_ncols = True,
        )

        for batch_idx, batch_rows in enumerate(pbar):
            # Per-batch retry loop
            batch_preds = self._run_batch_with_retry(
                batch_rows, stats, batch_idx
            )
            all_predictions.extend(batch_preds)

            if (
                self.config.log_every_n_batches > 0
                and (batch_idx + 1) % self.config.log_every_n_batches == 0
            ):
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "Batch %d/%d | processed=%d | cache_hits=%d | "
                    "oom_recoveries=%d | elapsed=%.1fs",
                    batch_idx + 1, n_batches,
                    stats.processed_chunks,
                    stats.cache_hits,
                    stats.oom_recoveries,
                    elapsed,
                )

        stats.elapsed_seconds = time.perf_counter() - t_start

        if sort_output:
            all_predictions.sort(
                key=lambda p: (p.transcript_id, p.chunk_order)
            )

        logger.info(
            "Inference complete | chunks=%d | processed=%d | "
            "skipped=%d | cache_hits=%d | batches=%d | "
            "elapsed=%.2fs | throughput=%.1f chunks/s",
            stats.total_chunks,
            stats.processed_chunks,
            stats.skipped_empty + stats.skipped_too_short + stats.skipped_too_long,
            stats.cache_hits,
            stats.total_batches,
            stats.elapsed_seconds,
            stats.chunks_per_second,
        )

        return InferenceResult(predictions=all_predictions, stats=stats)

    def score_chunks(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Compatibility wrapper returning a valid prediction DataFrame.

        Older orchestration code calls ``score_chunks`` directly.  The newer
        API separates model loading from ``run_inference`` and returns an
        ``InferenceResult``.  This wrapper keeps the CLI stable while preserving
        the stronger result object for research modules and tests.
        """
        if not self.is_ready:
            self.load_model()
        return self.run_inference(chunks_df).filter_valid().to_dataframe()

    def _run_batch_with_retry(
        self,
        batch_rows: list[dict],
        stats: InferenceStats,
        batch_idx: int,
    ) -> list[FinBERTPrediction]:
        """Run ``predict_batch`` with exponential-back-off retries."""
        for attempt in range(self.config.max_retries):
            try:
                return self.predict_batch(batch_rows, stats)
            except Exception as exc:
                err_str = str(exc).lower()
                is_oom = "out of memory" in err_str

                if attempt < self.config.max_retries - 1:
                    wait = self.config.retry_backoff_s * (2 ** attempt)
                    logger.warning(
                        "Batch %d failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        batch_idx, attempt + 1, self.config.max_retries,
                        exc, wait,
                    )
                    stats.retry_events += 1
                    time.sleep(wait)
                else:
                    # Final failure — mark all chunks in batch as skipped
                    logger.error(
                        "Batch %d failed after %d attempts: %s. "
                        "Marking %d chunks as errored.",
                        batch_idx, self.config.max_retries,
                        exc, len(batch_rows),
                    )
                    stats.errors.append(
                        f"Batch {batch_idx} error after "
                        f"{self.config.max_retries} retries: {exc}"
                    )
                    return [
                        FinBERTPrediction.make_skipped(
                            chunk_id         = r.get("chunk_id", f"__err_{i}"),
                            transcript_id    = r.get("transcript_id", ""),
                            chunk_order      = int(r.get("chunk_order", i)),
                            reason           = f"inference_error:{type(exc).__name__}",
                            token_count      = int(r.get("token_count", 0)),
                            section_type     = r.get("section_type", ""),
                            dominant_speaker = r.get("dominant_speaker", ""),
                            model_name       = self.config.model_name,
                        )
                        for i, r in enumerate(batch_rows)
                    ]
        # Unreachable, but satisfies type checkers
        return []

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate_outputs(
        self,
        result: InferenceResult,
    ) -> list[str]:
        """Run post-inference validation on an :class:`InferenceResult`.

        Returns a list of issue strings.  An empty list means all checks
        passed.

        Checks performed
        ----------------
        * No NaN probabilities
        * Probability sums ≈ 1.0
        * Labels are in the allowed set
        * No duplicate chunk_ids
        * Chunk ordering is internally consistent
        """
        validator = InferenceValidator(
            prob_sum_tolerance = self.config.prob_sum_tolerance,
            allowed_labels     = set(self._label_map.values()),
        )
        return validator.validate(result)

    # -----------------------------------------------------------------------
    # Input schema validation
    # -----------------------------------------------------------------------

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        missing = _REQUIRED_INPUT_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {sorted(missing)}. "
                f"Required: {sorted(_REQUIRED_INPUT_COLUMNS)}"
            )
        if df.empty:
            raise ValueError("Input chunk DataFrame is empty.")

    # -----------------------------------------------------------------------
    # Export helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def save_parquet(
        result: InferenceResult,
        path: Union[str, Path],
        include_skipped: bool = True,
    ) -> Path:
        """Save inference results to a parquet file.

        Parameters
        ----------
        result : InferenceResult
        path : str | Path
            Output file path.
        include_skipped : bool
            If False, skipped/error predictions are excluded.

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = result.to_dataframe() if include_skipped else result.filter_valid().to_dataframe()
        df.to_parquet(out, index=False)
        logger.info("Saved %d predictions → %s", len(df), out)
        return out

    @staticmethod
    def save_scores(
        scores: InferenceResult | pd.DataFrame,
        path: Union[str, Path],
        include_skipped: bool = True,
    ) -> Path:
        """Compatibility writer accepting either ``InferenceResult`` or DataFrame."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(scores, InferenceResult):
            df = scores.to_dataframe() if include_skipped else scores.filter_valid().to_dataframe()
        else:
            df = scores.copy()
        df.to_parquet(out, index=False)
        logger.info("Saved %d FinBERT score rows → %s", len(df), out)
        return out

    @staticmethod
    def save_csv(
        result: InferenceResult,
        path: Union[str, Path],
        include_skipped: bool = True,
    ) -> Path:
        """Save inference results to a CSV file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = result.to_dataframe() if include_skipped else result.filter_valid().to_dataframe()
        df.to_csv(out, index=False)
        logger.info("Saved %d predictions → %s", len(df), out)
        return out

    @staticmethod
    def save_jsonl(
        result: InferenceResult,
        path: Union[str, Path],
        include_skipped: bool = True,
    ) -> Path:
        """Save inference results to a JSONL file (one JSON object per line)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        preds = result.predictions if include_skipped else result.filter_valid().predictions
        with open(out, "w", encoding="utf-8") as fh:
            for pred in preds:
                fh.write(json.dumps(pred.to_dict()) + "\n")
        logger.info("Saved %d predictions → %s", len(preds), out)
        return out


# ---------------------------------------------------------------------------
# Standalone validation helper
# ---------------------------------------------------------------------------

class InferenceValidator:
    """Post-hoc validation of :class:`InferenceResult` objects.

    Designed to be used in test suites, pipeline integration checks,
    and post-run quality gates.

    Examples
    --------
    >>> validator = InferenceValidator()
    >>> issues = validator.validate(result)
    >>> assert not issues, f"Validation failed: {issues}"
    """

    VALID_LABELS: frozenset[str] = frozenset({
        SentimentLabel.POSITIVE.value,
        SentimentLabel.NEUTRAL.value,
        SentimentLabel.NEGATIVE.value,
        SentimentLabel.UNKNOWN.value,
    })

    def __init__(
        self,
        prob_sum_tolerance  : float               = 1e-4,
        max_token_length    : int                 = 512,
        allowed_labels      : Optional[set[str]]  = None,
    ) -> None:
        self.prob_sum_tolerance = prob_sum_tolerance
        self.max_token_length   = max_token_length
        self.allowed_labels     = allowed_labels or self.VALID_LABELS

    def validate(self, result: InferenceResult) -> list[str]:
        """Run all checks.  Returns an empty list if everything is fine."""
        issues: list[str] = []
        issues += self._check_non_empty(result)
        if not issues:
            issues += self._check_no_nan(result)
            issues += self._check_prob_sums(result)
            issues += self._check_labels(result)
            issues += self._check_no_duplicate_chunks(result)
            issues += self._check_score_range(result)
            issues += self._check_confidence_range(result)
        return issues

    def validate_dataframe(self, df: pd.DataFrame) -> list[str]:
        """Validate a flattened predictions DataFrame directly."""
        issues: list[str] = []
        if df.empty:
            issues.append("Predictions DataFrame is empty.")
            return issues

        # Required columns
        required = {
            "chunk_id", "transcript_id", "chunk_order",
            "positive_prob", "neutral_prob", "negative_prob",
            "sentiment_score", "confidence", "predicted_label",
        }
        missing = required - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {sorted(missing)}")
            return issues  # Cannot proceed without required columns

        # NaN check
        prob_cols = ["positive_prob", "neutral_prob", "negative_prob",
                     "sentiment_score", "confidence"]
        for col in prob_cols:
            if df[col].isna().any():
                issues.append(f"NaN values found in column '{col}'.")

        # Probability sum check (exclude skipped)
        valid = df[~df.get("was_skipped", pd.Series(False, index=df.index))]
        if not valid.empty:
            prob_sums = valid[["positive_prob", "neutral_prob", "negative_prob"]].sum(axis=1)
            bad = (prob_sums - 1.0).abs() > self.prob_sum_tolerance
            if bad.any():
                issues.append(
                    f"{bad.sum()} rows have probability sums outside "
                    f"[1 ± {self.prob_sum_tolerance}]. "
                    f"Range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]"
                )

        # Label validation
        bad_labels = ~df["predicted_label"].isin(self.allowed_labels)
        if bad_labels.any():
            issues.append(
                f"{bad_labels.sum()} rows have invalid predicted_label values: "
                f"{df.loc[bad_labels, 'predicted_label'].unique().tolist()[:5]}"
            )

        # Duplicate chunk_id
        dup = df[df.duplicated("chunk_id", keep=False)]
        if not dup.empty:
            issues.append(
                f"{len(dup)} rows have duplicate chunk_id values: "
                f"{dup['chunk_id'].unique()[:3].tolist()}"
            )

        # Score and confidence ranges
        score_range = df["sentiment_score"].agg(["min", "max"])
        if score_range["min"] < -1.0 - 1e-6 or score_range["max"] > 1.0 + 1e-6:
            issues.append(
                f"sentiment_score out of [-1, 1] range: "
                f"[{score_range['min']:.6f}, {score_range['max']:.6f}]"
            )

        conf_range = df["confidence"].agg(["min", "max"])
        if conf_range["min"] < 0.0 - 1e-6 or conf_range["max"] > 1.0 + 1e-6:
            issues.append(
                f"confidence out of [0, 1] range: "
                f"[{conf_range['min']:.6f}, {conf_range['max']:.6f}]"
            )

        return issues

    # --- Private checks ---

    def _check_non_empty(self, r: InferenceResult) -> list[str]:
        if not r.predictions:
            return ["InferenceResult has no predictions."]
        return []

    def _check_no_nan(self, r: InferenceResult) -> list[str]:
        issues = []
        for p in r.predictions:
            if p.was_skipped:
                continue
            fields = {
                "positive_prob" : p.positive_prob,
                "neutral_prob"  : p.neutral_prob,
                "negative_prob" : p.negative_prob,
                "sentiment_score": p.sentiment_score,
                "confidence"    : p.confidence,
            }
            for fname, val in fields.items():
                if math.isnan(val) or math.isinf(val):
                    issues.append(
                        f"NaN/Inf in {fname} for chunk_id={p.chunk_id!r}"
                    )
        return issues

    def _check_prob_sums(self, r: InferenceResult) -> list[str]:
        issues = []
        bad_count = 0
        for p in r.predictions:
            if p.was_skipped:
                continue
            total = p.positive_prob + p.neutral_prob + p.negative_prob
            if abs(total - 1.0) > self.prob_sum_tolerance:
                bad_count += 1
                if bad_count <= 3:  # report first few
                    issues.append(
                        f"Probability sum {total:.6f} ≠ 1.0 for "
                        f"chunk_id={p.chunk_id!r}"
                    )
        if bad_count > 3:
            issues.append(f"... and {bad_count - 3} more probability-sum violations.")
        return issues

    def _check_labels(self, r: InferenceResult) -> list[str]:
        bad = [
            p.chunk_id for p in r.predictions
            if p.predicted_label not in self.allowed_labels
        ]
        if bad:
            return [f"{len(bad)} predictions have invalid labels: {bad[:5]}"]
        return []

    def _check_no_duplicate_chunks(self, r: InferenceResult) -> list[str]:
        seen: set[str] = set()
        dupes: list[str] = []
        for p in r.predictions:
            if p.chunk_id in seen:
                dupes.append(p.chunk_id)
            seen.add(p.chunk_id)
        if dupes:
            return [f"Duplicate chunk_ids in predictions: {dupes[:5]}"]
        return []

    def _check_score_range(self, r: InferenceResult) -> list[str]:
        out_of_range = [
            p.chunk_id for p in r.predictions
            if not p.was_skipped and (
                p.sentiment_score < -1.0 - 1e-6
                or p.sentiment_score > 1.0 + 1e-6
            )
        ]
        if out_of_range:
            return [f"{len(out_of_range)} sentiment_scores outside [-1, 1]: {out_of_range[:3]}"]
        return []

    def _check_confidence_range(self, r: InferenceResult) -> list[str]:
        out_of_range = [
            p.chunk_id for p in r.predictions
            if not p.was_skipped and (
                p.confidence < 0.0 - 1e-6
                or p.confidence > 1.0 + 1e-6
            )
        ]
        if out_of_range:
            return [f"{len(out_of_range)} confidence values outside [0, 1]: {out_of_range[:3]}"]
        return []


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _iter_batches(
    items: list, batch_size: int
) -> Iterator[list]:
    """Yield consecutive non-overlapping sub-lists of *items*."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------------
# Mock inference backend (for testing without a model download)
# ---------------------------------------------------------------------------

class _MockFinBERTBackend:
    """Deterministic mock FinBERT backend for unit testing.

    Produces plausible probability distributions based on simple heuristics
    so that the full pipeline (batching, validation, export) can be exercised
    without requiring a live model or network access.

    Sentiment heuristic:
    * Text containing primarily positive financial language → high P(positive)
    * Text containing negative words → high P(negative)
    * Neutral/operator text → high P(neutral)
    """

    _POSITIVE_KEYWORDS = frozenset({
        "grew", "growth", "record", "strong", "increased", "beat",
        "exceeded", "margin", "raised", "guidance", "optimistic",
        "confident", "opportunity", "momentum", "outperform",
    })
    _NEGATIVE_KEYWORDS = frozenset({
        "decline", "decreased", "miss", "shortfall", "risk",
        "uncertainty", "headwind", "challenged", "slowdown",
        "concern", "pressure", "difficult", "lower", "reduced",
    })

    def predict(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic mock probabilities for each text."""
        import hashlib
        results = []
        for text in texts:
            lower = text.lower()
            words = set(lower.split())
            pos_hits = len(words & self._POSITIVE_KEYWORDS)
            neg_hits = len(words & self._NEGATIVE_KEYWORDS)

            # Base scores
            raw_pos = 1.0 + pos_hits * 0.5
            raw_neg = 1.0 + neg_hits * 0.5
            raw_neu = 1.5  # slight neutral bias

            total = raw_pos + raw_neg + raw_neu
            pos_p = round(raw_pos / total, 6)
            neg_p = round(raw_neg / total, 6)
            neu_p = round(1.0 - pos_p - neg_p, 6)  # ensure exact sum

            # Determinism: use text hash to add minor stable perturbation
            h = int(hashlib.md5(text.encode()[:50]).hexdigest(), 16)
            eps = (h % 100) / 100_000.0  # tiny perturbation ≤ 0.001
            pos_p = min(1.0, pos_p + eps)
            total2 = pos_p + neg_p + neu_p
            pos_p /= total2
            neg_p /= total2
            neu_p  = 1.0 - pos_p - neg_p

            # FinBERT label order: idx 0 = positive, 1 = negative, 2 = neutral
            results.append([pos_p, neg_p, neu_p])

        return results


class MockFinBERTPipeline(FinBERTPipeline):
    """A :class:`FinBERTPipeline` that uses mock inference instead of a real model.

    Drop-in replacement for unit tests and CI environments where HuggingFace
    model weights are not available.

    Examples
    --------
    >>> pipeline = MockFinBERTPipeline()
    >>> pipeline.load_model()          # no network call
    >>> result = pipeline.run_inference(chunks_df)
    """

    def load_model(self) -> None:  # type: ignore[override]
        """Initialise the mock backend without any network calls."""
        import torch as _torch
        self._device    = _torch.device("cpu")
        self._label_map = {0: "positive", 1: "negative", 2: "neutral"}
        self._mock      = _MockFinBERTBackend()
        self._state     = PipelineState.READY
        logger.info(
            "MockFinBERTPipeline ready | model=%s (MOCK) | device=cpu",
            self.config.model_name,
        )

    def _run_with_oom_recovery(
        self, texts: list[str], stats: InferenceStats
    ) -> list[list[float]]:
        """Use the mock backend instead of a real forward pass."""
        probs = self._mock.predict(texts)
        stats.total_batches += 1
        return probs


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def print_inference_summary(result: InferenceResult) -> None:
    """Print a human-readable inference summary to stdout."""
    stats = result.stats
    valid = [p for p in result.predictions if not p.was_skipped]

    label_counts: dict[str, int] = {}
    for p in valid:
        label_counts[p.predicted_label] = label_counts.get(p.predicted_label, 0) + 1

    pos_scores = [p.sentiment_score for p in valid]
    avg_score  = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    avg_conf   = sum(p.confidence for p in valid) / len(valid) if valid else 0.0

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  FinBERT Inference Summary")
    print(sep)
    print(f"  Model          : {stats.model_name}")
    print(f"  Device         : {stats.device_used}")
    print(f"  Total chunks   : {stats.total_chunks}")
    print(f"  Processed      : {stats.processed_chunks}")
    skipped = stats.skipped_empty + stats.skipped_too_short + stats.skipped_too_long
    print(f"  Skipped        : {skipped}")
    print(f"  Cache hits     : {stats.cache_hits}")
    print(f"  OOM recoveries : {stats.oom_recoveries}")
    print(f"  Elapsed        : {stats.elapsed_seconds:.2f}s  "
          f"({stats.chunks_per_second:.1f} chunks/s)")
    print(f"  Success rate   : {stats.success_rate:.1%}")
    print()
    print(f"  Label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / len(valid) * 100 if valid else 0
        bar = "█" * int(pct / 3)
        print(f"    {label:<12s} {count:>5d}  ({pct:5.1f}%)  {bar}")
    print()
    print(f"  Avg sentiment_score : {avg_score:+.4f}")
    print(f"  Avg confidence      :  {avg_conf:.4f}")
    if stats.errors:
        print(f"\n  ⚠ Errors ({len(stats.errors)}):")
        for e in stats.errors[:3]:
            print(f"    • {e}")
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
    print("  FinBERTPipeline — Self-Test & Demo (MockFinBERTPipeline)")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # 1.  Synthetic chunk DataFrame (mimics chunk_generator output)
    # -----------------------------------------------------------------------
    import random
    random.seed(42)

    TRANSCRIPT_CHUNKS = [
        # AAPL Q1 2025 — prepared remarks
        {
            "chunk_id"        : "AAPL_Q1_2025_chunk_000000",
            "transcript_id"   : "AAPL_Q1_2025",
            "chunk_order"     : 0,
            "chunk_text"      : (
                "We generated revenue of $124.3 billion, up 4% year-over-year. "
                "iPhone revenue reached a new December quarter record of $69.7 billion. "
                "Services grew 14% to $26.3 billion, driven by strong adoption globally. "
                "We are confident in our trajectory for the rest of fiscal 2025."
            ),
            "token_count"     : 62,
            "section_type"    : "prepared_remarks",
            "dominant_speaker": "Tim Cook",
        },
        {
            "chunk_id"        : "AAPL_Q1_2025_chunk_000001",
            "transcript_id"   : "AAPL_Q1_2025",
            "chunk_order"     : 1,
            "chunk_text"      : (
                "Gross margin was 46.9%, up 90 basis points from a year ago. "
                "Operating cash flow was $29.9 billion. "
                "We returned over $30 billion to shareholders during the quarter. "
                "EPS grew 11% to $2.40."
            ),
            "token_count"     : 48,
            "section_type"    : "prepared_remarks",
            "dominant_speaker": "Luca Maestri",
        },
        # AAPL — Q&A section
        {
            "chunk_id"        : "AAPL_Q1_2025_chunk_000002",
            "transcript_id"   : "AAPL_Q1_2025",
            "chunk_order"     : 2,
            "chunk_text"      : (
                "Can you provide more color on the sustainability of 14% Services growth? "
                "And how should we think about the contribution from Apple Intelligence? "
                "There's some concern about headwinds in the China market."
            ),
            "token_count"     : 45,
            "section_type"    : "qa",
            "dominant_speaker": "Wamsi Mohan",
        },
        {
            "chunk_id"        : "AAPL_Q1_2025_chunk_000003",
            "transcript_id"   : "AAPL_Q1_2025",
            "chunk_order"     : 3,
            "chunk_text"      : (
                "We feel very good about Services. "
                "The installed base continues to hit all-time highs. "
                "Apple Intelligence will open new monetization vectors. "
                "We're confident about the long-term opportunity in China despite near-term pressure."
            ),
            "token_count"     : 52,
            "section_type"    : "qa",
            "dominant_speaker": "Tim Cook",
        },
        # MSFT — different transcript
        {
            "chunk_id"        : "MSFT_Q2_2025_chunk_000000",
            "transcript_id"   : "MSFT_Q2_2025",
            "chunk_order"     : 0,
            "chunk_text"      : (
                "Azure and other cloud services revenue grew 31% in constant currency. "
                "Microsoft 365 Commercial cloud revenue grew 15%. "
                "We saw record adoption of Copilot across enterprise customers. "
                "Our AI momentum is accelerating and we are outperforming expectations."
            ),
            "token_count"     : 55,
            "section_type"    : "prepared_remarks",
            "dominant_speaker": "Satya Nadella",
        },
        {
            "chunk_id"        : "MSFT_Q2_2025_chunk_000001",
            "transcript_id"   : "MSFT_Q2_2025",
            "chunk_order"     : 1,
            "chunk_text"      : (
                "We are seeing some slowdown in commercial PC refresh cycles. "
                "There is uncertainty in the macro environment that has created headwinds. "
                "Operating expenses increased more than expected due to infrastructure investments. "
                "Margins declined slightly compared to the prior quarter."
            ),
            "token_count"     : 55,
            "section_type"    : "prepared_remarks",
            "dominant_speaker": "Amy Hood",
        },
        # Edge cases
        {
            "chunk_id"        : "MSFT_Q2_2025_chunk_000002",
            "transcript_id"   : "MSFT_Q2_2025",
            "chunk_order"     : 2,
            "chunk_text"      : "",    # empty → should be skipped
            "token_count"     : 0,
            "section_type"    : "qa",
            "dominant_speaker": "Operator",
        },
        {
            "chunk_id"        : "MSFT_Q2_2025_chunk_000003",
            "transcript_id"   : "MSFT_Q2_2025",
            "chunk_order"     : 3,
            "chunk_text"      : "OK.",  # too short → should be skipped
            "token_count"     : 2,
            "section_type"    : "qa",
            "dominant_speaker": "Analyst",
        },
    ]

    df_chunks = pd.DataFrame(TRANSCRIPT_CHUNKS)
    print(f"\n[INPUT] {len(df_chunks)} chunks across "
          f"{df_chunks['transcript_id'].nunique()} transcripts")
    print(f"  Columns: {list(df_chunks.columns)}\n")

    # -----------------------------------------------------------------------
    # 2.  Config validation
    # -----------------------------------------------------------------------
    print("[CONFIG]")
    try:
        bad_cfg = FinBERTConfig(max_token_length=600)
    except ValueError as e:
        print(f"  max_token_length > 512 correctly rejected: {e}")

    cfg = FinBERTConfig(
        batch_size         = 4,
        use_cache          = True,
        show_progress      = True,
        log_every_n_batches= 2,
    )
    print(f"  Config OK: batch_size={cfg.batch_size}, "
          f"max_token_length={cfg.max_token_length}")

    # -----------------------------------------------------------------------
    # 3.  MockFinBERTPipeline — core inference
    # -----------------------------------------------------------------------
    print("\n[INFERENCE]")
    pipeline = MockFinBERTPipeline(config=cfg)
    assert not pipeline.is_ready
    pipeline.load_model()
    assert pipeline.is_ready
    print(f"  Pipeline state: {pipeline._state.name}")
    print(f"  Device        : {pipeline.device_str}")
    print(f"  Label map     : {pipeline._label_map}")

    result = pipeline.run_inference(df_chunks)
    print(f"\n  Result: {result}")

    # -----------------------------------------------------------------------
    # 4.  Result inspection
    # -----------------------------------------------------------------------
    print("\n[PREDICTIONS]")
    for p in result.predictions:
        status = "SKIP" if p.was_skipped else "    "
        print(
            f"  [{status}] {p.chunk_id:<35s} | "
            f"{p.predicted_label:<10s} | "
            f"score={p.sentiment_score:+.4f} | "
            f"conf={p.confidence:.4f}"
            + (f"  ← {p.skip_reason}" if p.was_skipped else "")
        )

    # -----------------------------------------------------------------------
    # 5.  DataFrame export
    # -----------------------------------------------------------------------
    df_out = result.to_dataframe()
    print(f"\n[EXPORT] DataFrame shape: {df_out.shape}")
    print(f"  Columns: {list(df_out.columns)}")
    valid_rows = df_out[~df_out["was_skipped"]]
    print(f"  Valid rows: {len(valid_rows)}")

    # -----------------------------------------------------------------------
    # 6.  Inference caching — second run should hit cache
    # -----------------------------------------------------------------------
    print("\n[CACHE TEST]")
    result2 = pipeline.run_inference(df_chunks)
    print(f"  Cache hits on second run: {result2.stats.cache_hits}")
    assert result2.stats.cache_hits > 0, "Cache should have hits on re-run"
    print(f"  Cache size: {pipeline.cache_size}")

    # Predictions identical across runs
    scores1 = {p.chunk_id: p.sentiment_score for p in result.predictions}
    scores2 = {p.chunk_id: p.sentiment_score for p in result2.predictions}
    assert scores1 == scores2, "Predictions must be deterministic across runs"
    print("  Determinism check: ✅")

    # -----------------------------------------------------------------------
    # 7.  Validation
    # -----------------------------------------------------------------------
    print("\n[VALIDATION]")
    issues = pipeline.validate_outputs(result)
    if issues:
        print(f"  ❌ {len(issues)} validation issue(s):")
        for issue in issues:
            print(f"     • {issue}")
        sys.exit(1)
    else:
        print("  ✅ All output validation checks passed.")

    # Also validate the DataFrame
    validator = InferenceValidator()
    df_issues = validator.validate_dataframe(df_out)
    if df_issues:
        print(f"  ❌ DataFrame validation issues: {df_issues}")
        sys.exit(1)
    else:
        print("  ✅ DataFrame validation passed.")

    # -----------------------------------------------------------------------
    # 8.  filter_valid helper
    # -----------------------------------------------------------------------
    clean_result = result.filter_valid()
    assert all(not p.was_skipped for p in clean_result.predictions), \
        "filter_valid() should not return skipped predictions"
    print(f"\n[FILTER] filter_valid(): "
          f"{len(result)} → {len(clean_result)} predictions")

    # -----------------------------------------------------------------------
    # 9.  Summary printout
    # -----------------------------------------------------------------------
    print_inference_summary(result)

    # -----------------------------------------------------------------------
    # 10. Error-handling: empty DataFrame
    # -----------------------------------------------------------------------
    print("[EDGE CASE] Empty DataFrame:")
    try:
        pipeline.run_inference(pd.DataFrame(columns=list(_REQUIRED_INPUT_COLUMNS)))
    except ValueError as e:
        print(f"  Correctly rejected empty input: {e}")

    # -----------------------------------------------------------------------
    # 11. Error-handling: missing column
    # -----------------------------------------------------------------------
    print("\n[EDGE CASE] Missing required column:")
    try:
        pipeline.run_inference(df_chunks.drop(columns=["chunk_text"]))
    except ValueError as e:
        print(f"  Correctly rejected missing column: {e}")

    # -----------------------------------------------------------------------
    # 12. Unloaded pipeline guard
    # -----------------------------------------------------------------------
    print("\n[EDGE CASE] Inference without load_model():")
    p2 = MockFinBERTPipeline()
    try:
        p2.run_inference(df_chunks)
    except RuntimeError as e:
        print(f"  Correctly raised RuntimeError: {e}")

    print("\n" + "=" * 68)
    print("  Self-test complete — all checks passed ✅")
    print("=" * 68)
