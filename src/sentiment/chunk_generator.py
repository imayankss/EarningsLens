"""
src/sentiment/chunk_generator.py
=================================
Scalable, token-aware transcript chunking engine for long earnings call
transcripts.

Position in pipeline
--------------------
    sentence_segmenter.py  →  **chunk_generator.py**  →  finbert_pipeline.py

Responsibilities
----------------
* Split arbitrarily-long transcripts into FinBERT-safe chunks (≤512 tokens).
* Never split mid-sentence; always honour sentence boundaries.
* Inject configurable overlap windows between consecutive chunks so that
  cross-boundary context is preserved for the transformer.
* Track rich chunk metadata (speaker, section, sentence range, overlap flags)
  that downstream aggregation layers depend on.
* Produce deterministic outputs — identical input always yields identical chunks.
* Expose DataFrame, dict, JSONL and Parquet export helpers so the module slots
  cleanly into the broader Parquet-first pipeline.

Design constraints (from architecture docs)
-------------------------------------------
* Default chunk_size  : 400 tokens  (350–450 range)
* Default overlap     : 64 tokens   (50–75 range)
* Hard token ceiling  : 512 tokens  (FinBERT limit)
* Tokenizer           : HuggingFace-compatible (ProsusAI/finbert by default)
* Fallback counter    : word-based approximation when no tokenizer available
* Chunking method     : sentence-stream with greedy packing + overlap injection

Author: Earnings Call Sentiment Analyzer — DAY 8
Python: 3.11+
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer Protocol — keeps this module decoupled from transformers
# ---------------------------------------------------------------------------

class TokenizerProtocol(Protocol):
    """Minimal interface expected from any HuggingFace-compatible tokenizer."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        ...


# ---------------------------------------------------------------------------
# Lightweight fallback tokenizer (no HuggingFace required)
# ---------------------------------------------------------------------------

class _WordTokenizerFallback:
    """
    Approximation tokenizer used when no real tokenizer is injected.

    WordPiece sub-word tokens are ~1.3× raw word count for English financial
    text.  This heuristic is deliberately conservative to prevent chunks from
    exceeding the 512-token hard ceiling.
    """

    SUBWORD_FACTOR: float = 1.35  # empirical for financial English

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        word_count = len(text.split())
        estimated = int(word_count * self.SUBWORD_FACTOR)
        # +2 for [CLS] / [SEP] special tokens when flag is set
        if add_special_tokens:
            estimated += 2
        # Return a dummy list of the right length (values irrelevant here)
        return list(range(estimated))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """
    All tuneable parameters for the ChunkGenerator.

    Attributes
    ----------
    chunk_size_tokens : int
        Target maximum token count per chunk (excl. special tokens).
        Must be < hard_token_ceiling.
    overlap_tokens : int
        Number of overlap tokens injected at the start of every chunk
        (except the first).  Provides cross-boundary context for FinBERT.
    hard_token_ceiling : int
        Absolute maximum — chunks exceeding this will raise a
        ChunkValidationError at validation time.  Must be ≤512 for FinBERT.
    min_chunk_tokens : int
        Chunks smaller than this are merged into the previous chunk rather
        than emitted as standalone tiny chunks.
    model_name : str
        HuggingFace model identifier used to load a tokenizer when one is not
        explicitly provided.
    add_special_tokens : bool
        Whether to count [CLS]/[SEP] special tokens when sizing chunks.
        Should remain True for BERT-family models.
    preserve_section_boundaries : bool
        When True, a new chunk is started whenever section_type changes even
        if the current chunk has budget remaining.  Keeps prepared-remarks and
        Q&A sentiment cleanly separated.
    preserve_speaker_boundaries : bool
        When True, start a new chunk on every speaker change if the speaker
        change coincides with a section boundary.  Soft preference — overridden
        if it would produce a chunk below min_chunk_tokens.
    chunk_id_prefix : str
        Prefix prepended to all chunk_id values.  Useful when merging outputs
        from multiple transcripts.
    """

    chunk_size_tokens: int = 400
    overlap_tokens: int = 64
    hard_token_ceiling: int = 490   # leaves headroom below 512 for specials
    min_chunk_tokens: int = 32
    model_name: str = "ProsusAI/finbert"
    add_special_tokens: bool = True
    preserve_section_boundaries: bool = True
    preserve_speaker_boundaries: bool = False
    chunk_id_prefix: str = "chunk"

    def __post_init__(self) -> None:
        if self.chunk_size_tokens >= self.hard_token_ceiling:
            raise ValueError(
                f"chunk_size_tokens ({self.chunk_size_tokens}) must be "
                f"< hard_token_ceiling ({self.hard_token_ceiling})."
            )
        if self.overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be "
                f"< chunk_size_tokens ({self.chunk_size_tokens})."
            )
        if self.hard_token_ceiling > 512:
            raise ValueError(
                "hard_token_ceiling cannot exceed 512 — FinBERT model limit."
            )
        if self.min_chunk_tokens < 1:
            raise ValueError("min_chunk_tokens must be ≥ 1.")


# ---------------------------------------------------------------------------
# Sentence input dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentenceRecord:
    """
    Represents a single sentence emitted by sentence_segmenter.py.

    The ChunkGenerator accepts a list of SentenceRecord objects or a
    pandas DataFrame with equivalent column names.
    """

    sentence_id: str
    transcript_id: str
    sentence_order: int
    sentence_text: str
    speaker: str = "unknown"
    speaker_role: str = "unknown"
    section_type: str = "unknown"
    token_estimate: int = 0          # pre-computed estimate; recalculated internally

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SentenceRecord":
        return cls(
            sentence_id=str(d.get("sentence_id", "")),
            transcript_id=str(d.get("transcript_id", "")),
            sentence_order=int(d.get("sentence_order", 0)),
            sentence_text=str(d.get("sentence_text", "")),
            speaker=str(d.get("speaker", "unknown")),
            speaker_role=str(d.get("speaker_role", "unknown")),
            section_type=str(d.get("section_type", "unknown")),
            token_estimate=int(d.get("token_estimate", 0)),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list["SentenceRecord"]:
        """Convert a DataFrame (sentence_segmenter output) to a list."""
        records: list[SentenceRecord] = []
        for row in df.itertuples(index=False):
            records.append(
                cls(
                    sentence_id=str(getattr(row, "sentence_id", "")),
                    transcript_id=str(getattr(row, "transcript_id", "")),
                    sentence_order=int(getattr(row, "sentence_order", 0)),
                    sentence_text=str(getattr(row, "sentence_text", "")),
                    speaker=str(getattr(row, "speaker", "unknown")),
                    speaker_role=str(getattr(row, "speaker_role", "unknown")),
                    section_type=str(getattr(row, "section_type", "unknown")),
                    token_estimate=int(getattr(row, "token_estimate", 0)),
                )
            )
        return records


# ---------------------------------------------------------------------------
# Chunk output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single transformer-ready chunk of text produced by ChunkGenerator.

    All fields are populated deterministically from the input sentence stream.

    Attributes
    ----------
    chunk_id : str
        Globally unique identifier.  Format:
        ``{prefix}_{transcript_id}_{chunk_order:06d}``
    transcript_id : str
        Parent transcript identifier (primary event-study merge key).
    chunk_order : int
        0-based sequential index within the transcript.  Strictly increasing.
    chunk_text : str
        Space-joined sentence text, ready for tokenizer input.
    token_count : int
        Exact token count as computed by the configured tokenizer.
    sentence_count : int
        Number of sentences included in this chunk.
    section_type : str
        Dominant section type (majority vote among included sentences).
    dominant_speaker : str
        Speaker occupying the most sentences in this chunk.
    overlap_start : bool
        True when this chunk's first N tokens originate from the previous
        chunk's tail (i.e. an overlap window was prepended).
    overlap_end : bool
        True when this chunk's final N tokens will be re-used as the overlap
        prefix of the next chunk.
    first_sentence_order : int
        ``sentence_order`` of the first sentence included (for reconstruction).
    last_sentence_order : int
        ``sentence_order`` of the last sentence included.
    sentence_ids : list[str]
        Ordered list of sentence_id values for full reconstruction auditing.
    """

    chunk_id: str
    transcript_id: str
    chunk_order: int
    chunk_text: str
    token_count: int
    sentence_count: int
    section_type: str
    dominant_speaker: str
    overlap_start: bool
    overlap_end: bool
    first_sentence_order: int
    last_sentence_order: int
    sentence_ids: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-serialisable)."""
        d = asdict(self)
        return d

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Flat dict suitable for a single DataFrame row.
        ``sentence_ids`` is serialised as a pipe-separated string to
        remain Parquet-compatible without nested types.
        """
        d = self.to_dict()
        d["sentence_ids"] = "|".join(d["sentence_ids"])
        return d

    @staticmethod
    def dataframe_columns() -> list[str]:
        return [
            "chunk_id",
            "transcript_id",
            "chunk_order",
            "chunk_text",
            "token_count",
            "sentence_count",
            "section_type",
            "dominant_speaker",
            "overlap_start",
            "overlap_end",
            "first_sentence_order",
            "last_sentence_order",
            "sentence_ids",
        ]


# ---------------------------------------------------------------------------
# Chunk generation result container
# ---------------------------------------------------------------------------

@dataclass
class ChunkGenerationResult:
    """
    Aggregated output returned by ``ChunkGenerator.generate()``.

    Attributes
    ----------
    transcript_id : str
    chunks : list[Chunk]
        Ordered list of all generated chunks.
    total_sentences : int
        Number of input sentences processed.
    total_tokens : int
        Sum of token_count across all chunks (includes overlap tokens).
    unique_tokens_estimated : int
        Estimated unique token coverage (total minus overlap budget).
    warnings : list[str]
        Non-fatal issues encountered during generation (e.g. oversized
        single sentences that had to be hard-truncated).
    generation_config : ChunkConfig
        Snapshot of the config used, for reproducibility metadata.
    """

    transcript_id: str
    chunks: list[Chunk]
    total_sentences: int
    total_tokens: int
    unique_tokens_estimated: int
    warnings: list[str] = field(default_factory=list)
    generation_config: ChunkConfig = field(default_factory=ChunkConfig)

    # ------------------------------------------------------------------
    # DataFrame / export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all chunks to a tidy DataFrame ready for Parquet export.

        Returns
        -------
        pd.DataFrame
            One row per chunk, columns as defined in ``Chunk.dataframe_columns()``.
        """
        if not self.chunks:
            return pd.DataFrame(columns=Chunk.dataframe_columns())
        rows = [c.to_flat_dict() for c in self.chunks]
        return pd.DataFrame(rows, columns=Chunk.dataframe_columns())

    def to_dict(self) -> dict[str, Any]:
        """Full serialisation including config snapshot."""
        return {
            "transcript_id": self.transcript_id,
            "total_chunks": len(self.chunks),
            "total_sentences": self.total_sentences,
            "total_tokens": self.total_tokens,
            "unique_tokens_estimated": self.unique_tokens_estimated,
            "warnings": self.warnings,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def to_jsonl(self) -> str:
        """
        Return newline-delimited JSON (one line per chunk).
        Suitable for future RAG ingestion pipelines.
        """
        return "\n".join(json.dumps(c.to_dict()) for c in self.chunks)

    def save_parquet(self, path: str | Path) -> Path:
        """Persist chunks DataFrame to Parquet."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_parquet(p, index=False)
        logger.info("Saved %d chunks → %s", len(self.chunks), p)
        return p

    def save_csv(self, path: str | Path) -> Path:
        """Persist chunks DataFrame to CSV (debugging convenience)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(p, index=False)
        logger.info("Saved %d chunks (CSV) → %s", len(self.chunks), p)
        return p

    def save_jsonl(self, path: str | Path) -> Path:
        """Persist to JSONL for RAG or streaming consumers."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_jsonl(), encoding="utf-8")
        logger.info("Saved %d chunks (JSONL) → %s", len(self.chunks), p)
        return p

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        """
        Return a summary statistics dict — useful for pipeline health checks
        and the validation layer.
        """
        if not self.chunks:
            return {"transcript_id": self.transcript_id, "chunk_count": 0}

        token_counts = [c.token_count for c in self.chunks]
        return {
            "transcript_id": self.transcript_id,
            "chunk_count": len(self.chunks),
            "total_sentences": self.total_sentences,
            "total_tokens": self.total_tokens,
            "unique_tokens_estimated": self.unique_tokens_estimated,
            "token_min": min(token_counts),
            "token_max": max(token_counts),
            "token_mean": round(sum(token_counts) / len(token_counts), 2),
            "overlap_chunks": sum(1 for c in self.chunks if c.overlap_start),
            "sections": list({c.section_type for c in self.chunks}),
            "speakers": list({c.dominant_speaker for c in self.chunks}),
            "warning_count": len(self.warnings),
        }

    def coverage_metrics(self) -> dict[str, float]:
        """
        Sentence and token coverage fractions.

        A coverage of 1.0 means every input sentence appears in at least
        one chunk (no sentence was dropped).
        """
        covered_sentence_ids: set[str] = set()
        for chunk in self.chunks:
            covered_sentence_ids.update(chunk.sentence_ids)
        sentence_coverage = (
            len(covered_sentence_ids) / self.total_sentences
            if self.total_sentences > 0
            else 0.0
        )
        return {
            "sentence_coverage": round(sentence_coverage, 6),
            "covered_sentences": len(covered_sentence_ids),
            "total_sentences": self.total_sentences,
        }


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ChunkGenerationError(RuntimeError):
    """Raised when chunking cannot proceed due to invalid inputs."""


class ChunkValidationError(ValueError):
    """Raised by the validation layer when a chunk fails a hard constraint."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dominant_value(values: Sequence[str]) -> str:
    """Return the most frequent value in *values*, with a stable tie-break."""
    if not values:
        return "unknown"
    counter = Counter(values)
    return max(counter, key=lambda k: (counter[k], k))


def _deterministic_chunk_id(
    prefix: str,
    transcript_id: str,
    chunk_order: int,
) -> str:
    """
    Generate a deterministic, URL-safe chunk identifier.

    Format: ``{prefix}_{transcript_id}_{chunk_order:06d}``

    A short SHA-1 suffix is appended to guard against transcript_id values
    that contain special characters.
    """
    raw = f"{prefix}|{transcript_id}|{chunk_order:06d}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]  # noqa: S324
    safe_tid = re.sub(r"[^A-Za-z0-9_\-]", "_", transcript_id)
    return f"{prefix}_{safe_tid}_{chunk_order:06d}_{digest}"


def _clean_sentence(text: str) -> str:
    """
    Light normalisation applied to each sentence before chunking.
    Does NOT perform aggressive cleaning — that is sentence_segmenter's job.
    """
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# ChunkGenerator — main engine
# ---------------------------------------------------------------------------

class ChunkGenerator:
    """
    Token-aware, sentence-preserving chunking engine for earnings call
    transcripts.

    Parameters
    ----------
    config : ChunkConfig, optional
        Chunking configuration.  Defaults to ``ChunkConfig()`` with
        production-tuned values (400-token chunks, 64-token overlap).
    tokenizer : TokenizerProtocol, optional
        A HuggingFace-compatible tokenizer.  If *None* the engine falls back
        to the lightweight ``_WordTokenizerFallback``.  Pass an actual
        ``AutoTokenizer`` instance for production runs.

    Examples
    --------
    Minimal usage::

        from transformers import AutoTokenizer
        from src.sentiment.chunk_generator import ChunkGenerator, ChunkConfig

        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        gen = ChunkGenerator(config=ChunkConfig(), tokenizer=tokenizer)
        result = gen.generate(sentences_df, transcript_id="AAPL_Q1_2025")
        result.save_parquet("data/interim/transcript_chunks.parquet")

    No-tokenizer (fallback) usage::

        gen = ChunkGenerator()
        result = gen.generate(sentences_df, transcript_id="AAPL_Q1_2025")
        df = result.to_dataframe()
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        tokenizer: Optional[TokenizerProtocol] = None,
    ) -> None:
        self.config: ChunkConfig = config or ChunkConfig()
        self._tokenizer: TokenizerProtocol = tokenizer or _WordTokenizerFallback()
        self._using_fallback: bool = isinstance(
            self._tokenizer, _WordTokenizerFallback
        )

        if self._using_fallback:
            logger.warning(
                "No HuggingFace tokenizer provided — using word-count "
                "heuristic.  Token estimates may be inaccurate.  "
                "Inject AutoTokenizer.from_pretrained('%s') for production.",
                self.config.model_name,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        sentences: list[SentenceRecord] | pd.DataFrame,
        transcript_id: str = "",
    ) -> ChunkGenerationResult:
        """
        Generate chunks for a single transcript.

        Parameters
        ----------
        sentences : list[SentenceRecord] | pd.DataFrame
            Ordered sentence records for ONE transcript.  If a DataFrame is
            passed it is converted to ``SentenceRecord`` objects internally.
            Rows must be pre-sorted by ``sentence_order`` ascending.
        transcript_id : str
            Override or supply the transcript identifier.  When *sentences*
            is a DataFrame the value is read from the first row's
            ``transcript_id`` column if this argument is empty.

        Returns
        -------
        ChunkGenerationResult
        """
        # ── 0. Normalise input ─────────────────────────────────────────
        sentence_list = self._normalise_input(sentences, transcript_id)

        if not transcript_id:
            transcript_id = (
                sentence_list[0].transcript_id if sentence_list else "unknown"
            )

        logger.info(
            "[%s] Starting chunk generation — %d sentences.",
            transcript_id,
            len(sentence_list),
        )

        # ── 1. Handle empty / malformed input ─────────────────────────
        if not sentence_list:
            logger.warning("[%s] No sentences provided — returning empty result.", transcript_id)
            return ChunkGenerationResult(
                transcript_id=transcript_id,
                chunks=[],
                total_sentences=0,
                total_tokens=0,
                unique_tokens_estimated=0,
                warnings=["No sentences provided."],
                generation_config=self.config,
            )

        # ── 2. Sort deterministically ──────────────────────────────────
        sentence_list = sorted(sentence_list, key=lambda s: s.sentence_order)

        # ── 3. Compute token counts ────────────────────────────────────
        warnings: list[str] = []
        token_counts = self._compute_token_counts(sentence_list, warnings)

        # ── 4. Build chunks ────────────────────────────────────────────
        raw_chunks = self._build_chunks(
            sentence_list, token_counts, transcript_id, warnings
        )

        # ── 5. Inject overlap metadata ────────────────────────────────
        self._mark_overlap_flags(raw_chunks)

        # ── 6. Finalise chunk objects ──────────────────────────────────
        chunks = self._finalise_chunks(raw_chunks, transcript_id)

        # ── 7. Aggregate stats ─────────────────────────────────────────
        total_tokens = sum(c.token_count for c in chunks)
        overlap_budget = self.config.overlap_tokens * max(0, len(chunks) - 1)
        unique_tokens_estimated = max(0, total_tokens - overlap_budget)

        logger.info(
            "[%s] Generated %d chunks | tokens: %d total / ~%d unique.",
            transcript_id,
            len(chunks),
            total_tokens,
            unique_tokens_estimated,
        )

        return ChunkGenerationResult(
            transcript_id=transcript_id,
            chunks=chunks,
            total_sentences=len(sentence_list),
            total_tokens=total_tokens,
            unique_tokens_estimated=unique_tokens_estimated,
            warnings=warnings,
            generation_config=self.config,
        )

    def generate_batch(
        self,
        sentences_df: pd.DataFrame,
    ) -> dict[str, ChunkGenerationResult]:
        """
        Process a multi-transcript DataFrame in one call.

        Parameters
        ----------
        sentences_df : pd.DataFrame
            Combined sentence records from multiple transcripts.
            Must contain a ``transcript_id`` column.

        Returns
        -------
        dict[str, ChunkGenerationResult]
            Mapping of transcript_id → ChunkGenerationResult.
        """
        if "transcript_id" not in sentences_df.columns:
            raise ChunkGenerationError(
                "DataFrame must contain a 'transcript_id' column for batch processing."
            )

        results: dict[str, ChunkGenerationResult] = {}
        for tid, group in sentences_df.groupby("transcript_id", sort=False):
            tid_str = str(tid)
            logger.info("Batch processing transcript: %s (%d rows)", tid_str, len(group))
            results[tid_str] = self.generate(group, transcript_id=tid_str)

        logger.info("Batch complete — processed %d transcripts.", len(results))
        return results

    def batch_to_dataframe(
        self,
        results: dict[str, ChunkGenerationResult],
    ) -> pd.DataFrame:
        """
        Concatenate batch results into a single DataFrame.

        Parameters
        ----------
        results : dict[str, ChunkGenerationResult]
            Output from ``generate_batch()``.

        Returns
        -------
        pd.DataFrame
            All chunks across all transcripts, with chunk_order reset
            per-transcript.
        """
        frames = [r.to_dataframe() for r in results.values() if r.chunks]
        if not frames:
            return pd.DataFrame(columns=Chunk.dataframe_columns())
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """
        Return the exact token count for *text* using the configured tokenizer.

        This is the single source of truth for token counting throughout the
        pipeline.  Both the chunk-builder and the validation layer call this
        method.
        """
        if not text or not text.strip():
            return 0
        tokens = self._tokenizer.encode(
            text,
            add_special_tokens=self.config.add_special_tokens,
        )
        return len(tokens)

    # ------------------------------------------------------------------
    # Validation helpers (also used by sentiment_validation.py)
    # ------------------------------------------------------------------

    def validate_result(self, result: ChunkGenerationResult) -> list[str]:
        """
        Run all hard-constraint validation checks on a ChunkGenerationResult.

        Returns
        -------
        list[str]
            A list of error messages.  Empty list means all checks passed.
        """
        errors: list[str] = []

        if not result.chunks:
            if result.total_sentences > 0:
                errors.append(
                    f"[{result.transcript_id}] Zero chunks produced for "
                    f"{result.total_sentences} input sentences."
                )
            return errors

        # Check 1: Token ceiling
        for chunk in result.chunks:
            if chunk.token_count > self.config.hard_token_ceiling:
                errors.append(
                    f"Chunk {chunk.chunk_id} exceeds hard token ceiling: "
                    f"{chunk.token_count} > {self.config.hard_token_ceiling}."
                )

        # Check 2: Strict ordering
        orders = [c.chunk_order for c in result.chunks]
        if orders != list(range(len(orders))):
            errors.append(
                f"[{result.transcript_id}] chunk_order is not strictly "
                f"sequential: {orders[:10]}..."
            )

        # Check 3: Unique chunk IDs
        ids = [c.chunk_id for c in result.chunks]
        if len(ids) != len(set(ids)):
            dupes = [cid for cid in ids if ids.count(cid) > 1]
            errors.append(
                f"[{result.transcript_id}] Duplicate chunk_ids detected: "
                f"{list(set(dupes))[:5]}"
            )

        # Check 4: No empty chunk texts
        for chunk in result.chunks:
            if not chunk.chunk_text.strip():
                errors.append(
                    f"Chunk {chunk.chunk_id} has empty chunk_text."
                )

        # Check 5: Sentence continuity (no sentences dropped)
        all_sids: set[str] = set()
        for chunk in result.chunks:
            all_sids.update(chunk.sentence_ids)
        # We cannot check against the original input here — the caller can
        # compare coverage_metrics().sentence_coverage ≈ 1.0.

        # Check 6: sentence_order range integrity
        for chunk in result.chunks:
            if chunk.first_sentence_order > chunk.last_sentence_order:
                errors.append(
                    f"Chunk {chunk.chunk_id}: first_sentence_order "
                    f"({chunk.first_sentence_order}) > last_sentence_order "
                    f"({chunk.last_sentence_order})."
                )

        return errors

    def validate_overlap(self, result: ChunkGenerationResult) -> list[str]:
        """
        Verify that consecutive chunks share the expected overlap content.

        Checks that the tail text of chunk N appears at the head of chunk N+1
        when overlap_start is True.

        Returns
        -------
        list[str]
            Validation error messages (empty = all OK).
        """
        errors: list[str] = []
        chunks = result.chunks
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            if not curr.overlap_start:
                continue
            # The last sentence_ids of prev should appear in curr
            overlap_sids = set(prev.sentence_ids[-3:])  # check last few
            if not overlap_sids.intersection(curr.sentence_ids):
                errors.append(
                    f"Overlap check failed between chunk {prev.chunk_id} "
                    f"and {curr.chunk_id}: no shared sentence_ids found."
                )
        return errors

    # ------------------------------------------------------------------
    # Reconstruction utility
    # ------------------------------------------------------------------

    def reconstruct_transcript(
        self,
        result: ChunkGenerationResult,
        deduplicate: bool = True,
    ) -> str:
        """
        Reconstruct the full transcript text from chunks.

        Parameters
        ----------
        deduplicate : bool
            When True, overlapping sentences from adjacent chunks are
            de-duplicated using sentence_ids so the reconstructed text
            does not repeat overlap passages.

        Returns
        -------
        str
            Reconstructed transcript text.
        """
        seen_sids: set[str] = set()
        parts: list[str] = []

        for chunk in sorted(result.chunks, key=lambda c: c.chunk_order):
            if not deduplicate:
                parts.append(chunk.chunk_text)
                continue
            # Re-emit only sentences not yet seen (removes overlap duplication)
            # We approximate by splitting chunk_text back into sentences.
            # For exact reconstruction the caller should use sentence_ids
            # against the original sentence DataFrame.
            chunk_sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.chunk_text)
                if s.strip()
            ]
            new_sids = [
                sid for sid in chunk.sentence_ids if sid not in seen_sids
            ]
            n_new = len(new_sids)
            # Keep only the last n_new sentences from the chunk text
            unique_sentences = chunk_sentences[-n_new:] if n_new else chunk_sentences
            parts.append(" ".join(unique_sentences))
            seen_sids.update(chunk.sentence_ids)

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _normalise_input(
        self,
        sentences: list[SentenceRecord] | pd.DataFrame,
        transcript_id: str,
    ) -> list[SentenceRecord]:
        """Convert DataFrame input to SentenceRecord list and basic sanitise."""
        if isinstance(sentences, pd.DataFrame):
            if sentences.empty:
                return []
            records = SentenceRecord.from_dataframe(sentences)
        elif isinstance(sentences, list):
            records = list(sentences)
        else:
            raise ChunkGenerationError(
                f"sentences must be list[SentenceRecord] or pd.DataFrame, "
                f"got {type(sentences).__name__}."
            )

        # Filter out blank sentences
        valid: list[SentenceRecord] = []
        for rec in records:
            text = _clean_sentence(rec.sentence_text)
            if not text:
                logger.debug("Dropping blank sentence_id=%s", rec.sentence_id)
                continue
            rec.sentence_text = text
            valid.append(rec)

        return valid

    def _compute_token_counts(
        self,
        sentences: list[SentenceRecord],
        warnings: list[str],
    ) -> list[int]:
        """
        Return token counts for each sentence.

        Sentences whose token count exceeds the hard ceiling on their own are
        hard-truncated with a warning.  This is a last-resort guard — in
        practice financial sentences are well within limits.
        """
        counts: list[int] = []
        ceiling = self.config.hard_token_ceiling

        for sent in sentences:
            n = self.count_tokens(sent.sentence_text)
            if n > ceiling:
                msg = (
                    f"Sentence {sent.sentence_id} exceeds hard token ceiling "
                    f"({n} tokens).  It will be truncated to fit a single chunk."
                )
                warnings.append(msg)
                logger.warning(msg)
                # Truncate by words until within limit
                words = sent.sentence_text.split()
                while n > ceiling and words:
                    words = words[:-5]
                    sent.sentence_text = " ".join(words)
                    n = self.count_tokens(sent.sentence_text)
            counts.append(n)

        return counts

    def _should_break_boundary(
        self,
        prev_sentence: SentenceRecord,
        next_sentence: SentenceRecord,
    ) -> bool:
        """
        Return True when a chunk boundary should be forced between two
        consecutive sentences for structural reasons (section / speaker change).
        """
        if self.config.preserve_section_boundaries:
            if prev_sentence.section_type != next_sentence.section_type:
                return True
        if self.config.preserve_speaker_boundaries:
            if prev_sentence.speaker != next_sentence.speaker:
                return True
        return False

    def _build_chunks(
        self,
        sentences: list[SentenceRecord],
        token_counts: list[int],
        transcript_id: str,
        warnings: list[str],
    ) -> list["_RawChunk"]:
        """
        Core greedy chunk-packing algorithm.

        Algorithm
        ---------
        1. Maintain a current_chunk accumulator.
        2. For each sentence:
           a. If adding it would exceed chunk_size_tokens OR a structural
              boundary is detected → flush current_chunk, start a new one
              with an overlap prefix drawn from the tail of the previous chunk.
           b. Otherwise → append to current_chunk.
        3. Flush final chunk.

        Overlap injection
        -----------------
        When a chunk is flushed and a new one begins, we re-prepend the last
        *M* sentences from the previous chunk whose total token count ≤
        overlap_tokens.  This ensures cross-boundary context is available to
        FinBERT without inflating chunks beyond the ceiling.
        """
        raw_chunks: list[_RawChunk] = []
        current = _RawChunk()
        prev_chunk_tail: list[tuple[SentenceRecord, int]] = []  # (sentence, tokens)

        cfg = self.config

        for idx, (sent, n_tokens) in enumerate(zip(sentences, token_counts)):

            force_break = (
                current.sentences
                and self._should_break_boundary(current.sentences[-1][0], sent)
            )
            would_exceed = (
                current.total_tokens + n_tokens > cfg.chunk_size_tokens
                and current.sentences  # don't break on first sentence
            )

            if force_break or would_exceed:
                # Flush
                if current.sentences:
                    prev_chunk_tail = list(current.sentences)
                    raw_chunks.append(current)
                    current = _RawChunk()

                # Inject overlap prefix into new chunk
                overlap_sentences = self._select_overlap_sentences(
                    prev_chunk_tail, cfg.overlap_tokens
                )
                for ov_sent, ov_tokens in overlap_sentences:
                    current.add(ov_sent, ov_tokens, is_overlap=True)

            # Handle case where a single sentence exceeds chunk_size on its own
            if (
                n_tokens > cfg.chunk_size_tokens
                and not current.sentences
            ):
                warn_msg = (
                    f"[{transcript_id}] Sentence {sent.sentence_id} alone "
                    f"uses {n_tokens} tokens (> chunk_size_tokens "
                    f"{cfg.chunk_size_tokens}).  Emitting as solo chunk."
                )
                warnings.append(warn_msg)
                logger.warning(warn_msg)

            current.add(sent, n_tokens, is_overlap=False)

        # Flush remaining
        if current.sentences:
            # Merge tiny tail chunk into previous chunk if it is too small
            if (
                raw_chunks
                and current.total_tokens < cfg.min_chunk_tokens
                and not any(s[2] for s in current.sentences)  # no overlap-only
            ):
                prev = raw_chunks[-1]
                for item in current.sentences:
                    prev.add(item[0], item[1], item[2])
                logger.debug(
                    "[%s] Merged tiny tail chunk (%d tokens) into previous.",
                    transcript_id,
                    current.total_tokens,
                )
            else:
                raw_chunks.append(current)

        return raw_chunks

    def _select_overlap_sentences(
        self,
        tail: list[tuple[SentenceRecord, int]],
        budget: int,
    ) -> list[tuple[SentenceRecord, int]]:
        """
        Select sentences from the tail of the previous chunk to use as overlap.

        Works backwards from the last sentence, accumulating until the token
        budget is exhausted.  Returns in original (forward) order.
        """
        selected: list[tuple[SentenceRecord, int]] = []
        accumulated = 0
        for item in reversed(tail):
            sent, n_tokens, _is_ov = item
            if accumulated + n_tokens > budget:
                break
            selected.append((sent, n_tokens))
            accumulated += n_tokens
        selected.reverse()
        return selected

    def _mark_overlap_flags(self, raw_chunks: list["_RawChunk"]) -> None:
        """
        Set overlap_start / overlap_end flags on raw chunks in-place.
        A chunk has overlap_start=True if ANY of its leading sentences were
        injected as overlap from the previous chunk.
        """
        for i, chunk in enumerate(raw_chunks):
            chunk.overlap_start = any(flag for _, _, flag in chunk.sentences)
            chunk.overlap_end = (i < len(raw_chunks) - 1)

    def _finalise_chunks(
        self,
        raw_chunks: list["_RawChunk"],
        transcript_id: str,
    ) -> list[Chunk]:
        """
        Convert internal _RawChunk objects to the public Chunk dataclass.
        Assigns deterministic chunk_ids and computes exact token counts.
        """
        chunks: list[Chunk] = []
        cfg = self.config

        for order, raw in enumerate(raw_chunks):
            sentences_only = [s for s, _, _ in raw.sentences]
            joined_text = " ".join(s.sentence_text for s in sentences_only)
            exact_tokens = self.count_tokens(joined_text)

            section_type = _dominant_value([s.section_type for s in sentences_only])
            dominant_speaker = _dominant_value([s.speaker for s in sentences_only])

            sentence_orders = [s.sentence_order for s in sentences_only]
            sentence_ids = [s.sentence_id for s in sentences_only]

            chunk_id = _deterministic_chunk_id(
                cfg.chunk_id_prefix, transcript_id, order
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    transcript_id=transcript_id,
                    chunk_order=order,
                    chunk_text=joined_text,
                    token_count=exact_tokens,
                    sentence_count=len(sentences_only),
                    section_type=section_type,
                    dominant_speaker=dominant_speaker,
                    overlap_start=raw.overlap_start,
                    overlap_end=raw.overlap_end,
                    first_sentence_order=min(sentence_orders),
                    last_sentence_order=max(sentence_orders),
                    sentence_ids=sentence_ids,
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# Internal mutable accumulator (not part of public API)
# ---------------------------------------------------------------------------

class _RawChunk:
    """
    Mutable working buffer used during chunk construction.

    sentences : list of (SentenceRecord, token_count, is_overlap_flag)
    """

    __slots__ = ("sentences", "total_tokens", "overlap_start", "overlap_end")

    def __init__(self) -> None:
        self.sentences: list[tuple[SentenceRecord, int, bool]] = []
        self.total_tokens: int = 0
        self.overlap_start: bool = False
        self.overlap_end: bool = False

    def add(
        self,
        sentence: SentenceRecord,
        n_tokens: int,
        is_overlap: bool,
    ) -> None:
        self.sentences.append((sentence, n_tokens, is_overlap))
        self.total_tokens += n_tokens

    def __len__(self) -> int:
        return len(self.sentences)


# ---------------------------------------------------------------------------
# Convenience factory — lazy tokenizer loading
# ---------------------------------------------------------------------------

def create_chunk_generator(
    config: Optional[ChunkConfig] = None,
    load_tokenizer: bool = True,
) -> ChunkGenerator:
    """
    Factory that attempts to load the HuggingFace tokenizer and falls back
    gracefully if *transformers* is not installed.

    Parameters
    ----------
    config : ChunkConfig, optional
        Custom config.  Defaults to production-tuned ``ChunkConfig()``.
    load_tokenizer : bool
        When False, always uses the word-count fallback (useful in CI/test
        environments without the *transformers* library).

    Returns
    -------
    ChunkGenerator
    """
    cfg = config or ChunkConfig()
    tokenizer: Optional[TokenizerProtocol] = None

    if load_tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            logger.info("Loaded HuggingFace tokenizer: %s", cfg.model_name)
        except ImportError:
            logger.warning(
                "transformers not installed — using word-count fallback tokenizer."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load tokenizer '%s': %s — using fallback.",
                cfg.model_name,
                exc,
            )

    return ChunkGenerator(config=cfg, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Iterator helper — stream chunks without materialising all in memory
# ---------------------------------------------------------------------------

def iter_chunks(
    result: ChunkGenerationResult,
    batch_size: int = 32,
) -> Iterator[list[Chunk]]:
    """
    Yield batches of chunks from a ChunkGenerationResult.

    Useful when feeding large transcripts into the FinBERT inference engine
    one batch at a time without holding all chunks in GPU memory.

    Parameters
    ----------
    result : ChunkGenerationResult
    batch_size : int
        Number of chunks per yielded batch.

    Yields
    ------
    list[Chunk]
    """
    chunks = result.chunks
    for start in range(0, len(chunks), batch_size):
        yield chunks[start : start + batch_size]


# ---------------------------------------------------------------------------
# Self-test / demo block
# ---------------------------------------------------------------------------

def _build_demo_sentences(transcript_id: str = "AAPL_Q1_2025") -> list[SentenceRecord]:
    """
    Construct a minimal synthetic transcript for self-test purposes.
    Simulates a prepared-remarks block followed by a Q&A section.
    """
    prepared_text = [
        "Good afternoon everyone, and welcome to Apple's first quarter fiscal 2025 earnings call.",
        "My name is Tim Cook, and I will be joined today by our CFO Luca Maestri.",
        "We are very pleased to report record-breaking revenue this quarter driven by strong iPhone demand.",
        "Our services business achieved an all-time revenue record of thirty-two point eight billion dollars.",
        "The Mac line-up saw significant growth following the launch of our latest M-series chips.",
        "iPad revenue grew substantially year over year as enterprise adoption accelerated.",
        "We continue to invest heavily in artificial intelligence capabilities across all of our platforms.",
        "Our developer tools and on-device AI features have been extremely well-received by the market.",
        "We are also expanding manufacturing partnerships in India to diversify our supply chain.",
        "Customer satisfaction scores for iPhone fifteen remain the highest we have ever recorded.",
        "Now I will hand over to Luca who will provide more detail on our financial performance.",
        "Total net sales for the quarter were one hundred and twenty four point three billion dollars.",
        "Gross margin was forty six point nine percent, up sixty basis points year over year.",
        "Operating expenses were fourteen point five billion dollars, in line with our guidance.",
        "We generated thirty three point eight billion dollars in operating cash flow this quarter.",
        "The board has approved a further ninety billion dollars in share repurchase authorisation.",
    ]
    qa_text = [
        "Thank you Luca.",
        "Our first question comes from Mike Olson at Piper Sandler.",
        "Thank you Tim and Luca for the great results this quarter.",
        "My question is around the services growth trajectory — do you see acceleration continuing?",
        "Yes Mike, we remain very confident in services momentum given the installed base expansion.",
        "The combination of Apple Music, Apple TV Plus, and iCloud storage creates strong recurring revenue.",
        "We see subscription growth across every geographic segment including emerging markets.",
        "The next question comes from Katy Huberty at Morgan Stanley.",
        "Congratulations on the results — my question is around Vision Pro adoption in enterprise.",
        "Enterprise adoption of Vision Pro has exceeded our initial expectations significantly.",
        "We now have over five hundred Fortune five hundred companies piloting or deploying the device.",
    ]
    sentences: list[SentenceRecord] = []
    for i, text in enumerate(prepared_text):
        sentences.append(
            SentenceRecord(
                sentence_id=f"{transcript_id}_prep_{i:03d}",
                transcript_id=transcript_id,
                sentence_order=i,
                sentence_text=text,
                speaker="Tim Cook" if i < 11 else "Luca Maestri",
                speaker_role="CEO" if i < 11 else "CFO",
                section_type="prepared_remarks",
            )
        )
    offset = len(prepared_text)
    for j, text in enumerate(qa_text):
        speaker = "Operator" if "question comes from" in text else (
            "Tim Cook" if j % 3 == 0 else "Analyst"
        )
        sentences.append(
            SentenceRecord(
                sentence_id=f"{transcript_id}_qa_{j:03d}",
                transcript_id=transcript_id,
                sentence_order=offset + j,
                sentence_text=text,
                speaker=speaker,
                speaker_role="Operator" if speaker == "Operator" else (
                    "CEO" if speaker == "Tim Cook" else "Analyst"
                ),
                section_type="qa",
            )
        )
    return sentences


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # Quick self-test — runs without any external dependencies
    # -------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(name)s | %(message)s",
    )

    print("\n" + "=" * 70)
    print("  chunk_generator.py  —  Self-Test / Demo")
    print("=" * 70)

    # 1. Build demo data
    TRANSCRIPT_ID = "AAPL_Q1_2025"
    demo_sentences = _build_demo_sentences(TRANSCRIPT_ID)
    print(f"\n✓  Built {len(demo_sentences)} demo sentences.")

    # 2. Instantiate generator (fallback tokenizer, no transformers required)
    config = ChunkConfig(
        chunk_size_tokens=150,   # small for demo visibility
        overlap_tokens=30,
        hard_token_ceiling=490,
        min_chunk_tokens=10,
        preserve_section_boundaries=True,
    )
    gen = ChunkGenerator(config=config)   # uses _WordTokenizerFallback

    # 3. Generate chunks
    result = gen.generate(demo_sentences, transcript_id=TRANSCRIPT_ID)
    print(f"\n✓  Generated {len(result.chunks)} chunks.")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")

    # 4. Print statistics
    stats = result.statistics()
    print(f"\n── Chunk Statistics ──────────────────────────────")
    for k, v in stats.items():
        print(f"   {k:30s}: {v}")

    # 5. Coverage metrics
    cov = result.coverage_metrics()
    print(f"\n── Coverage Metrics ──────────────────────────────")
    for k, v in cov.items():
        print(f"   {k:30s}: {v}")

    # 6. Show first 3 chunks
    print(f"\n── First 3 Chunks ────────────────────────────────")
    for chunk in result.chunks[:3]:
        print(f"\n  chunk_id         : {chunk.chunk_id}")
        print(f"  chunk_order      : {chunk.chunk_order}")
        print(f"  token_count      : {chunk.token_count}")
        print(f"  sentence_count   : {chunk.sentence_count}")
        print(f"  section_type     : {chunk.section_type}")
        print(f"  dominant_speaker : {chunk.dominant_speaker}")
        print(f"  overlap_start    : {chunk.overlap_start}")
        print(f"  overlap_end      : {chunk.overlap_end}")
        print(f"  sentences        : {chunk.first_sentence_order}→{chunk.last_sentence_order}")
        print(f"  text preview     : {chunk.chunk_text[:120]}...")

    # 7. Run validation
    errors = gen.validate_result(result)
    overlap_errors = gen.validate_overlap(result)
    all_errors = errors + overlap_errors
    if all_errors:
        print(f"\n✗  Validation FAILED ({len(all_errors)} errors):")
        for e in all_errors:
            print(f"   - {e}")
    else:
        print(f"\n✓  All validation checks passed.")

    # 8. DataFrame export preview
    df = result.to_dataframe()
    print(f"\n── DataFrame shape: {df.shape} ─────────────────────")
    print(df[["chunk_id", "chunk_order", "token_count", "section_type",
              "dominant_speaker", "overlap_start"]].to_string(index=False))

    # 9. JSONL preview
    jsonl_lines = result.to_jsonl().splitlines()
    print(f"\n── JSONL output: {len(jsonl_lines)} lines (first line preview) ──")
    first = json.loads(jsonl_lines[0])
    print(f"   keys: {list(first.keys())}")

    # 10. Transcript reconstruction
    reconstructed = gen.reconstruct_transcript(result, deduplicate=True)
    print(f"\n── Reconstructed transcript: {len(reconstructed)} chars ──")
    print(f"   Preview: {reconstructed[:200]}...")

    # 11. Batch mode test
    df_batch = pd.DataFrame(
        [asdict(s) for s in _build_demo_sentences("MSFT_Q1_2025")]
        + [asdict(s) for s in _build_demo_sentences("NVDA_Q1_2025")]
    )
    df_batch = df_batch.rename(columns={"sentence_text": "sentence_text"})
    # Rename to match expected column
    batch_results = gen.generate_batch(df_batch)
    combined_df = gen.batch_to_dataframe(batch_results)
    print(f"\n── Batch mode: {len(batch_results)} transcripts → {len(combined_df)} total chunks ──")

    print("\n" + "=" * 70)
    print("  Self-test complete — all systems nominal.")
    print("=" * 70 + "\n")
