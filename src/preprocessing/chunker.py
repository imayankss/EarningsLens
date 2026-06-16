"""
Compatibility transcript chunker used by the top-level pipeline script.

The research-grade speaker-aware chunker lives in ``src.preprocessing.chunking``
and expects segmented speaker turns.  ``scripts/run_pipeline.py`` operates on
one row per transcript, so this adapter provides the older DataFrame-oriented
interface while emitting the canonical columns consumed by FinBERT:
``chunk_id``, ``transcript_id``, ``chunk_order``, ``chunk_text``,
``token_count``, ``section_type`` and ``dominant_speaker``.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Chunk:
    """A single transcript chunk ready for FinBERT inference."""

    chunk_id: str
    transcript_id: str
    ticker: str
    earnings_date: str
    chunk_order: int
    chunk_text: str
    token_count: int
    section_type: str
    dominant_speaker: str = ""


class TranscriptChunker:
    """Sentence-aware sliding-window chunker for transcript-level rows."""

    def __init__(self) -> None:
        config = load_config()
        finbert_cfg = config.get("finbert", {})
        self.chunk_size = int(finbert_cfg.get("chunk_size", 400))
        self.chunk_overlap = int(finbert_cfg.get("chunk_overlap", 50))
        self.model_name = str(finbert_cfg.get("model_name", "ProsusAI/finbert"))
        self._tokenizer: Any | None = None
        self._tokenizer_failed = False

    @property
    def tokenizer(self) -> Any | None:
        """Lazy-load the tokenizer; fall back to word counts if unavailable."""
        if self._tokenizer is None and not self._tokenizer_failed:
            try:
                from transformers import AutoTokenizer

                log.info(f"Loading tokenizer: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception as exc:
                self._tokenizer_failed = True
                log.warning(
                    "Tokenizer unavailable (%s); falling back to whitespace token counts.",
                    exc,
                )
        return self._tokenizer

    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        section_col: str = "transcript_text_clean",
    ) -> pd.DataFrame:
        """Chunk all transcript rows in ``df`` into a flat chunk DataFrame."""
        all_chunks: list[dict[str, Any]] = []
        text_col = self._resolve_text_col(df, section_col)
        log.info(f"Chunking {len(df)} transcripts from column '{text_col}'.")

        for _, row in df.iterrows():
            text = str(row.get(text_col, "") or "").strip()
            if len(text) < 50:
                continue

            chunks = self.chunk_text(
                text=text,
                transcript_id=str(row.get("transcript_id", "")),
                ticker=str(row.get("ticker", "")),
                earnings_date=str(row.get("earnings_date", "")),
                section_type=text_col,
                dominant_speaker=str(row.get("speaker", "")),
            )
            all_chunks.extend(asdict(chunk) for chunk in chunks)

        result = pd.DataFrame(all_chunks)
        log.info(f"Generated {len(result)} chunks from {len(df)} transcripts.")
        return result

    def chunk_text(
        self,
        text: str,
        transcript_id: str,
        ticker: str = "",
        earnings_date: str = "",
        section_type: str = "full",
        dominant_speaker: str = "",
    ) -> list[Chunk]:
        """Split one transcript into sentence-aware overlapping chunks."""
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        current_sents: list[str] = []
        current_tokens = 0
        chunk_order = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if current_sents and current_tokens + sentence_tokens > self.chunk_size:
                chunks.append(self._build_chunk(
                    transcript_id=transcript_id,
                    ticker=ticker,
                    earnings_date=earnings_date,
                    chunk_order=chunk_order,
                    sentences=current_sents,
                    token_count=current_tokens,
                    section_type=section_type,
                    dominant_speaker=dominant_speaker,
                ))
                chunk_order += 1
                current_sents = self._overlap_tail(current_sents)
                current_tokens = sum(self._count_tokens(sent) for sent in current_sents)

            current_sents.append(sentence)
            current_tokens += sentence_tokens

        if current_sents and self._count_tokens(" ".join(current_sents)) >= 20:
            chunks.append(self._build_chunk(
                transcript_id=transcript_id,
                ticker=ticker,
                earnings_date=earnings_date,
                chunk_order=chunk_order,
                sentences=current_sents,
                token_count=current_tokens,
                section_type=section_type,
                dominant_speaker=dominant_speaker,
            ))

        return chunks

    def _build_chunk(
        self,
        *,
        transcript_id: str,
        ticker: str,
        earnings_date: str,
        chunk_order: int,
        sentences: list[str],
        token_count: int,
        section_type: str,
        dominant_speaker: str,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{transcript_id}_chunk_{chunk_order:04d}",
            transcript_id=transcript_id,
            ticker=ticker,
            earnings_date=earnings_date,
            chunk_order=chunk_order,
            chunk_text=" ".join(sentences),
            token_count=token_count,
            section_type=section_type,
            dominant_speaker=dominant_speaker,
        )

    def _count_tokens(self, text: str) -> int:
        tokenizer = self.tokenizer
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
        return [part.strip() for part in parts if len(part.strip()) > 10]

    def _overlap_tail(self, sentences: list[str]) -> list[str]:
        tail: list[str] = []
        used = 0
        for sentence in reversed(sentences):
            tokens = self._count_tokens(sentence)
            if used + tokens > self.chunk_overlap:
                break
            tail.insert(0, sentence)
            used += tokens
        return tail

    @staticmethod
    def _resolve_text_col(df: pd.DataFrame, requested: str) -> str:
        for candidate in (requested, "transcript_text_clean", "prepared_remarks", "transcript_text"):
            if candidate in df.columns:
                return candidate
        raise ValueError(
            "No transcript text column found. Expected one of: "
            f"{[requested, 'transcript_text_clean', 'prepared_remarks', 'transcript_text']}"
        )


__all__ = ["Chunk", "TranscriptChunker"]
