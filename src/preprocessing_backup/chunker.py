"""
src/preprocessing/chunker.py
==============================
Splits long transcripts into overlapping token-bounded chunks
suitable for FinBERT inference (512-token limit).

Strategy:
  - Sentence-boundary aware splitting (no mid-sentence cuts)
  - 400-token chunks with 50-token overlap
  - Uses the FinBERT tokenizer to count tokens precisely

Why chunking matters:
  Naively truncating at 512 tokens discards most of the transcript.
  Overlapping sliding-window chunking preserves all content while
  respecting the model's context window.

Usage:
    chunker = TranscriptChunker()
    chunks_df = chunker.chunk_dataframe(df, section_col="prepared_remarks")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    """A single transcript chunk ready for FinBERT inference."""
    chunk_id    : str
    transcript_id: str
    ticker      : str
    earnings_date: str
    chunk_index : int
    chunk_text  : str
    token_count : int
    section     : str   # "prepared_remarks" | "qa_section" | "full"


class TranscriptChunker:
    """
    Sentence-aware sliding-window chunker for FinBERT.
    Tokenizer is lazy-loaded on first use.
    """

    def __init__(self) -> None:
        config = load_config()
        self.chunk_size    = config["finbert"]["chunk_size"]    # 400
        self.chunk_overlap = config["finbert"]["chunk_overlap"] # 50
        self.model_name    = config["finbert"]["model_name"]
        self._tokenizer    = None   # lazy-loaded

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore
            log.info(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    # ── Public API ──────────────────────────────────────────────

    def chunk_text(
        self,
        text      : str,
        transcript_id: str,
        ticker    : str,
        earnings_date: str,
        section   : str = "full",
    ) -> list[Chunk]:
        """
        Split a single transcript into overlapping chunks.

        Args:
            text         : Cleaned transcript text
            transcript_id: Unique transcript ID
            ticker       : Stock ticker symbol
            earnings_date: Earnings call date string
            section      : Section label for output metadata

        Returns:
            List of Chunk objects
        """
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        current_sents: list[str] = []
        current_tokens = 0
        idx = 0

        for sentence in sentences:
            s_tokens = self._count_tokens(sentence)

            if current_tokens + s_tokens > self.chunk_size and current_sents:
                # Emit chunk
                chunks.append(Chunk(
                    chunk_id     = f"{transcript_id}_chunk_{idx:04d}",
                    transcript_id= transcript_id,
                    ticker       = ticker,
                    earnings_date= earnings_date,
                    chunk_index  = idx,
                    chunk_text   = " ".join(current_sents),
                    token_count  = current_tokens,
                    section      = section,
                ))
                idx += 1

                # Carry-over overlap sentences
                overlap = self._overlap_tail(current_sents)
                current_sents  = overlap
                current_tokens = sum(self._count_tokens(s) for s in overlap)

            current_sents.append(sentence)
            current_tokens += s_tokens

        # Final chunk
        if current_sents:
            final_text = " ".join(current_sents)
            if self._count_tokens(final_text) >= 20:
                chunks.append(Chunk(
                    chunk_id     = f"{transcript_id}_chunk_{idx:04d}",
                    transcript_id= transcript_id,
                    ticker       = ticker,
                    earnings_date= earnings_date,
                    chunk_index  = idx,
                    chunk_text   = final_text,
                    token_count  = current_tokens,
                    section      = section,
                ))

        log.debug(
            f"{transcript_id}: {len(sentences)} sentences → {len(chunks)} chunks"
        )
        return chunks

    def chunk_dataframe(
        self,
        df         : pd.DataFrame,
        section_col: str = "prepared_remarks",
    ) -> pd.DataFrame:
        """
        Chunk all transcripts in a DataFrame.

        Args:
            df         : DataFrame with transcript rows
            section_col: Which text column to chunk

        Returns:
            Flat DataFrame of all chunks
        """
        all_chunks: list[dict] = []
        log.info(f"Chunking {len(df):,} transcripts (section: {section_col})...")

        for _, row in df.iterrows():
            text = str(row.get(section_col, row.get("transcript_text", ""))).strip()
            if len(text) < 50:
                continue

            chunks = self.chunk_text(
                text          = text,
                transcript_id = str(row.get("transcript_id", "")),
                ticker        = str(row.get("ticker", "")),
                earnings_date = str(row.get("earnings_date", "")),
                section       = section_col,
            )
            all_chunks.extend(vars(c) for c in chunks)

        result = pd.DataFrame(all_chunks)
        log.info(
            f"Generated {len(result):,} chunks from {len(df):,} transcripts"
        )
        return result

    # ── Private helpers ─────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_sentences(self, text: str) -> list[str]:
        """Sentence split on punctuation + capital letter boundary."""
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in parts if len(s.strip()) > 10]

    def _overlap_tail(self, sentences: list[str]) -> list[str]:
        """Return the tail sentences that fit within chunk_overlap tokens."""
        result: list[str] = []
        used = 0
        for s in reversed(sentences):
            t = self._count_tokens(s)
            if used + t > self.chunk_overlap:
                break
            result.insert(0, s)
            used += t
        return result
