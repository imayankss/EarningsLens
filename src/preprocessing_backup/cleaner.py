"""
src/preprocessing/cleaner.py
==============================
Cleans raw earnings call transcripts for NLP consumption.

Operations (in order):
  1. Unicode normalisation
  2. Boilerplate removal  (legal disclaimers, operator cues)
  3. Whitespace normalisation
  4. Q&A section separation

Design: pure stateless functions → fully testable with no side effects.

Usage:
    cleaner = TranscriptCleaner()
    clean_text = cleaner.clean(raw_text)
    sections   = cleaner.separate_sections(raw_text)
    df         = cleaner.clean_dataframe(df, text_col="transcript_text")
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Common boilerplate patterns in earnings call transcripts
_BOILERPLATE_PATTERNS: list[str] = [
    r"(?i)this transcript is produced by.*?\.",
    r"(?i)this call (?:is being )?recorded.*?\.",
    r"(?i)safe harbor statement.*?(?:forward.looking statements?)",
    r"(?i)\[operator instructions?\]",
    r"(?i)operator[:\s]+(?:good (?:morning|afternoon|evening)[^.]*\.)",
    r"(?i)thank you for standing by[^.]*\.",
    r"(?i)ladies and gentlemen[^.]*(?:thank you|welcome)[^.]*\.",
    r"(?i)this concludes today's[^.]*\.",
    r"(?i)please go ahead[,.]?",
    r"(?i)your (?:next|first) question comes from[^.]*\.",
]

# Compiled QA section markers
_QA_MARKERS: list[str] = [
    r"(?i)question.and.answer",
    r"(?i)q&a session",
    r"(?i)now open (?:the )?(?:floor|line)s? for questions",
    r"(?i)we will now begin the question",
    r"(?i)operator.*?take.*?question",
]


class TranscriptCleaner:
    """Cleans and segments earnings call transcripts."""

    def clean(self, text: str) -> str:
        """
        Full single-pass cleaning pipeline.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned text string
        """
        if not text or not str(text).strip():
            return ""
        text = self._normalise_unicode(text)
        text = self._remove_boilerplate(text)
        text = self._normalise_whitespace(text)
        return text.strip()

    def separate_sections(self, text: str) -> dict[str, str]:
        """
        Split transcript into prepared remarks and Q&A section.

        Returns:
            {
              "prepared": str,  — management prepared remarks
              "qa"      : str,  — analyst Q&A section
              "full"    : str,  — full cleaned transcript
            }
        """
        text = self.clean(text)
        split_idx = len(text)

        for marker in _QA_MARKERS:
            match = re.search(marker, text)
            if match:
                split_idx = min(split_idx, match.start())

        return {
            "prepared": text[:split_idx].strip(),
            "qa"      : text[split_idx:].strip(),
            "full"    : text,
        }

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "transcript_text",
    ) -> pd.DataFrame:
        """
        Apply cleaning pipeline to every row in a DataFrame.

        Adds columns:
            prepared_remarks  — management presentation section
            qa_section        — analyst Q&A section
            word_count_clean  — word count post-cleaning

        Args:
            df      : Input DataFrame
            text_col: Column containing raw transcript text

        Returns:
            DataFrame with added cleaning columns
        """
        log.info(f"Cleaning {len(df):,} transcripts...")
        df = df.copy()
        df[text_col] = df[text_col].fillna("").astype(str).apply(self.clean)

        sections = df[text_col].apply(self.separate_sections)
        df["prepared_remarks"] = sections.apply(lambda x: x["prepared"])
        df["qa_section"]       = sections.apply(lambda x: x["qa"])
        df["word_count_clean"] = df[text_col].str.split().str.len()

        log.info(
            f"Cleaning complete. "
            f"Avg words: {df['word_count_clean'].mean():.0f}"
        )
        return df

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        return unicodedata.normalize("NFKD", text)

    @staticmethod
    def _remove_boilerplate(text: str) -> str:
        for pattern in _BOILERPLATE_PATTERNS:
            text = re.sub(pattern, " ", text)
        return text

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text
