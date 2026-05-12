"""
src/sentiment/lm_baseline.py
==============================
Loughran-McDonald (2011) financial sentiment lexicon baseline.

LM Tone formula:
    LM_Tone = (Positive − Negative) / (Positive + Negative)  ∈ [-1, +1]

This is the standard academic benchmark for financial text sentiment.
Used to validate and compare against FinBERT output.

Dictionary download:
    URL : https://sraf.nd.edu/loughranmcdonald-master-dictionary/
    File: Loughran-McDonald_MasterDictionary_1993-2023.csv
    Save: data/dictionaries/loughran_mcdonald/LM_MasterDictionary.csv

Usage:
    lm = LoughranMcDonaldBaseline()
    score = lm.score_text("Revenue grew but uncertainty remains elevated.")
    df    = lm.score_dataframe(df, text_col="transcript_text")
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class LoughranMcDonaldBaseline:
    """
    Lexicon-based sentiment scoring using the LM financial wordlist.
    Dictionary is lazy-loaded on first use.
    """

    def __init__(self) -> None:
        cfg = load_config()
        self.dict_path    = Path(cfg["paths"]["lm_dictionary"])
        self.pos_col      = cfg["lm_baseline"]["positive_col"]
        self.neg_col      = cfg["lm_baseline"]["negative_col"]
        self.word_col     = cfg["lm_baseline"]["word_col"]
        self._pos_words: set[str] | None = None
        self._neg_words: set[str] | None = None

    # ── Public API ──────────────────────────────────────────────

    def score_text(self, text: str) -> dict[str, float | int]:
        """
        Compute LM sentiment scores for a single text string.

        Args:
            text: Cleaned transcript or chunk text

        Returns:
            {
              lm_positive_count : int,
              lm_negative_count : int,
              lm_total_words    : int,
              lm_tone           : float  ∈ [-1, +1]
            }
        """
        tokens    = re.findall(r"\b[a-z]+\b", text.lower())
        pos_count = sum(1 for t in tokens if t in self.positive_words)
        neg_count = sum(1 for t in tokens if t in self.negative_words)
        total     = len(tokens)
        denom     = pos_count + neg_count

        return {
            "lm_positive_count": pos_count,
            "lm_negative_count": neg_count,
            "lm_total_words"   : total,
            "lm_tone"          : (pos_count - neg_count) / denom if denom > 0 else 0.0,
        }

    def score_dataframe(
        self,
        df      : pd.DataFrame,
        text_col: str = "transcript_text",
    ) -> pd.DataFrame:
        """
        Score all transcripts in a DataFrame.

        Appends columns: lm_positive_count, lm_negative_count,
                         lm_total_words, lm_tone

        Args:
            df      : Input DataFrame
            text_col: Column containing transcript text

        Returns:
            DataFrame with appended LM score columns
        """
        log.info(f"Computing LM scores for {len(df):,} rows...")
        scores = df[text_col].fillna("").apply(self.score_text)
        scores_df = pd.DataFrame(scores.tolist())
        result = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)],
            axis=1,
        )
        log.info(f"LM scoring done. Mean tone: {result['lm_tone'].mean():.4f}")
        return result

    # ── Properties (lazy load dictionary) ──────────────────────

    @property
    def positive_words(self) -> set[str]:
        if self._pos_words is None:
            self._load_dictionary()
        return self._pos_words  # type: ignore

    @property
    def negative_words(self) -> set[str]:
        if self._neg_words is None:
            self._load_dictionary()
        return self._neg_words  # type: ignore

    def _load_dictionary(self) -> None:
        if not self.dict_path.exists():
            raise FileNotFoundError(
                f"LM dictionary not found: {self.dict_path}\n"
                f"Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/\n"
                f"Save as: {self.dict_path}"
            )
        log.info(f"Loading LM dictionary: {self.dict_path}")
        df = pd.read_csv(self.dict_path, low_memory=False)
        df.columns = df.columns.str.strip()

        self._pos_words = set(
            df[df[self.pos_col] > 0][self.word_col].str.lower().tolist()
        )
        self._neg_words = set(
            df[df[self.neg_col] > 0][self.word_col].str.lower().tolist()
        )
        log.info(
            f"Dictionary loaded: {len(self._pos_words):,} positive, "
            f"{len(self._neg_words):,} negative words"
        )
