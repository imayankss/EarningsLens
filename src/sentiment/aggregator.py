"""
src/sentiment/aggregator.py
=============================
Aggregates chunk-level FinBERT scores to transcript-level scores.

Aggregation method: confidence-weighted average.
    weight_i = confidence_i  (= max softmax probability for chunk i)

    finbert_score = Σ(score_i × weight_i) / Σ(weight_i)

Rationale: chunks where FinBERT is uncertain (confidence ≈ 0.34)
contribute less than chunks where it is highly confident (≈ 0.95).
This outperforms simple averaging on noisy financial text.

Also merges FinBERT and LM scores into the final sentiment feature store.

Usage:
    agg = SentimentAggregator()
    transcript_scores = agg.aggregate_finbert(chunks_df)
    feature_store     = agg.merge_features(transcripts_df, transcript_scores, lm_df)
"""
from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class SentimentAggregator:
    """Aggregates chunk-level scores to transcript-level features."""

    def aggregate_finbert(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Confidence-weighted aggregation of chunk-level FinBERT scores.

        Args:
            chunks_df: DataFrame with per-chunk FinBERT scores
                       Must contain: transcript_id, sentiment_score,
                       confidence, positive_prob, negative_prob, neutral_prob

        Returns:
            Transcript-level DataFrame with columns:
                finbert_score, finbert_positive, finbert_negative,
                finbert_neutral, chunk_count, mean_confidence
        """
        required = [
            "transcript_id", "sentiment_score", "confidence",
            "positive_prob", "negative_prob", "neutral_prob",
        ]
        missing = [c for c in required if c not in chunks_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        def _weighted_agg(group: pd.DataFrame) -> pd.Series:
            w     = group["confidence"]
            w_sum = w.sum()
            if w_sum == 0:
                w     = pd.Series([1.0] * len(group), index=group.index)
                w_sum = float(len(group))
            return pd.Series({
                "finbert_score"   : (group["sentiment_score"] * w).sum() / w_sum,
                "finbert_positive": (group["positive_prob"]   * w).sum() / w_sum,
                "finbert_negative": (group["negative_prob"]   * w).sum() / w_sum,
                "finbert_neutral" : (group["neutral_prob"]    * w).sum() / w_sum,
                "chunk_count"     : len(group),
                "mean_confidence" : w.mean(),
            })

        agg = (
            chunks_df
            .groupby("transcript_id")
            .apply(_weighted_agg)
            .reset_index()
        )
        log.info(
            f"Aggregated {len(chunks_df):,} chunks → {len(agg):,} transcripts"
        )
        return agg

    def merge_features(
        self,
        transcripts_df : pd.DataFrame,
        finbert_df     : pd.DataFrame,
        lm_df          : pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Merge transcript metadata + FinBERT scores + LM scores into the
        final sentiment feature store.

        Args:
            transcripts_df: Transcript metadata (must include transcript_id)
            finbert_df    : Aggregated FinBERT scores per transcript
            lm_df         : LM baseline scores per transcript (optional)

        Returns:
            Merged feature store DataFrame
        """
        meta_cols = [
            c for c in ["transcript_id", "ticker", "earnings_date", "fiscal_quarter"]
            if c in transcripts_df.columns
        ]
        merged = transcripts_df[meta_cols].merge(
            finbert_df, on="transcript_id", how="inner"
        )

        if lm_df is not None:
            lm_cols = ["transcript_id"] + [
                c for c in lm_df.columns if c.startswith("lm_")
            ]
            merged = merged.merge(lm_df[lm_cols], on="transcript_id", how="left")

        log.info(
            f"Feature store: {merged.shape[0]:,} rows × {merged.shape[1]} cols"
        )
        return merged
