"""
src/sentiment/finbert_pipeline.py
===================================
Production FinBERT inference on transcript chunks.

Model  : ProsusAI/finbert (BERT fine-tuned on financial news)
Device : MPS (Apple Silicon) → CPU fallback
Output : positive_prob, negative_prob, neutral_prob per chunk

Chunk-level sentiment score:
    score = positive_prob - negative_prob   ∈ [-1, +1]

Inference is batched and wrapped with tqdm for progress tracking.
Model and tokenizer are lazy-loaded on first use.

Usage:
    pipeline = FinBERTPipeline()
    scored_df = pipeline.score_chunks(chunks_df)
    pipeline.save_scores(scored_df, "data/processed/sentiment/scores.parquet")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.storage import save_parquet

log = get_logger(__name__)

# FinBERT label order (verify with model.config.id2label if behaviour changes)
_LABEL_MAP: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}


class FinBERTPipeline:
    """
    Batched FinBERT inference on transcript chunks.

    Supports MPS (Apple Silicon), CUDA, and CPU devices.
    Uses model caching — load once, score many DataFrames.
    """

    def __init__(self) -> None:
        cfg = load_config()["finbert"]
        self.model_name = cfg["model_name"]
        self.max_length = cfg["max_length"]
        self.batch_size = cfg["batch_size"]
        self.device     = self._resolve_device(cfg["device"])
        self._model     = None
        self._tokenizer = None
        log.info(f"FinBERT device: {self.device}")

    # ── Public API ──────────────────────────────────────────────

    @torch.no_grad()
    def score_chunks(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run FinBERT on all chunks in a DataFrame.

        Args:
            chunks_df: DataFrame containing at least "chunk_text" and "chunk_id"

        Returns:
            Input DataFrame with appended columns:
                positive_prob, negative_prob, neutral_prob,
                sentiment_score, predicted_label, confidence
        """
        if "chunk_text" not in chunks_df.columns:
            raise ValueError("DataFrame must contain 'chunk_text' column")

        texts = chunks_df["chunk_text"].tolist()
        log.info(
            f"FinBERT inference: {len(texts):,} chunks "
            f"(batch_size={self.batch_size}, device={self.device})"
        )

        results: list[dict] = []
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="FinBERT",
            unit="batch",
        ):
            batch = texts[i : i + self.batch_size]
            results.extend(self._score_batch(batch))

        scores_df = pd.DataFrame(results)
        output = pd.concat(
            [chunks_df.reset_index(drop=True), scores_df.reset_index(drop=True)],
            axis=1,
        )
        log.info(
            f"Inference complete. "
            f"Mean score: {output['sentiment_score'].mean():.4f}"
        )
        return output

    def save_scores(self, df: pd.DataFrame, path: str) -> Path:
        """Save scored chunks to Parquet."""
        return save_parquet(df, path)

    # ── Properties (lazy load) ──────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification  # type: ignore
            log.info(f"Loading FinBERT model: {self.model_name}")
            self._model = (
                AutoModelForSequenceClassification
                .from_pretrained(self.model_name)
                .to(self.device)
            )
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    # ── Private helpers ─────────────────────────────────────────

    def _score_batch(self, texts: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs  = softmax(logits, dim=-1).cpu().numpy()

        out = []
        for row in probs:
            pos, neg, neu = float(row[0]), float(row[1]), float(row[2])
            pred_idx = int(row.argmax())
            out.append({
                "positive_prob"  : pos,
                "negative_prob"  : neg,
                "neutral_prob"   : neu,
                "sentiment_score": pos - neg,          # ∈ [-1, +1]
                "predicted_label": _LABEL_MAP[pred_idx],
                "confidence"     : float(row.max()),
            })
        return out

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested not in ("cpu",):
            log.warning(f"Device '{requested}' unavailable. Falling back to CPU.")
        return torch.device("cpu")
