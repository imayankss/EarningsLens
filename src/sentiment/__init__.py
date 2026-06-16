"""Sentiment analysis: FinBERT pipeline, LM baseline, aggregation."""
from .finbert_pipeline import FinBERTPipeline
from .lm_baseline import LoughranMcDonaldBaseline
from .lm_pipeline import LMPipeline, LMPipelineConfig
from .aggregator import SentimentAggregator
