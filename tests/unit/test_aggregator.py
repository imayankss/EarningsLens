"""Unit tests for sentiment aggregator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pytest
from src.sentiment.aggregator import SentimentAggregator


class TestSentimentAggregator:
    def setup_method(self):
        self.agg = SentimentAggregator()

    def _make_chunks(self):
        return pd.DataFrame({
            "transcript_id" : ["T1", "T1", "T1"],
            "sentiment_score": [0.8, 0.6, 0.4],
            "confidence"     : [0.9, 0.5, 0.3],
            "positive_prob"  : [0.85, 0.65, 0.45],
            "negative_prob"  : [0.05, 0.05, 0.05],
            "neutral_prob"   : [0.10, 0.30, 0.50],
        })

    def test_aggregation_returns_one_row(self):
        result = self.agg.aggregate_finbert(self._make_chunks())
        assert len(result) == 1

    def test_score_in_range(self):
        result = self.agg.aggregate_finbert(self._make_chunks())
        score = result["finbert_score"].iloc[0]
        assert -1 <= score <= 1

    def test_missing_column_raises(self):
        bad_df = pd.DataFrame({"transcript_id": ["T1"], "sentiment_score": [0.5]})
        with pytest.raises(ValueError):
            self.agg.aggregate_finbert(bad_df)
