# Architecture

See README.md for full system design and methodology.

## Pipeline Flow

```
HuggingFace Transcripts
        │
        ▼
  TranscriptLoader  ──►  data/raw/transcripts/
        │
        ▼
  TranscriptCleaner + TranscriptChunker  ──►  data/interim/
        │
   ┌────┴────────┐
   ▼             ▼
FinBERT       LM Baseline
Pipeline      (Loughran-McDonald)
   │             │
   └────┬────────┘
        ▼
  SentimentAggregator  ──►  data/processed/sentiment/
        │
        ▼
  MarketDataLoader (yfinance)  ──►  data/raw/market/
        │
        ▼
  EventStudyEngine  ──►  data/processed/event_study/
        │
        ▼
  Streamlit Dashboard + Plotly Charts
```
