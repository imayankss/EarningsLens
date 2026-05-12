# Earnings Call Sentiment Analyzer

> Research-grade financial NLP platform — FinBERT + Loughran-McDonald + Event Study.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Model](https://img.shields.io/badge/Model-FinBERT-orange)](https://huggingface.co/ProsusAI/finbert)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black)](https://apple.com)

---

## Setup (macOS Apple Silicon)

```bash
# 1. Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# 3. Copy and fill environment variables
cp .env.example .env

# 4. Download LM Dictionary (required for baseline)
#    URL: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
#    Save as: data/dictionaries/loughran_mcdonald/LM_MasterDictionary.csv

# 5. Run tests
pytest tests/ -v

# 6. Run full pipeline
python scripts/run_pipeline.py

# 7. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Quick Dev Run (fast test without full data)

```bash
python scripts/run_pipeline.py --max-rows 50 --tickers AAPL MSFT --skip-lm
```

---

## Project Structure

```
earnings-call-sentiment-analyzer/
├── configs/                    # YAML config — single source of truth
│   ├── config.yaml
│   └── tickers.yaml
├── src/
│   ├── utils/                  # Config, logging, storage
│   ├── ingestion/              # Transcript + market data loaders
│   ├── preprocessing/          # Cleaner + chunker
│   ├── sentiment/              # FinBERT pipeline, LM baseline, aggregator
│   ├── finance/                # Return computations
│   ├── event_study/            # Event study engine
│   └── visualization/          # Plotly chart library
├── scripts/
│   └── run_pipeline.py         # Full pipeline orchestrator
├── app/
│   └── streamlit_app.py        # Dashboard
├── tests/unit/                 # pytest test suite
├── notebooks/                  # Exploration notebooks
└── data/                       # Gitignored — populated by pipeline
```

---

## Methodology

| Component | Detail |
|---|---|
| NLP Model | `ProsusAI/finbert` — BERT fine-tuned on financial news |
| Lexicon | Loughran-McDonald (2011) Master Dictionary |
| Chunking | 400-token sliding window, 50-token sentence-aware overlap |
| Sentiment Score | `positive_prob − negative_prob ∈ [−1, +1]` |
| Aggregation | Confidence-weighted average across chunks |
| Benchmark | S&P 500 (`^GSPC`) |
| Abnormal Return | `AR = R_stock − R_benchmark` |
| Event Windows | 1D, 2D, 3D, 5D, 10D |
| Timing | After-market → T+1 event date |

---

## Dataset Sources

| Component | Source |
|---|---|
| Transcripts | [Bose345/sp500_earnings_transcripts](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts) |
| Market Data | yfinance (adjusted close) |
| LM Dictionary | [SRAF Notre Dame](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) |

---

## Stack

`Python 3.12` · `PyTorch (MPS)` · `transformers` · `pandas` · `DuckDB` · `Parquet` · `yfinance` · `statsmodels` · `plotly` · `Streamlit`
