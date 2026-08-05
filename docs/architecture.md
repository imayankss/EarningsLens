# Architecture

## High-Level Architecture

```mermaid
flowchart TD
    A[Raw transcripts] --> B[src/preprocessing]
    B --> C[src/sentiment FinBERT]
    B --> D[src/sentiment LM baseline]
    C --> E[src/analysis comparison]
    D --> E
    F[Market data] --> G[src/finance]
    G --> H[src/event_study]
    E --> I[src/analysis master dataset]
    H --> I
    I --> J[src/analysis statistical analysis]
    I --> K[src/models predictive modeling]
    B --> L[src/nlp advanced NLP]
    I --> M[src/visualization professional charts]
    J --> M
    M --> N[reports/figures]
    J --> O[reports/tables]
    K --> O
    L --> O
    I --> P[app.py Streamlit dashboard]
    N --> P
    O --> P
    I --> Q[scripts/export_web_data.py]
    L --> Q
    O --> Q
    Q --> R[web/public/data/dashboard.json]
    R --> S[Next.js static dashboard]
```

## Data Flow

```mermaid
flowchart LR
    T[data/interim/transcripts/transcripts_cleaned.parquet]
    C[data/interim/chunks/chunks.parquet]
    F[data/processed/sentiment/sentiment_features.parquet]
    L[data/processed/sentiment/lm_scores.parquet]
    E[data/processed/event_study/event_study.parquet]
    M[data/processed/market_data.parquet]
    X[data/processed/analysis/sentiment_model_comparison.parquet]
    D[data/processed/master_dataset.parquet]
    N[data/processed/nlp/advanced_nlp_features.parquet]
    R[reports/tables/*.csv]
    G[reports/figures/*.png]
    S[app.py]
    W[scripts/export_web_data.py]
    J[web/public/data/dashboard.json]
    V[Next.js on Vercel]

    T --> D
    F --> X
    L --> X
    E --> X
    X --> D
    E --> D
    M --> D
    C --> N
    D --> R
    N --> R
    D --> G
    R --> S
    G --> S
    D --> W
    N --> W
    R --> W
    W --> J
    J --> V
```

## Module-Level Explanation

### `src/sentiment/`

Contains FinBERT scoring, Loughran-McDonald scoring, chunk aggregation, validation, and supporting NLP utilities.

### `src/finance/`

Handles market data loading, return computation, feature engineering, trading calendar helpers, and market dataset construction.

### `src/event_study/`

Implements event-window generation, abnormal returns, CAR calculations, sentiment/event merging, validation, and event-study orchestration.

### `src/analysis/`

Contains post-pipeline analysis modules:

- `sentiment_model_comparison.py`
- `master_dataset_builder.py`
- `statistical_analysis.py`

### `src/nlp/`

Adds advanced text features: keywords, topics, bigrams, uncertainty scoring, and speaker/section sentiment summaries.

### `src/models/`

Contains guarded prediction modeling. The current dataset has one row, so model training is skipped and metadata records the reason.

### `src/visualization/`

Generates static matplotlib PNG charts for reports and dashboard display.

### `app.py`

Streamlit dashboard that reads generated tables, figures, metadata, and processed datasets. It does not modify pipeline outputs.

### `scripts/export_web_data.py`

Reads existing checked CSV/JSON artifacts and writes a small JSON contract for the web application. It performs no transcript ingestion, market download, transformer inference, or model training and rejects non-finite JSON values.

### `web/`

Next.js App Router application deployed with `web/` as the Vercel Root Directory. Normal requests serve static UI assets and `web/public/data/dashboard.json`; the application does not require the Python runtime.

## Output Artifact Map

| Artifact | Purpose |
|---|---|
| `data/processed/master_dataset.parquet` | ML-ready merged dataset |
| `data/processed/master_dataset.csv` | CSV version of master dataset |
| `data/processed/nlp/advanced_nlp_features.csv` | Advanced NLP feature table |
| `reports/tables/statistical_summary.csv` | Statistical summary metrics |
| `reports/tables/correlation_analysis.csv` | Guarded correlation table |
| `reports/tables/regression_results.csv` | Guarded regression table |
| `reports/tables/prediction_model_summary.csv` | Prediction modeling summary |
| `reports/tables/prediction_model_metrics.csv` | Model status/metrics table |
| `reports/figures/*.png` | Static professional figures |
| `models/predictive_model_metadata.json` | Prediction metadata and skip reasons |
| `web/public/data/dashboard.json` | Versioned deployment-safe dashboard contract |
