# Methodology

This document describes the modeling and analysis design used by the earnings-call sentiment analyzer. The current real dataset contains one transcript/event, so methodology is implemented and pipeline-ready, while statistical inference and ML validation remain limited.

## Transcript Preprocessing

Raw transcripts are cleaned into structured intermediate artifacts. The pipeline preserves identifiers such as `transcript_id`, `ticker`, `earnings_date`, fiscal period fields, company name, and cleaned word counts. Transcript text is chunked for transformer-friendly sentiment scoring while maintaining chunk-level metadata when available.

## FinBERT Sentiment

FinBERT is used as the contextual sentiment model for financial language. Chunk-level positive, negative, and neutral probabilities are aggregated into transcript-level features including:

- `finbert_score`
- `finbert_positive`
- `finbert_negative`
- `finbert_neutral`
- `chunk_count`
- `mean_confidence`

The main sentiment score is interpreted as positive tone minus negative tone.

## Loughran-McDonald Baseline

The Loughran-McDonald financial dictionary provides an explainable lexical baseline. It counts finance-specific positive and negative terms and produces features such as:

- `lm_tone_score`
- `lm_tone`
- `lm_positive_count`
- `lm_negative_count`
- `lm_word_count`
- `lm_label`

The dictionary file is expected at `data/dictionaries/loughran_mcdonald/LM_MasterDictionary.csv`.

## FinBERT vs LM Comparison

The comparison layer merges FinBERT and LM outputs and derives agreement diagnostics:

- FinBERT direction
- LM direction
- Directional agreement
- Score difference
- Absolute difference
- Divergence flag

This creates an interpretable bridge between contextual model behavior and dictionary-based financial sentiment.

## Market and Event-Study Integration

Market data is aligned to earnings dates. Event-study outputs include raw returns and abnormal returns over post-event horizons:

- `return_1d`, `ar_1d`
- `return_2d`, `ar_2d`
- `return_3d`, `ar_3d`
- `return_5d`, `ar_5d`
- `return_10d`, `ar_10d`

The current event-study output is a single-event snapshot, not a basis for inference.

## Master Dataset Construction

The master dataset builder uses `data/interim/transcripts/transcripts_cleaned.parquet` as the base table and left-joins sentiment, LM, event-study, market aggregate, and comparison features on `transcript_id`. It writes:

- `data/processed/master_dataset.parquet`
- `data/processed/master_dataset.csv`
- `reports/tables/master_dataset_summary.csv`

It validates duplicate transcript IDs, warns on missing important columns, and preserves missing values rather than faking them.

## Statistical Analysis

The statistical module reads the master dataset and creates:

- `reports/tables/statistical_summary.csv`
- `reports/tables/correlation_analysis.csv`
- `reports/tables/regression_results.csv`
- `reports/tables/statistical_warnings.csv`

Correlation analysis requires at least two paired observations. Regression requires enough observations relative to feature count. With `n=1`, tests are correctly skipped and warnings are written.

## Advanced NLP Features

The advanced NLP layer extracts lightweight explainability features without requiring downloads from spaCy or NLTK:

- Top keywords
- Financial topic bucket counts
- Common bigrams
- Uncertainty and hedging language
- Speaker or section sentiment summaries when chunk-level fields are available

Outputs include `data/processed/nlp/advanced_nlp_features.csv` and report tables such as `keyword_summary.csv`, `topic_frequency.csv`, and `uncertainty_summary.csv`.

## Prediction Modeling

The prediction module creates binary targets such as:

- `target_positive_ar_1d`
- `target_positive_ar_3d`
- `target_positive_return_1d`

Candidate features include sentiment, NLP, market, and text/meta variables. Training is guarded:

- Fewer than 10 rows: skip with `insufficient_sample_size`
- Single-class target: skip with `single_class_target`
- Missing sklearn/xgboost: skip gracefully

The current dataset has one row, so model training is correctly skipped and no model performance is claimed.

## Dashboard Layer

`app.py` renders a Streamlit dashboard over generated artifacts. It reads existing tables and PNG figures rather than recalculating analysis. Missing files produce clean UI warnings instead of tracebacks.

## Honest n=1 Limitation

The current real output is useful as a pipeline and portfolio demonstration. It is not sufficient for statistical claims, trained model validation, or investment conclusions. The code is designed to scale once additional transcript/events are loaded.
