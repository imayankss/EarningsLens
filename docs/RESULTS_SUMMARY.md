# Results Summary

## Current Dataset Summary

The current real generated master dataset contains:

- Rows: 1
- Event: `AAPL_20201029`
- Ticker: `AAPL`
- Company: Apple Inc.
- Master outputs:
  - `data/processed/master_dataset.parquet`
  - `data/processed/master_dataset.csv`

Because there is only one event, all statistics and ML outputs must be interpreted as pipeline demonstrations.

## Generated Outputs Summary

Verified important outputs include:

- `data/processed/master_dataset.csv`
- `data/processed/master_dataset.parquet`
- `data/processed/nlp/advanced_nlp_features.csv`
- `reports/tables/statistical_summary.csv`
- `reports/tables/correlation_analysis.csv`
- `reports/tables/regression_results.csv`
- `reports/tables/prediction_model_summary.csv`
- `reports/figures/*.png`

## Sentiment Comparison Summary

The master dataset includes FinBERT and LM sentiment fields:

- `finbert_score`: approximately `0.4324`
- `lm_tone_score`: approximately `0.4921`
- Directional agreement: `True`

This means both models indicate positive direction for the current event, but this is one example only.

## Event-Study Summary

The master dataset includes return and abnormal return columns such as:

- `return_1d`, `ar_1d`
- `return_3d`, `ar_3d`
- `return_5d`, `ar_5d`

The current event-study chart is descriptive for one event and should not be generalized.

## Advanced NLP Output Summary

Advanced NLP outputs include:

- `data/processed/nlp/advanced_nlp_features.csv`
- `reports/tables/keyword_summary.csv`
- `reports/tables/topic_frequency.csv`
- `reports/tables/bigram_summary.csv`
- `reports/tables/uncertainty_summary.csv`
- `reports/tables/speaker_sentiment_summary.csv`

Current top keywords include terms such as `quarter`, `iphone`, `year`, `apple`, and `services`.

## Statistical Analysis Summary

The statistical module writes:

- `reports/tables/statistical_summary.csv`
- `reports/tables/correlation_analysis.csv`
- `reports/tables/regression_results.csv`
- `reports/tables/statistical_warnings.csv`

With `n=1`, correlations, regressions, and t-tests are skipped or marked as insufficient sample size.

## Prediction Modeling Status

The prediction module writes:

- `reports/tables/prediction_model_summary.csv`
- `reports/tables/prediction_model_metrics.csv`
- `reports/tables/prediction_model_warnings.csv`
- `reports/tables/prediction_feature_importance.csv`
- `models/predictive_model_metadata.json`

Current status: model training is skipped because the dataset has fewer than 10 rows. No model performance is claimed.

## Dashboard Features

The Streamlit dashboard includes:

- Overview and pipeline status
- Sentiment comparison
- Event-study snapshots
- Advanced NLP tables
- Statistical analysis warnings and outputs
- Prediction modeling status and metadata
- Figures gallery
- Download buttons for generated CSV reports

## Why n=1 Limits Inference

One event cannot support statistical significance, stable correlations, regression estimates, or ML validation. The project is therefore presented as a complete and tested analytical system that is ready to scale, not as a source of validated investment findings from the current sample.
