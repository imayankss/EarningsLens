# Project Report: Earnings Call Sentiment Analyzer

## Executive Summary

This project builds a complete financial NLP workflow for earnings-call analysis. It combines contextual transformer sentiment, an explainable finance dictionary baseline, event-study return alignment, advanced NLP summaries, guarded statistical analysis, prediction-model scaffolding, professional visualizations, and a Streamlit dashboard.

The current checked artifact set contains one real transcript/event, `AAPL_20201029`. The repository therefore emphasizes workflow quality, testability, and honest limitations rather than overstated statistical or model claims.

## Motivation

Earnings calls are dense sources of qualitative financial information. Management tone, uncertainty, and topic emphasis can add context to reported earnings. A robust research workflow should connect language signals to market reactions while preserving interpretability and avoiding unsupported conclusions.

## System Design

The system is artifact-driven. Each stage writes explicit processed files or report tables, and downstream stages read those artifacts rather than relying on hidden in-memory state. The dashboard is intentionally read-only over generated outputs.

## Analytical Components

1. Transcript preprocessing and chunking.
2. FinBERT contextual sentiment scoring.
3. Loughran-McDonald dictionary baseline scoring.
4. Model comparison and directional agreement analysis.
5. Market return and event-study feature integration.
6. Master dataset creation.
7. Guarded statistical diagnostics.
8. Advanced NLP feature extraction.
9. Prediction modeling scaffolding with sample-size safeguards.
10. Streamlit dashboard and figure gallery.

## Current Results

The current one-event dataset demonstrates the full flow but does not support inference. The pipeline produces a master dataset, report tables, PNG figures, advanced NLP outputs, and prediction metadata. Statistical and ML modules correctly skip underpowered analyses.

## Engineering Notes

The repository includes pytest coverage for the major post-Day-10 modules, safe loaders, Streamlit helpers, and a repo health check. The dashboard and documentation are designed for portfolio review and clear communication of limitations.

## Limitations and Next Steps

The main limitation is sample size. Future work should scale transcript ingestion across more tickers and quarters, then revisit statistical inference, model validation, and deployment.
