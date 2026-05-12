"""
scripts/run_pipeline.py
========================
Full end-to-end pipeline orchestrator.

Runs all stages in order:
  Stage 1 — Download transcripts from HuggingFace
  Stage 2 — Download market prices from yfinance
  Stage 3 — Clean and preprocess transcripts
  Stage 4 — Chunk transcripts for FinBERT
  Stage 5 — Run FinBERT inference
  Stage 6 — Compute LM baseline scores
  Stage 7 — Aggregate sentiment to transcript level
  Stage 8 — Build event study dataset
  Stage 9 — Compute correlations and summary statistics

Usage (from project root with venv active):
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --max-rows 100   # fast dev run
    python scripts/run_pipeline.py --tickers AAPL MSFT GOOGL

Output:
    data/processed/sentiment/sentiment_features.parquet
    data/processed/event_study/event_study.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

# ── Make src importable from scripts/ ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config, get_all_tickers, get_date_range
from src.utils.logger import get_logger
from src.utils.storage import save_parquet

from src.ingestion.transcript_loader import TranscriptLoader
from src.ingestion.market_data_loader import MarketDataLoader
from src.preprocessing.cleaner import TranscriptCleaner
from src.preprocessing.chunker import TranscriptChunker
from src.sentiment.finbert_pipeline import FinBERTPipeline
from src.sentiment.lm_baseline import LoughranMcDonaldBaseline
from src.sentiment.aggregator import SentimentAggregator
from src.event_study.engine import EventStudyEngine

log = get_logger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Earnings Call Sentiment Pipeline")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override tickers (e.g. --tickers AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit transcript rows for fast dev/test runs"
    )
    parser.add_argument(
        "--skip-finbert", action="store_true",
        help="Skip FinBERT inference (use if model not downloaded)"
    )
    parser.add_argument(
        "--skip-lm", action="store_true",
        help="Skip LM baseline (use if dictionary not downloaded)"
    )
    return parser.parse_args()


def stage(name: str) -> None:
    log.info("")
    log.info(f"{'='*60}")
    log.info(f"  {name}")
    log.info(f"{'='*60}")


def main() -> None:
    args   = parse_args()
    config = load_config()
    t0     = perf_counter()

    tickers        = args.tickers or get_all_tickers()
    start, end     = get_date_range()
    processed_dir  = Path(config["paths"]["processed_sentiment"])
    event_dir      = Path(config["paths"]["processed_event"])
    interim_dir    = Path(config["paths"]["interim_chunks"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Tickers : {tickers}")
    log.info(f"Dates   : {start} → {end}")
    log.info(f"Max rows: {args.max_rows or 'all'}")

    # ── Stage 1: Download transcripts ──────────────────────────
    stage("Stage 1 — Download Transcripts")
    loader = TranscriptLoader()
    transcripts_df = loader.load_from_huggingface(
        tickers=tickers,
        max_rows=args.max_rows,
    )
    loader.save_raw(transcripts_df)
    log.info(f"Transcripts loaded: {len(transcripts_df):,} rows")

    # ── Stage 2: Download market prices ─────────────────────────
    stage("Stage 2 — Download Market Prices")
    market_loader = MarketDataLoader()
    prices  = market_loader.download_prices(tickers, start, end)
    returns = market_loader.compute_returns(prices)
    market_loader.save_prices(prices)
    log.info(f"Prices: {prices.shape}")

    # ── Stage 3: Clean transcripts ──────────────────────────────
    stage("Stage 3 — Clean Transcripts")
    cleaner     = TranscriptCleaner()
    cleaned_df  = cleaner.clean_dataframe(transcripts_df)
    save_parquet(
        cleaned_df,
        "data/interim/transcripts/transcripts_cleaned.parquet",
    )

    # ── Stage 4: Chunk transcripts ──────────────────────────────
    stage("Stage 4 — Chunk Transcripts")
    chunker   = TranscriptChunker()
    chunks_df = chunker.chunk_dataframe(cleaned_df, section_col="prepared_remarks")
    save_parquet(chunks_df, "data/interim/chunks/chunks.parquet")
    log.info(f"Chunks generated: {len(chunks_df):,}")

    # ── Stage 5: FinBERT inference ──────────────────────────────
    finbert_scores = None
    if not args.skip_finbert:
        stage("Stage 5 — FinBERT Inference")
        pipeline       = FinBERTPipeline()
        finbert_scores = pipeline.score_chunks(chunks_df)
        pipeline.save_scores(
            finbert_scores,
            "data/processed/sentiment/finbert_chunk_scores.parquet",
        )
    else:
        log.warning("Stage 5 skipped (--skip-finbert flag set)")

    # ── Stage 6: LM baseline ────────────────────────────────────
    lm_scores = None
    if not args.skip_lm:
        stage("Stage 6 — Loughran-McDonald Baseline")
        try:
            lm = LoughranMcDonaldBaseline()
            lm_scores = lm.score_dataframe(cleaned_df, text_col="transcript_text")
            save_parquet(
                lm_scores,
                "data/processed/sentiment/lm_scores.parquet",
            )
        except FileNotFoundError as e:
            log.warning(f"LM dictionary not found — skipping. {e}")
            lm_scores = None
    else:
        log.warning("Stage 6 skipped (--skip-lm flag set)")

    # ── Stage 7: Aggregate sentiment ────────────────────────────
    stage("Stage 7 — Aggregate Sentiment")
    aggregator = SentimentAggregator()

    if finbert_scores is not None:
        finbert_agg = aggregator.aggregate_finbert(finbert_scores)
        feature_store = aggregator.merge_features(
            transcripts_df = cleaned_df,
            finbert_df     = finbert_agg,
            lm_df          = lm_scores,
        )
        save_parquet(
            feature_store,
            "data/processed/sentiment/sentiment_features.parquet",
        )
        log.info(f"Feature store: {feature_store.shape}")
    else:
        log.warning("No FinBERT scores — skipping feature store merge")
        feature_store = cleaned_df  # fallback: carry metadata only

    # ── Stage 8: Event study ────────────────────────────────────
    stage("Stage 8 — Event Study")
    engine = EventStudyEngine()

    if "finbert_score" in feature_store.columns:
        event_df = engine.build_event_dataset(feature_store, returns)
        save_parquet(
            event_df,
            "data/processed/event_study/event_study.parquet",
        )

        # ── Stage 9: Correlations ────────────────────────────────
        stage("Stage 9 — Correlations & Summary Stats")
        corr_df    = engine.sentiment_return_correlation(event_df)
        summary_df = engine.summary_stats(event_df)
        save_parquet(corr_df,    "data/processed/event_study/correlations.parquet")
        save_parquet(summary_df, "data/processed/event_study/summary_stats.parquet")

        log.info("\nCorrelation Matrix:")
        log.info(f"\n{corr_df.to_string(index=False)}")

        log.info("\nSummary Statistics:")
        log.info(f"\n{summary_df.to_string(index=False)}")
    else:
        log.warning("No finbert_score in feature store — skipping event study")

    elapsed = perf_counter() - t0
    log.info("")
    log.info(f"Pipeline complete in {elapsed/60:.1f} minutes")
    log.info("Launch dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
