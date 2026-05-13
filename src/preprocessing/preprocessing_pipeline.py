from __future__ import annotations

QA_PATTERNS = [
    r"question(?:s)?\\s*(?:and|&)\\s*answer(?:s)?",
    r"q\\s*&\\s*a",
    r"question-and-answer",
    r"questions?\\s+and\\s+answers?",
]


MAX_TOKENS = 420
OVERLAP = 60

"""
src/preprocessing/preprocessing_pipeline.py
=============================================
Stage orchestrator — chains all preprocessing stages in the correct order.

Pipeline stages (in order):
    1. TextCleaner          — remove HTML, boilerplate, normalise whitespace
    2. TranscriptSegmenter  — split prepared remarks from Q&A, parse speakers
    3. TranscriptChunker    — sliding-window tokenisation for FinBERT
    4. TranscriptValidator  — quality checks, schema enforcement, report

This module is the single entry point for all preprocessing.
Downstream sentiment modules import from here, never from individual stages.

Usage (simple):
    pipeline = PreprocessingPipeline()
    result   = pipeline.run(transcripts_df)
    chunks   = result.chunks_df
    report   = result.validation_report

Usage (custom):
    pipeline = PreprocessingPipeline(
        clean_kwargs   = {"remove_speaker_labels": False},
        chunk_col      = "qa_text",
        skip_validation= False,
    )
    result = pipeline.run(transcripts_df)

PipelineResult fields:
    transcripts_df    — cleaned transcripts with section columns added
    segments_df       — speaker-turn segments
    chunks_df         — FinBERT-ready chunks
    validation_report — row-level QA report
    summary           — dict of pipeline statistics

TODO:
    - [ ] Add checkpoint saving (resume from any stage after crash)
    - [ ] Add parallel processing via multiprocessing.Pool
    - [ ] Add configurable stage skipping (skip_segmentation=True)
    - [ ] Add MLflow run logging for experiment tracking
    - [ ] Add data drift detection (compare chunk stats to baseline run)
    - [ ] Support streaming mode for very large transcript collections
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.storage import save_parquet

from .chunking import TranscriptChunker
from .text_cleaner import TextCleaner
from .transcript_segmenter import TranscriptSegmenter
from .validation import TranscriptValidator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """
    Container for all outputs from a PreprocessingPipeline.run() call.

    Attributes:
        transcripts_df    : Cleaned + segmented transcript DataFrame
        segments_df       : Speaker-turn segment DataFrame
        chunks_df         : FinBERT-ready chunk DataFrame
        validation_report : Row-level validation results
        summary           : Pipeline statistics dict
        elapsed_seconds   : Wall-clock time for the full pipeline
    """

    transcripts_df: pd.DataFrame
    segments_df: pd.DataFrame
    chunks_df: pd.DataFrame
    validation_report: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def log_summary(self) -> None:
        """Print a human-readable summary to the logger."""
        log.info("=" * 55)
        log.info("  Preprocessing Pipeline — Summary")
        log.info("=" * 55)
        for k, v in self.summary.items():
            log.info(f"  {k:<35}: {v}")
        log.info(f"  {'Elapsed':<35}: {self.elapsed_seconds:.1f}s")
        log.info("=" * 55)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """
    Orchestrates all preprocessing stages end-to-end.

    Instantiate once and call run() for each batch of transcripts.
    All intermediate outputs are optionally persisted to Parquet.

    Args:
        clean_kwargs    : Passed to TextCleaner.__init__()
        chunk_col       : Which text column to use for chunking
                          Default: "prepared_text" (prepared remarks only)
                          Alternative: "transcript_text_clean" (full transcript)
        skip_validation : Set True to skip validation for speed (not recommended)
        save_intermediates: Persist interim DataFrames to data/interim/
    """

    def __init__(
        self,
        clean_kwargs: dict[str, Any] | None = None,
        chunk_col: str = "prepared_text",
        skip_validation: bool = False,
        save_intermediates: bool = True,
    ) -> None:
        self.chunk_col = chunk_col
        self.skip_validation = skip_validation
        self.save_intermediates = save_intermediates

        cfg = load_config()
        self.interim_dir = Path(cfg["paths"]["interim_transcripts"])
        self.chunks_dir = Path(cfg["paths"]["interim_chunks"])
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        cleaner_options = {"remove_speaker_labels": False}
        cleaner_options.update(clean_kwargs or {})
        self.cleaner = TextCleaner(**cleaner_options)
        self.segmenter = TranscriptSegmenter()
        self.chunker = TranscriptChunker()
        self.validator = TranscriptValidator()

        log.info(
            f"PreprocessingPipeline ready "
            f"(chunk_col='{chunk_col}', skip_validation={skip_validation})"
        )

    # ── Public API ───────────────────────────────────────────────

    def run(
        self,
        transcripts_df: pd.DataFrame,
        text_col: str = "transcript_text",
        id_col: str = "transcript_id",
        ticker_col: str = "ticker",
        date_col: str = "earnings_date",
    ) -> PipelineResult:
        """
        Execute all preprocessing stages on a transcript DataFrame.

        Args:
            transcripts_df: Raw transcript DataFrame (output of TranscriptLoader)
            text_col      : Column containing raw transcript text
            id_col        : Unique transcript ID column
            ticker_col    : Ticker symbol column
            date_col      : Earnings date column

        Returns:
            PipelineResult with all outputs and statistics
        """
        t_start = time.perf_counter()
        log.info(f"Pipeline started: {len(transcripts_df):,} transcripts")

        # ── Stage 1: Clean ───────────────────────────────────────
        log.info("Stage 1/4 — Text cleaning")
        cleaned_df = self._stage_clean(transcripts_df, text_col)

        # ── Stage 2: Segment ─────────────────────────────────────
        log.info("Stage 2/4 — Segmentation")
        sections_df, segments_df = self._stage_segment(
            cleaned_df, id_col, ticker_col, date_col
        )

        # ── Stage 3: Chunk ───────────────────────────────────────
        log.info("Stage 3/4 — Chunking")
        chunks_df = self._stage_chunk(segments_df, id_col, ticker_col, date_col)

        # ── Stage 4: Validate ────────────────────────────────────
        log.info("Stage 4/4 — Validation")
        validation_report = self._stage_validate(sections_df, chunks_df)

        elapsed = time.perf_counter() - t_start
        summary = self._build_summary(
            transcripts_df,
            sections_df,
            segments_df,
            chunks_df,
            validation_report,
            elapsed,
        )

        result = PipelineResult(
            transcripts_df=sections_df,
            segments_df=segments_df,
            chunks_df=chunks_df,
            validation_report=validation_report,
            summary=summary,
            elapsed_seconds=elapsed,
        )
        result.log_summary()
        return result

    def run_single(
        self,
        text: str,
        transcript_id: str,
        ticker: str = "",
        earnings_date: str = "",
    ) -> PipelineResult:
        """
        Run the full pipeline on a single transcript string.
        Useful for ad-hoc inspection and debugging.

        Args:
            text         : Raw transcript text
            transcript_id: Unique ID for this transcript
            ticker       : Stock ticker
            earnings_date: Earnings call date string

        Returns:
            PipelineResult for this single transcript
        """
        row = {
            "transcript_id": transcript_id,
            "ticker": ticker,
            "earnings_date": earnings_date,
            "transcript_text": text,
        }
        df = pd.DataFrame([row])
        return self.run(df)

    # ── Stage implementations ────────────────────────────────────

    def _stage_clean(
        self,
        df: pd.DataFrame,
        text_col: str,
    ) -> pd.DataFrame:
        """Stage 1: Clean raw transcript text."""
        cleaned = self.cleaner.clean_dataframe(
            df,
            text_col=text_col,
            out_col="transcript_text_clean",
        )
        if self.save_intermediates:
            save_parquet(cleaned, self.interim_dir / "transcripts_cleaned.parquet")
        return cleaned

    def _stage_segment(
        self,
        df: pd.DataFrame,
        id_col: str,
        ticker_col: str,
        date_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 2: Split sections and extract speaker turns."""
        all_segments = []

        for _, row in df.iterrows():
            segments = self.segmenter.segment_transcript(
                raw_text=row["transcript_text_clean"],
                transcript_id=row[id_col],
            )
            all_segments.extend(segments)

        segments_df = self.segmenter.to_dataframe(all_segments)

        sections_df = df.copy()
        if self.save_intermediates:
            save_parquet(sections_df, self.interim_dir / "transcripts_sections.parquet")
            save_parquet(segments_df, self.interim_dir / "transcript_segments.parquet")
        return sections_df, segments_df

    def _stage_chunk(
        self,
        segments_df: pd.DataFrame,
        id_col: str,
        ticker_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Stage 3: Chunk transcripts for FinBERT inference."""
        all_chunks = []

        for transcript_id, group in segments_df.groupby("transcript_id"):
            ticker = (
                group["ticker"].iloc[0]
                if "ticker" in group.columns else ""
            )

            chunk_df = self.chunker.chunk_transcript(
                segments_df=group,
                transcript_id=transcript_id,
                ticker=ticker,
            )

            all_chunks.append(chunk_df)

        chunks_df = (
            pd.concat(all_chunks, ignore_index=True)
            if all_chunks
            else pd.DataFrame()
        )
        if self.save_intermediates:
            save_parquet(chunks_df, self.chunks_dir / "chunks.parquet")
        return chunks_df

    def _stage_validate(
        self,
        transcripts_df: pd.DataFrame,
        chunks_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stage 4: Validate transcripts and chunks."""
        if self.skip_validation:
            log.warning("Validation skipped (skip_validation=True)")
            return pd.DataFrame()

        transcript_report = self.validator.validate_transcripts(transcripts_df)
        chunk_report = self.validator.validate_chunks(chunks_df)

        # Merge both reports side-by-side on transcript_id
        combined = transcript_report.rename(
            columns={
                "is_valid": "transcript_valid",
                "error_count": "transcript_errors",
                "warning_count": "transcript_warnings",
                "errors": "transcript_error_msgs",
                "warnings": "transcript_warning_msgs",
            }
        )
        if self.save_intermediates:
            save_parquet(combined, self.interim_dir / "validation_report.parquet")

        return combined

    # ── Summary ──────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        raw_df: pd.DataFrame,
        sections_df: pd.DataFrame,
        segments_df: pd.DataFrame,
        chunks_df: pd.DataFrame,
        validation_report: pd.DataFrame,
        elapsed: float,
    ) -> dict[str, Any]:
        """Build a statistics dict for PipelineResult.summary."""
        summary: dict[str, Any] = {
            "input_transcripts": len(raw_df),
            "output_transcripts": len(sections_df),
            "total_segments": len(segments_df),
            "total_chunks": len(chunks_df),
        }

        if not chunks_df.empty and "token_count" in chunks_df.columns:
            summary["avg_tokens_per_chunk"] = round(chunks_df["token_count"].mean(), 1)
            summary["min_tokens"] = int(chunks_df["token_count"].min())
            summary["max_tokens"] = int(chunks_df["token_count"].max())

        if (
            not validation_report.empty
            and "transcript_valid" in validation_report.columns
        ):
            n_valid = validation_report["transcript_valid"].sum()
            summary["valid_transcripts"] = int(n_valid)
            summary["invalid_transcripts"] = int(len(validation_report) - n_valid)

        if not segments_df.empty and "speaker_role" in segments_df.columns:
            role_counts = segments_df["speaker_role"].value_counts().to_dict()
            summary["segments_by_role"] = role_counts

        return summary

    # TODO: add .run_parallel(df, n_workers=4) using multiprocessing
    # TODO: add .resume_from_checkpoint(stage_name) for crash recovery
    # TODO: add .dry_run(df) that validates config without executing
    # TODO: add .get_stage_timings() -> dict[str, float]
    # TODO: add MLflow logging: mlflow.log_params(summary)
