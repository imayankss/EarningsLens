"""
src/sentiment/lm_pipeline.py
=============================
Master orchestration pipeline for the Loughran-McDonald explainable
financial sentiment baseline system.

Architecture role
-----------------
Coordinates all LM subsystem components into a single deterministic,
cache-aware, validation-driven pipeline that transforms raw transcript
chunk data into research-grade sentiment outputs.

Pipeline position
-----------------
  data/interim/transcript_chunks.parquet
  data/interim/segmented_transcripts.parquet
        │
        ▼
  LMPipeline                    ← THIS FILE
  ├── LMDictionaryLoader
  ├── LMPreprocessor
  ├── LMDictionaryMatcher
  ├── LMScorer
  ├── LMAggregation
  └── LMValidation
        │
        ▼
  data/processed/sentiment/
  ├── lm_scores.parquet              # compatibility contract for event study
  ├── lm_chunk_scores.parquet / .csv
  ├── lm_transcript_scores.parquet / .csv
  ├── lm_section_scores.parquet / .csv
  ├── lm_speaker_scores.parquet / .csv
  └── lm_diagnostics.parquet / .csv

Integration contract
--------------------
All subsystem modules are imported defensively — if a component is not
yet implemented, the pipeline degrades gracefully with stub hooks and
clear error messages rather than crashing at import time.

Design principles
-----------------
* Deterministic: same inputs → same outputs every run.
* Cache-aware: completed parquet outputs are reused unless overwrite=True.
* Resume-safe: each stage writes its own intermediate before advancing.
* Validation-driven: every stage gate is guarded by schema + range checks.
* Parquet-first: all inter-stage communication uses parquet; CSV is a
  human-readable companion export, never the primary format.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive subsystem imports
# ---------------------------------------------------------------------------
# Each import is wrapped so the orchestrator can report a clear missing-
# module error at pipeline.run() time rather than at import time.
# Replace each try/except with a direct import once the module exists.
# ---------------------------------------------------------------------------

try:
    from .lm_preprocessing import LMPreprocessor, LMPreprocessingConfig
    _PREPROCESSOR_AVAILABLE = True
except ImportError:
    try:
        from lm_preprocessing import LMPreprocessor, LMPreprocessingConfig  # type: ignore
        _PREPROCESSOR_AVAILABLE = True
    except ImportError:
        _PREPROCESSOR_AVAILABLE = False
        logger.debug("lm_preprocessing not yet available — stub mode.")

try:
    from .lm_scoring import LMScorer, LMScoringConfig, SentimentLabel
    _SCORER_AVAILABLE = True
except ImportError:
    try:
        from lm_scoring import LMScorer, LMScoringConfig, SentimentLabel  # type: ignore
        _SCORER_AVAILABLE = True
    except ImportError:
        _SCORER_AVAILABLE = False
        logger.debug("lm_scoring not yet available — stub mode.")

try:
    from .lm_dictionary_loader import LMDictionaryConfig, LMDictionaryLoader
    _DICTIONARY_LOADER_AVAILABLE = True
except ImportError:
    try:
        from lm_dictionary_loader import LMDictionaryConfig, LMDictionaryLoader  # type: ignore
        _DICTIONARY_LOADER_AVAILABLE = True
    except ImportError:
        _DICTIONARY_LOADER_AVAILABLE = False
        logger.debug("lm_dictionary_loader not available.")

try:
    from .lm_matcher import LMMatcher, LMMatcherConfig
    _MATCHER_AVAILABLE = True
except ImportError:
    try:
        from lm_matcher import LMMatcher, LMMatcherConfig  # type: ignore
        _MATCHER_AVAILABLE = True
    except ImportError:
        _MATCHER_AVAILABLE = False
        logger.debug("lm_matcher not available.")


# ===========================================================================
# ENUMS
# ===========================================================================


class PipelineStage(str, Enum):
    INIT         = "init"
    LOAD_DICT    = "load_dictionary"
    LOAD_INPUTS  = "load_inputs"
    PREPROCESS   = "preprocess"
    MATCH        = "match"
    SCORE        = "score"
    AGGREGATE    = "aggregate"
    VALIDATE     = "validate"
    EXPORT       = "export"
    DIAGNOSTICS  = "diagnostics"
    COMPLETE     = "complete"


class StageStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    SKIPPED  = "skipped"   # cache hit
    FAILED   = "failed"


# ===========================================================================
# CONFIGURATION
# ===========================================================================


@dataclass
class LMPipelineConfig:
    """
    Immutable end-to-end pipeline policy.

    All path, batch, and caching decisions live here so that lm_pipeline.py
    itself contains no hardcoded paths or magic numbers.
    """

    # ── Input paths ────────────────────────────────────────────────────
    chunks_path: Path = Path("data/interim/chunks/chunks.parquet")
    segments_path: Path = Path("data/interim/segmented_transcripts.parquet")
    dictionary_path: Path = Path(
        "data/dictionaries/loughran_mcdonald/LM_MasterDictionary.csv"
    )

    # ── Output root ────────────────────────────────────────────────────
    output_dir: Path = Path("data/processed/sentiment")

    # ── Output file stems (no suffix — appended at export) ─────────────
    chunk_scores_stem:      str = "lm_chunk_scores"
    transcript_scores_stem: str = "lm_transcript_scores"
    compatibility_scores_stem: str = "lm_scores"
    section_scores_stem:    str = "lm_section_scores"
    speaker_scores_stem:    str = "lm_speaker_scores"
    diagnostics_stem:       str = "lm_diagnostics"

    # ── Intermediate cache ─────────────────────────────────────────────
    interim_dir: Path = Path("data/interim")
    cache_preprocessed: bool = True    # save preprocessed token lists
    cache_match_results: bool = True   # save raw match-count rows

    # ── Execution policy ──────────────────────────────────────────────
    overwrite: bool = False            # False = reuse existing parquet outputs
    batch_size: int = 256              # chunks per processing batch
    max_chunks: Optional[int] = None   # None = process all (set for dev runs)
    allow_stub_dictionary: bool = False

    # ── Column names (allow schema remapping without code changes) ─────
    chunk_id_col:      str = "chunk_id"
    transcript_id_col: str = "transcript_id"
    ticker_col:        str = "ticker"
    chunk_text_col:    str = "chunk_text"
    section_type_col:  str = "section_type"
    speaker_col:       str = "dominant_speaker"
    speaker_role_col:  str = "speaker_role"
    token_count_col:   str = "token_count"

    # ── Required input columns ────────────────────────────────────────
    required_chunk_cols: Tuple[str, ...] = (
        "chunk_id", "transcript_id", "chunk_text",
        "section_type", "token_count",
    )
    required_segment_cols: Tuple[str, ...] = (
        "transcript_id", "speaker", "speaker_role",
        "section_type", "text",
    )

    # ── Sub-component configs ─────────────────────────────────────────
    preprocessing_config: Optional[object] = None   # LMPreprocessingConfig
    scoring_config: Optional[object]       = None   # LMScoringConfig

    # ── Validation thresholds ─────────────────────────────────────────
    min_coverage_ratio: float = 0.005   # global low-coverage warning floor
    max_invalid_score_fraction: float = 0.10  # fail if >10% scores invalid

    # ── Export ────────────────────────────────────────────────────────
    export_csv: bool = True
    export_parquet: bool = True

    def __post_init__(self) -> None:
        self.chunks_path  = Path(self.chunks_path)
        self.segments_path = Path(self.segments_path)
        self.dictionary_path = Path(self.dictionary_path)
        self.output_dir   = Path(self.output_dir)
        self.interim_dir  = Path(self.interim_dir)

    def output_path(self, stem: str, suffix: str) -> Path:
        return self.output_dir / f"{stem}.{suffix}"

    def input_candidates(self) -> Tuple[Path, ...]:
        """Chunk input search order for the official LM pipeline contract."""
        ordered = (
            self.chunks_path,
            Path("data/interim/transcript_chunks.parquet"),
            self.segments_path,
        )
        unique: List[Path] = []
        for path in ordered:
            p = Path(path)
            if p not in unique:
                unique.append(p)
        return tuple(unique)


# ===========================================================================
# STAGE RESULT
# ===========================================================================


@dataclass
class PipelineStageResult:
    """Diagnostic record for a single pipeline stage execution."""

    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    rows_in: int = 0
    rows_out: int = 0
    elapsed_seconds: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def mark_running(self) -> float:
        self.status = StageStatus.RUNNING
        return time.perf_counter()

    def mark_complete(self, t0: float, rows_out: int = 0) -> None:
        self.status = StageStatus.COMPLETE
        self.elapsed_seconds = time.perf_counter() - t0
        self.rows_out = rows_out

    def mark_failed(self, t0: float, error: str) -> None:
        self.status = StageStatus.FAILED
        self.elapsed_seconds = time.perf_counter() - t0
        self.error = error

    def mark_skipped(self, reason: str = "cache hit") -> None:
        self.status = StageStatus.SKIPPED
        self.cache_hit = True
        self.notes.append(reason)

    def summary_line(self) -> str:
        flag = "✓" if self.status == StageStatus.COMPLETE else (
               "⟳" if self.status == StageStatus.SKIPPED else (
               "✗" if self.status == StageStatus.FAILED else "…"))
        cache = " [cached]" if self.cache_hit else ""
        err   = f" ERROR: {self.error}" if self.error else ""
        return (
            f"  {flag} {self.stage.value:<20} | "
            f"in={self.rows_in:>7,} out={self.rows_out:>7,} | "
            f"{self.elapsed_seconds:>6.2f}s{cache}{err}"
        )


# ===========================================================================
# PIPELINE RESULT
# ===========================================================================


@dataclass
class LMPipelineResult:
    """
    End-to-end result object returned by LMPipeline.run().

    Contains stage audit trail, exported paths, and top-level diagnostics.
    Consumed by lm_validation.py and downstream event-study merge steps.
    """

    success: bool = False
    total_elapsed_seconds: float = 0.0

    # Stage audit trail
    stages: Dict[str, PipelineStageResult] = field(default_factory=dict)

    # Row counts at each layer
    chunks_loaded:      int = 0
    chunks_processed:   int = 0
    transcripts_scored: int = 0
    sections_scored:    int = 0
    speakers_scored:    int = 0

    # Validation
    validation_passed: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    low_coverage_transcripts: List[str] = field(default_factory=list)
    invalid_score_transcripts: List[str] = field(default_factory=list)

    # Score distribution snapshots
    score_mean:   float = 0.0
    score_std:    float = 0.0
    score_median: float = 0.0

    label_counts: Dict[str, int] = field(
        default_factory=lambda: {"positive": 0, "neutral": 0,
                                  "negative": 0, "unknown": 0}
    )
    coverage_mean:    float = 0.0
    low_coverage_pct: float = 0.0

    # Exported file paths
    exported_paths: Dict[str, Path] = field(default_factory=dict)

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            "",
            f"══ LMPipeline Result — {status} "
            f"({'%.2f' % self.total_elapsed_seconds}s) ══",
            "  Stage audit:",
        ]
        for sr in self.stages.values():
            lines.append(sr.summary_line())

        lines += [
            "  ──────────────────────────────────────────────────────",
            f"  Chunks loaded      : {self.chunks_loaded:,}",
            f"  Chunks processed   : {self.chunks_processed:,}",
            f"  Transcripts scored : {self.transcripts_scored:,}",
            f"  Sections scored    : {self.sections_scored:,}",
            f"  Speakers scored    : {self.speakers_scored:,}",
            f"  Validation passed  : {self.validation_passed}",
            f"  Validation warnings: {len(self.validation_warnings)}",
            f"  Low-cov transcripts: {len(self.low_coverage_transcripts)}",
            "  Score distribution :",
            f"    mean={self.score_mean:+.4f}  "
            f"std={self.score_std:.4f}  "
            f"median={self.score_median:+.4f}",
            f"  Label counts       : {self.label_counts}",
            f"  Coverage mean      : {self.coverage_mean:.4f}",
            f"  Low-cov %          : {self.low_coverage_pct:.1f}%",
            "  Exported paths:",
        ]
        for key, path in self.exported_paths.items():
            lines.append(f"    {key:<30}: {path}")
        lines.append("══════════════════════════════════════════════════════")
        return "\n".join(lines)


# ===========================================================================
# PIPELINE
# ===========================================================================


class LMPipeline:
    """
    Master orchestrator for the Loughran-McDonald explainable sentiment pipeline.

    Responsibilities
    ----------------
    1. Load LM dictionary via LMDictionaryLoader
    2. Load chunk / segment parquet inputs
    3. Preprocess text via LMPreprocessor (batch)
    4. Match tokens against LM categories via LMDictionaryMatcher
    5. Compute LMScore per chunk via LMScorer
    6. Aggregate to transcript / section / speaker level via LMAggregation
    7. Validate all output layers via LMValidation
    8. Export parquet + CSV outputs
    9. Generate and log diagnostics

    Execution model
    ---------------
    run() is the single entry point. Each stage records a PipelineStageResult
    and writes its output before the next stage begins, so a failure at any
    stage leaves all prior outputs intact (resume-safe).

    Caching
    -------
    If config.overwrite=False and the target parquet file already exists, the
    stage is marked SKIPPED and its existing output is loaded directly. This
    makes iterative development fast: fix the aggregation layer, re-run, and
    only re-execute from Stage 6 onward.

    Usage
    -----
    >>> config = LMPipelineConfig(overwrite=True, batch_size=128)
    >>> pipeline = LMPipeline(config)
    >>> result = pipeline.run()
    >>> print(result.summary())
    """

    def __init__(self, config: Optional[LMPipelineConfig] = None) -> None:
        self.config = config or LMPipelineConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.interim_dir.mkdir(parents=True, exist_ok=True)

        # Sub-components (initialised lazily inside run())
        self._preprocessor = None
        self._scorer = None
        self._matcher = None
        self._dictionary = None   # LMDictionary object from loader

        logger.info(
            "LMPipeline initialised | overwrite=%s | batch_size=%d",
            self.config.overwrite,
            self.config.batch_size,
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> LMPipelineResult:
        """
        Execute the full LM sentiment pipeline end-to-end.

        Returns LMPipelineResult regardless of whether individual stages
        failed — inspect result.success and result.stages for details.
        """
        result  = LMPipelineResult()
        wall_t0 = time.perf_counter()

        try:
            # Stage 1 — Load dictionary
            dict_stage, lm_dict = self._stage_load_dictionary()
            result.stages[PipelineStage.LOAD_DICT.value] = dict_stage
            if dict_stage.status == StageStatus.FAILED:
                result.success = False
                return result

            # Stage 2 — Load inputs
            input_stage, chunks_df = self.load_inputs()
            result.stages[PipelineStage.LOAD_INPUTS.value] = input_stage
            result.chunks_loaded = len(chunks_df)
            if input_stage.status == StageStatus.FAILED:
                result.success = False
                return result

            if self.config.max_chunks:
                chunks_df = chunks_df.iloc[: self.config.max_chunks].copy()
                logger.info("max_chunks=%d applied; %d rows retained.",
                            self.config.max_chunks, len(chunks_df))

            # Stage 3 — Preprocess + Match + Score (combined hot path)
            score_stage, chunk_scores_df = self.process_chunks(chunks_df, lm_dict)
            result.stages[PipelineStage.SCORE.value] = score_stage
            result.chunks_processed = len(chunk_scores_df)
            if score_stage.status == StageStatus.FAILED:
                result.success = False
                return result

            # Stage 4 — Aggregate
            agg_stage, agg_outputs = self.aggregate_outputs(chunk_scores_df)
            result.stages[PipelineStage.AGGREGATE.value] = agg_stage
            transcript_df = agg_outputs.get("transcript", pd.DataFrame())
            section_df    = agg_outputs.get("section",    pd.DataFrame())
            speaker_df    = agg_outputs.get("speaker",    pd.DataFrame())
            result.transcripts_scored = len(transcript_df)
            result.sections_scored    = len(section_df)
            result.speakers_scored    = len(speaker_df)

            # Stage 5 — Validate
            val_stage = self.validate_outputs(
                chunk_scores_df, transcript_df, result
            )
            result.stages[PipelineStage.VALIDATE.value] = val_stage
            result.validation_passed = val_stage.status != StageStatus.FAILED

            # Stage 6 — Export
            export_stage = self.export_outputs(
                chunk_scores_df, transcript_df, section_df, speaker_df, result
            )
            result.stages[PipelineStage.EXPORT.value] = export_stage

            # Stage 7 — Diagnostics
            diag_stage = self.generate_diagnostics(
                chunk_scores_df, transcript_df, result
            )
            result.stages[PipelineStage.DIAGNOSTICS.value] = diag_stage

            result.success = (
                score_stage.status in {StageStatus.COMPLETE, StageStatus.SKIPPED}
                and export_stage.status != StageStatus.FAILED
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("LMPipeline.run() raised an unhandled exception.")
            result.success = False
            result.validation_warnings.append(f"Unhandled pipeline error: {exc}")

        finally:
            result.total_elapsed_seconds = time.perf_counter() - wall_t0
            logger.info(
                "Pipeline finished | success=%s | elapsed=%.2fs",
                result.success,
                result.total_elapsed_seconds,
            )

        return result

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_load_dictionary(
        self,
    ) -> Tuple[PipelineStageResult, Optional[object]]:
        """
        Stage 1 — Load and validate the LM master dictionary.

        Loads the real LM dictionary by default. Stub mode is available only
        when explicitly enabled for tests or demos.
        """
        stage = PipelineStageResult(stage=PipelineStage.LOAD_DICT)
        t0 = stage.mark_running()
        logger.info("[Stage 1/7] Loading LM dictionary from %s",
                    self.config.dictionary_path)

        try:
            if self.config.dictionary_path.exists():
                if not _DICTIONARY_LOADER_AVAILABLE:
                    raise ImportError(
                        "LMDictionaryLoader is unavailable; cannot load the "
                        "Loughran-McDonald dictionary."
                    )
                dict_cfg = LMDictionaryConfig(
                    dictionary_dir=self.config.dictionary_path.parent,
                    filename=self.config.dictionary_path.name,
                )
                loader = LMDictionaryLoader(dict_cfg)
                lm_dict = loader.load()
            elif self.config.allow_stub_dictionary:
                lm_dict = _StubDictionary()
                stage.notes.append("Using explicitly enabled stub dictionary.")
                logger.warning(
                    "Stub LM dictionary active because allow_stub_dictionary=True."
                )
            else:
                raise FileNotFoundError(
                    "Download the Loughran-McDonald Master Dictionary CSV and "
                    f"place it at {self.config.dictionary_path}"
                )

            stage.mark_complete(t0, rows_out=len(lm_dict))
            logger.info("Dictionary loaded: %d terms.", len(lm_dict))
            return stage, lm_dict

        except Exception as exc:
            stage.mark_failed(t0, str(exc))
            logger.error("Dictionary load failed: %s", exc)
            return stage, None

    def load_inputs(self) -> Tuple[PipelineStageResult, pd.DataFrame]:
        """
        Stage 2 — Load and validate chunk parquet inputs.

        Falls back to segments parquet when chunks parquet is absent.
        Applies required-column validation before returning.
        """
        stage = PipelineStageResult(stage=PipelineStage.LOAD_INPUTS)
        t0 = stage.mark_running()
        logger.info("[Stage 2/7] Loading inputs")

        try:
            df = pd.DataFrame()
            loaded_path: Optional[Path] = None
            checked_paths = self.config.input_candidates()

            for candidate in checked_paths:
                if not candidate.exists():
                    continue
                loaded_path = candidate
                logger.info("Loading LM chunks from %s", candidate)
                df = pd.read_parquet(candidate)
                if self.config.chunk_text_col not in df.columns and "text" in df.columns:
                    df = self._remap_segments_to_chunks(df)
                    stage.notes.append(
                        f"Loaded {len(df):,} rows from {candidate} and remapped segments."
                    )
                else:
                    stage.notes.append(f"Loaded {len(df):,} rows from {candidate}.")
                break

            if loaded_path is None:
                raise FileNotFoundError(
                    "No LM input parquet found. Checked: "
                    + ", ".join(str(p) for p in checked_paths)
                )

            stage.rows_in = len(df)

            # Schema validation
            missing = [
                c for c in self.config.required_chunk_cols
                if c not in df.columns
            ]
            if missing:
                raise ValueError(
                    f"Input parquet missing required columns: {missing}"
                )

            # Drop rows with null text
            before = len(df)
            df = df[df[self.config.chunk_text_col].notna()].copy()
            df = df[df[self.config.chunk_text_col].str.strip().astype(bool)].copy()
            dropped = before - len(df)
            if dropped:
                logger.warning("Dropped %d rows with null/empty chunk_text.", dropped)
                stage.notes.append(f"{dropped} null/empty text rows dropped.")

            # Deterministic ordering
            df = df.sort_values(
                [self.config.transcript_id_col, self.config.chunk_id_col],
                ascending=True,
            ).reset_index(drop=True)

            stage.mark_complete(t0, rows_out=len(df))
            logger.info("Inputs loaded: %d chunks from %d transcripts.",
                        len(df),
                        df[self.config.transcript_id_col].nunique())
            return stage, df

        except Exception as exc:
            stage.mark_failed(t0, str(exc))
            logger.error("Input loading failed: %s", exc)
            return stage, pd.DataFrame()

    def process_chunks(
        self,
        chunks_df: pd.DataFrame,
        lm_dict: object,
    ) -> Tuple[PipelineStageResult, pd.DataFrame]:
        """
        Stage 3 — Preprocess, match, and score every chunk.

        Executes in config.batch_size batches to bound memory usage.
        Writes intermediate parquet after processing (cache point).

        Returns a DataFrame with one scored row per chunk, containing
        all LM metric columns alongside the original chunk metadata.
        """
        stage = PipelineStageResult(stage=PipelineStage.SCORE)
        t0 = stage.mark_running()
        stage.rows_in = len(chunks_df)
        logger.info("[Stage 3/7] Processing %d chunks (batch_size=%d)",
                    len(chunks_df), self.config.batch_size)

        cache_path = self.config.interim_dir / "lm_chunk_scores_cache.parquet"

        # Cache hit check
        if not self.config.overwrite and cache_path.exists():
            logger.info("Cache hit: loading chunk scores from %s", cache_path)
            df = pd.read_parquet(cache_path)
            stage.mark_skipped(f"Loaded {len(df):,} rows from cache.")
            stage.rows_out = len(df)
            return stage, df

        # Initialise sub-components
        self._init_preprocessor()
        self._init_scorer()

        scored_batches: List[pd.DataFrame] = []
        total = len(chunks_df)
        batch_size = self.config.batch_size

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch     = chunks_df.iloc[batch_start:batch_end]

            logger.debug(
                "Batch %d–%d of %d", batch_start + 1, batch_end, total
            )
            batch_scored = self._score_batch(batch, lm_dict)
            scored_batches.append(batch_scored)

        if not scored_batches:
            stage.mark_failed(t0, "No batches produced — empty input?")
            return stage, pd.DataFrame()

        chunk_scores_df = pd.concat(scored_batches, ignore_index=True)

        # Write intermediate cache
        if self.config.cache_match_results:
            _write_parquet(chunk_scores_df, cache_path)
            logger.info("Chunk scores cached to %s", cache_path)

        stage.mark_complete(t0, rows_out=len(chunk_scores_df))
        logger.info(
            "Chunk scoring complete: %d rows in %.2fs "
            "(%.0f chunks/s)",
            len(chunk_scores_df),
            stage.elapsed_seconds,
            len(chunk_scores_df) / max(stage.elapsed_seconds, 1e-9),
        )
        return stage, chunk_scores_df

    def compute_scores(self, tokens: List[str], lm_dict: object) -> Dict[str, object]:
        """
        Match tokens against LM dictionary categories and compute LMScore.

        This is a chunk-level helper called inside _score_batch().
        Integrates with LMDictionaryMatcher + LMScorer.

        Returns a flat dict of score columns ready for DataFrame insertion.
        """
        # ── Real matcher path (uncomment when lm_matcher.py exists) ──
        # from lm_matcher import LMDictionaryMatcher
        # matcher = LMDictionaryMatcher(lm_dict)
        # match_result = matcher.match(tokens)
        # ─────────────────────────────────────────────────────────────

        if hasattr(lm_dict, "match"):
            match_result = lm_dict.match(tokens)  # type: ignore[union-attr]
        else:
            if not _MATCHER_AVAILABLE:
                raise ImportError(
                    "LMMatcher is unavailable; cannot score a real LM dictionary."
                )
            if self._matcher is None:
                self._matcher = LMMatcher(lm_dict, LMMatcherConfig())
            match_result = self._matcher.match(tokens)

        if _SCORER_AVAILABLE and self._scorer is not None:
            score_obj = self._scorer.compute_score(
                positive_count     = match_result.get("positive_count",     0),
                negative_count     = match_result.get("negative_count",     0),
                uncertainty_count  = match_result.get("uncertainty_count",  0),
                litigious_count    = match_result.get("litigious_count",    0),
                strong_modal_count = match_result.get("strong_modal_count", 0),
                weak_modal_count   = match_result.get("weak_modal_count",   0),
                constraining_count = match_result.get("constraining_count", 0),
                total_tokens       = match_result.get("total_tokens",       len(tokens)),
                matched_tokens     = match_result.get("matched_tokens",     0),
            )
            return score_obj.to_dict()

        # Fallback when scorer unavailable
        return _stub_score_dict(match_result)

    def aggregate_outputs(
        self,
        chunk_scores_df: pd.DataFrame,
    ) -> Tuple[PipelineStageResult, Dict[str, pd.DataFrame]]:
        """
        Stage 4 — Aggregate chunk-level scores to transcript / section /
        speaker level using pandas groupby operations.

        Calls LMAggregation when available; falls back to an internal
        pandas implementation so the pipeline remains functional during
        incremental development.

        Returns
        -------
        stage   : PipelineStageResult
        outputs : dict with keys "transcript", "section", "speaker"
        """
        stage = PipelineStageResult(stage=PipelineStage.AGGREGATE)
        t0 = stage.mark_running()
        stage.rows_in = len(chunk_scores_df)
        logger.info("[Stage 4/7] Aggregating chunk scores")

        try:
            # ── Real aggregation path ─────────────────────────────────
            # from lm_aggregation import LMAggregation
            # aggregator = LMAggregation()
            # transcript_df = aggregator.aggregate_transcript(chunk_scores_df)
            # section_df    = aggregator.aggregate_section(chunk_scores_df)
            # speaker_df    = aggregator.aggregate_speaker(chunk_scores_df)
            # ─────────────────────────────────────────────────────────

            transcript_df = self._aggregate_transcript(chunk_scores_df)
            section_df    = self._aggregate_section(chunk_scores_df)
            speaker_df    = self._aggregate_speaker(chunk_scores_df)

            outputs = {
                "transcript": transcript_df,
                "section":    section_df,
                "speaker":    speaker_df,
            }

            total_rows = sum(len(v) for v in outputs.values())
            stage.mark_complete(t0, rows_out=total_rows)
            logger.info(
                "Aggregation complete: %d transcript, %d section, %d speaker rows.",
                len(transcript_df), len(section_df), len(speaker_df),
            )
            return stage, outputs

        except Exception as exc:
            stage.mark_failed(t0, str(exc))
            logger.error("Aggregation failed: %s", exc)
            return stage, {}

    def validate_outputs(
        self,
        chunk_scores_df: pd.DataFrame,
        transcript_df: pd.DataFrame,
        result: LMPipelineResult,
    ) -> PipelineStageResult:
        """
        Stage 5 — Validate schema, score ranges, coverage, and duplicates
        across all output layers.

        Populates result.validation_warnings and result.low_coverage_transcripts.
        Does not raise — validation failures are recorded and reported so the
        pipeline can still export whatever was produced.
        """
        stage = PipelineStageResult(stage=PipelineStage.VALIDATE)
        t0 = stage.mark_running()
        stage.rows_in = len(chunk_scores_df)
        logger.info("[Stage 5/7] Validating outputs")

        warnings_list: List[str] = []

        try:
            # ── Real validation path ──────────────────────────────────
            # from lm_validation import LMValidation
            # validator = LMValidation(self.config)
            # val_report = validator.validate(chunk_scores_df, transcript_df)
            # ─────────────────────────────────────────────────────────

            # ── Built-in validation checks ───────────────────────────

            # 1. Required chunk columns
            required = [
                "lm_tone_score", "lm_label", "lm_coverage_ratio",
                self.config.transcript_id_col, self.config.chunk_id_col,
            ]
            for col in required:
                if col not in chunk_scores_df.columns:
                    warnings_list.append(f"Missing column in chunk scores: '{col}'")

            if "lm_tone_score" in chunk_scores_df.columns:
                # 2. Score range [-1, +1]
                out_of_range = chunk_scores_df[
                    chunk_scores_df["lm_tone_score"].abs() > 1.0 + 1e-9
                ]
                if not out_of_range.empty:
                    ids = out_of_range[self.config.transcript_id_col].unique().tolist()
                    warnings_list.append(
                        f"{len(out_of_range)} chunks have |lm_tone_score| > 1.0 "
                        f"(transcripts: {ids[:5]})"
                    )
                    result.invalid_score_transcripts.extend(ids)

                # 3. Score distribution
                scores = chunk_scores_df["lm_tone_score"].dropna()
                result.score_mean   = float(scores.mean())
                result.score_std    = float(scores.std())
                result.score_median = float(scores.median())

            # 4. Coverage check
            if "lm_coverage_ratio" in chunk_scores_df.columns:
                low_cov = chunk_scores_df[
                    chunk_scores_df["lm_coverage_ratio"]
                    < self.config.min_coverage_ratio
                ]
                result.coverage_mean = float(
                    chunk_scores_df["lm_coverage_ratio"].mean()
                )
                result.low_coverage_pct = (
                    100.0 * len(low_cov) / max(len(chunk_scores_df), 1)
                )
                if not low_cov.empty:
                    ids = low_cov[self.config.transcript_id_col].unique().tolist()
                    result.low_coverage_transcripts.extend(ids)
                    warnings_list.append(
                        f"{len(low_cov)} chunks below min_coverage_ratio="
                        f"{self.config.min_coverage_ratio} "
                        f"({result.low_coverage_pct:.1f}%)"
                    )

            # 5. Duplicate chunk IDs
            if self.config.chunk_id_col in chunk_scores_df.columns:
                dup_count = chunk_scores_df[self.config.chunk_id_col].duplicated().sum()
                if dup_count:
                    warnings_list.append(
                        f"{dup_count} duplicate chunk_ids detected."
                    )

            # 6. Transcript-level duplicates
            if not transcript_df.empty and self.config.transcript_id_col in transcript_df.columns:
                dup_t = transcript_df[self.config.transcript_id_col].duplicated().sum()
                if dup_t:
                    warnings_list.append(
                        f"{dup_t} duplicate transcript_ids in transcript scores."
                    )

            # 7. Invalid-score fraction gate
            if result.invalid_score_transcripts:
                frac = len(result.invalid_score_transcripts) / max(
                    chunk_scores_df[self.config.transcript_id_col].nunique(), 1
                )
                if frac > self.config.max_invalid_score_fraction:
                    warnings_list.append(
                        f"CRITICAL: {frac*100:.1f}% of transcripts have invalid "
                        f"scores (threshold={self.config.max_invalid_score_fraction*100:.0f}%)."
                    )

            # 8. Label counts
            if "lm_label" in chunk_scores_df.columns:
                counts = chunk_scores_df["lm_label"].value_counts().to_dict()
                for label in ("positive", "neutral", "negative", "unknown"):
                    result.label_counts[label] = int(counts.get(label, 0))

            result.validation_warnings = warnings_list

            if warnings_list:
                for w in warnings_list:
                    logger.warning("Validation: %s", w)
            else:
                logger.info("Validation passed — no issues detected.")

            stage.mark_complete(t0, rows_out=len(chunk_scores_df))
            return stage

        except Exception as exc:
            stage.mark_failed(t0, str(exc))
            logger.error("Validation stage error: %s", exc)
            return stage

    def export_outputs(
        self,
        chunk_scores_df: pd.DataFrame,
        transcript_df: pd.DataFrame,
        section_df: pd.DataFrame,
        speaker_df: pd.DataFrame,
        result: LMPipelineResult,
    ) -> PipelineStageResult:
        """
        Stage 6 — Write all output layers as parquet (and optionally CSV).

        Respects config.overwrite: if False and the target file exists,
        the write is skipped and the existing file path is recorded.
        """
        stage = PipelineStageResult(stage=PipelineStage.EXPORT)
        t0 = stage.mark_running()
        logger.info("[Stage 6/7] Exporting outputs to %s", self.config.output_dir)

        exports = {
            self.config.chunk_scores_stem:      chunk_scores_df,
            self.config.transcript_scores_stem: transcript_df,
            self.config.compatibility_scores_stem:
                self._build_compatibility_scores(transcript_df),
            self.config.section_scores_stem:    section_df,
            self.config.speaker_scores_stem:    speaker_df,
        }

        total_written = 0
        try:
            for stem, df in exports.items():
                if df.empty:
                    logger.warning("Skipping export for '%s' — empty DataFrame.", stem)
                    continue

                # Parquet
                if self.config.export_parquet:
                    pq_path = self.config.output_path(stem, "parquet")
                    if pq_path.exists() and not self.config.overwrite:
                        logger.info("Skipping %s (exists, overwrite=False).", pq_path)
                        result.exported_paths[f"{stem}.parquet"] = pq_path
                    else:
                        _write_parquet(df, pq_path)
                        result.exported_paths[f"{stem}.parquet"] = pq_path
                        total_written += len(df)
                        logger.info("Exported %s (%d rows).", pq_path, len(df))

                # CSV
                if self.config.export_csv:
                    csv_path = self.config.output_path(stem, "csv")
                    if csv_path.exists() and not self.config.overwrite:
                        logger.info("Skipping %s (exists, overwrite=False).", csv_path)
                        result.exported_paths[f"{stem}.csv"] = csv_path
                    else:
                        df.to_csv(csv_path, index=False)
                        result.exported_paths[f"{stem}.csv"] = csv_path
                        logger.info("Exported %s (%d rows).", csv_path, len(df))

            stage.mark_complete(t0, rows_out=total_written)
            return stage

        except Exception as exc:
            stage.mark_failed(t0, str(exc))
            logger.error("Export failed: %s", exc)
            return stage

    def generate_diagnostics(
        self,
        chunk_scores_df: pd.DataFrame,
        transcript_df: pd.DataFrame,
        result: LMPipelineResult,
    ) -> PipelineStageResult:
        """
        Stage 7 — Compute and export a diagnostics summary parquet.

        Rows represent per-transcript coverage and score statistics,
        useful for downstream quality monitoring without opening the
        full chunk-level dataset.
        """
        stage = PipelineStageResult(stage=PipelineStage.DIAGNOSTICS)
        t0 = stage.mark_running()
        logger.info("[Stage 7/7] Generating diagnostics")

        try:
            diag_rows = []

            if not chunk_scores_df.empty and self.config.transcript_id_col in chunk_scores_df.columns:
                score_cols = [
                    c for c in (
                        "lm_tone_score", "lm_coverage_ratio",
                        "lm_positive_count", "lm_negative_count",
                        "lm_uncertainty_count", "lm_matched_tokens",
                        "lm_total_tokens",
                    )
                    if c in chunk_scores_df.columns
                ]
                grp = chunk_scores_df.groupby(self.config.transcript_id_col)

                for tid, group in grp:
                    row: Dict[str, object] = {
                        self.config.transcript_id_col: tid,
                        "chunk_count": len(group),
                    }
                    for col in score_cols:
                        row[f"{col}_mean"]   = float(group[col].mean())
                        row[f"{col}_std"]    = float(group[col].std())
                        row[f"{col}_median"] = float(group[col].median())
                        row[f"{col}_min"]    = float(group[col].min())
                        row[f"{col}_max"]    = float(group[col].max())

                    if "lm_label" in group.columns:
                        label_dist = group["lm_label"].value_counts().to_dict()
                        for label in ("positive", "neutral", "negative", "unknown"):
                            row[f"label_{label}_count"] = int(
                                label_dist.get(label, 0)
                            )

                    row["is_low_coverage"] = (
                        tid in result.low_coverage_transcripts
                    )
                    row["has_invalid_score"] = (
                        tid in result.invalid_score_transcripts
                    )
                    diag_rows.append(row)

            diag_df = pd.DataFrame(diag_rows)

            if not diag_df.empty:
                diag_path_pq  = self.config.output_path(
                    self.config.diagnostics_stem, "parquet"
                )
                if self.config.export_parquet:
                    _write_parquet(diag_df, diag_path_pq)
                    result.exported_paths[f"{self.config.diagnostics_stem}.parquet"] = diag_path_pq
                if self.config.export_csv:
                    diag_path_csv = self.config.output_path(
                        self.config.diagnostics_stem, "csv"
                    )
                    diag_df.to_csv(diag_path_csv, index=False)
                    result.exported_paths[f"{self.config.diagnostics_stem}.csv"] = diag_path_csv
                logger.info(
                    "Diagnostics exported: %d transcripts → %s",
                    len(diag_df), diag_path_pq,
                )

            stage.mark_complete(t0, rows_out=len(diag_df))
            return stage

        except Exception as exc:
            stage.mark_failed(t0, str(exc))
            logger.error("Diagnostics generation failed: %s", exc)
            return stage

    def summary(self, result: LMPipelineResult) -> str:
        """Delegate to LMPipelineResult.summary() for consistent formatting."""
        return result.summary()

    # ------------------------------------------------------------------
    # Private helpers — sub-component init
    # ------------------------------------------------------------------

    def _init_preprocessor(self) -> None:
        if self._preprocessor is None:
            if _PREPROCESSOR_AVAILABLE:
                cfg = self.config.preprocessing_config or LMPreprocessingConfig()
                self._preprocessor = LMPreprocessor(cfg)
                logger.debug("LMPreprocessor initialised.")
            else:
                self._preprocessor = _StubPreprocessor()
                logger.warning("LMPreprocessor unavailable — stub active.")

    def _init_scorer(self) -> None:
        if self._scorer is None:
            if _SCORER_AVAILABLE:
                cfg = self.config.scoring_config or LMScoringConfig()
                self._scorer = LMScorer(cfg)
                logger.debug("LMScorer initialised.")
            else:
                self._scorer = None
                logger.warning("LMScorer unavailable — stub scores will be used.")

    # ------------------------------------------------------------------
    # Private helpers — scoring hot path
    # ------------------------------------------------------------------

    def _score_batch(
        self,
        batch: pd.DataFrame,
        lm_dict: object,
    ) -> pd.DataFrame:
        """
        Process one batch of chunks through preprocess → match → score.

        Returns the batch DataFrame with score columns appended in-place.
        All rows are preserved; failed individual chunks receive null scores.
        """
        score_rows: List[Dict[str, object]] = []

        for _, row in batch.iterrows():
            text = row.get(self.config.chunk_text_col, "")

            try:
                # Preprocess
                if self._preprocessor is not None:
                    prep_result = self._preprocessor.preprocess(str(text))
                    tokens = prep_result.tokens if prep_result else []
                else:
                    tokens = str(text).lower().split()

                # Match + score
                scores = self.compute_scores(tokens, lm_dict)

            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Chunk scoring error for chunk_id=%s: %s",
                    row.get(self.config.chunk_id_col, "?"),
                    exc,
                )
                scores = _null_score_dict()

            score_rows.append(scores)

        scores_df = pd.DataFrame(score_rows)
        result    = pd.concat(
            [batch.reset_index(drop=True), scores_df],
            axis=1,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers — aggregation fallbacks
    # ------------------------------------------------------------------

    def _aggregate_transcript(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate chunk scores to transcript level."""
        return self._groupby_aggregate(
            df,
            group_cols=[self.config.transcript_id_col],
            extra_carry=[
                self.config.ticker_col,
                "earnings_date",
                "date",
                "quarter",
                "year",
            ],
        )

    def _aggregate_section(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate chunk scores to transcript × section level."""
        return self._groupby_aggregate(
            df,
            group_cols=[
                self.config.transcript_id_col,
                self.config.section_type_col,
            ],
            extra_carry=[self.config.ticker_col, "earnings_date", "date"],
        )

    def _aggregate_speaker(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate chunk scores to transcript × speaker level."""
        group_cols = [self.config.transcript_id_col]
        if self.config.speaker_col in df.columns:
            group_cols.append(self.config.speaker_col)
        if self.config.speaker_role_col in df.columns:
            group_cols.append(self.config.speaker_role_col)
        return self._groupby_aggregate(
            df,
            group_cols=group_cols,
            extra_carry=[self.config.ticker_col, "earnings_date", "date"],
        )

    def _build_compatibility_scores(self, transcript_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert rich transcript aggregation into the stable downstream contract:
        data/processed/sentiment/lm_scores.parquet.
        """
        if transcript_df.empty or self.config.transcript_id_col not in transcript_df.columns:
            return pd.DataFrame()

        out = pd.DataFrame()
        out[self.config.transcript_id_col] = transcript_df[self.config.transcript_id_col]

        for col in ("ticker", "earnings_date", "date", "quarter", "year"):
            if col in transcript_df.columns:
                out[col] = transcript_df[col]

        out["lm_positive_count"] = _first_existing_numeric(
            transcript_df,
            ("lm_positive_count_sum", "lm_positive_count", "lm_positive_total"),
        )
        out["lm_negative_count"] = _first_existing_numeric(
            transcript_df,
            ("lm_negative_count_sum", "lm_negative_count", "lm_negative_total"),
        )
        out["lm_total_tokens"] = _first_existing_numeric(
            transcript_df,
            ("lm_total_tokens_sum", "lm_total_tokens", "lm_token_total"),
        )
        out["lm_word_count"] = out["lm_total_tokens"]
        out["lm_total_words"] = out["lm_total_tokens"]
        out["lm_tone_score"] = _first_existing_numeric(
            transcript_df,
            ("lm_weighted_tone_score", "lm_tone_score_mean", "lm_tone_score"),
        )
        out["lm_tone"] = out["lm_tone_score"]
        out["lm_label"] = out["lm_tone_score"].apply(_tone_to_label)

        ordered = [
            self.config.transcript_id_col,
            "ticker",
            "earnings_date",
            "date",
            "quarter",
            "year",
            "lm_positive_count",
            "lm_negative_count",
            "lm_tone_score",
            "lm_tone",
            "lm_word_count",
            "lm_total_words",
            "lm_total_tokens",
            "lm_label",
        ]
        return out[[c for c in ordered if c in out.columns]]

    def _groupby_aggregate(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        extra_carry: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generic weighted aggregation of LM score columns.

        Weight = lm_total_tokens (longer chunks contribute more).
        Non-existent group columns are silently skipped so the pipeline
        degrades gracefully on partial schemas.
        """
        score_cols = [
            c for c in (
                "lm_tone_score", "lm_coverage_ratio",
                "lm_positive_count", "lm_negative_count",
                "lm_uncertainty_count", "lm_litigious_count",
                "lm_strong_modal", "lm_weak_modal", "lm_constraining",
                "lm_matched_tokens", "lm_total_tokens",
                "lm_positive_ratio", "lm_negative_ratio",
                "lm_uncertainty_ratio", "lm_sentiment_intensity",
            )
            if c in df.columns
        ]

        valid_group = [c for c in group_cols if c in df.columns]
        if not valid_group:
            return pd.DataFrame()

        if not score_cols:
            return df[valid_group].drop_duplicates().copy()

        agg_funcs = {c: ["mean", "median", "std", "sum"] for c in score_cols}
        # Keep count col
        agg_funcs["lm_total_tokens"] = ["sum", "mean", "count"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            grouped = df.groupby(valid_group, sort=True).agg(agg_funcs)

        # Flatten multi-level column names
        grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
        grouped = grouped.reset_index()

        # Carry through ticker if available and not already a group key
        if extra_carry:
            for col in extra_carry:
                if col in df.columns and col not in valid_group:
                    ticker_map = (
                        df.groupby(valid_group)[col]
                        .first()
                        .reset_index()
                    )
                    grouped = grouped.merge(ticker_map, on=valid_group, how="left")

        # Weighted mean tone score (primary signal)
        if "lm_tone_score" in df.columns and "lm_total_tokens" in df.columns:
            def _weighted_mean(g: pd.DataFrame) -> float:
                w = g["lm_total_tokens"].clip(lower=0)
                if w.sum() == 0:
                    return float(g["lm_tone_score"].mean())
                return float((g["lm_tone_score"] * w).sum() / w.sum())

            wm = df.groupby(valid_group).apply(_weighted_mean).reset_index()
            wm.columns = list(valid_group) + ["lm_weighted_tone_score"]
            grouped = grouped.merge(wm, on=valid_group, how="left")

        return grouped

    # ------------------------------------------------------------------
    # Internal helpers — schema remapping
    # ------------------------------------------------------------------

    def _remap_segments_to_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map segmented_transcripts schema → chunk schema when chunks parquet
        is unavailable, so the rest of the pipeline sees a consistent layout.
        """
        rename_map: Dict[str, str] = {}
        if "text" in df.columns and self.config.chunk_text_col not in df.columns:
            rename_map["text"] = self.config.chunk_text_col
        if "speaker" in df.columns and self.config.speaker_col not in df.columns:
            rename_map["speaker"] = self.config.speaker_col

        df = df.rename(columns=rename_map)

        # Synthesise chunk_id if absent
        if self.config.chunk_id_col not in df.columns:
            df = df.copy()
            df[self.config.chunk_id_col] = (
                df[self.config.transcript_id_col].astype(str)
                + "_seg_"
                + df.index.astype(str)
            )

        # Synthesise token_count if absent
        if self.config.token_count_col not in df.columns:
            df[self.config.token_count_col] = (
                df[self.config.chunk_text_col].str.split().str.len().fillna(0)
            )

        if "section_type" not in df.columns:
            df["section_type"] = "unknown"

        return df


# ===========================================================================
# STUB HELPERS
# (replace with real module imports as development progresses)
# ===========================================================================


class _StubDictionary:
    """
    Minimal stand-in for LMDictionary while lm_dictionary_loader.py
    is not yet implemented.  Provides a small hard-coded LM word set for
    smoke-testing the full pipeline scaffold.
    """

    _POSITIVE = frozenset({
        "growth", "record", "strong", "exceeded", "expanded", "improved",
        "confident", "momentum", "outperform", "robust", "exceptional",
        "increased", "profitable", "optimistic", "delivered", "achieved",
        "raised", "beat", "favorable", "opportunities",
    })
    _NEGATIVE = frozenset({
        "decline", "loss", "impairment", "litigation", "concerns", "risk",
        "headwind", "uncertainty", "weaker", "reduced", "challenges",
        "miss", "disappointing", "adverse", "restructuring", "write-off",
        "unfavorable", "difficult", "constraint", "deterioration",
    })
    _UNCERTAINTY = frozenset({
        "uncertain", "approximately", "may", "might", "could", "if",
        "subject", "estimate", "project", "expect",
    })
    _LITIGIOUS = frozenset({
        "litigation", "lawsuit", "regulatory", "compliance", "liability",
        "settlement", "claim", "allegation", "dispute",
    })

    def __len__(self) -> int:
        return (
            len(self._POSITIVE) + len(self._NEGATIVE)
            + len(self._UNCERTAINTY) + len(self._LITIGIOUS)
        )

    def match(self, tokens: List[str]) -> Dict[str, int]:
        token_set = set(tokens)
        pos   = len(token_set & self._POSITIVE)
        neg   = len(token_set & self._NEGATIVE)
        unc   = len(token_set & self._UNCERTAINTY)
        lit   = len(token_set & self._LITIGIOUS)
        total = len(tokens)
        matched = pos + neg + unc + lit
        return {
            "positive_count":     pos,
            "negative_count":     neg,
            "uncertainty_count":  unc,
            "litigious_count":    lit,
            "strong_modal_count": 0,
            "weak_modal_count":   0,
            "constraining_count": 0,
            "total_tokens":       total,
            "matched_tokens":     matched,
        }


class _StubPreprocessor:
    """Minimal tokeniser for when LMPreprocessor is unavailable."""

    def preprocess(self, text: str) -> object:
        import re
        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
        tokens = [t for t in tokens if len(t) > 1]

        class _R:
            pass

        r = _R()
        r.tokens = tokens  # type: ignore[attr-defined]
        return r


def _stub_score_dict(match_result: Dict[str, int]) -> Dict[str, object]:
    """Minimal score dict when LMScorer is unavailable."""
    pos   = match_result.get("positive_count", 0)
    neg   = match_result.get("negative_count", 0)
    total = match_result.get("total_tokens", 1) or 1
    denom = pos + neg or 1
    tone  = (pos - neg) / denom
    return {
        "lm_tone_score":        round(tone, 6),
        "lm_label":             ("positive" if tone > 0.05 else
                                  "negative" if tone < -0.05 else "neutral"),
        "lm_confidence":        round(min(abs(tone), 1.0), 6),
        "lm_positive_count":    pos,
        "lm_negative_count":    neg,
        "lm_uncertainty_count": match_result.get("uncertainty_count", 0),
        "lm_litigious_count":   match_result.get("litigious_count", 0),
        "lm_strong_modal":      0,
        "lm_weak_modal":        0,
        "lm_constraining":      0,
        "lm_total_tokens":      match_result.get("total_tokens", 0),
        "lm_matched_tokens":    match_result.get("matched_tokens", 0),
        "lm_coverage_ratio":    round(
            match_result.get("matched_tokens", 0) / total, 6
        ),
        "lm_positive_ratio":    round(pos / total, 6),
        "lm_negative_ratio":    round(neg / total, 6),
        "lm_uncertainty_ratio": round(
            match_result.get("uncertainty_count", 0) / total, 6
        ),
        "lm_polarity_ratio":    round(pos / denom, 6),
        "lm_sentiment_intensity": round((pos + neg) / total, 6),
        "lm_normalized_positive": round(
            pos / max(match_result.get("matched_tokens", 1), 1), 6
        ),
        "lm_normalized_negative": round(
            neg / max(match_result.get("matched_tokens", 1), 1), 6
        ),
        "lm_is_low_coverage":   match_result.get("matched_tokens", 0) == 0,
        "lm_zero_div":          (pos + neg) == 0,
    }


def _null_score_dict() -> Dict[str, object]:
    """All-zero score dict for chunks that fail processing."""
    d = _stub_score_dict({
        "positive_count": 0, "negative_count": 0,
        "uncertainty_count": 0, "litigious_count": 0,
        "total_tokens": 0, "matched_tokens": 0,
    })
    d["lm_label"] = "unknown"
    return d


# ===========================================================================
# PARQUET I/O HELPER
# ===========================================================================


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path), compression="snappy")


def _first_existing_numeric(
    df: pd.DataFrame,
    candidates: Tuple[str, ...],
) -> pd.Series:
    """Return the first matching numeric column, or zeros when absent."""
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series([0] * len(df), index=df.index)


def _tone_to_label(score: object) -> str:
    """Map an LM tone score to the stable positive/neutral/negative label."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if value > 0.02:
        return "positive"
    if value < -0.02:
        return "negative"
    return "neutral"


# ===========================================================================
# DEMO / SELF-TEST
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    import tempfile

    print("\n" + "=" * 65)
    print("  LMPipeline — Demo Run")
    print("=" * 65)

    # ── Build synthetic transcript chunk dataset ───────────────────────
    SYNTHETIC_CHUNKS = [
        # AAPL Q1 2025 — prepared remarks (positive)
        {
            "chunk_id": "AAPL_Q1_2025_prep_001",
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "section_type": "prepared_remarks",
            "dominant_speaker": "Tim Cook",
            "speaker_role": "CEO",
            "chunk_text": (
                "We delivered record revenue growth driven by strong iPhone momentum. "
                "Our services segment achieved exceptional results and exceeded "
                "expectations. We remain confident in our long-term opportunities "
                "and expanded gross margins improved significantly this quarter. "
                "The team delivered outstanding performance across all geographies."
            ),
            "token_count": 52,
        },
        {
            "chunk_id": "AAPL_Q1_2025_prep_002",
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "section_type": "prepared_remarks",
            "dominant_speaker": "Luca Maestri",
            "speaker_role": "CFO",
            "chunk_text": (
                "Revenue increased twelve percent year-over-year to a record high. "
                "Gross margin expanded to forty-seven percent driven by favorable "
                "mix and increased services contribution. We raised our dividend and "
                "repurchased twenty-five billion dollars of stock demonstrating "
                "confidence in our profitable and robust business model."
            ),
            "token_count": 58,
        },
        {
            "chunk_id": "AAPL_Q1_2025_qa_001",
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "section_type": "qa",
            "dominant_speaker": "Analyst",
            "speaker_role": "Analyst",
            "chunk_text": (
                "Can you discuss the uncertainty around China headwinds and whether "
                "the risk of regulatory constraints could adversely affect the "
                "challenging macro environment and potential revenue decline?"
            ),
            "token_count": 35,
        },
        # MSFT Q2 2025 — mixed sentiment
        {
            "chunk_id": "MSFT_Q2_2025_prep_001",
            "transcript_id": "MSFT_Q2_2025",
            "ticker": "MSFT",
            "section_type": "prepared_remarks",
            "dominant_speaker": "Satya Nadella",
            "speaker_role": "CEO",
            "chunk_text": (
                "Azure growth remained strong and we achieved record cloud revenue. "
                "AI momentum continues with Copilot driving improved productivity. "
                "We face some uncertainty in the consumer segment with weaker demand "
                "but our commercial business delivered robust and expanded performance."
            ),
            "token_count": 48,
        },
        {
            "chunk_id": "MSFT_Q2_2025_qa_001",
            "transcript_id": "MSFT_Q2_2025",
            "ticker": "MSFT",
            "section_type": "qa",
            "dominant_speaker": "Amy Hood",
            "speaker_role": "CFO",
            "chunk_text": (
                "Capital expenditure will increase as we invest in AI infrastructure. "
                "We project operating income growth of approximately eighteen percent. "
                "Litigation risk from regulatory scrutiny remains a concern but we "
                "are confident in compliance and long-term opportunities."
            ),
            "token_count": 44,
        },
        # NVDA Q3 2025 — very positive
        {
            "chunk_id": "NVDA_Q3_2025_prep_001",
            "transcript_id": "NVDA_Q3_2025",
            "ticker": "NVDA",
            "section_type": "prepared_remarks",
            "dominant_speaker": "Jensen Huang",
            "speaker_role": "CEO",
            "chunk_text": (
                "Nvidia delivered exceptional record revenue and achieved outstanding "
                "growth across data center and AI accelerator segments. Demand momentum "
                "is robust and we are optimistic about expanded opportunities. "
                "We beat analyst expectations and raised full-year guidance confidently."
            ),
            "token_count": 50,
        },
        # Edge case: operator block (no sentiment content)
        {
            "chunk_id": "AAPL_Q1_2025_op_001",
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "section_type": "operator",
            "dominant_speaker": "Operator",
            "speaker_role": "Operator",
            "chunk_text": (
                "Ladies and gentlemen thank you for joining today's call. "
                "Please be advised that this conference is being recorded."
            ),
            "token_count": 22,
        },
    ]

    chunks_df = pd.DataFrame(SYNTHETIC_CHUNKS)

    # ── Write synthetic data to a temp parquet ─────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        interim_dir = tmp / "data" / "interim"
        output_dir  = tmp / "data" / "processed" / "sentiment"
        interim_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks_path = interim_dir / "transcript_chunks.parquet"
        _write_parquet(chunks_df, chunks_path)
        print(f"\nSynthetic dataset written: {len(chunks_df)} chunks to {chunks_path}")

        # ── Configure and run pipeline ─────────────────────────────────
        config = LMPipelineConfig(
            chunks_path  = chunks_path,
            interim_dir  = interim_dir,
            output_dir   = output_dir,
            overwrite    = True,
            batch_size   = 4,
            export_csv   = True,
            export_parquet = True,
        )

        pipeline = LMPipeline(config)
        result   = pipeline.run()

        # ── Print full summary ─────────────────────────────────────────
        print(result.summary())

        # ── Show chunk scores head ────────────────────────────────────
        chunk_pq = output_dir / "lm_chunk_scores.parquet"
        if chunk_pq.exists():
            df_out = pd.read_parquet(chunk_pq)
            print("\nChunk scores sample (key columns):")
            display_cols = [
                c for c in (
                    "chunk_id", "transcript_id", "section_type",
                    "lm_tone_score", "lm_label", "lm_coverage_ratio",
                    "lm_positive_count", "lm_negative_count",
                )
                if c in df_out.columns
            ]
            print(df_out[display_cols].to_string(index=False))

        # ── Show transcript scores ─────────────────────────────────────
        transcript_pq = output_dir / "lm_transcript_scores.parquet"
        if transcript_pq.exists():
            df_t = pd.read_parquet(transcript_pq)
            print("\nTranscript-level scores:")
            t_cols = [
                c for c in (
                    "transcript_id",
                    "lm_tone_score_mean",
                    "lm_weighted_tone_score",
                    "lm_coverage_ratio_mean",
                    "lm_total_tokens_sum",
                )
                if c in df_t.columns
            ]
            print(df_t[t_cols].to_string(index=False))

        # ── Cache reuse demo ───────────────────────────────────────────
        print("\n── Cache reuse demo (overwrite=False) ──────────────────")
        config_cached = LMPipelineConfig(
            chunks_path  = chunks_path,
            interim_dir  = interim_dir,
            output_dir   = output_dir,
            overwrite    = False,   # will reuse existing outputs
            batch_size   = 4,
        )
        pipeline_cached = LMPipeline(config_cached)
        result_cached   = pipeline_cached.run()
        for stage_name, sr in result_cached.stages.items():
            print(f"  {stage_name:<20} status={sr.status.value:<10} "
                  f"cache_hit={sr.cache_hit}")

    print("\nDemo complete.\n")
