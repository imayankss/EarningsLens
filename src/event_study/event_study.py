"""
event_study.py
==============
DAY 6 — Master Orchestration Engine
Earnings Call Sentiment Analyzer — Event Study Pipeline

This module is the single entry point for the complete DAY 6 event-study
workflow. It owns pipeline orchestration only: it delegates all financial
computations (abnormal returns, CAR, event-window generation, sentiment
merging, validation) to their dedicated sibling modules.

Architecture position
---------------------
    DAY 4 structured_transcripts.parquet
    DAY 5 market_data.parquet
    FinBERT call_level_sentiment.parquet
    LM     lm_scores.parquet
                │
                ▼
        EventStudyEngine          ← this file
                │
    ┌───────────┼────────────────────────┐
    ▼           ▼            ▼           ▼
event_window  abnormal_   car_        sentiment_
_generator    returns     calculator  event_merger
                │
                ▼
        validation.py
                │
                ▼
    data/interim/  ──  data/processed/
    aligned_event_windows.parquet
    abnormal_returns.parquet
    car_metrics.parquet
    event_study.parquet   ← final deliverable
    event_study.csv

Design principles (from architecture docs)
------------------------------------------
- Deterministic: same input always produces same output.
- Parquet-first: intermediate and final outputs are .parquet.
- Modular: no finance formula lives here; only orchestration logic.
- Fail-fast validation: schema and sanity checks at every stage boundary.
- Cached intermediates: expensive stages are skipped when cached output exists.
- Typed interfaces: dataclasses throughout; no bare dicts crossing boundaries.
- Extensive logging: every stage emits structured log messages.

Usage
-----
    # Programmatic
    from src.event_study.event_study import EventStudyEngine, EventStudyConfig

    config = EventStudyConfig()
    engine = EventStudyEngine(config)
    result = engine.run()

    # CLI
    python -m src.event_study.event_study

Author : Earnings Call Sentiment Analyzer project
Python : 3.11+
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency guard – graceful import errors with actionable messages
# ---------------------------------------------------------------------------
try:
    import pyarrow  # noqa: F401  – required for parquet I/O
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyarrow is required for parquet I/O.  "
        "Install it with:  pip install pyarrow"
    ) from exc

# ---------------------------------------------------------------------------
# Sibling-module imports (lazy where possible to keep startup fast)
# ---------------------------------------------------------------------------
# These are declared at the top so type checkers resolve them, but each
# import is individually guarded so partial installs still surface clear errors.

if TYPE_CHECKING:
    # Only used for type annotations; avoid circular at runtime.
    from src.event_study.abnormal_returns import AbnormalReturnCalculator
    from src.event_study.car_calculator import CARCalculator
    from src.event_study.event_window_generator import EventWindowGenerator
    from src.event_study.sentiment_event_merger import SentimentEventMerger
    from src.event_study.validation import EventStudyValidator


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PIPELINE_VERSION: Final[str] = "1.0.0"

# Expected event-window horizons (trading days after event date)
_DEFAULT_RETURN_HORIZONS: Final[tuple[int, ...]] = (1, 3, 5)

# Minimum rows required after each merge to continue
_MIN_ROWS_AFTER_MERGE: Final[int] = 1

# Final schema — all columns that MUST be present in the output
FINAL_SCHEMA_COLUMNS: Final[tuple[str, ...]] = (
    "transcript_id",
    "ticker",
    "event_date",
    "finbert_sentiment_score",
    "lm_tone_score",
    "sentiment_label",
    "return_1d",
    "return_3d",
    "return_5d",
    "abnormal_return_1d",
    "abnormal_return_3d",
    "abnormal_return_5d",
    "car_3d",
    "car_5d",
    "rolling_volatility_20d",
    "sma_20",
    "volume",
)

# Required columns from each upstream dataset
_MARKET_REQUIRED_COLS: Final[frozenset[str]] = frozenset(
    {
        "transcript_id",
        "ticker",
        "aligned_event_date",
        "daily_return",
        "benchmark_return",
        "abnormal_return",
        "adj_close",
        "volume",
        "rolling_volatility_20d",
        "sma_20",
    }
)

_SENTIMENT_REQUIRED_COLS: Final[frozenset[str]] = frozenset(
    {
        "transcript_id",
        "finbert_sentiment_score",
        "lm_tone_score",
        "sentiment_label",
    }
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EventStudyConfig:
    """
    Immutable configuration for the EventStudyEngine.

    All paths are resolved relative to ``project_root`` at construction
    time so the engine can run from any working directory.

    Parameters
    ----------
    project_root:
        Absolute path to the repository root.  Defaults to three levels
        above this file (src/event_study/event_study.py → repo root).
    return_horizons:
        Trading-day horizons for which forward returns, abnormal returns,
        and CAR metrics are computed.  Must be a non-empty tuple of positive
        integers in ascending order.
    benchmark_ticker:
        Ticker symbol used as the market benchmark.
    force_recompute:
        When True, ignore cached intermediate outputs and rerun every stage.
    save_intermediates:
        When True, write aligned_event_windows, abnormal_returns, and
        car_metrics to ``data/interim/``.
    raise_on_validation_failure:
        When True, a validation failure aborts the pipeline with a
        RuntimeError.  When False, failures are logged as warnings.
    min_sentiment_coverage:
        Minimum fraction (0–1) of events that must have sentiment scores
        after the merge.  Below this threshold the pipeline raises.
    """

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    return_horizons: tuple[int, ...] = _DEFAULT_RETURN_HORIZONS
    benchmark_ticker: str = "^GSPC"
    force_recompute: bool = False
    save_intermediates: bool = True
    raise_on_validation_failure: bool = True
    min_sentiment_coverage: float = 0.80  # 80 % of events must have sentiment

    # ---- derived paths (populated in __post_init__) ----
    market_data_path: Path = field(init=False)
    sentiment_path: Path = field(init=False)
    lm_scores_path: Path = field(init=False)
    interim_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    output_parquet: Path = field(init=False)
    output_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        root = self.project_root
        self.market_data_path = root / "data" / "processed" / "market_data.parquet"
        self.sentiment_path = (
            root / "data" / "processed" / "call_level_sentiment.parquet"
        )
        self.lm_scores_path = root / "data" / "processed" / "lm_scores.parquet"
        self.interim_dir = root / "data" / "interim"
        self.processed_dir = root / "data" / "processed"
        self.output_parquet = root / "data" / "processed" / "event_study.parquet"
        self.output_csv = root / "data" / "processed" / "event_study.csv"

        self._validate()

    def _validate(self) -> None:
        if not self.return_horizons:
            raise ValueError("return_horizons must be a non-empty tuple.")
        if sorted(self.return_horizons) != list(self.return_horizons):
            raise ValueError("return_horizons must be in ascending order.")
        if any(h <= 0 for h in self.return_horizons):
            raise ValueError("All return_horizons must be positive integers.")
        if not (0.0 < self.min_sentiment_coverage <= 1.0):
            raise ValueError("min_sentiment_coverage must be in (0, 1].")

    def ensure_output_dirs(self) -> None:
        """Create interim and processed directories if they do not exist."""
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "Output directories ensured: interim=%s  processed=%s",
            self.interim_dir,
            self.processed_dir,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"EventStudyConfig("
            f"horizons={self.return_horizons}, "
            f"benchmark={self.benchmark_ticker!r}, "
            f"force_recompute={self.force_recompute})"
        )


# ---------------------------------------------------------------------------


@dataclass
class StageTiming:
    """Wall-clock timing record for a single pipeline stage."""

    stage: str
    start_ts: float = field(default_factory=time.monotonic)
    end_ts: float = 0.0
    success: bool = False
    rows_in: int = 0
    rows_out: int = 0
    note: str = ""

    def finish(
        self,
        *,
        success: bool,
        rows_out: int = 0,
        note: str = "",
    ) -> None:
        self.end_ts = time.monotonic()
        self.success = success
        self.rows_out = rows_out
        self.note = note

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed wall time in seconds."""
        end = self.end_ts if self.end_ts else time.monotonic()
        return end - self.start_ts

    def __str__(self) -> str:  # noqa: D105
        status = "✓" if self.success else "✗"
        return (
            f"[{status}] {self.stage:<35s} "
            f"{self.elapsed_seconds:6.2f}s  "
            f"rows: {self.rows_in} → {self.rows_out}"
            + (f"  [{self.note}]" if self.note else "")
        )


# ---------------------------------------------------------------------------


@dataclass
class EventStudyResult:
    """
    Encapsulates the final output of a completed EventStudyEngine run.

    Attributes
    ----------
    event_study_df:
        The final analysis-ready DataFrame conforming to FINAL_SCHEMA_COLUMNS.
    parquet_path:
        Path where ``event_study.parquet`` was written.
    csv_path:
        Path where ``event_study.csv`` was written.
    total_events:
        Number of unique earnings events (transcript_ids) in the output.
    sentiment_coverage:
        Fraction of events that received sentiment scores after the merge.
    validation_passed:
        True when all validation checks passed without critical failures.
    pipeline_version:
        Semantic version string of the pipeline that produced this result.
    run_timestamp:
        UTC timestamp of the run that produced this result.
    """

    event_study_df: pd.DataFrame
    parquet_path: Path
    csv_path: Path
    total_events: int
    sentiment_coverage: float
    validation_passed: bool
    pipeline_version: str = _PIPELINE_VERSION
    run_timestamp: datetime = field(default_factory=datetime.utcnow)

    def summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary suitable for logging / reporting."""
        return {
            "pipeline_version": self.pipeline_version,
            "run_timestamp": self.run_timestamp.isoformat(),
            "total_events": self.total_events,
            "sentiment_coverage_pct": round(self.sentiment_coverage * 100, 2),
            "validation_passed": self.validation_passed,
            "output_rows": len(self.event_study_df),
            "output_columns": list(self.event_study_df.columns),
            "parquet_path": str(self.parquet_path),
            "csv_path": str(self.csv_path),
        }


# ---------------------------------------------------------------------------


@dataclass
class PipelineSummary:
    """
    Human-readable summary of the entire pipeline run.

    Created by EventStudyEngine.run() and printed/logged at completion.

    Attributes
    ----------
    stage_timings:
        Ordered list of StageTiming objects — one per pipeline stage.
    config:
        The EventStudyConfig that was used.
    result:
        The EventStudyResult if the run succeeded, else None.
    error:
        Exception captured if the run failed, else None.
    wall_seconds:
        Total elapsed seconds for the full run.
    """

    stage_timings: list[StageTiming] = field(default_factory=list)
    config: Optional[EventStudyConfig] = None
    result: Optional[EventStudyResult] = None
    error: Optional[Exception] = None
    wall_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        """True when the pipeline completed without an unhandled exception."""
        return self.error is None and self.result is not None

    def print_report(self) -> None:
        """Print a formatted pipeline report to stdout."""
        separator = "─" * 70
        print(f"\n{'═' * 70}")
        print(f"  EVENT STUDY PIPELINE REPORT  (v{_PIPELINE_VERSION})")
        print(f"{'═' * 70}")

        print(f"\n  Config : {self.config}")
        print(f"  Status : {'SUCCESS ✓' if self.succeeded else 'FAILED ✗'}")
        print(f"  Elapsed: {self.wall_seconds:.2f}s\n")

        print(f"  {separator}")
        print("  STAGE BREAKDOWN")
        print(f"  {separator}")
        for t in self.stage_timings:
            print(f"  {t}")

        if self.result:
            print(f"\n  {separator}")
            print("  OUTPUT SUMMARY")
            print(f"  {separator}")
            summary = self.result.summary_dict()
            for k, v in summary.items():
                print(f"  {k:<35s}: {v}")

        if self.error:
            print(f"\n  {separator}")
            print("  ERROR DETAIL")
            print(f"  {separator}")
            print(f"  {type(self.error).__name__}: {self.error}")

        print(f"\n{'═' * 70}\n")


# ---------------------------------------------------------------------------
# EventStudyEngine
# ---------------------------------------------------------------------------


class EventStudyEngine:
    """
    Master orchestrator for the DAY 6 event-study pipeline.

    This class owns the sequencing of all pipeline stages and the movement
    of data between them.  It does NOT contain any financial formulas — those
    live exclusively in the sibling modules listed below.

    Sibling module responsibilities
    --------------------------------
    event_window_generator.py
        Generate forward trading-session windows (t+1, t+3, t+5 …) from
        aligned event dates.

    abnormal_returns.py
        Compute benchmark-adjusted abnormal returns for each horizon.

    car_calculator.py
        Aggregate daily abnormal returns into cumulative abnormal returns.

    sentiment_event_merger.py
        Left-join FinBERT and LM sentiment scores onto the event frame.

    validation.py
        Run schema, sanity, and integrity checks at each stage boundary.

    Parameters
    ----------
    config:
        Pipeline configuration.  Defaults to EventStudyConfig() which reads
        all paths relative to the repository root.

    Examples
    --------
    >>> config = EventStudyConfig(return_horizons=(1, 3, 5))
    >>> engine = EventStudyEngine(config)
    >>> result = engine.run()
    >>> print(result.total_events)
    42
    """

    def __init__(self, config: Optional[EventStudyConfig] = None) -> None:
        self.config: EventStudyConfig = config or EventStudyConfig()
        self._summary: PipelineSummary = PipelineSummary(config=self.config)
        logger.info("EventStudyEngine initialised — %s", self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> EventStudyResult:
        """
        Execute the complete event-study pipeline end-to-end.

        Stage sequence
        --------------
        1.  Ensure output directories exist.
        2.  Load upstream datasets (market, FinBERT sentiment, LM scores).
        3.  Validate input schemas.
        4.  Generate event windows.
        5.  Compute forward returns per horizon.
        6.  Compute abnormal returns per horizon.
        7.  Compute CAR metrics per horizon.
        8.  Merge sentiment scores.
        9.  Final schema enforcement and column ordering.
        10. Run validation suite.
        11. Export parquet + CSV.
        12. Build and log PipelineSummary.

        Returns
        -------
        EventStudyResult
            Contains the final DataFrame, output paths, and run metadata.

        Raises
        ------
        RuntimeError
            When a critical stage fails and ``config.raise_on_validation_failure``
            is True.
        FileNotFoundError
            When a required input dataset is missing.
        """
        wall_start = time.monotonic()
        logger.info("=" * 60)
        logger.info("DAY 6 Event Study Pipeline starting")
        logger.info("Config: %s", self.config)
        logger.info("=" * 60)

        try:
            self.config.ensure_output_dirs()

            # ── Stage 1: Load inputs ──────────────────────────────────────
            market_df = self._stage_load_market_data()
            sentiment_df = self._stage_load_sentiment_data()
            lm_df = self._stage_load_lm_scores()

            # ── Stage 2: Input schema validation ─────────────────────────
            self._stage_validate_inputs(market_df, sentiment_df, lm_df)

            # ── Stage 3: Event window generation ─────────────────────────
            event_windows_df = self._stage_generate_event_windows(market_df)

            # ── Stage 4: Forward returns ──────────────────────────────────
            returns_df = self._stage_compute_forward_returns(
                market_df, event_windows_df
            )

            # ── Stage 5: Abnormal returns ─────────────────────────────────
            ar_df = self._stage_compute_abnormal_returns(returns_df, market_df)

            # ── Stage 6: CAR metrics ──────────────────────────────────────
            car_df = self._stage_compute_car(ar_df)

            # ── Stage 7: Sentiment merge ──────────────────────────────────
            merged_df = self._stage_merge_sentiment(car_df, sentiment_df, lm_df)

            # ── Stage 8: Final schema enforcement ────────────────────────
            final_df = self._stage_enforce_schema(merged_df)

            # ── Stage 9: Validation suite ─────────────────────────────────
            validation_passed = self._stage_validate_output(final_df)

            # ── Stage 10: Export ──────────────────────────────────────────
            self._stage_export(final_df)

            # ── Build result ──────────────────────────────────────────────
            total_events = final_df["transcript_id"].nunique()
            sentiment_coverage = self._compute_sentiment_coverage(final_df)

            result = EventStudyResult(
                event_study_df=final_df,
                parquet_path=self.config.output_parquet,
                csv_path=self.config.output_csv,
                total_events=total_events,
                sentiment_coverage=sentiment_coverage,
                validation_passed=validation_passed,
            )

            self._summary.result = result
            self._summary.wall_seconds = time.monotonic() - wall_start
            self._summary.print_report()

            logger.info(
                "Pipeline completed successfully — %d events, "
                "%.1f%% sentiment coverage, %.2fs elapsed",
                total_events,
                sentiment_coverage * 100,
                self._summary.wall_seconds,
            )
            return result

        except Exception as exc:  # noqa: BLE001
            self._summary.error = exc
            self._summary.wall_seconds = time.monotonic() - wall_start
            self._summary.print_report()
            logger.error(
                "Pipeline FAILED after %.2fs — %s: %s",
                self._summary.wall_seconds,
                type(exc).__name__,
                exc,
            )
            logger.debug(traceback.format_exc())
            raise

    @property
    def summary(self) -> PipelineSummary:
        """Access the PipelineSummary populated after run()."""
        return self._summary

    # ------------------------------------------------------------------
    # Stage implementations (private)
    # ------------------------------------------------------------------

    # ── Stage 1a ─────────────────────────────────────────────────────────

    def _stage_load_market_data(self) -> pd.DataFrame:
        """
        Load the processed market dataset produced by the DAY 5 pipeline.

        Returns
        -------
        pd.DataFrame
            OHLCV + return features aligned to trading sessions.
        """
        timing = StageTiming(stage="1a. Load market data")
        logger.info("[Stage 1a] Loading market data from %s", self.config.market_data_path)

        try:
            df = self._load_parquet(self.config.market_data_path)
            timing.rows_in = len(df)
            timing.finish(success=True, rows_out=len(df))
            logger.info(
                "[Stage 1a] Market data loaded — %d rows, %d tickers",
                len(df),
                df["ticker"].nunique() if "ticker" in df.columns else -1,
            )
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 1b ─────────────────────────────────────────────────────────

    def _stage_load_sentiment_data(self) -> pd.DataFrame:
        """
        Load FinBERT call-level sentiment scores.

        Returns
        -------
        pd.DataFrame
            DataFrame keyed by ``transcript_id`` with FinBERT scores.
        """
        timing = StageTiming(stage="1b. Load FinBERT sentiment")
        logger.info(
            "[Stage 1b] Loading FinBERT sentiment from %s", self.config.sentiment_path
        )

        try:
            df = self._load_parquet(self.config.sentiment_path)
            timing.rows_in = len(df)
            timing.finish(success=True, rows_out=len(df))
            logger.info("[Stage 1b] Sentiment data loaded — %d rows", len(df))
        except FileNotFoundError:
            # Sentiment may not yet exist; return empty frame, warn loudly.
            logger.warning(
                "[Stage 1b] Sentiment file not found: %s — "
                "proceeding with empty sentiment (merge will produce NaNs).",
                self.config.sentiment_path,
            )
            df = pd.DataFrame(columns=list(_SENTIMENT_REQUIRED_COLS))
            timing.finish(success=True, rows_out=0, note="file missing – empty frame")
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 1c ─────────────────────────────────────────────────────────

    def _stage_load_lm_scores(self) -> pd.DataFrame:
        """
        Load Loughran-McDonald dictionary sentiment scores.

        Returns
        -------
        pd.DataFrame
            DataFrame keyed by ``transcript_id`` with LM tone scores.
        """
        timing = StageTiming(stage="1c. Load LM scores")
        logger.info("[Stage 1c] Loading LM scores from %s", self.config.lm_scores_path)

        try:
            df = self._load_parquet(self.config.lm_scores_path)
            timing.rows_in = len(df)
            timing.finish(success=True, rows_out=len(df))
            logger.info("[Stage 1c] LM scores loaded — %d rows", len(df))
        except FileNotFoundError:
            logger.warning(
                "[Stage 1c] LM scores file not found: %s — "
                "proceeding with empty LM scores.",
                self.config.lm_scores_path,
            )
            df = pd.DataFrame(columns=["transcript_id", "lm_tone_score"])
            timing.finish(success=True, rows_out=0, note="file missing – empty frame")
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 2 ──────────────────────────────────────────────────────────

    def _stage_validate_inputs(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        lm_df: pd.DataFrame,
    ) -> None:
        """
        Validate that all upstream datasets contain required columns and
        have non-zero row counts.

        Raises
        ------
        ValueError
            On missing columns or empty market dataset.
        """
        timing = StageTiming(stage="2.  Validate input schemas")
        logger.info("[Stage 2] Validating input schemas")

        try:
            errors: list[str] = []

            # Market dataset – hard requirement
            missing_market = _MARKET_REQUIRED_COLS - set(market_df.columns)
            if missing_market:
                errors.append(
                    f"market_data missing columns: {sorted(missing_market)}"
                )
            if len(market_df) == 0:
                errors.append("market_data is empty — nothing to process.")

            # Sentiment datasets – soft requirement (warn, don't abort)
            if len(sentiment_df) > 0:
                missing_sent = (
                    _SENTIMENT_REQUIRED_COLS - set(sentiment_df.columns)
                )
                if missing_sent:
                    errors.append(
                        f"call_level_sentiment missing columns: {sorted(missing_sent)}"
                    )

            if errors:
                msg = "Input validation failed:\n  " + "\n  ".join(errors)
                logger.error("[Stage 2] %s", msg)
                raise ValueError(msg)

            timing.finish(success=True, rows_out=len(market_df))
            logger.info(
                "[Stage 2] Input validation passed — market_df=%d rows, "
                "sentiment_df=%d rows, lm_df=%d rows",
                len(market_df),
                len(sentiment_df),
                len(lm_df),
            )
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

    # ── Stage 3 ──────────────────────────────────────────────────────────

    def _stage_generate_event_windows(
        self, market_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Delegate event-window generation to ``event_window_generator.py``.

        Each unique (transcript_id, aligned_event_date) pair becomes a set
        of forward trading-session offsets: t+1, t+3, t+5, …

        If the cached intermediate exists and ``force_recompute`` is False
        the stage is skipped and the cached file is returned.

        Returns
        -------
        pd.DataFrame
            One row per (event, horizon) with columns:
            transcript_id, ticker, aligned_event_date, horizon, window_date.
        """
        timing = StageTiming(stage="3.  Generate event windows")
        cache_path = self.config.interim_dir / "aligned_event_windows.parquet"
        logger.info("[Stage 3] Generating event windows — horizons=%s", self.config.return_horizons)

        try:
            if not self.config.force_recompute and cache_path.exists():
                logger.info(
                    "[Stage 3] Cache hit — loading from %s", cache_path
                )
                df = self._load_parquet(cache_path)
                timing.rows_in = len(market_df)
                timing.finish(success=True, rows_out=len(df), note="from cache")
                self._summary.stage_timings.append(timing)
                return df

            generator = self._import_event_window_generator()
            df = generator.generate(
                market_df=market_df,
                horizons=self.config.return_horizons,
            )

            self._assert_non_empty(df, "event windows")
            timing.rows_in = len(market_df)
            timing.finish(success=True, rows_out=len(df))

            if self.config.save_intermediates:
                self._save_parquet(df, cache_path, label="event windows")
            logger.info("[Stage 3] Event windows generated — %d rows", len(df))

        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 4 ──────────────────────────────────────────────────────────

    def _stage_compute_forward_returns(
        self,
        market_df: pd.DataFrame,
        event_windows_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute compound forward returns for each event horizon.

        Delegates to ``event_window_generator.attach_forward_returns()``
        or the equivalent sub-function in that module.

        Returns
        -------
        pd.DataFrame
            Event-level frame with columns: return_1d, return_3d, return_5d
            (and any additional horizons from config.return_horizons).
        """
        timing = StageTiming(stage="4.  Compute forward returns")
        logger.info("[Stage 4] Computing compound forward returns")

        try:
            generator = self._import_event_window_generator()
            df = generator.attach_forward_returns(
                market_df=market_df,
                event_windows_df=event_windows_df,
                horizons=self.config.return_horizons,
            )

            self._assert_non_empty(df, "forward returns")
            self._assert_return_columns_exist(df, prefix="return")

            timing.rows_in = len(event_windows_df)
            timing.finish(success=True, rows_out=len(df))
            logger.info(
                "[Stage 4] Forward returns attached — %d event-rows", len(df)
            )
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 5 ──────────────────────────────────────────────────────────

    def _stage_compute_abnormal_returns(
        self,
        returns_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute benchmark-adjusted abnormal returns for each horizon.

        Delegates entirely to ``abnormal_returns.py``.
        No financial formula lives in this method.

        Returns
        -------
        pd.DataFrame
            Event-level frame enriched with abnormal_return_1d,
            abnormal_return_3d, abnormal_return_5d columns.
        """
        timing = StageTiming(stage="5.  Compute abnormal returns")
        cache_path = self.config.interim_dir / "abnormal_returns.parquet"
        logger.info("[Stage 5] Computing abnormal returns")

        try:
            if not self.config.force_recompute and cache_path.exists():
                logger.info(
                    "[Stage 5] Cache hit — loading from %s", cache_path
                )
                df = self._load_parquet(cache_path)
                timing.rows_in = len(returns_df)
                timing.finish(success=True, rows_out=len(df), note="from cache")
                self._summary.stage_timings.append(timing)
                return df

            calculator = self._import_abnormal_return_calculator()
            df = calculator.compute(
                event_returns_df=returns_df,
                market_df=market_df,
                horizons=self.config.return_horizons,
            )

            self._assert_non_empty(df, "abnormal returns")
            self._assert_return_columns_exist(df, prefix="abnormal_return")

            timing.rows_in = len(returns_df)
            timing.finish(success=True, rows_out=len(df))

            if self.config.save_intermediates:
                self._save_parquet(df, cache_path, label="abnormal returns")
            logger.info(
                "[Stage 5] Abnormal returns computed — %d event-rows", len(df)
            )
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 6 ──────────────────────────────────────────────────────────

    def _stage_compute_car(self, ar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate abnormal returns into Cumulative Abnormal Returns (CAR).

        Delegates entirely to ``car_calculator.py``.

        Returns
        -------
        pd.DataFrame
            Event-level frame enriched with car_3d and car_5d columns.
        """
        timing = StageTiming(stage="6.  Compute CAR metrics")
        cache_path = self.config.interim_dir / "car_metrics.parquet"
        logger.info("[Stage 6] Computing Cumulative Abnormal Returns (CAR)")

        try:
            if not self.config.force_recompute and cache_path.exists():
                logger.info(
                    "[Stage 6] Cache hit — loading from %s", cache_path
                )
                df = self._load_parquet(cache_path)
                timing.rows_in = len(ar_df)
                timing.finish(success=True, rows_out=len(df), note="from cache")
                self._summary.stage_timings.append(timing)
                return df

            calculator = self._import_car_calculator()
            df = calculator.compute(
                ar_df=ar_df,
                horizons=self.config.return_horizons,
            )

            self._assert_non_empty(df, "CAR metrics")
            # CAR horizons subset (3d and 5d are always required)
            for col in ("car_3d", "car_5d"):
                if col not in df.columns:
                    raise ValueError(
                        f"car_calculator did not produce required column '{col}'."
                    )

            timing.rows_in = len(ar_df)
            timing.finish(success=True, rows_out=len(df))

            if self.config.save_intermediates:
                self._save_parquet(df, cache_path, label="CAR metrics")
            logger.info("[Stage 6] CAR metrics computed — %d event-rows", len(df))
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 7 ──────────────────────────────────────────────────────────

    def _stage_merge_sentiment(
        self,
        car_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        lm_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join FinBERT and LM sentiment scores onto the event frame.

        Delegates to ``sentiment_event_merger.py``.  Uses ``transcript_id``
        as the primary merge key.

        Checks sentiment coverage against ``config.min_sentiment_coverage``
        and raises if coverage is too low.

        Returns
        -------
        pd.DataFrame
            Merged frame containing all finance metrics and sentiment scores.
        """
        timing = StageTiming(stage="7.  Merge sentiment scores")
        logger.info("[Stage 7] Merging sentiment scores onto event frame")

        try:
            merger = self._import_sentiment_event_merger()
            df = merger.merge(
                event_df=car_df,
                finbert_df=sentiment_df,
                lm_df=lm_df,
                merge_key="transcript_id",
            )

            self._assert_non_empty(df, "sentiment-merged events")

            # Coverage check
            coverage = self._compute_sentiment_coverage(df)
            logger.info(
                "[Stage 7] Sentiment coverage: %.1f%% (threshold %.1f%%)",
                coverage * 100,
                self.config.min_sentiment_coverage * 100,
            )

            if coverage < self.config.min_sentiment_coverage:
                msg = (
                    f"Sentiment coverage {coverage:.1%} is below the configured "
                    f"threshold of {self.config.min_sentiment_coverage:.1%}. "
                    "Check that FinBERT outputs align with market events."
                )
                if self.config.raise_on_validation_failure:
                    raise RuntimeError(msg)
                else:
                    warnings.warn(msg, stacklevel=2)

            timing.rows_in = len(car_df)
            timing.finish(success=True, rows_out=len(df))
            logger.info(
                "[Stage 7] Sentiment merge complete — %d rows", len(df)
            )
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 8 ──────────────────────────────────────────────────────────

    def _stage_enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce the final column schema defined in FINAL_SCHEMA_COLUMNS.

        - Adds missing columns as NaN (with a warning per missing column).
        - Reorders columns so schema columns appear first, then extras.
        - Renames ``aligned_event_date`` → ``event_date`` if needed.

        Returns
        -------
        pd.DataFrame
            Schema-compliant DataFrame.
        """
        timing = StageTiming(stage="8.  Enforce final schema")
        logger.info("[Stage 8] Enforcing final output schema")

        try:
            # Rename aligned_event_date → event_date (schema canonical name)
            if "aligned_event_date" in df.columns and "event_date" not in df.columns:
                df = df.rename(columns={"aligned_event_date": "event_date"})
                logger.debug("[Stage 8] Renamed aligned_event_date → event_date")

            # Check + add any missing schema columns
            missing_cols = [c for c in FINAL_SCHEMA_COLUMNS if c not in df.columns]
            if missing_cols:
                logger.warning(
                    "[Stage 8] Final schema columns not found — will be NaN: %s",
                    missing_cols,
                )
                for col in missing_cols:
                    df[col] = float("nan")

            # Reorder: schema columns first, then any extras
            extra_cols = [c for c in df.columns if c not in FINAL_SCHEMA_COLUMNS]
            ordered = list(FINAL_SCHEMA_COLUMNS) + extra_cols
            df = df[ordered]

            timing.rows_in = len(df)
            timing.finish(success=True, rows_out=len(df))
            logger.info(
                "[Stage 8] Schema enforced — %d columns, %d rows",
                len(df.columns),
                len(df),
            )
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return df

    # ── Stage 9 ──────────────────────────────────────────────────────────

    def _stage_validate_output(self, df: pd.DataFrame) -> bool:
        """
        Run the complete validation suite via ``validation.py``.

        Returns
        -------
        bool
            True when all checks pass, False when any non-critical check
            fails and ``raise_on_validation_failure`` is False.

        Raises
        ------
        RuntimeError
            When a critical check fails and ``raise_on_validation_failure``
            is True.
        """
        timing = StageTiming(stage="9.  Validate output dataset")
        logger.info("[Stage 9] Running output validation suite")

        try:
            validator = self._import_validator()
            report = validator.validate(
                df=df,
                required_columns=list(FINAL_SCHEMA_COLUMNS),
                return_horizons=self.config.return_horizons,
            )

            passed = report.all_passed

            if not passed:
                failed_checks = [c for c in report.checks if not c.passed]
                summary_lines = "\n  ".join(
                    f"{c.name}: {c.message}" for c in failed_checks
                )
                msg = f"Validation failures:\n  {summary_lines}"
                if self.config.raise_on_validation_failure:
                    timing.finish(success=False, rows_out=len(df), note="validation failed")
                    raise RuntimeError(msg)
                else:
                    logger.warning("[Stage 9] %s", msg)

            timing.finish(success=passed, rows_out=len(df))
            logger.info(
                "[Stage 9] Validation complete — passed=%s", passed
            )
        except RuntimeError:
            raise
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

        return passed

    # ── Stage 10 ─────────────────────────────────────────────────────────

    def _stage_export(self, df: pd.DataFrame) -> None:
        """
        Write the final event-study dataset to parquet and CSV.

        Parquet is the primary output format.  CSV is written for
        human-readable debugging.  Both are written atomically
        (temp file → rename) to prevent partial writes.

        Raises
        ------
        IOError
            If either write fails.
        """
        timing = StageTiming(stage="10. Export parquet + CSV")
        logger.info("[Stage 10] Exporting final event-study dataset")

        try:
            # Parquet (primary)
            self._save_parquet(df, self.config.output_parquet, label="event_study")

            # CSV (secondary, for debugging)
            csv_tmp = self.config.output_csv.with_suffix(".tmp.csv")
            df.to_csv(csv_tmp, index=False)
            csv_tmp.replace(self.config.output_csv)
            logger.info(
                "[Stage 10] CSV written → %s  (%d rows)",
                self.config.output_csv,
                len(df),
            )

            timing.rows_in = len(df)
            timing.finish(success=True, rows_out=len(df))
        except Exception as exc:
            timing.finish(success=False, note=str(exc))
            raise
        finally:
            self._summary.stage_timings.append(timing)

    # ------------------------------------------------------------------
    # Helper utilities (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_parquet(path: Path) -> pd.DataFrame:
        """Load a parquet file and return a DataFrame. Raises FileNotFoundError."""
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")
        df = pd.read_parquet(path)
        logger.debug("Loaded parquet — %s  (%d rows, %d cols)", path, len(df), len(df.columns))
        return df

    @staticmethod
    def _save_parquet(df: pd.DataFrame, path: Path, *, label: str = "") -> None:
        """
        Write DataFrame to parquet using a temp-file-then-rename pattern
        to prevent partial/corrupt writes.
        """
        tmp_path = path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        tmp_path.replace(path)
        logger.info(
            "Parquet written%s → %s  (%d rows)",
            f" [{label}]" if label else "",
            path,
            len(df),
        )

    @staticmethod
    def _assert_non_empty(df: pd.DataFrame, label: str) -> None:
        """Raise ValueError when a stage produces zero rows."""
        if len(df) < _MIN_ROWS_AFTER_MERGE:
            raise ValueError(
                f"Stage produced zero rows for '{label}'. "
                "Check upstream data and pipeline logs."
            )

    def _assert_return_columns_exist(
        self, df: pd.DataFrame, prefix: str
    ) -> None:
        """
        Verify that return columns for all configured horizons are present.

        For prefix='return' checks: return_1d, return_3d, return_5d.
        For prefix='abnormal_return' checks: abnormal_return_1d, etc.
        """
        expected = [f"{prefix}_{h}d" for h in self.config.return_horizons]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"Expected columns missing after {prefix} computation: {missing}"
            )

    @staticmethod
    def _compute_sentiment_coverage(df: pd.DataFrame) -> float:
        """
        Return fraction of rows that have a non-null finbert_sentiment_score.
        """
        if "finbert_sentiment_score" not in df.columns or len(df) == 0:
            return 0.0
        return df["finbert_sentiment_score"].notna().mean()

    # ------------------------------------------------------------------
    # Lazy sibling-module imports (private)
    # ------------------------------------------------------------------
    # Each method imports the sibling module on first use, allowing the
    # engine to start even when some sibling modules are not yet written.
    # This also makes it easy to swap in mock implementations in tests.

    def _import_event_window_generator(self) -> Any:
        """Import and instantiate EventWindowGenerator."""
        try:
            from src.event_study.event_window_generator import (  # type: ignore[import]
                EventWindowGenerator,
            )
            return EventWindowGenerator(config=self.config)
        except ImportError as exc:
            raise ImportError(
                "event_window_generator.py is not yet implemented or importable. "
                f"Details: {exc}"
            ) from exc

    def _import_abnormal_return_calculator(self) -> Any:
        """Import and instantiate AbnormalReturnCalculator."""
        try:
            from src.event_study.abnormal_returns import (  # type: ignore[import]
                AbnormalReturnCalculator,
            )
            return AbnormalReturnCalculator(config=self.config)
        except ImportError as exc:
            raise ImportError(
                "abnormal_returns.py is not yet implemented or importable. "
                f"Details: {exc}"
            ) from exc

    def _import_car_calculator(self) -> Any:
        """Import and instantiate CARCalculator."""
        try:
            from src.event_study.car_calculator import (  # type: ignore[import]
                CARCalculator,
            )
            return CARCalculator(config=self.config)
        except ImportError as exc:
            raise ImportError(
                "car_calculator.py is not yet implemented or importable. "
                f"Details: {exc}"
            ) from exc

    def _import_sentiment_event_merger(self) -> Any:
        """Import and instantiate SentimentEventMerger."""
        try:
            from src.event_study.sentiment_event_merger import (  # type: ignore[import]
                SentimentEventMerger,
            )
            return SentimentEventMerger(config=self.config)
        except ImportError as exc:
            raise ImportError(
                "sentiment_event_merger.py is not yet implemented or importable. "
                f"Details: {exc}"
            ) from exc

    def _import_validator(self) -> Any:
        """Import and instantiate EventStudyValidator."""
        try:
            from src.event_study.validation import (  # type: ignore[import]
                EventStudyValidator,
            )
            return EventStudyValidator(config=self.config)
        except ImportError as exc:
            raise ImportError(
                "validation.py is not yet implemented or importable. "
                f"Details: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Logging setup helper
# ---------------------------------------------------------------------------


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with a timestamped format.

    Call this once in __main__ or at application startup.
    Individual modules use ``logging.getLogger(__name__)`` automatically.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# Self-test / demo block
# ---------------------------------------------------------------------------


def _build_demo_market_df() -> pd.DataFrame:
    """
    Create a tiny synthetic market DataFrame that mimics the DAY 5 output.

    Used exclusively by the __main__ self-test to validate the engine
    without requiring real data files.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    tickers = ["AAPL", "MSFT", "NVDA"]
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        t_id = f"{ticker}_Q1_2025"
        base_price = {"AAPL": 180.0, "MSFT": 380.0, "NVDA": 550.0}[ticker]
        event_date = pd.Timestamp("2025-01-30")

        # 25 trading days of data per ticker
        for i in range(-20, 6):
            price = base_price * (1 + rng.normal(0, 0.01))
            rows.append(
                {
                    "transcript_id": t_id,
                    "ticker": ticker,
                    "date": event_date + pd.tseries.offsets.BDay(i),
                    "aligned_event_date": event_date,
                    "adj_close": price,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": int(rng.integers(10_000_000, 50_000_000)),
                    "daily_return": rng.normal(0.001, 0.015),
                    "benchmark_return": rng.normal(0.0005, 0.010),
                    "abnormal_return": rng.normal(0.0005, 0.012),
                    "rolling_volatility_20d": abs(rng.normal(0.20, 0.03)),
                    "sma_20": price * 0.99,
                    "sma_50": price * 0.97,
                    "relative_volume": abs(rng.normal(1.0, 0.2)),
                }
            )

    return pd.DataFrame(rows)


def _build_demo_sentiment_df() -> pd.DataFrame:
    """Minimal FinBERT sentiment scores for the self-test."""
    return pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_Q1_2025",
                "finbert_sentiment_score": 0.72,
                "lm_tone_score": 0.31,
                "sentiment_label": "positive",
            },
            {
                "transcript_id": "MSFT_Q1_2025",
                "finbert_sentiment_score": 0.55,
                "lm_tone_score": 0.18,
                "sentiment_label": "positive",
            },
            {
                "transcript_id": "NVDA_Q1_2025",
                "finbert_sentiment_score": 0.88,
                "lm_tone_score": 0.45,
                "sentiment_label": "positive",
            },
        ]
    )


def _build_demo_lm_df() -> pd.DataFrame:
    """Minimal LM scores for the self-test (lm_tone_score may duplicate from
    sentiment_df in a real pipeline; here we treat them independently)."""
    return pd.DataFrame(
        [
            {"transcript_id": "AAPL_Q1_2025", "lm_tone_score": 0.31},
            {"transcript_id": "MSFT_Q1_2025", "lm_tone_score": 0.18},
            {"transcript_id": "NVDA_Q1_2025", "lm_tone_score": 0.45},
        ]
    )


class _StubEventWindowGenerator:
    """Minimal stub satisfying the EventWindowGenerator interface for the demo."""

    def __init__(self, config: EventStudyConfig) -> None:
        self.config = config

    def generate(
        self,
        market_df: pd.DataFrame,
        horizons: tuple[int, ...],
    ) -> pd.DataFrame:
        """Return a copy of market_df — event windows already encoded in dates."""
        return market_df.copy()

    def attach_forward_returns(
        self,
        market_df: pd.DataFrame,
        event_windows_df: pd.DataFrame,
        horizons: tuple[int, ...],
    ) -> pd.DataFrame:
        """Pivot market data to event-level with stub forward returns."""
        import numpy as np

        events = (
            market_df[["transcript_id", "ticker", "aligned_event_date"]]
            .drop_duplicates("transcript_id")
            .copy()
        )
        rng = np.random.default_rng(99)
        for h in horizons:
            events[f"return_{h}d"] = rng.normal(0.005, 0.02, size=len(events))
        events["rolling_volatility_20d"] = 0.22
        events["sma_20"] = 350.0
        events["volume"] = 20_000_000
        return events


class _StubAbnormalReturnCalculator:
    """Minimal stub for AbnormalReturnCalculator."""

    def __init__(self, config: EventStudyConfig) -> None:
        self.config = config

    def compute(
        self,
        event_returns_df: pd.DataFrame,
        market_df: pd.DataFrame,
        horizons: tuple[int, ...],
    ) -> pd.DataFrame:
        import numpy as np

        df = event_returns_df.copy()
        rng = np.random.default_rng(77)
        for h in horizons:
            df[f"abnormal_return_{h}d"] = rng.normal(0.002, 0.015, size=len(df))
        return df


class _StubCARCalculator:
    """Minimal stub for CARCalculator."""

    def __init__(self, config: EventStudyConfig) -> None:
        self.config = config

    def compute(
        self,
        ar_df: pd.DataFrame,
        horizons: tuple[int, ...],
    ) -> pd.DataFrame:
        import numpy as np

        df = ar_df.copy()
        rng = np.random.default_rng(55)
        df["car_3d"] = rng.normal(0.006, 0.025, size=len(df))
        df["car_5d"] = rng.normal(0.010, 0.030, size=len(df))
        return df


class _StubSentimentEventMerger:
    """Minimal stub for SentimentEventMerger."""

    def __init__(self, config: EventStudyConfig) -> None:
        self.config = config

    def merge(
        self,
        event_df: pd.DataFrame,
        finbert_df: pd.DataFrame,
        lm_df: pd.DataFrame,
        merge_key: str,
    ) -> pd.DataFrame:
        df = event_df.copy()
        sent_cols = ["finbert_sentiment_score", "lm_tone_score", "sentiment_label"]
        for col in sent_cols:
            if col not in finbert_df.columns:
                continue
        df = df.merge(
            finbert_df[["transcript_id"] + sent_cols].rename(
                columns={"lm_tone_score": "_lm_from_sent"}
            ),
            on="transcript_id",
            how="left",
        )
        if "lm_tone_score" in lm_df.columns:
            df = df.merge(
                lm_df[["transcript_id", "lm_tone_score"]],
                on="transcript_id",
                how="left",
            )
        elif "_lm_from_sent" in df.columns:
            df = df.rename(columns={"_lm_from_sent": "lm_tone_score"})
        return df


class _StubValidationReport:
    all_passed = True
    checks: list[Any] = []


class _StubValidator:
    """Minimal stub for EventStudyValidator."""

    def __init__(self, config: EventStudyConfig) -> None:
        self.config = config

    def validate(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
        return_horizons: tuple[int, ...],
    ) -> _StubValidationReport:
        return _StubValidationReport()


def _run_demo() -> None:
    """
    Self-contained demonstration of EventStudyEngine using synthetic data
    and stub implementations of all sibling modules.

    Run with:   python -m src.event_study.event_study
    or:         python event_study.py
    """
    import tempfile

    configure_logging(logging.DEBUG)
    logger.info("=" * 60)
    logger.info("SELF-TEST / DEMO MODE — using synthetic data + stub modules")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Build config pointing at temp directories ──────────────────
        config = EventStudyConfig.__new__(EventStudyConfig)
        config.project_root = tmp
        config.return_horizons = (1, 3, 5)
        config.benchmark_ticker = "^GSPC"
        config.force_recompute = True
        config.save_intermediates = True
        config.raise_on_validation_failure = False
        config.min_sentiment_coverage = 0.80

        config.market_data_path = tmp / "data" / "processed" / "market_data.parquet"
        config.sentiment_path = (
            tmp / "data" / "processed" / "call_level_sentiment.parquet"
        )
        config.lm_scores_path = tmp / "data" / "processed" / "lm_scores.parquet"
        config.interim_dir = tmp / "data" / "interim"
        config.processed_dir = tmp / "data" / "processed"
        config.output_parquet = tmp / "data" / "processed" / "event_study.parquet"
        config.output_csv = tmp / "data" / "processed" / "event_study.csv"

        config.ensure_output_dirs()

        # ── Write synthetic upstream parquets ──────────────────────────
        market_df = _build_demo_market_df()
        sentiment_df = _build_demo_sentiment_df()
        lm_df = _build_demo_lm_df()

        market_df.to_parquet(config.market_data_path, index=False)
        sentiment_df.to_parquet(config.sentiment_path, index=False)
        lm_df.to_parquet(config.lm_scores_path, index=False)

        logger.info("Synthetic datasets written to temp dir: %s", tmp)

        # ── Patch engine to use stubs instead of real sibling modules ──
        engine = EventStudyEngine(config)
        engine._import_event_window_generator = (  # type: ignore[method-assign]
            lambda: _StubEventWindowGenerator(config)
        )
        engine._import_abnormal_return_calculator = (  # type: ignore[method-assign]
            lambda: _StubAbnormalReturnCalculator(config)
        )
        engine._import_car_calculator = (  # type: ignore[method-assign]
            lambda: _StubCARCalculator(config)
        )
        engine._import_sentiment_event_merger = (  # type: ignore[method-assign]
            lambda: _StubSentimentEventMerger(config)
        )
        engine._import_validator = (  # type: ignore[method-assign]
            lambda: _StubValidator(config)
        )

        # ── Run ────────────────────────────────────────────────────────
        try:
            result = engine.run()
            print("\n[DEMO] Pipeline result summary:")
            for k, v in result.summary_dict().items():
                print(f"  {k:<35s}: {v}")
            print("\n[DEMO] First 3 rows of event_study_df:")
            print(result.event_study_df.head(3).to_string())
            print("\n[DEMO] ✓ Self-test passed.\n")
        except Exception as exc:
            print(f"\n[DEMO] ✗ Self-test FAILED: {exc}\n")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_demo()
