"""
lm_aggregation.py
=================
Hierarchical Aggregation Layer for Loughran-McDonald Sentiment Outputs.

Responsibilities
----------------
* Accept chunk-level or segment-level LM score DataFrames (produced by
  lm_scoring.py) and aggregate them at four hierarchical levels:
    1. Chunk        — one row per chunk_id  (passthrough + enrichment)
    2. Transcript   — one row per transcript_id
    3. Section      — one row per (transcript_id, section_type) pair
    4. Speaker      — one row per (transcript_id, speaker_role) pair

* Produce fully validated, deterministically ordered output DataFrames
  ready for downstream event-study merges and parquet export.

* Expose AggregationStatistics for batch diagnostics and pipeline health
  monitoring.

Does NOT implement:
    scoring, matching, preprocessing, pipeline orchestration, parquet
    export, FinBERT comparison, regression analysis.

Integration contract
--------------------
    lm_scoring.py   → supplies the input DataFrame; expected columns are
                      documented in _REQUIRED_CHUNK_COLUMNS / _REQUIRED_SEG_COLUMNS.
    lm_pipeline.py  → calls each aggregate_* method in sequence; receives
                      output DataFrames and handles I/O.
    event_study     → merges on transcript_id / aligned_event_date.

Column naming convention
------------------------
    All output columns use the ``lm_`` prefix (e.g. ``lm_tone_score``) so
    downstream merges with FinBERT columns (``finbert_*``) are unambiguous.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Section types recognised by the aggregator.
VALID_SECTIONS: FrozenSet[str] = frozenset(
    {"prepared_remarks", "qa", "closing_remarks", "opening_remarks"}
)

#: Speaker roles that represent management (included in speaker aggregation).
MANAGEMENT_ROLES: FrozenSet[str] = frozenset(
    {"ceo", "cfo", "coo", "president", "management", "executive"}
)

#: Analyst roles.
ANALYST_ROLES: FrozenSet[str] = frozenset({"analyst", "research analyst"})

#: Roles excluded from speaker aggregation by default.
EXCLUDED_ROLES_DEFAULT: FrozenSet[str] = frozenset({"operator", "unknown", ""})

#: Minimum number of tokens a group must contain to be included in output.
_DEFAULT_MIN_TOKENS: int = 10

#: Minimum dictionary coverage ratio to avoid "low-coverage" flag.
_DEFAULT_MIN_COVERAGE: float = 0.01

#: Columns that the input chunk-level DataFrame MUST contain.
_REQUIRED_CHUNK_COLUMNS: Tuple[str, ...] = (
    "transcript_id",
    "chunk_id",
    "lm_tone_score",
    "lm_positive_count",
    "lm_negative_count",
    "lm_uncertainty_count",
    "token_count",
    "matched_token_count",
)

#: Additional columns required for section/speaker aggregation.
_REQUIRED_SEG_COLUMNS: Tuple[str, ...] = (
    "section_type",
    "speaker_role",
)

#: All LM count columns that are summed during aggregation.
_COUNT_COLUMNS: Tuple[str, ...] = (
    "lm_positive_count",
    "lm_negative_count",
    "lm_uncertainty_count",
    "lm_litigious_count",
    "lm_strong_modal_count",
    "lm_weak_modal_count",
    "lm_constraining_count",
    "token_count",
    "matched_token_count",
)

#: Columns that represent optional count fields (may not always be present).
_OPTIONAL_COUNT_COLUMNS: Tuple[str, ...] = (
    "lm_litigious_count",
    "lm_strong_modal_count",
    "lm_weak_modal_count",
    "lm_constraining_count",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LMAggregationConfig:
    """
    Immutable configuration for LMAggregator.

    Parameters
    ----------
    weighting_mode:
        ``"uniform"`` — each chunk contributes equally to the group mean.
        ``"token"``   — chunks weighted by their token_count.
        ``"coverage"``— chunks weighted by their matched_token_count.
        ``"hybrid"``  — token_count × coverage_ratio (recommended default).
    min_token_threshold:
        Groups with fewer total tokens than this value are flagged and
        optionally excluded from output.
    min_coverage_threshold:
        Groups whose coverage_ratio falls below this threshold are flagged
        as low-quality.
    exclude_roles:
        Speaker roles to drop before speaker-level aggregation.
        Defaults to ``EXCLUDED_ROLES_DEFAULT``.
    include_sections:
        Section types to include.  ``None`` includes all recognised sections.
    drop_low_token_groups:
        When ``True`` groups below ``min_token_threshold`` are silently
        dropped.  When ``False`` they are retained but flagged in the
        ``AggregationStatistics``.
    sort_output:
        When ``True`` all output DataFrames are sorted by their primary key
        columns for deterministic ordering.
    label_thresholds:
        ``(positive_threshold, negative_threshold)`` for converting a
        mean tone score into a label.  Default ``(0.02, -0.02)`` matches
        the lm_scoring.py convention.
    """

    weighting_mode: str = "hybrid"
    min_token_threshold: int = _DEFAULT_MIN_TOKENS
    min_coverage_threshold: float = _DEFAULT_MIN_COVERAGE
    exclude_roles: FrozenSet[str] = EXCLUDED_ROLES_DEFAULT
    include_sections: Optional[FrozenSet[str]] = None   # None → all
    drop_low_token_groups: bool = False
    sort_output: bool = True
    label_thresholds: Tuple[float, float] = (0.02, -0.02)

    def __post_init__(self) -> None:
        valid_modes = {"uniform", "token", "coverage", "hybrid"}
        if self.weighting_mode not in valid_modes:
            raise ValueError(
                f"Invalid weighting_mode '{self.weighting_mode}'. "
                f"Choose from {valid_modes}."
            )

    @property
    def active_sections(self) -> Optional[FrozenSet[str]]:
        return self.include_sections if self.include_sections else None


# ---------------------------------------------------------------------------
# AggregatedLMResult — structured output for one aggregation group
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregatedLMResult:
    """
    Immutable result for a single aggregation group (transcript, section, or
    speaker).  Used internally before DataFrame construction; also useful for
    unit tests that need to inspect individual group outputs without a full
    DataFrame roundtrip.

    Attributes
    ----------
    group_key : tuple
        The grouping key values (e.g. ``("AAPL_Q1_2025",)`` for transcript-
        level or ``("AAPL_Q1_2025", "qa")`` for section-level).
    mean_tone : float
        Weighted mean LM tone score across all chunks in the group.
    median_tone : float
        Median LM tone score (unweighted; robust to outlier chunks).
    std_tone : float
        Standard deviation of chunk-level tone scores.
    token_total : int
        Sum of token_count across all chunks.
    matched_token_total : int
        Sum of matched_token_count across all chunks.
    positive_total : int
        Sum of lm_positive_count.
    negative_total : int
        Sum of lm_negative_count.
    uncertainty_total : int
        Sum of lm_uncertainty_count.
    coverage_ratio : float
        ``matched_token_total / token_total``.  0.0 when token_total == 0.
    sentiment_label : str
        ``"positive"`` / ``"neutral"`` / ``"negative"`` derived from mean_tone.
    chunk_count : int
        Number of chunks in this group.
    is_low_coverage : bool
        ``True`` when coverage_ratio < config.min_coverage_threshold.
    is_low_token : bool
        ``True`` when token_total < config.min_token_threshold.
    sentiment_distribution : Dict[str, float]
        Fraction of chunks with each label.
    """

    group_key: Tuple
    mean_tone: float
    median_tone: float
    std_tone: float
    token_total: int
    matched_token_total: int
    positive_total: int
    negative_total: int
    uncertainty_total: int
    coverage_ratio: float
    sentiment_label: str
    chunk_count: int
    is_low_coverage: bool
    is_low_token: bool
    sentiment_distribution: Dict[str, float]

    def to_flat_dict(self) -> Dict:
        """Return a flat dict excluding nested structures (for DataFrame rows)."""
        return {
            "mean_tone": self.mean_tone,
            "median_tone": self.median_tone,
            "std_tone": self.std_tone,
            "token_total": self.token_total,
            "matched_token_total": self.matched_token_total,
            "positive_total": self.positive_total,
            "negative_total": self.negative_total,
            "uncertainty_total": self.uncertainty_total,
            "coverage_ratio": self.coverage_ratio,
            "sentiment_label": self.sentiment_label,
            "chunk_count": self.chunk_count,
            "is_low_coverage": self.is_low_coverage,
            "is_low_token": self.is_low_token,
        }


# ---------------------------------------------------------------------------
# AggregationStatistics — mutable batch-level diagnostics
# ---------------------------------------------------------------------------

@dataclass
class AggregationStatistics:
    """
    Mutable accumulator for diagnostics across an aggregation run.

    Attributes
    ----------
    total_input_rows : int
        Rows in the raw input DataFrame.
    total_groups : int
        Number of unique aggregation groups found.
    groups_produced : int
        Groups that passed all thresholds and appear in output.
    groups_dropped_low_token : int
        Groups dropped because total tokens < min_token_threshold.
    groups_flagged_low_coverage : int
        Groups flagged (but not necessarily dropped) as low-coverage.
    duplicate_chunk_ids : int
        Duplicate chunk_id values detected in input.
    duplicate_transcript_ids : int
        Duplicate transcript_id values in the output (should be 0).
    skipped_roles : Dict[str, int]
        Count of rows excluded per speaker role.
    section_counts : Dict[str, int]
        Number of output rows per section_type.
    elapsed_ms : float
        Total wall-clock aggregation time in milliseconds.
    """

    total_input_rows: int = 0
    total_groups: int = 0
    groups_produced: int = 0
    groups_dropped_low_token: int = 0
    groups_flagged_low_coverage: int = 0
    duplicate_chunk_ids: int = 0
    duplicate_transcript_ids: int = 0
    skipped_roles: Dict[str, int] = field(default_factory=dict)
    section_counts: Dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            "=== AggregationStatistics ===",
            f"  total_input_rows            : {self.total_input_rows}",
            f"  total_groups                : {self.total_groups}",
            f"  groups_produced             : {self.groups_produced}",
            f"  groups_dropped_low_token    : {self.groups_dropped_low_token}",
            f"  groups_flagged_low_coverage : {self.groups_flagged_low_coverage}",
            f"  duplicate_chunk_ids         : {self.duplicate_chunk_ids}",
            f"  duplicate_transcript_ids    : {self.duplicate_transcript_ids}",
            f"  elapsed_ms                  : {self.elapsed_ms:.2f}",
        ]
        if self.skipped_roles:
            lines.append(f"  skipped_roles               : {dict(self.skipped_roles)}")
        if self.section_counts:
            lines.append(f"  section_counts              : {dict(self.section_counts)}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.total_input_rows = 0
        self.total_groups = 0
        self.groups_produced = 0
        self.groups_dropped_low_token = 0
        self.groups_flagged_low_coverage = 0
        self.duplicate_chunk_ids = 0
        self.duplicate_transcript_ids = 0
        self.skipped_roles = {}
        self.section_counts = {}
        self.elapsed_ms = 0.0


# ---------------------------------------------------------------------------
# LMAggregator
# ---------------------------------------------------------------------------

class LMAggregator:
    """
    Hierarchical aggregation engine for Loughran-McDonald chunk-level scores.

    Accepts a chunk-level scoring DataFrame (one row per chunk, as produced by
    ``lm_scoring.py``) and produces four aggregated DataFrames:

    * **chunk**      — validated and enriched passthrough (coverage flags, etc.)
    * **transcript** — one row per transcript_id
    * **section**    — one row per (transcript_id, section_type)
    * **speaker**    — one row per (transcript_id, speaker_role)

    All outputs are deterministic (stable sort, no random operations).

    Parameters
    ----------
    config : LMAggregationConfig, optional
        Aggregation behaviour.  Uses research-grade defaults when omitted.

    Usage
    -----
    ::

        aggregator = LMAggregator()
        chunk_df    = aggregator.aggregate_chunks(raw_scores_df)
        tx_df       = aggregator.aggregate_transcripts(chunk_df)
        section_df  = aggregator.aggregate_sections(chunk_df)
        speaker_df  = aggregator.aggregate_speakers(chunk_df)

        stats = aggregator.last_run_statistics
        print(stats.summary())
    """

    def __init__(self, config: Optional[LMAggregationConfig] = None) -> None:
        self._config: LMAggregationConfig = config or LMAggregationConfig()
        self._stats: AggregationStatistics = AggregationStatistics()
        logger.info(
            "[LMAggregator] Initialised — weighting_mode=%s, "
            "min_tokens=%d, min_coverage=%.3f.",
            self._config.weighting_mode,
            self._config.min_token_threshold,
            self._config.min_coverage_threshold,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def last_run_statistics(self) -> AggregationStatistics:
        """Statistics from the most recent aggregation run."""
        return self._stats

    # ------------------------------------------------------------------ #
    #  1. Chunk-level                                                      #
    # ------------------------------------------------------------------ #

    def aggregate_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate, enrich, and return the chunk-level DataFrame.

        Adds derived columns (``coverage_ratio``, ``lm_label``,
        ``is_low_coverage``, ``is_low_token``) without grouping.

        Parameters
        ----------
        df : pd.DataFrame
            Raw chunk-level LM scores.  Must contain ``_REQUIRED_CHUNK_COLUMNS``.

        Returns
        -------
        pd.DataFrame
            Enriched chunk-level DataFrame sorted by
            ``(transcript_id, chunk_id)`` when ``config.sort_output=True``.
        """
        t0 = time.perf_counter()
        self._stats.reset()
        self._stats.total_input_rows = len(df)

        logger.info(
            "[LMAggregator] aggregate_chunks: %d input rows.", len(df)
        )

        df = df.copy()
        self._ensure_required_columns(df, _REQUIRED_CHUNK_COLUMNS)
        self._fill_optional_count_columns(df)

        # Duplicate chunk detection
        dup_mask = df["chunk_id"].duplicated(keep=False)
        n_dups = int(dup_mask.sum())
        if n_dups:
            self._stats.duplicate_chunk_ids = n_dups
            logger.warning(
                "[LMAggregator] %d duplicate chunk_id rows detected.", n_dups
            )

        # Derive coverage_ratio
        df["coverage_ratio"] = _safe_divide(
            df["matched_token_count"].astype(float),
            df["token_count"].astype(float),
        )

        # Derive lm_label from lm_tone_score
        df["lm_label"] = df["lm_tone_score"].apply(self._score_to_label)

        # Quality flags
        df["is_low_coverage"] = (
            df["coverage_ratio"] < self._config.min_coverage_threshold
        )
        df["is_low_token"] = (
            df["token_count"] < self._config.min_token_threshold
        )

        if self._config.sort_output:
            df = df.sort_values(
                ["transcript_id", "chunk_id"], kind="mergesort"
            ).reset_index(drop=True)

        elapsed = (time.perf_counter() - t0) * 1_000
        self._stats.elapsed_ms += elapsed
        logger.info(
            "[LMAggregator] aggregate_chunks complete: %d rows [%.1f ms].",
            len(df),
            elapsed,
        )
        return df

    # ------------------------------------------------------------------ #
    #  2. Transcript-level                                                 #
    # ------------------------------------------------------------------ #

    def aggregate_transcripts(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate chunk-level scores to one row per transcript.

        Parameters
        ----------
        chunk_df : pd.DataFrame
            Chunk-level DataFrame (output of :meth:`aggregate_chunks` or
            any DataFrame with ``_REQUIRED_CHUNK_COLUMNS``).

        Returns
        -------
        pd.DataFrame
            One row per ``transcript_id`` with ``lm_`` prefixed columns.
        """
        t0 = time.perf_counter()
        logger.info(
            "[LMAggregator] aggregate_transcripts: %d chunk rows → groupby transcript_id.",
            len(chunk_df),
        )

        self._ensure_required_columns(chunk_df, _REQUIRED_CHUNK_COLUMNS)
        self._fill_optional_count_columns(chunk_df)
        df = chunk_df.copy()
        self._fill_optional_count_columns(df)

        result_rows: List[Dict] = []
        groups = df.groupby("transcript_id", sort=True)
        self._stats.total_groups += len(groups)

        for transcript_id, group in groups:
            agg = self._aggregate_group(group, group_key=(transcript_id,))
            if agg is None:
                continue

            row: Dict = {"transcript_id": transcript_id}
            row.update(self._agg_result_to_lm_columns(agg))
            row.update(self._sum_count_columns(group))
            result_rows.append(row)
            self._stats.groups_produced += 1

        out = pd.DataFrame(result_rows)
        if out.empty:
            logger.warning("[LMAggregator] aggregate_transcripts produced empty output.")
            return out

        # Duplicate output check
        n_dup = int(out["transcript_id"].duplicated().sum())
        if n_dup:
            self._stats.duplicate_transcript_ids = n_dup
            logger.warning(
                "[LMAggregator] %d duplicate transcript_ids in output.", n_dup
            )

        elapsed = (time.perf_counter() - t0) * 1_000
        self._stats.elapsed_ms += elapsed
        logger.info(
            "[LMAggregator] aggregate_transcripts → %d rows [%.1f ms].",
            len(out),
            elapsed,
        )
        return out

    # ------------------------------------------------------------------ #
    #  3. Section-level                                                    #
    # ------------------------------------------------------------------ #

    def aggregate_sections(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate to one row per ``(transcript_id, section_type)`` pair.

        Parameters
        ----------
        chunk_df : pd.DataFrame
            Chunk-level DataFrame.  Must also contain ``section_type``.

        Returns
        -------
        pd.DataFrame
            One row per (transcript_id, section_type) with ``lm_`` columns.
        """
        t0 = time.perf_counter()

        required = list(_REQUIRED_CHUNK_COLUMNS) + ["section_type"]
        self._ensure_required_columns(chunk_df, tuple(required))

        df = chunk_df.copy()
        self._fill_optional_count_columns(df)

        # Filter to included sections if configured
        active_sections = self._config.active_sections
        if active_sections:
            before = len(df)
            df = df[df["section_type"].isin(active_sections)]
            logger.info(
                "[LMAggregator] Section filter: %d → %d rows "
                "(kept %s).",
                before, len(df), sorted(active_sections),
            )

        logger.info(
            "[LMAggregator] aggregate_sections: %d rows → groupby "
            "(transcript_id, section_type).",
            len(df),
        )

        result_rows: List[Dict] = []
        groups = df.groupby(["transcript_id", "section_type"], sort=True)

        for (transcript_id, section_type), group in groups:
            agg = self._aggregate_group(
                group, group_key=(transcript_id, section_type)
            )
            if agg is None:
                continue

            row: Dict = {
                "transcript_id": transcript_id,
                "section_type": section_type,
            }
            row.update(self._agg_result_to_lm_columns(agg))
            row.update(self._sum_count_columns(group))
            result_rows.append(row)

            # Track section distribution
            self._stats.section_counts[section_type] = (
                self._stats.section_counts.get(section_type, 0) + 1
            )

        out = pd.DataFrame(result_rows)
        if out.empty:
            logger.warning("[LMAggregator] aggregate_sections produced empty output.")
            return out

        if self._config.sort_output:
            out = out.sort_values(
                ["transcript_id", "section_type"], kind="mergesort"
            ).reset_index(drop=True)

        elapsed = (time.perf_counter() - t0) * 1_000
        self._stats.elapsed_ms += elapsed
        logger.info(
            "[LMAggregator] aggregate_sections → %d rows [%.1f ms].",
            len(out),
            elapsed,
        )
        return out

    # ------------------------------------------------------------------ #
    #  4. Speaker-level                                                    #
    # ------------------------------------------------------------------ #

    def aggregate_speakers(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate to one row per ``(transcript_id, speaker_role)`` pair,
        excluding configured operator/unknown roles.

        Parameters
        ----------
        chunk_df : pd.DataFrame
            Chunk-level DataFrame.  Must contain ``speaker_role``.

        Returns
        -------
        pd.DataFrame
            One row per (transcript_id, speaker_role) with ``lm_`` columns.
        """
        t0 = time.perf_counter()

        required = list(_REQUIRED_CHUNK_COLUMNS) + ["speaker_role"]
        self._ensure_required_columns(chunk_df, tuple(required))

        df = chunk_df.copy()
        self._fill_optional_count_columns(df)

        # Normalise roles to lowercase for deterministic grouping
        df["_role_norm"] = df["speaker_role"].str.lower().str.strip().fillna("")

        # Exclude unwanted roles
        excluded = self._config.exclude_roles
        exclude_mask = df["_role_norm"].isin(excluded)
        if exclude_mask.any():
            role_counts: Dict[str, int] = (
                df.loc[exclude_mask, "_role_norm"]
                .value_counts()
                .to_dict()
            )
            self._stats.skipped_roles.update(role_counts)
            logger.info(
                "[LMAggregator] Excluded %d rows by role: %s",
                int(exclude_mask.sum()),
                role_counts,
            )
            df = df[~exclude_mask].copy()

        logger.info(
            "[LMAggregator] aggregate_speakers: %d rows → groupby "
            "(transcript_id, speaker_role).",
            len(df),
        )

        result_rows: List[Dict] = []
        groups = df.groupby(["transcript_id", "_role_norm"], sort=True)

        for (transcript_id, role), group in groups:
            agg = self._aggregate_group(
                group, group_key=(transcript_id, role)
            )
            if agg is None:
                continue

            row: Dict = {
                "transcript_id": transcript_id,
                "speaker_role": role,
            }
            row.update(self._agg_result_to_lm_columns(agg))
            row.update(self._sum_count_columns(group))
            result_rows.append(row)

        out = pd.DataFrame(result_rows)
        if out.empty:
            logger.warning("[LMAggregator] aggregate_speakers produced empty output.")
            return out

        # Drop helper column if it survived
        out = out.drop(columns=["_role_norm"], errors="ignore")

        if self._config.sort_output:
            out = out.sort_values(
                ["transcript_id", "speaker_role"], kind="mergesort"
            ).reset_index(drop=True)

        elapsed = (time.perf_counter() - t0) * 1_000
        self._stats.elapsed_ms += elapsed
        logger.info(
            "[LMAggregator] aggregate_speakers → %d rows [%.1f ms].",
            len(out),
            elapsed,
        )
        return out

    # ------------------------------------------------------------------ #
    #  5. Public statistics helper                                         #
    # ------------------------------------------------------------------ #

    def compute_statistics(self, df: pd.DataFrame, level: str) -> Dict:
        """
        Compute descriptive statistics over a finalised aggregation output.

        Parameters
        ----------
        df : pd.DataFrame
            Output from one of the ``aggregate_*`` methods.
        level : str
            Human-readable label for logging (e.g. ``"transcript"``).

        Returns
        -------
        Dict
            Keys: mean_tone, std_tone, median_tone, min_tone, max_tone,
            positive_pct, negative_pct, neutral_pct, n_groups,
            n_low_coverage, n_low_token.
        """
        if df.empty or "lm_mean_tone" not in df.columns:
            logger.warning(
                "[LMAggregator] compute_statistics: empty or missing "
                "lm_mean_tone in '%s' level.", level
            )
            return {}

        tones = df["lm_mean_tone"].dropna()
        n = len(tones)

        label_col = "lm_sentiment_label"
        if label_col in df.columns:
            vc = df[label_col].value_counts(normalize=True).to_dict()
        else:
            vc = {}

        stats: Dict = {
            "level": level,
            "n_groups": n,
            "mean_tone": float(tones.mean()) if n else float("nan"),
            "median_tone": float(tones.median()) if n else float("nan"),
            "std_tone": float(tones.std()) if n > 1 else 0.0,
            "min_tone": float(tones.min()) if n else float("nan"),
            "max_tone": float(tones.max()) if n else float("nan"),
            "positive_pct": float(vc.get("positive", 0.0)),
            "negative_pct": float(vc.get("negative", 0.0)),
            "neutral_pct": float(vc.get("neutral", 0.0)),
            "n_low_coverage": int(
                df.get("lm_is_low_coverage", pd.Series([], dtype=bool)).sum()
            ),
            "n_low_token": int(
                df.get("lm_is_low_token", pd.Series([], dtype=bool)).sum()
            ),
        }
        logger.info(
            "[LMAggregator] compute_statistics [%s]: n=%d, "
            "mean=%.4f, std=%.4f, +%%=%.2f, -%%=%.2f.",
            level,
            stats["n_groups"],
            stats["mean_tone"],
            stats["std_tone"],
            stats["positive_pct"],
            stats["negative_pct"],
        )
        return stats

    # ------------------------------------------------------------------ #
    #  6. Validation                                                       #
    # ------------------------------------------------------------------ #

    def validate_aggregation(
        self,
        df: pd.DataFrame,
        level: str,
        key_columns: Tuple[str, ...],
    ) -> List[str]:
        """
        Validate a finalised aggregation output DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated output to validate.
        level : str
            Aggregation level name for logging context.
        key_columns : tuple[str, ...]
            Columns that form the unique key (e.g. ``("transcript_id",)``
            or ``("transcript_id", "section_type")``).

        Returns
        -------
        List[str]
            Validation issue descriptions.  Empty list → output is clean.
        """
        issues: List[str] = []

        if df.empty:
            issues.append(f"[{level}] Output DataFrame is empty.")
            logger.warning("[LMAggregator] validate_aggregation [%s]: empty output.", level)
            return issues

        # Duplicate key check
        dup_mask = df.duplicated(subset=list(key_columns), keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup:
            issues.append(
                f"[{level}] {n_dup} duplicate rows on key {key_columns}."
            )

        # Tone score range
        if "lm_mean_tone" in df.columns:
            out_of_range = df["lm_mean_tone"].dropna()
            bad = out_of_range[(out_of_range < -1.0) | (out_of_range > 1.0)]
            if len(bad):
                issues.append(
                    f"[{level}] {len(bad)} rows with lm_mean_tone outside [-1, 1]: "
                    f"{bad.tolist()[:5]}"
                )

        # Non-negative counts
        for col in ("lm_positive_total", "lm_negative_total", "lm_token_total"):
            if col in df.columns:
                neg_rows = int((df[col] < 0).sum())
                if neg_rows:
                    issues.append(
                        f"[{level}] {neg_rows} rows with negative {col}."
                    )

        # Coverage ratio bounds
        if "lm_coverage_ratio" in df.columns:
            bad_cov = df["lm_coverage_ratio"].dropna()
            bad_cov = bad_cov[(bad_cov < 0) | (bad_cov > 1)]
            if len(bad_cov):
                issues.append(
                    f"[{level}] {len(bad_cov)} rows with lm_coverage_ratio outside [0, 1]."
                )

        # NaN tone scores
        if "lm_mean_tone" in df.columns:
            n_nan = int(df["lm_mean_tone"].isna().sum())
            if n_nan:
                issues.append(
                    f"[{level}] {n_nan} rows with NaN lm_mean_tone."
                )

        # Label validity
        valid_labels = {"positive", "neutral", "negative"}
        if "lm_sentiment_label" in df.columns:
            invalid_labels = set(df["lm_sentiment_label"].dropna().unique()) - valid_labels
            if invalid_labels:
                issues.append(
                    f"[{level}] Invalid sentiment labels: {invalid_labels}."
                )

        if issues:
            logger.warning(
                "[LMAggregator] validate_aggregation [%s]: %d issue(s).",
                level, len(issues),
            )
        else:
            logger.info(
                "[LMAggregator] validate_aggregation [%s]: PASSED.", level
            )
        return issues

    def summary(self) -> str:
        """Return a combined configuration + statistics summary string."""
        cfg = self._config
        lines = [
            "=== LMAggregator ===",
            f"  weighting_mode        : {cfg.weighting_mode}",
            f"  min_token_threshold   : {cfg.min_token_threshold}",
            f"  min_coverage_threshold: {cfg.min_coverage_threshold}",
            f"  drop_low_token_groups : {cfg.drop_low_token_groups}",
            f"  sort_output           : {cfg.sort_output}",
            f"  label_thresholds      : {cfg.label_thresholds}",
            "",
        ]
        lines.append(self._stats.summary())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_group(
        self,
        group: pd.DataFrame,
        group_key: Tuple,
    ) -> Optional[AggregatedLMResult]:
        """
        Compute an ``AggregatedLMResult`` for a single pandas group.

        Returns ``None`` when the group is dropped due to threshold policy.
        """
        n_chunks = len(group)
        token_total = int(group["token_count"].sum())
        matched_total = int(
            group["matched_token_count"].sum()
            if "matched_token_count" in group.columns
            else 0
        )
        coverage = _safe_divide(float(matched_total), float(token_total))

        is_low_token = token_total < self._config.min_token_threshold
        is_low_cov = coverage < self._config.min_coverage_threshold

        if is_low_token:
            self._stats.groups_dropped_low_token += 1
            if self._config.drop_low_token_groups:
                logger.debug(
                    "[LMAggregator] Dropping group %s — "
                    "tokens=%d < threshold=%d.",
                    group_key, token_total, self._config.min_token_threshold,
                )
                return None

        if is_low_cov:
            self._stats.groups_flagged_low_coverage += 1
            logger.debug(
                "[LMAggregator] Low-coverage group %s: coverage=%.4f.",
                group_key, coverage,
            )

        # Compute weights
        weights = self._compute_weights(group)
        scores = group["lm_tone_score"].fillna(0.0).values

        mean_tone = float(np.average(scores, weights=weights))
        median_tone = float(np.median(scores))
        std_tone = (
            float(np.sqrt(np.average((scores - mean_tone) ** 2, weights=weights)))
            if n_chunks > 1
            else 0.0
        )

        # Sentiment distribution over chunks
        if "lm_label" in group.columns:
            label_series = group["lm_label"]
        else:
            label_series = pd.Series(
                [self._score_to_label(s) for s in scores],
                index=group.index,
            )
        dist = label_series.value_counts(normalize=True).to_dict()
        sentiment_distribution = {
            "positive": float(dist.get("positive", 0.0)),
            "neutral":  float(dist.get("neutral", 0.0)),
            "negative": float(dist.get("negative", 0.0)),
        }

        return AggregatedLMResult(
            group_key=group_key,
            mean_tone=mean_tone,
            median_tone=median_tone,
            std_tone=std_tone,
            token_total=token_total,
            matched_token_total=matched_total,
            positive_total=int(group.get("lm_positive_count", pd.Series([0] * n_chunks)).sum()),
            negative_total=int(group.get("lm_negative_count", pd.Series([0] * n_chunks)).sum()),
            uncertainty_total=int(group.get("lm_uncertainty_count", pd.Series([0] * n_chunks)).sum()),
            coverage_ratio=coverage,
            sentiment_label=self._score_to_label(mean_tone),
            chunk_count=n_chunks,
            is_low_coverage=is_low_cov,
            is_low_token=is_low_token,
            sentiment_distribution=sentiment_distribution,
        )

    def _compute_weights(self, group: pd.DataFrame) -> np.ndarray:
        """
        Return a weight vector for the rows in *group* based on
        ``config.weighting_mode``.

        All returned arrays are non-negative and sum to at least 1 (uniform
        fallback is applied when all computed weights are zero).
        """
        mode = self._config.weighting_mode
        n = len(group)

        if mode == "uniform":
            return np.ones(n, dtype=float)

        if mode == "token":
            w = group["token_count"].fillna(1.0).values.astype(float)

        elif mode == "coverage":
            if "coverage_ratio" in group.columns:
                w = group["coverage_ratio"].fillna(0.0).values.astype(float)
            elif "matched_token_count" in group.columns and "token_count" in group.columns:
                tok = group["token_count"].fillna(1.0).values.astype(float)
                matched = group["matched_token_count"].fillna(0.0).values.astype(float)
                w = np.where(tok > 0, matched / tok, 0.0)
            else:
                w = np.ones(n, dtype=float)

        elif mode == "hybrid":
            tok = group["token_count"].fillna(1.0).values.astype(float)
            if "coverage_ratio" in group.columns:
                cov = group["coverage_ratio"].fillna(0.0).values.astype(float)
            elif "matched_token_count" in group.columns:
                matched = group["matched_token_count"].fillna(0.0).values.astype(float)
                cov = np.where(tok > 0, matched / tok, 0.0)
            else:
                cov = np.ones(n, dtype=float)
            w = tok * cov

        else:
            w = np.ones(n, dtype=float)

        # Fallback: if all weights are zero use uniform
        total = w.sum()
        if total == 0:
            logger.debug(
                "[LMAggregator] All weights zero in group — falling back to uniform."
            )
            return np.ones(n, dtype=float)
        return w

    def _score_to_label(self, score: float) -> str:
        """Convert a numeric tone score to a sentiment label string."""
        if pd.isna(score):
            return "neutral"
        pos_thresh, neg_thresh = self._config.label_thresholds
        if score >= pos_thresh:
            return "positive"
        if score <= neg_thresh:
            return "negative"
        return "neutral"

    @staticmethod
    def _agg_result_to_lm_columns(agg: AggregatedLMResult) -> Dict:
        """
        Map an AggregatedLMResult to the ``lm_`` prefixed output column dict.
        """
        return {
            "lm_mean_tone":          agg.mean_tone,
            "lm_median_tone":        agg.median_tone,
            "lm_std_tone":           agg.std_tone,
            "lm_token_total":        agg.token_total,
            "lm_matched_token_total":agg.matched_token_total,
            "lm_coverage_ratio":     agg.coverage_ratio,
            "lm_sentiment_label":    agg.sentiment_label,
            "lm_chunk_count":        agg.chunk_count,
            "lm_is_low_coverage":    agg.is_low_coverage,
            "lm_is_low_token":       agg.is_low_token,
            "lm_dist_positive":      agg.sentiment_distribution.get("positive", 0.0),
            "lm_dist_neutral":       agg.sentiment_distribution.get("neutral", 0.0),
            "lm_dist_negative":      agg.sentiment_distribution.get("negative", 0.0),
        }

    @staticmethod
    def _sum_count_columns(group: pd.DataFrame) -> Dict:
        """
        Sum all LM count columns present in *group* and return a flat dict
        with ``lm_`` prefixed output names.
        """
        output_map = {
            "lm_positive_count":     "lm_positive_total",
            "lm_negative_count":     "lm_negative_total",
            "lm_uncertainty_count":  "lm_uncertainty_total",
            "lm_litigious_count":    "lm_litigious_total",
            "lm_strong_modal_count": "lm_strong_modal_total",
            "lm_weak_modal_count":   "lm_weak_modal_total",
            "lm_constraining_count": "lm_constraining_total",
        }
        row: Dict = {}
        for src_col, dst_col in output_map.items():
            if src_col in group.columns:
                row[dst_col] = int(group[src_col].fillna(0).sum())
        return row

    @staticmethod
    def _ensure_required_columns(
        df: pd.DataFrame,
        required: Tuple[str, ...],
    ) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    @staticmethod
    def _fill_optional_count_columns(df: pd.DataFrame) -> None:
        """Zero-fill optional LM count columns if absent (in-place)."""
        for col in _OPTIONAL_COUNT_COLUMNS:
            if col not in df.columns:
                df[col] = 0


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def _safe_divide(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` or ``0.0`` on zero division."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Self-test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    print("\n" + "=" * 72)
    print("LMAggregator — self-test / demo")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 1. Build a synthetic chunk-level LM score DataFrame                 #
    # ------------------------------------------------------------------ #
    print("\n[1] Building synthetic chunk-level LM scores …")

    rng = np.random.default_rng(42)

    def _make_chunks(
        transcript_id: str,
        ticker: str,
        n_chunks: int,
        section_types: List[str],
        speaker_roles: List[str],
        tone_mean: float,
        tone_std: float,
    ) -> List[Dict]:
        rows = []
        for i in range(n_chunks):
            tone = float(np.clip(rng.normal(tone_mean, tone_std), -1, 1))
            pos = max(0, int((tone + 1) * 5 + rng.integers(0, 3)))
            neg = max(0, int((1 - tone) * 3 + rng.integers(0, 3)))
            unc = rng.integers(0, 4)
            tok = rng.integers(80, 200)
            matched = max(1, int(tok * rng.uniform(0.04, 0.18)))
            rows.append({
                "transcript_id":      transcript_id,
                "ticker":             ticker,
                "chunk_id":           f"{transcript_id}_chunk_{i:04d}",
                "chunk_order":        i,
                "section_type":       section_types[i % len(section_types)],
                "speaker_role":       speaker_roles[i % len(speaker_roles)],
                "lm_tone_score":      round(tone, 6),
                "lm_positive_count":  pos,
                "lm_negative_count":  neg,
                "lm_uncertainty_count": unc,
                "lm_litigious_count": rng.integers(0, 2),
                "lm_strong_modal_count": rng.integers(0, 3),
                "lm_weak_modal_count": rng.integers(0, 3),
                "lm_constraining_count": rng.integers(0, 2),
                "token_count":        int(tok),
                "matched_token_count": int(matched),
            })
        return rows

    all_rows: List[Dict] = []
    all_rows.extend(_make_chunks(
        "AAPL_Q1_2025", "AAPL", 24,
        ["prepared_remarks", "qa"],
        ["ceo", "cfo", "analyst"],
        tone_mean=0.10, tone_std=0.08,
    ))
    all_rows.extend(_make_chunks(
        "MSFT_Q2_2025", "MSFT", 20,
        ["prepared_remarks", "qa"],
        ["ceo", "cfo", "operator"],
        tone_mean=-0.05, tone_std=0.12,
    ))
    all_rows.extend(_make_chunks(
        "NVDA_Q3_2025", "NVDA", 30,
        ["prepared_remarks", "qa", "closing_remarks"],
        ["ceo", "analyst", "analyst"],
        tone_mean=0.18, tone_std=0.06,
    ))
    # Duplicate chunk to test detection
    dupe_row = dict(all_rows[0])
    all_rows.append(dupe_row)

    raw_df = pd.DataFrame(all_rows)
    print(f"  Synthetic DataFrame: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    print(f"  Transcripts: {raw_df['transcript_id'].unique().tolist()}")
    print(f"  Sections   : {raw_df['section_type'].unique().tolist()}")
    print(f"  Roles      : {raw_df['speaker_role'].unique().tolist()}")

    # ------------------------------------------------------------------ #
    # 2. Instantiate aggregator (hybrid weighting)                        #
    # ------------------------------------------------------------------ #
    print("\n[2] Instantiating LMAggregator (hybrid mode) …")
    config = LMAggregationConfig(
        weighting_mode="hybrid",
        min_token_threshold=50,
        min_coverage_threshold=0.02,
        drop_low_token_groups=False,
        sort_output=True,
    )
    aggregator = LMAggregator(config)

    # ------------------------------------------------------------------ #
    # 3. Chunk-level aggregation                                          #
    # ------------------------------------------------------------------ #
    print("\n[3] aggregate_chunks() …")
    chunk_df = aggregator.aggregate_chunks(raw_df)
    print(f"  Output shape : {chunk_df.shape}")
    print("  New columns  :", [c for c in chunk_df.columns if c not in raw_df.columns])
    print(f"  Low coverage : {chunk_df['is_low_coverage'].sum()}")
    print(f"  Low token    : {chunk_df['is_low_token'].sum()}")

    # ------------------------------------------------------------------ #
    # 4. Transcript-level aggregation                                     #
    # ------------------------------------------------------------------ #
    print("\n[4] aggregate_transcripts() …")
    tx_df = aggregator.aggregate_transcripts(chunk_df)
    print(f"  Output shape : {tx_df.shape}")
    print(tx_df[["transcript_id", "lm_mean_tone", "lm_sentiment_label",
                  "lm_chunk_count", "lm_coverage_ratio"]].to_string(index=False))

    # Validate
    issues = aggregator.validate_aggregation(tx_df, "transcript", ("transcript_id",))
    print(f"  Validation issues: {issues}")

    # Statistics
    tx_stats = aggregator.compute_statistics(tx_df, "transcript")
    print("  Transcript stats:", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                             for k, v in tx_stats.items()}, indent=2))

    # ------------------------------------------------------------------ #
    # 5. Section-level aggregation                                        #
    # ------------------------------------------------------------------ #
    print("\n[5] aggregate_sections() …")
    sec_df = aggregator.aggregate_sections(chunk_df)
    print(f"  Output shape : {sec_df.shape}")
    print(sec_df[["transcript_id", "section_type", "lm_mean_tone",
                   "lm_sentiment_label", "lm_chunk_count"]].to_string(index=False))

    issues_sec = aggregator.validate_aggregation(
        sec_df, "section", ("transcript_id", "section_type")
    )
    print(f"  Validation issues: {issues_sec}")

    # ------------------------------------------------------------------ #
    # 6. Speaker-level aggregation                                        #
    # ------------------------------------------------------------------ #
    print("\n[6] aggregate_speakers() …")
    spk_df = aggregator.aggregate_speakers(chunk_df)
    print(f"  Output shape : {spk_df.shape}")
    print(spk_df[["transcript_id", "speaker_role", "lm_mean_tone",
                   "lm_sentiment_label", "lm_chunk_count"]].to_string(index=False))

    issues_spk = aggregator.validate_aggregation(
        spk_df, "speaker", ("transcript_id", "speaker_role")
    )
    print(f"  Validation issues: {issues_spk}")

    # ------------------------------------------------------------------ #
    # 7. Compare weighting modes                                          #
    # ------------------------------------------------------------------ #
    print("\n[7] Comparing weighting modes for AAPL_Q1_2025 …")
    aapl_chunks = chunk_df[chunk_df["transcript_id"] == "AAPL_Q1_2025"].copy()

    for mode in ("uniform", "token", "coverage", "hybrid"):
        cfg_mode = LMAggregationConfig(weighting_mode=mode)
        agg_mode = LMAggregator(cfg_mode)
        tx = agg_mode.aggregate_transcripts(aapl_chunks)
        row = tx.iloc[0]
        print(
            f"  [{mode:<8}]  mean_tone={row['lm_mean_tone']:+.6f}  "
            f"label={row['lm_sentiment_label']}"
        )

    # ------------------------------------------------------------------ #
    # 8. Aggregator summary                                               #
    # ------------------------------------------------------------------ #
    print("\n[8] Aggregator summary (after all runs) …")
    print(aggregator.summary())

    # ------------------------------------------------------------------ #
    # 9. AggregatedLMResult.to_flat_dict()                               #
    # ------------------------------------------------------------------ #
    print("\n[9] AggregatedLMResult.to_flat_dict() sample …")
    sample_agg = AggregatedLMResult(
        group_key=("AAPL_Q1_2025",),
        mean_tone=0.12,
        median_tone=0.10,
        std_tone=0.05,
        token_total=2400,
        matched_token_total=312,
        positive_total=88,
        negative_total=22,
        uncertainty_total=14,
        coverage_ratio=0.13,
        sentiment_label="positive",
        chunk_count=24,
        is_low_coverage=False,
        is_low_token=False,
        sentiment_distribution={"positive": 0.75, "neutral": 0.20, "negative": 0.05},
    )
    print(json.dumps(sample_agg.to_flat_dict(), indent=2))

    print("\n" + "=" * 72)
    print("Self-test complete.")
    print("=" * 72 + "\n")
