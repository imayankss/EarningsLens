"""
src/sentiment/lm_utils.py
==========================
Reusable utility and helper module for the Loughran-McDonald financial
sentiment system.

Architecture role
-----------------
Shared foundation imported by every LM subsystem module. Contains ONLY
pure, stateless helper functions and lightweight dataclasses — no business
logic, no model state, no pipeline orchestration.

Dependency rule
---------------
This module may not import from any other lm_*.py module.
All other lm_*.py modules may freely import from this one.

Utility groups
--------------
1.  Text normalisation       — unicode, whitespace, control chars
2.  Token utilities          — validation, filtering, frequency
3.  Numeric utilities        — safe arithmetic, clipping, normalisation
4.  Coverage utilities       — match-rate and coverage diagnostics
5.  Validation helpers       — range checks, duplicate detection
6.  Diagnostics helpers      — distribution, null, label summaries
7.  Export helpers           — parquet / CSV with error handling
8.  Timing / logging helpers — decorator + DataFrame profiling
"""

from __future__ import annotations

import functools
import logging
import math
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Module-level compiled patterns (avoids re-compilation on every call)
# ---------------------------------------------------------------------------
import re as _re

_RE_WHITESPACE     = _re.compile(r"\s+")
_RE_CONTROL        = _re.compile(r"[\x00-\x1f\x7f-\x9f]")
_RE_NON_PRINTABLE  = _re.compile(r"[^\x20-\x7e\u00a0-\ufffd]")
_RE_MULTI_PUNCT    = _re.compile(r"[.!?,;:]{2,}")
_RE_LEADING_PUNCT  = _re.compile(r"^[^a-zA-Z0-9]+")
_RE_TRAILING_PUNCT = _re.compile(r"[^a-zA-Z0-9%]+$")


# ===========================================================================
# 1. TEXT NORMALISATION
# ===========================================================================


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Apply Unicode normalisation to canonical form.

    NFC is the default — composes diacritics into single code points,
    which is what most tokenisers and dictionary lookup tables expect.

    Parameters
    ----------
    text : raw string (may contain multi-codepoint sequences)
    form : Unicode normalisation form (NFC | NFD | NFKC | NFKD)

    Returns
    -------
    Normalised string; empty string on null / non-string input.
    """
    if not isinstance(text, str) or not text:
        return ""
    try:
        return unicodedata.normalize(form, text)
    except (TypeError, ValueError) as exc:
        logger.debug("normalize_unicode failed: %s", exc)
        return text


def remove_control_chars(text: str) -> str:
    """
    Strip C0/C1 control characters and non-printable code points.

    Preserves standard printable ASCII and the majority of the Latin
    extended range (U+00A0 – U+FFFD).  Hard tabs and newlines are
    converted to spaces rather than deleted so downstream whitespace
    normalisation can handle them uniformly.

    Returns empty string on null / non-string input.
    """
    if not isinstance(text, str) or not text:
        return ""
    # Tabs and newlines → space
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = _RE_CONTROL.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse all internal whitespace runs to a single space and strip
    leading / trailing whitespace.

    Returns empty string on null / non-string input.
    """
    if not isinstance(text, str) or not text:
        return ""
    return _RE_WHITESPACE.sub(" ", text).strip()


def safe_lower(text: str) -> str:
    """
    Lowercase a string with null / type safety.

    Returns empty string for None or non-string inputs rather than raising.
    """
    if not isinstance(text, str):
        return ""
    return text.lower()


def clean_token(token: str, *, lowercase: bool = True) -> str:
    """
    Apply the standard single-token cleaning sequence used by the LM
    preprocessing and matching layers.

    Steps
    -----
    1. Null / type guard → return ""
    2. Strip leading / trailing non-alphanumeric characters
       (preserves % inside tokens for financial percentages)
    3. Optionally lowercase

    This is intentionally lightweight — heavy normalisation lives in
    LMPreprocessor.  clean_token() is for individual token post-processing.
    """
    if not isinstance(token, str) or not token:
        return ""
    token = _RE_LEADING_PUNCT.sub("", token)
    token = _RE_TRAILING_PUNCT.sub("", token)
    if lowercase:
        token = token.lower()
    return token


def full_text_clean(
    text: str,
    *,
    lowercase: bool = True,
    unicode_form: str = "NFC",
) -> str:
    """
    Convenience pipeline: unicode → control chars → whitespace → lowercase.

    Intended for quick ad-hoc cleaning when the full LMPreprocessor is
    not required (e.g. inside diagnostic helpers).
    """
    text = normalize_unicode(text, form=unicode_form)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)
    if lowercase:
        text = safe_lower(text)
    return text


# ===========================================================================
# 2. TOKEN UTILITIES
# ===========================================================================


def validate_token(
    token: str,
    *,
    min_length: int = 2,
    max_length: int = 60,
    allow_numeric: bool = True,
) -> bool:
    """
    Return True if *token* passes the standard LM validity criteria.

    Rules
    -----
    * Must be a non-empty string
    * Length within [min_length, max_length]
    * Not composed entirely of digits longer than 10 chars (avoids phone
      numbers and long IDs passing as tokens)
    * Contains at least one alphabetic character unless allow_numeric=True
    """
    if not isinstance(token, str) or not token.strip():
        return False
    if not (min_length <= len(token) <= max_length):
        return False
    if token.isdigit() and len(token) > 10:
        return False
    if not allow_numeric and not any(c.isalpha() for c in token):
        return False
    return True


def filter_tokens(
    tokens: Sequence[str],
    *,
    min_length: int = 2,
    max_length: int = 60,
    stopwords: Optional[Collection[str]] = None,
    allow_numeric: bool = True,
) -> List[str]:
    """
    Apply validate_token() and optional stopword removal to a token list.

    Returns a new list; the input is never mutated.
    Stopword matching is case-insensitive.
    """
    sw = frozenset(s.lower() for s in stopwords) if stopwords else frozenset()
    return [
        t for t in tokens
        if validate_token(t, min_length=min_length,
                          max_length=max_length,
                          allow_numeric=allow_numeric)
        and t.lower() not in sw
    ]


def token_frequency(tokens: Iterable[str]) -> Counter:
    """Return a Counter of lowercased token frequencies."""
    return Counter(t.lower() for t in tokens if isinstance(t, str))


def unique_token_ratio(tokens: Sequence[str]) -> float:
    """
    Compute type-token ratio (TTR): unique tokens / total tokens.

    Returns 0.0 for empty input.  TTR close to 1.0 indicates high lexical
    diversity; close to 0.0 indicates heavy repetition (e.g. boilerplate).
    """
    if not tokens:
        return 0.0
    return len(set(t.lower() for t in tokens)) / len(tokens)


def top_n_tokens(
    tokens: Iterable[str],
    n: int = 10,
    *,
    exclude: Optional[Collection[str]] = None,
) -> List[Tuple[str, int]]:
    """
    Return the top-N most frequent tokens as (token, count) pairs.

    Parameters
    ----------
    tokens  : iterable of token strings
    n       : number of results to return
    exclude : optional set of tokens to suppress from results
    """
    exclude_set = frozenset(e.lower() for e in exclude) if exclude else frozenset()
    freq = token_frequency(tokens)
    filtered = [(tok, cnt) for tok, cnt in freq.items()
                if tok not in exclude_set]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:n]


# ===========================================================================
# 3. NUMERIC UTILITIES
# ===========================================================================


def safe_divide(
    numerator: float,
    denominator: float,
    *,
    default: float = 0.0,
    log_zero_div: bool = False,
) -> float:
    """
    Divide numerator by denominator, returning *default* on zero-division
    or non-finite inputs rather than raising.

    Parameters
    ----------
    numerator   : dividend
    denominator : divisor
    default     : value returned when denominator == 0 or inputs are non-finite
    log_zero_div: if True, emit a DEBUG log when zero-division occurs
    """
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return default
    if denominator == 0.0:
        if log_zero_div:
            logger.debug(
                "safe_divide: zero denominator (numerator=%.4f) → %.4f",
                numerator, default,
            )
        return default
    return numerator / denominator


def safe_mean(
    values: Sequence[float],
    *,
    ignore_nan: bool = True,
    default: float = 0.0,
) -> float:
    """
    Compute mean of a sequence, handling empty input and NaN values.

    Parameters
    ----------
    values     : numeric sequence
    ignore_nan : if True, NaN values are excluded before computing mean
    default    : returned when the (filtered) sequence is empty
    """
    if not values:
        return default
    arr = np.asarray(values, dtype=float)
    if ignore_nan:
        arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return default
    result = float(np.mean(arr))
    return result if math.isfinite(result) else default


def safe_std(
    values: Sequence[float],
    *,
    ignore_nan: bool = True,
    default: float = 0.0,
    ddof: int = 1,
) -> float:
    """
    Compute standard deviation, handling empty input, NaN, and single-element
    sequences (which return 0.0 rather than NaN with ddof=1).
    """
    if not values:
        return default
    arr = np.asarray(values, dtype=float)
    if ignore_nan:
        arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return default
    result = float(np.std(arr, ddof=ddof))
    return result if math.isfinite(result) else default


def clip_score_range(
    score: float,
    lo: float = -1.0,
    hi: float = 1.0,
) -> float:
    """
    Clip a numeric score to [lo, hi].

    Returns lo for non-finite inputs (guards against inf / NaN propagation).
    """
    if not math.isfinite(score):
        logger.debug("clip_score_range: non-finite input %.6g → %.4f", score, lo)
        return lo
    return float(np.clip(score, lo, hi))


def normalize_probability(values: Sequence[float]) -> List[float]:
    """
    Normalise a sequence of non-negative values to sum to 1.0.

    Returns a list of zeros if all values are zero or the sum is not finite.
    Negative values are clipped to 0.0 before normalisation.
    """
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total == 0.0 or not math.isfinite(total):
        return [0.0] * len(values)
    return (arr / total).tolist()


def weighted_mean(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    default: float = 0.0,
) -> float:
    """
    Compute a weighted mean, ignoring pairs where either value is NaN.

    Falls back to unweighted mean when all weights are zero.
    """
    if not values or not weights or len(values) != len(weights):
        return default

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not valid.any():
        finite_v = v[np.isfinite(v)]
        return float(np.mean(finite_v)) if finite_v.size else default

    result = float(np.average(v[valid], weights=w[valid]))
    return result if math.isfinite(result) else default


# ===========================================================================
# 4. COVERAGE UTILITIES
# ===========================================================================


def compute_coverage_ratio(
    matched_tokens: int,
    total_tokens: int,
    *,
    default: float = 0.0,
) -> float:
    """
    Proportion of total tokens matched against the LM dictionary.

    Returns *default* when total_tokens == 0.
    Clips result to [0.0, 1.0] as a safety net.
    """
    ratio = safe_divide(float(matched_tokens), float(total_tokens), default=default)
    return float(np.clip(ratio, 0.0, 1.0))


def compute_match_rate(
    matched_count: int,
    vocabulary_size: int,
    *,
    default: float = 0.0,
) -> float:
    """
    Proportion of unique dictionary terms matched in a text.

    match_rate = unique_matched_types / vocabulary_size

    Useful for understanding breadth of dictionary utilisation vs.
    coverage_ratio which measures token-level depth.
    """
    return safe_divide(
        float(matched_count), float(vocabulary_size), default=default
    )


def low_coverage_flag(
    coverage_ratio: float,
    threshold: float = 0.01,
) -> bool:
    """
    Return True if coverage_ratio falls below *threshold*.

    Handles NaN / non-finite inputs by returning True (assume low coverage).
    """
    if not math.isfinite(coverage_ratio):
        return True
    return coverage_ratio < threshold


@dataclass
class CoverageSummary:
    """Aggregate coverage statistics across a batch of processed texts."""

    n_texts: int = 0
    mean_coverage: float = 0.0
    median_coverage: float = 0.0
    std_coverage: float = 0.0
    min_coverage: float = 0.0
    max_coverage: float = 0.0
    low_coverage_count: int = 0
    low_coverage_pct: float = 0.0
    zero_coverage_count: int = 0

    def summary(self) -> str:
        lines = [
            "  ── CoverageSummary ──────────────────────────────",
            f"  Texts              : {self.n_texts:,}",
            f"  Mean coverage      : {self.mean_coverage:.4f}",
            f"  Median coverage    : {self.median_coverage:.4f}",
            f"  Std coverage       : {self.std_coverage:.4f}",
            f"  Min / Max          : {self.min_coverage:.4f} / {self.max_coverage:.4f}",
            f"  Low coverage (<1%) : {self.low_coverage_count} ({self.low_coverage_pct:.1f}%)",
            f"  Zero coverage      : {self.zero_coverage_count}",
            "  ─────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


def summarize_coverage(
    coverage_ratios: Sequence[float],
    threshold: float = 0.01,
) -> CoverageSummary:
    """
    Build a CoverageSummary from a sequence of per-text coverage ratios.
    NaN values are excluded from all statistics.
    """
    arr = np.asarray(coverage_ratios, dtype=float)
    arr = arr[~np.isnan(arr)]
    n   = int(arr.size)

    if n == 0:
        return CoverageSummary(n_texts=0)

    low  = int((arr < threshold).sum())
    zero = int((arr == 0.0).sum())

    return CoverageSummary(
        n_texts             = n,
        mean_coverage       = float(np.mean(arr)),
        median_coverage     = float(np.median(arr)),
        std_coverage        = float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        min_coverage        = float(arr.min()),
        max_coverage        = float(arr.max()),
        low_coverage_count  = low,
        low_coverage_pct    = 100.0 * low / n,
        zero_coverage_count = zero,
    )


# ===========================================================================
# 5. VALIDATION HELPERS
# ===========================================================================


def validate_score_range(
    score: float,
    lo: float = -1.0,
    hi: float = 1.0,
    *,
    name: str = "score",
    raise_on_fail: bool = True,
) -> bool:
    """
    Assert that *score* is a finite float within [lo, hi].

    Parameters
    ----------
    raise_on_fail : if True, raises ValueError; otherwise returns False.
    """
    if not math.isfinite(score):
        msg = f"'{name}' is non-finite: {score}"
        if raise_on_fail:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    if not (lo <= score <= hi):
        msg = f"'{name}' = {score:.6f} is outside [{lo}, {hi}]"
        if raise_on_fail:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    return True


def validate_non_negative(
    value: float,
    *,
    name: str = "value",
    raise_on_fail: bool = True,
) -> bool:
    """Assert that *value* is a non-negative, finite number."""
    if not math.isfinite(value):
        msg = f"'{name}' is non-finite: {value}"
        if raise_on_fail:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    if value < 0:
        msg = f"'{name}' = {value} is negative"
        if raise_on_fail:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    return True


def validate_probability_sum(
    probabilities: Sequence[float],
    *,
    atol: float = 1e-5,
    name: str = "probabilities",
    raise_on_fail: bool = True,
) -> bool:
    """
    Assert that a sequence of probabilities sums to approximately 1.0.

    Tolerates floating-point drift within *atol*.
    """
    total = float(sum(probabilities))
    if not math.isclose(total, 1.0, abs_tol=atol):
        msg = f"'{name}' sums to {total:.8f}, expected 1.0 (atol={atol})"
        if raise_on_fail:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    return True


def detect_duplicates(
    series: pd.Series,
    *,
    column_name: str = "column",
) -> Tuple[bool, int, List[Any]]:
    """
    Check a pandas Series for duplicate values.

    Returns
    -------
    has_duplicates : bool
    duplicate_count : int
    duplicate_values : list of the first 10 duplicated values
    """
    dups = series[series.duplicated(keep=False)]
    count = int(dups.nunique())
    values = dups.unique().tolist()[:10]
    if count:
        logger.warning(
            "detect_duplicates: %d duplicate values in '%s': %s%s",
            count, column_name, values[:5],
            " ..." if len(values) > 5 else "",
        )
    return bool(count > 0), count, values


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    *,
    context: str = "DataFrame",
    raise_on_fail: bool = True,
) -> List[str]:
    """
    Check that all *required* column names are present in *df*.

    Returns list of missing column names (empty list = all present).
    Raises ValueError if raise_on_fail=True and columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"{context} is missing required columns: {missing}"
        if raise_on_fail:
            raise ValueError(msg)
        logger.warning(msg)
    return missing


# ===========================================================================
# 6. DIAGNOSTICS HELPERS
# ===========================================================================


@dataclass
class DistributionSummary:
    """Descriptive statistics for a numeric column."""

    column: str
    count: int = 0
    null_count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p05: float = 0.0
    p95: float = 0.0

    def summary(self) -> str:
        null_pct = 100.0 * self.null_count / max(self.count + self.null_count, 1)
        return (
            f"  {self.column:<30} | "
            f"n={self.count:>7,} null={self.null_count}({null_pct:.1f}%) | "
            f"mean={self.mean:+.4f} "
            f"std={self.std:.4f} "
            f"med={self.median:+.4f} "
            f"[{self.min:+.4f}, {self.max:+.4f}] "
            f"p05={self.p05:+.4f} p95={self.p95:+.4f}"
        )


def summarize_distribution(
    series: pd.Series,
    *,
    column: Optional[str] = None,
) -> DistributionSummary:
    """
    Compute a DistributionSummary for a numeric pandas Series.

    NaN values are excluded from all statistics; null_count records them.
    """
    name    = column or str(series.name or "series")
    null_ct = int(series.isna().sum())
    clean   = series.dropna().astype(float)
    n       = len(clean)

    if n == 0:
        return DistributionSummary(column=name, null_count=null_ct)

    return DistributionSummary(
        column     = name,
        count      = n,
        null_count = null_ct,
        mean       = float(clean.mean()),
        median     = float(clean.median()),
        std        = float(clean.std(ddof=1)) if n > 1 else 0.0,
        min        = float(clean.min()),
        max        = float(clean.max()),
        p25        = float(clean.quantile(0.25)),
        p75        = float(clean.quantile(0.75)),
        p05        = float(clean.quantile(0.05)),
        p95        = float(clean.quantile(0.95)),
    )


def summarize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame of null counts and percentages per column.

    Columns: column | null_count | null_pct | dtype
    Sorted descending by null_count.
    """
    null_counts = df.isna().sum()
    null_pcts   = 100.0 * null_counts / max(len(df), 1)
    summary = pd.DataFrame({
        "column"     : null_counts.index,
        "null_count" : null_counts.values,
        "null_pct"   : null_pcts.values.round(2),
        "dtype"      : [str(df[c].dtype) for c in null_counts.index],
    })
    return summary.sort_values("null_count", ascending=False).reset_index(drop=True)


def summarize_labels(
    series: pd.Series,
    *,
    expected_labels: Optional[Sequence[str]] = None,
    column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute label distribution statistics for a categorical series.

    Returns
    -------
    dict with keys: counts, percentages, most_common, unexpected_labels
    """
    name   = column or str(series.name or "labels")
    counts = series.value_counts(dropna=False).to_dict()
    total  = sum(counts.values())
    pcts   = {
        k: round(100.0 * v / total, 2) if total else 0.0
        for k, v in counts.items()
    }
    most_common = max(counts, key=counts.get) if counts else None  # type: ignore[arg-type]

    unexpected: List[str] = []
    if expected_labels is not None:
        exp_set    = set(expected_labels)
        unexpected = [str(k) for k in counts if str(k) not in exp_set]
        if unexpected:
            logger.warning(
                "summarize_labels('%s'): unexpected labels found: %s",
                name, unexpected,
            )

    return {
        "column"            : name,
        "counts"            : counts,
        "percentages"       : pcts,
        "most_common"       : most_common,
        "unexpected_labels" : unexpected,
        "total"             : total,
    }


def summarize_category_counts(
    data: Dict[str, int],
    *,
    total_tokens: int = 0,
    label: str = "category",
) -> str:
    """
    Format a dictionary of category → count into a human-readable summary
    line for logging and diagnostic reports.
    """
    parts = []
    for cat, cnt in sorted(data.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total_tokens if total_tokens > 0 else 0.0
        parts.append(f"{cat}={cnt}({pct:.1f}%)")
    return f"[{label}] " + "  ".join(parts)


# ===========================================================================
# 7. EXPORT HELPERS
# ===========================================================================


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Create *path* (and all parents) if it does not exist.

    Returns the resolved Path object.
    Does not raise if the directory already exists.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    logger.debug("Directory ensured: %s", p)
    return p


def safe_parquet_export(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    compression: str = "snappy",
    overwrite: bool = True,
    log_memory: bool = False,
) -> bool:
    """
    Write a DataFrame to a parquet file with error handling.

    Parameters
    ----------
    df          : DataFrame to export
    path        : destination file path (parent dirs created automatically)
    compression : pyarrow compression codec (snappy | gzip | brotli | none)
    overwrite   : if False and file exists, skip writing and return True
    log_memory  : if True, log DataFrame memory usage before writing

    Returns
    -------
    True on success, False on any failure.
    """
    path = Path(path)

    if df.empty:
        logger.warning("safe_parquet_export: empty DataFrame, skipping %s", path)
        return False

    if not overwrite and path.exists():
        logger.info("safe_parquet_export: file exists, skipping %s", path)
        return True

    if log_memory:
        mb = dataframe_memory_usage(df)
        logger.debug("safe_parquet_export: DataFrame %.2f MB → %s", mb, path)

    ensure_directory(path.parent)

    try:
        if _PYARROW_AVAILABLE:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, str(path), compression=compression)
        else:
            df.to_parquet(str(path), compression=compression, index=False)

        logger.info(
            "Exported parquet: %s (%d rows, %d cols)",
            path, len(df), len(df.columns),
        )
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("safe_parquet_export failed for %s: %s", path, exc)
        return False


def safe_csv_export(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    overwrite: bool = True,
    encoding: str = "utf-8",
    float_format: str = "%.6f",
) -> bool:
    """
    Write a DataFrame to a CSV file with error handling.

    Parameters
    ----------
    overwrite : if False and file exists, skip writing and return True

    Returns
    -------
    True on success, False on any failure.
    """
    path = Path(path)

    if df.empty:
        logger.warning("safe_csv_export: empty DataFrame, skipping %s", path)
        return False

    if not overwrite and path.exists():
        logger.info("safe_csv_export: file exists, skipping %s", path)
        return True

    ensure_directory(path.parent)

    try:
        df.to_csv(
            str(path),
            index=False,
            encoding=encoding,
            float_format=float_format,
        )
        logger.info(
            "Exported CSV: %s (%d rows, %d cols)", path, len(df), len(df.columns)
        )
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("safe_csv_export failed for %s: %s", path, exc)
        return False


def export_parquet_and_csv(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    stem: str,
    *,
    overwrite: bool = True,
    compression: str = "snappy",
) -> Dict[str, Path]:
    """
    Convenience wrapper: export as both parquet and CSV in one call.

    Returns
    -------
    dict with keys "parquet" and "csv" mapping to exported Path objects.
    Only includes a key if the export succeeded.
    """
    output_dir = Path(output_dir)
    exported: Dict[str, Path] = {}

    pq_path  = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"

    if safe_parquet_export(df, pq_path, compression=compression, overwrite=overwrite):
        exported["parquet"] = pq_path

    if safe_csv_export(df, csv_path, overwrite=overwrite):
        exported["csv"] = csv_path

    return exported


# ===========================================================================
# 8. TIMING / LOGGING HELPERS
# ===========================================================================


def timed_execution(
    func: Optional[F] = None,
    *,
    log_level: int = logging.INFO,
    label: Optional[str] = None,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator that logs the wall-clock execution time of the wrapped function.

    Can be used with or without arguments:

        @timed_execution
        def my_func(): ...

        @timed_execution(log_level=logging.DEBUG, label="custom label")
        def my_func(): ...

    The elapsed time and function name are emitted via the module logger.
    The original return value is passed through unchanged.
    """
    def decorator(fn: F) -> F:
        fn_label = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                logger.log(
                    log_level,
                    "[timed] %s completed in %.3fs", fn_label, elapsed,
                )
                return result
            except Exception:
                elapsed = time.perf_counter() - t0
                logger.log(
                    logging.ERROR,
                    "[timed] %s raised after %.3fs", fn_label, elapsed,
                )
                raise

        return wrapper  # type: ignore[return-value]

    # Called as @timed_execution (no parentheses)
    if func is not None:
        return decorator(func)

    # Called as @timed_execution(...) (with parentheses)
    return decorator  # type: ignore[return-value]


def dataframe_memory_usage(df: pd.DataFrame, *, deep: bool = True) -> float:
    """
    Return the total memory usage of a DataFrame in megabytes.

    Parameters
    ----------
    deep : if True, introspects object columns for their actual size
           (slower but accurate for string-heavy DataFrames)
    """
    if df.empty:
        return 0.0
    bytes_used = df.memory_usage(deep=deep).sum()
    return bytes_used / (1024 ** 2)


def processing_rate(
    n_items: int,
    elapsed_seconds: float,
    *,
    unit: str = "items",
) -> str:
    """
    Format a human-readable throughput string.

    Example output: "1,234 items/s"
    """
    if elapsed_seconds <= 0 or n_items <= 0:
        return f"0 {unit}/s"
    rate = n_items / elapsed_seconds
    if rate >= 1_000:
        return f"{rate:,.0f} {unit}/s"
    return f"{rate:.1f} {unit}/s"


@dataclass
class TimingRecord:
    """Lightweight record of a named operation's execution time."""

    label: str
    elapsed_seconds: float
    items_processed: int = 0
    unit: str = "items"

    def rate_str(self) -> str:
        return processing_rate(self.items_processed, self.elapsed_seconds,
                               unit=self.unit)

    def summary_line(self) -> str:
        return (
            f"  {self.label:<35} | "
            f"{self.elapsed_seconds:>7.3f}s | "
            f"{self.items_processed:>8,} {self.unit} | "
            f"{self.rate_str()}"
        )


class StageTimer:
    """
    Simple multi-stage timer for pipeline stage profiling.

    Usage
    -----
    >>> timer = StageTimer()
    >>> with timer.measure("load"):
    ...     load_data()
    >>> with timer.measure("score"):
    ...     score_chunks()
    >>> print(timer.summary())
    """

    def __init__(self) -> None:
        self._records: List[TimingRecord] = []
        self._current_label: Optional[str] = None
        self._t0: float = 0.0

    class _Context:
        def __init__(self, timer: "StageTimer", label: str) -> None:
            self._timer = timer
            self._label = label

        def __enter__(self) -> None:
            self._timer._current_label = self._label
            self._timer._t0 = time.perf_counter()

        def __exit__(self, *_: Any) -> None:
            elapsed = time.perf_counter() - self._timer._t0
            self._timer._records.append(
                TimingRecord(label=self._label, elapsed_seconds=elapsed)
            )
            self._timer._current_label = None

    def measure(self, label: str) -> "_Context":
        return self._Context(self, label)

    def record(self, label: str, elapsed: float, items: int = 0) -> None:
        self._records.append(
            TimingRecord(label=label, elapsed_seconds=elapsed, items_processed=items)
        )

    def total_elapsed(self) -> float:
        return sum(r.elapsed_seconds for r in self._records)

    def summary(self) -> str:
        if not self._records:
            return "StageTimer: no records."
        lines = [
            "── StageTimer ────────────────────────────────────────────",
        ]
        for r in self._records:
            lines.append(r.summary_line())
        lines += [
            f"  {'TOTAL':<35} | {self.total_elapsed():>7.3f}s",
            "──────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ===========================================================================
# DEMO / SELF-TEST
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    import tempfile

    print("\n" + "=" * 65)
    print("  lm_utils.py — Demo Run")
    print("=" * 65)

    # ── 1. Text normalisation ─────────────────────────────────────────
    print("\n[1] Text normalisation")
    messy = "  Revenue\tgrew  12%\nto\r$4.5B…\x00\x1f  "
    print(f"  Input  : {repr(messy)}")
    print(f"  Output : {repr(full_text_clean(messy))}")

    print(f"  normalize_unicode : {repr(normalize_unicode('café'))}")
    print(f"  remove_ctrl_chars : {repr(remove_control_chars('hello\x00world'))}")
    print(f"  normalize_ws      : {repr(normalize_whitespace('  foo   bar  '))}")
    print(f"  safe_lower (None) : {repr(safe_lower(None))}")   # type: ignore[arg-type]
    print(f"  clean_token       : {repr(clean_token('...growth!'))}")

    # ── 2. Token utilities ────────────────────────────────────────────
    print("\n[2] Token utilities")
    tokens = [
        "revenue", "grew", "12%", "record", "growth", "revenue",
        "strong", "a", "the", "AAPL", "growth", "ebitda", "x",
    ]
    print(f"  validate_token('a')       : {validate_token('a')}")
    print(f"  validate_token('revenue') : {validate_token('revenue')}")
    filtered = filter_tokens(tokens, stopwords={"a", "the"}, min_length=2)
    print(f"  filter_tokens             : {filtered}")
    freq = token_frequency(tokens)
    print(f"  token_frequency (top3)    : {freq.most_common(3)}")
    print(f"  unique_token_ratio        : {unique_token_ratio(tokens):.3f}")
    print(f"  top_n_tokens (n=3)        : {top_n_tokens(tokens, n=3)}")

    # ── 3. Numeric utilities ──────────────────────────────────────────
    print("\n[3] Numeric utilities")
    print(f"  safe_divide(10, 4)        : {safe_divide(10, 4):.4f}")
    print(f"  safe_divide(10, 0)        : {safe_divide(10, 0, default=-1)}")
    print(f"  safe_divide(inf, 4)       : {safe_divide(float('inf'), 4)}")
    print(f"  safe_mean([1,2,NaN,4])    : {safe_mean([1.0, 2.0, float('nan'), 4.0]):.4f}")
    print(f"  safe_mean([])             : {safe_mean([])}")
    print(f"  safe_std([1,2,3])         : {safe_std([1.0, 2.0, 3.0]):.4f}")
    print(f"  clip_score_range(1.5)     : {clip_score_range(1.5)}")
    print(f"  clip_score_range(NaN)     : {clip_score_range(float('nan'))}")
    probs = normalize_probability([3.0, 1.0, 0.0, 2.0])
    print(f"  normalize_probability     : {[round(p, 4) for p in probs]}")
    print(f"  weighted_mean             : {weighted_mean([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]):.4f}")

    # ── 4. Coverage utilities ─────────────────────────────────────────
    print("\n[4] Coverage utilities")
    print(f"  compute_coverage_ratio(10,200) : {compute_coverage_ratio(10, 200):.4f}")
    print(f"  compute_coverage_ratio(0,0)    : {compute_coverage_ratio(0, 0)}")
    print(f"  low_coverage_flag(0.005)       : {low_coverage_flag(0.005)}")
    print(f"  low_coverage_flag(0.05)        : {low_coverage_flag(0.05)}")

    ratios = [0.0, 0.005, 0.02, 0.04, 0.08, 0.12, 0.15, float("nan")]
    cov_summary = summarize_coverage(ratios)
    print("\n" + cov_summary.summary())

    # ── 5. Validation helpers ─────────────────────────────────────────
    print("\n[5] Validation helpers")
    print(f"  validate_score_range(0.5)   : {validate_score_range(0.5, raise_on_fail=False)}")
    print(f"  validate_score_range(1.5)   : {validate_score_range(1.5, raise_on_fail=False)}")
    print(f"  validate_non_negative(3.0)  : {validate_non_negative(3.0, raise_on_fail=False)}")
    print(f"  validate_non_negative(-1.0) : {validate_non_negative(-1.0, raise_on_fail=False)}")
    print(
        f"  validate_probability_sum    : "
        f"{validate_probability_sum([0.6, 0.3, 0.1], raise_on_fail=False)}"
    )
    print(
        f"  validate_probability_sum    : "
        f"{validate_probability_sum([0.6, 0.3, 0.5], raise_on_fail=False)}"
    )

    df_dup = pd.DataFrame({
        "transcript_id": ["A", "B", "A", "C", "B"],
        "score":         [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    has_dup, dup_count, dup_vals = detect_duplicates(
        df_dup["transcript_id"], column_name="transcript_id"
    )
    print(f"  detect_duplicates           : has={has_dup} count={dup_count} vals={dup_vals}")

    missing = validate_required_columns(
        df_dup, ["transcript_id", "score", "lm_tone_score"], raise_on_fail=False
    )
    print(f"  validate_required_columns   : missing={missing}")

    # ── 6. Diagnostics helpers ────────────────────────────────────────
    print("\n[6] Diagnostics helpers")
    scores = pd.Series(
        [0.32, -0.15, 0.08, -0.42, 0.61, float("nan"), 0.19, -0.07, 0.45, -0.23],
        name="lm_tone_score",
    )
    dist = summarize_distribution(scores)
    print(dist.summary())

    labels_series = pd.Series(
        ["positive", "neutral", "negative", "positive", "positive",
         "unknown", "neutral", "negative", "positive"]
    )
    label_info = summarize_labels(
        labels_series,
        expected_labels=["positive", "neutral", "negative"],
        column="lm_label",
    )
    print(f"  label counts    : {label_info['counts']}")
    print(f"  label pcts      : {label_info['percentages']}")
    print(f"  most common     : {label_info['most_common']}")
    print(f"  unexpected      : {label_info['unexpected_labels']}")

    df_nulls = pd.DataFrame({
        "a": [1, None, 3],
        "b": [None, None, 6],
        "c": [7, 8, 9],
    })
    print("\n  summarize_nulls:")
    print(summarize_nulls(df_nulls).to_string(index=False))

    cat_counts = {"positive": 28, "negative": 7, "uncertainty": 4}
    print("\n  " + summarize_category_counts(cat_counts, total_tokens=150))

    # ── 7. Export helpers ─────────────────────────────────────────────
    print("\n[7] Export helpers")
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "sentiment" / "test_export"

        sample_df = pd.DataFrame({
            "transcript_id" : ["AAPL_Q1", "MSFT_Q2"],
            "lm_tone_score" : [0.42, -0.18],
            "lm_label"      : ["positive", "negative"],
            "lm_coverage"   : [0.08, 0.05],
        })

        exported = export_parquet_and_csv(sample_df, out.parent, out.name)
        for fmt, path in exported.items():
            print(f"  Exported {fmt:<8}: {path}  (exists={path.exists()})")

        # overwrite=False test
        ok = safe_parquet_export(
            sample_df, out.parent / f"{out.name}.parquet", overwrite=False
        )
        print(f"  overwrite=False skip : {ok}")

    # ── 8. Timing / logging helpers ───────────────────────────────────
    print("\n[8] Timing / logging helpers")

    @timed_execution(log_level=logging.DEBUG, label="synthetic_work")
    def fake_workload(n: int) -> int:
        time.sleep(0.05)
        return sum(range(n))

    result_val = fake_workload(100_000)
    print(f"  timed_execution result : {result_val:,}")

    timer = StageTimer()
    with timer.measure("preprocess"):
        time.sleep(0.02)
    with timer.measure("match"):
        time.sleep(0.03)
    with timer.measure("score"):
        time.sleep(0.01)
    timer.record("export", 0.008, items=1024)
    print("\n" + timer.summary())

    print(f"  processing_rate(1024, 0.05) : {processing_rate(1024, 0.05, unit='chunks')}")
    print(f"  dataframe_memory_usage      : {dataframe_memory_usage(df_nulls):.4f} MB")

    print("\nDemo complete.\n")
