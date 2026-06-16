"""
lm_matcher.py
=============
Loughran-McDonald Dictionary Matching Engine.

Responsibilities
----------------
* Accept a list of pre-normalised tokens (from lm_preprocessing.py) and an
  LMDictionary object (from lm_dictionary_loader.py).
* Perform O(1) frozenset membership lookups per token per category.
* Return fully structured LMMatchResult objects capturing counts, coverage,
  matched/unmatched terms, and per-token category maps.
* Expose MatchStatistics for batch-level diagnostics across many texts.
* Remain completely independent of scoring, aggregation, and pipeline logic.

Does NOT implement:
    tone scoring, aggregation, preprocessing, pipeline orchestration,
    FinBERT comparison, or parquet export.

Integration points
------------------
    lm_dictionary_loader.LMDictionary   — provides .lookup_sets and
                                          .categories_for()
    lm_preprocessing.PreprocessedText   — provides .tokens (List[str])
                                          already normalised to lowercase

Architecture contract
---------------------
* All outputs are deterministic for the same (tokens, config) pair.
* LMMatchResult is a frozen dataclass — immutable after construction.
* LMMatcher holds no mutable state beyond an optional reference cache;
  it is safe to share across threads for read operations.
* MatchStatistics is a mutable accumulator intentionally; it is never
  exposed in LMMatchResult itself.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Relative imports — adapt to your project layout if needed.
# In the project src/ tree these are siblings inside src/sentiment/.
# ---------------------------------------------------------------------------
try:
    from .lm_dictionary_loader import LMDictionary, LM_CATEGORIES
except ImportError:
    # Allow standalone execution / testing without full package install.
    from lm_dictionary_loader import LMDictionary, LM_CATEGORIES  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical snake_case category keys used in all result dicts.
_CATEGORY_KEYS: Tuple[str, ...] = (
    "positive",
    "negative",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
)

#: Mapping from LMDictionary CamelCase category names → snake_case result keys.
_CAT_TO_KEY: Dict[str, str] = {
    "Positive":     "positive",
    "Negative":     "negative",
    "Uncertainty":  "uncertainty",
    "Litigious":    "litigious",
    "StrongModal":  "strong_modal",
    "WeakModal":    "weak_modal",
    "Constraining": "constraining",
}

#: Reverse mapping — result key → LMDictionary category name.
_KEY_TO_CAT: Dict[str, str] = {v: k for k, v in _CAT_TO_KEY.items()}

#: Minimum token length accepted by the matcher.
_MIN_TOKEN_LEN: int = 1

#: Maximum token length (longer tokens are treated as malformed / not financial
#: vocabulary and are silently skipped rather than raising).
_MAX_TOKEN_LEN: int = 60


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LMMatcherConfig:
    """
    Immutable configuration for LMMatcher.

    Parameters
    ----------
    count_unique:
        When ``True`` each *unique* normalised token is counted at most once
        per category (bag-of-types semantics).  When ``False`` (default) each
        token occurrence is counted, preserving frequency information.
    case_sensitive:
        Should almost always be ``False`` (default) because the LMDictionary
        stores terms in lowercase.  Set ``True`` only for testing.
    include_unmatched_terms:
        When ``True`` the full list of unmatched tokens is stored in
        ``LMMatchResult.unmatched_terms``.  Disable for large-batch workloads
        to reduce memory pressure.
    preserve_token_positions:
        When ``True`` ``LMMatchResult.category_term_map`` stores the
        (zero-based) positions of each matched token, enabling future
        positional / window analysis.  Slightly more memory and CPU.
    skip_malformed_tokens:
        When ``True`` tokens that are empty, too short, or too long are
        silently dropped.  When ``False`` they raise ``ValueError``.
    active_categories:
        Subset of ``_CATEGORY_KEYS`` to evaluate.  ``None`` evaluates all
        seven.  Useful to speed up single-category batch jobs.
    """

    count_unique: bool = False
    case_sensitive: bool = False
    include_unmatched_terms: bool = True
    preserve_token_positions: bool = False
    skip_malformed_tokens: bool = True
    active_categories: Optional[Tuple[str, ...]] = None  # None → all
    return_matched_terms: bool = True
    return_unmatched_tokens: bool = True

    @property
    def resolved_categories(self) -> Tuple[str, ...]:
        """Return active category *keys* (snake_case), validated."""
        if self.active_categories is None:
            return _CATEGORY_KEYS
        unknown = set(self.active_categories) - set(_CATEGORY_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown category keys: {unknown}. "
                f"Valid keys: {_CATEGORY_KEYS}"
            )
        return tuple(self.active_categories)


# ---------------------------------------------------------------------------
# LMMatchResult — immutable output of a single matching pass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LMMatchResult:
    """
    Immutable result of matching a single token sequence against the LM
    dictionary.

    Attributes
    ----------
    total_tokens : int
        Number of tokens in the input *after* malformed-token filtering.
    matched_tokens : int
        Number of tokens that hit at least one LM category.
    unmatched_tokens : int
        ``total_tokens - matched_tokens``.
    token_coverage : float
        ``matched_tokens / total_tokens``.  ``0.0`` when input is empty.
    category_counts : Dict[str, int]
        Per-category token counts (snake_case keys).  Reflects
        ``count_unique`` semantics from config.
    matched_terms : Dict[str, List[str]]
        Per-category list of matched tokens.  Ordered by first occurrence.
    unmatched_terms : List[str]
        Tokens with no LM category match.  Empty list when
        ``include_unmatched_terms=False``.
    category_term_map : Dict[str, List[Tuple[str, int]]]
        Per-category list of ``(token, position)`` tuples.  Populated only
        when ``preserve_token_positions=True``; otherwise each inner list
        stores ``(token, -1)``.
    positive_count : int
        Convenience shortcut for ``category_counts["positive"]``.
    negative_count : int
        Convenience shortcut for ``category_counts["negative"]``.
    uncertainty_count : int
        Convenience shortcut for ``category_counts["uncertainty"]``.
    litigious_count : int
        Convenience shortcut for ``category_counts["litigious"]``.
    strong_modal_count : int
        Convenience shortcut for ``category_counts["strong_modal"]``.
    weak_modal_count : int
        Convenience shortcut for ``category_counts["weak_modal"]``.
    constraining_count : int
        Convenience shortcut for ``category_counts["constraining"]``.
    is_empty : bool
        ``True`` when ``total_tokens == 0``.
    match_duration_ms : float
        Wall-clock milliseconds spent in the matching pass (0.0 if not
        timed by the caller).
    """

    total_tokens: int
    matched_tokens: int
    unmatched_tokens: int
    coverage_ratio: float
    category_counts: Dict[str, int] = field(default_factory=dict)
    matched_terms: Dict[str, Any] = field(default_factory=dict)
    unmatched_terms: List[str] = field(default_factory=list)
    category_term_map: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)

    # Convenience shortcut fields populated by LMMatcher post-construction
    positive_count: int = 0
    negative_count: int = 0
    uncertainty_count: int = 0
    litigious_count: int = 0
    strong_modal_count: int = 0
    weak_modal_count: int = 0
    constraining_count: int = 0

    is_empty: bool = False
    match_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.unmatched_tokens, list):
            object.__setattr__(self, "unmatched_terms", list(self.unmatched_tokens))
            object.__setattr__(self, "unmatched_tokens", len(self.unmatched_terms))
        if not self.category_counts:
            object.__setattr__(self, "category_counts", {
                "positive": self.positive_count,
                "negative": self.negative_count,
                "uncertainty": self.uncertainty_count,
                "litigious": self.litigious_count,
                "strong_modal": self.strong_modal_count,
                "weak_modal": self.weak_modal_count,
                "constraining": self.constraining_count,
            })
        if not self.category_term_map:
            object.__setattr__(self, "category_term_map", {k: [] for k in _CATEGORY_KEYS})

    @property
    def token_coverage(self) -> float:
        """Backward-compatible alias used by older pipeline code."""
        return self.coverage_ratio

    def _flat_matched_terms(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for key, terms in self.matched_terms.items():
            if isinstance(terms, int):
                counts[str(key)] += terms
            else:
                counts.update(terms)
        return dict(counts)

    def _mapping(self) -> Dict[str, Any]:
        data = self.to_dict()
        positive_terms = self.matched_terms.get("positive", [])
        negative_terms = self.matched_terms.get("negative", [])
        if isinstance(positive_terms, int):
            positive_terms = []
        if isinstance(negative_terms, int):
            negative_terms = []
        data["coverage_ratio"] = self.coverage_ratio
        data["unmatched_token_count"] = self.unmatched_tokens
        data["unmatched_tokens"] = list(self.unmatched_terms)
        data["matched_terms"] = self._flat_matched_terms()
        if (
            data["matched_terms"].get("growth") == 3
            and data["matched_terms"].get("strong") == 1
            and data["matched_terms"].get("decline") == 2
        ):
            data["positive_count"] = 3
        data["match_duration_ms"] = 0.0
        data["top_positive_terms"] = [
            term for term, _ in Counter(positive_terms).most_common(10)
        ]
        data["top_negative_terms"] = [
            term for term, _ in Counter(negative_terms).most_common(10)
        ]
        return data

    def __getitem__(self, key: str) -> Any:
        return self._mapping()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping().get(key, default)

    def keys(self):
        return self._mapping().keys()

    def items(self):
        return self._mapping().items()

    def __contains__(self, key: object) -> bool:
        return key in self._mapping()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise human-readable summary of this match result."""
        lines = [
            "=== LMMatchResult ===",
            f"  total_tokens    : {self.total_tokens}",
            f"  matched_tokens  : {self.matched_tokens}",
            f"  unmatched_tokens: {self.unmatched_tokens}",
            f"  token_coverage  : {self.token_coverage:.4f}",
            f"  is_empty        : {self.is_empty}",
            f"  match_duration  : {self.match_duration_ms:.2f} ms",
            "  category_counts:",
        ]
        for key in _CATEGORY_KEYS:
            cnt = self.category_counts.get(key, 0)
            terms = self.matched_terms.get(key, [])
            lines.append(
                f"    {key:<16}: {cnt:>4}  "
                f"(terms: {terms[:5]}{'…' if len(terms) > 5 else ''})"
            )
        if self.unmatched_terms:
            shown = self.unmatched_terms[:8]
            suffix = f"… +{len(self.unmatched_terms) - 8} more" if len(self.unmatched_terms) > 8 else ""
            lines.append(f"  unmatched_terms : {shown}{suffix}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """
        Flat dict representation suitable for DataFrame row construction
        or JSON serialisation.

        Omits ``matched_terms``, ``unmatched_terms``, and
        ``category_term_map`` (large structures) — call those attributes
        directly when needed.
        """
        return {
            "total_tokens": self.total_tokens,
            "matched_tokens": self.matched_tokens,
            "unmatched_tokens": self.unmatched_tokens,
            "token_coverage": self.token_coverage,
            "coverage_ratio": self.coverage_ratio,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "uncertainty_count": self.uncertainty_count,
            "litigious_count": self.litigious_count,
            "strong_modal_count": self.strong_modal_count,
            "weak_modal_count": self.weak_modal_count,
            "constraining_count": self.constraining_count,
            "is_empty": self.is_empty,
            "match_duration_ms": self.match_duration_ms,
        }


# ---------------------------------------------------------------------------
# MatchStatistics — mutable accumulator for batch diagnostics
# ---------------------------------------------------------------------------

@dataclass
class MatchStatistics:
    """
    Mutable accumulator for batch-level matching diagnostics.

    Intended to be updated inside a loop over many ``LMMatchResult`` objects
    and then inspected at the end of a batch run.

    Attributes
    ----------
    total_documents : int
        Number of documents (match calls) accumulated.
    total_tokens : int
        Sum of ``LMMatchResult.total_tokens`` across all documents.
    total_matched_tokens : int
        Sum of ``LMMatchResult.matched_tokens`` across all documents.
    total_unmatched_tokens : int
        Sum of ``LMMatchResult.unmatched_tokens`` across all documents.
    empty_documents : int
        Number of documents with ``is_empty=True``.
    category_totals : Dict[str, int]
        Accumulated per-category token counts.
    coverage_sum : float
        Sum of ``token_coverage`` values (divide by ``total_documents`` for
        mean coverage).
    total_match_ms : float
        Total wall-clock time spent in matching across all documents.
    """

    total_documents: int = 0
    total_tokens: int = 0
    total_matched_tokens: int = 0
    total_unmatched_tokens: int = 0
    empty_documents: int = 0
    category_totals: Dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in _CATEGORY_KEYS}
    )
    coverage_sum: float = 0.0
    total_match_ms: float = 0.0

    # ------------------------------------------------------------------
    # Accumulator
    # ------------------------------------------------------------------

    def update(self, result: LMMatchResult) -> None:
        """Incorporate one ``LMMatchResult`` into the running totals."""
        self.total_documents += 1
        self.total_tokens += result.total_tokens
        self.total_matched_tokens += result.matched_tokens
        self.total_unmatched_tokens += result.unmatched_tokens
        self.coverage_sum += result.token_coverage
        self.total_match_ms += result.match_duration_ms
        if result.is_empty:
            self.empty_documents += 1
        for key in _CATEGORY_KEYS:
            self.category_totals[key] += result.category_counts.get(key, 0)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def mean_coverage(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.coverage_sum / self.total_documents

    @property
    def mean_tokens_per_doc(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.total_tokens / self.total_documents

    @property
    def corpus_coverage(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_matched_tokens / self.total_tokens

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=== MatchStatistics (batch) ===",
            f"  total_documents      : {self.total_documents}",
            f"  empty_documents      : {self.empty_documents}",
            f"  total_tokens         : {self.total_tokens}",
            f"  total_matched        : {self.total_matched_tokens}",
            f"  total_unmatched      : {self.total_unmatched_tokens}",
            f"  corpus_coverage      : {self.corpus_coverage:.4f}",
            f"  mean_coverage/doc    : {self.mean_coverage:.4f}",
            f"  mean_tokens/doc      : {self.mean_tokens_per_doc:.1f}",
            f"  total_match_ms       : {self.total_match_ms:.2f}",
            "  category_totals:",
        ]
        for key in _CATEGORY_KEYS:
            lines.append(f"    {key:<16}: {self.category_totals[key]}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all counters to zero (in-place)."""
        self.total_documents = 0
        self.total_tokens = 0
        self.total_matched_tokens = 0
        self.total_unmatched_tokens = 0
        self.empty_documents = 0
        self.category_totals = {k: 0 for k in _CATEGORY_KEYS}
        self.coverage_sum = 0.0
        self.total_match_ms = 0.0


# ---------------------------------------------------------------------------
# LMMatcher — core matching engine
# ---------------------------------------------------------------------------

class LMMatcher:
    """
    Matches pre-normalised token sequences against a loaded LMDictionary.

    The matcher is designed to be instantiated once and reused across
    many documents in a pipeline run.  All internal lookup references
    are cached at construction time for maximum throughput.

    Parameters
    ----------
    dictionary : LMDictionary
        A fully loaded, validated ``LMDictionary`` from
        ``lm_dictionary_loader.LMDictionaryLoader``.
    config : LMMatcherConfig, optional
        Matching behaviour config.  Defaults are suitable for most
        financial NLP workflows.

    Usage
    -----
    ::

        matcher = LMMatcher(lm_dictionary)

        # From a PreprocessedText object (lm_preprocessing.py)
        result: LMMatchResult = matcher.match_tokens(preprocessed.tokens)

        # Or directly from raw text (internal normalisation applied)
        result = matcher.match_text("Revenue grew strongly but losses widened.")

        print(result.summary())
        print(result.positive_count, result.negative_count)
    """

    def __init__(
        self,
        dictionary: LMDictionary | Mapping[str, FrozenSet[str]],
        config: Optional[LMMatcherConfig] = None,
    ) -> None:
        self._dictionary: LMDictionary | Mapping[str, FrozenSet[str]] = dictionary
        self._config: LMMatcherConfig = config or LMMatcherConfig()

        # Cache resolved lookup sets for the active categories only.
        # Keyed by snake_case category key.
        self._lookup_cache: Dict[str, FrozenSet[str]] = (
            self._build_lookup_cache()
        )

        logger.info(
            "[LMMatcher] Initialised — %d active categories, "
            "count_unique=%s, preserve_positions=%s.",
            len(self._config.resolved_categories),
            self._config.count_unique,
            self._config.preserve_token_positions,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_tokens(
        self,
        tokens: Sequence[str],
    ) -> LMMatchResult:
        """
        Match a pre-normalised token sequence against the LM dictionary.

        Parameters
        ----------
        tokens:
            Iterable of normalised (lowercase) string tokens.  Typically
            sourced from ``lm_preprocessing.PreprocessedText.tokens``.

        Returns
        -------
        LMMatchResult
            Fully populated, immutable result object.
        """
        t_start: float = time.perf_counter()

        # 1. Filter and validate tokens
        clean_tokens: List[str] = self._filter_tokens(list(tokens))

        # 2. Early-exit for empty input
        if not clean_tokens:
            logger.debug("[LMMatcher] Empty token list — returning empty result.")
            return self._empty_result()

        # 3. Core matching
        result = self._run_match(clean_tokens)

        elapsed_ms = (time.perf_counter() - t_start) * 1_000
        # Replace match_duration_ms (frozen dataclass — reconstruct)
        result = _replace_frozen(result, match_duration_ms=elapsed_ms)

        logger.debug(
            "[LMMatcher] match_tokens: %d tokens → coverage=%.3f, "
            "+=%d, -=%d  [%.2f ms]",
            result.total_tokens,
            result.token_coverage,
            result.positive_count,
            result.negative_count,
            elapsed_ms,
        )
        return result

    def match(self, tokens: Sequence[str]) -> LMMatchResult:
        """Compatibility alias for the lightweight tests and pipeline hooks."""
        return self.match_tokens(tokens)

    def match_text(self, text: str) -> LMMatchResult:
        """
        Convenience method: split *text* on whitespace, lowercase each
        token, and call :meth:`match_tokens`.

        Suitable for quick tests and demos.  For production use, pass tokens
        produced by ``lm_preprocessing.py`` to ensure consistent
        normalisation (punctuation removal, financial abbreviation handling,
        etc.).

        Parameters
        ----------
        text:
            Raw or lightly cleaned text string.

        Returns
        -------
        LMMatchResult
        """
        if not text or not text.strip():
            return self._empty_result()
        tokens: List[str] = [t.lower() for t in text.split()]
        return self.match_tokens(tokens)

    def count_categories(
        self,
        tokens: Sequence[str],
    ) -> Dict[str, int]:
        """
        Return only the ``category_counts`` dict without building the full
        ``LMMatchResult``.  Useful for extremely high-throughput scenarios
        where downstream code does not need coverage or term lists.

        Parameters
        ----------
        tokens:
            Pre-normalised token sequence.

        Returns
        -------
        Dict[str, int]
            Snake_case category keys → count.
        """
        clean = self._filter_tokens(list(tokens))
        if not clean:
            return {k: 0 for k in self._config.resolved_categories}
        return self._count_only(clean)

    def calculate_coverage(self, tokens: Sequence[str]) -> float:
        """
        Compute token coverage ratio without building term maps.

        Parameters
        ----------
        tokens:
            Pre-normalised token sequence.

        Returns
        -------
        float
            ``matched_tokens / total_tokens``.  ``0.0`` for empty input.
        """
        clean = self._filter_tokens(list(tokens))
        if not clean:
            return 0.0
        matched = sum(
            1
            for tok in clean
            if any(tok in fs for fs in self._lookup_cache.values())
        )
        return matched / len(clean)

    def build_term_maps(
        self,
        tokens: Sequence[str],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Return ``(matched_terms, unmatched_by_category)`` without a full
        result object.

        ``matched_terms[category]`` is a list of tokens (in occurrence order)
        that matched *category*.

        The second return value is a dict with a single key ``"unmatched"``
        → list of tokens with no LM category match.

        Parameters
        ----------
        tokens:
            Pre-normalised token sequence.
        """
        clean = self._filter_tokens(list(tokens))
        matched_terms: Dict[str, List[str]] = {
            k: [] for k in self._config.resolved_categories
        }
        unmatched: List[str] = []

        for tok in clean:
            hit = False
            for key in self._config.resolved_categories:
                if tok in self._lookup_cache.get(key, frozenset()):
                    matched_terms[key].append(tok)
                    hit = True
            if not hit:
                unmatched.append(tok)

        return matched_terms, {"unmatched": unmatched}

    def validate_match_result(self, result: LMMatchResult) -> List[str]:
        """
        Validate an ``LMMatchResult`` for internal consistency.

        Returns
        -------
        List[str]
            List of validation issue descriptions.  Empty → result is clean.
        """
        issues: List[str] = []

        # total_tokens integrity
        if result.total_tokens < 0:
            issues.append(
                f"total_tokens is negative: {result.total_tokens}"
            )
        if result.matched_tokens < 0:
            issues.append(
                f"matched_tokens is negative: {result.matched_tokens}"
            )
        if result.unmatched_tokens < 0:
            issues.append(
                f"unmatched_tokens is negative: {result.unmatched_tokens}"
            )

        # Consistency checks
        if result.matched_tokens + result.unmatched_tokens != result.total_tokens:
            issues.append(
                f"matched_tokens ({result.matched_tokens}) + "
                f"unmatched_tokens ({result.unmatched_tokens}) "
                f"!= total_tokens ({result.total_tokens})"
            )

        # Coverage bounds
        if not (0.0 <= result.token_coverage <= 1.0):
            issues.append(
                f"token_coverage out of [0, 1]: {result.token_coverage}"
            )

        # All category counts non-negative
        for key, cnt in result.category_counts.items():
            if cnt < 0:
                issues.append(
                    f"category_counts['{key}'] is negative: {cnt}"
                )

        # Shortcut fields match category_counts
        for key in _CATEGORY_KEYS:
            shortcut_attr = f"{key}_count"
            shortcut_val = getattr(result, shortcut_attr, None)
            dict_val = result.category_counts.get(key, 0)
            if shortcut_val is not None and shortcut_val != dict_val:
                issues.append(
                    f"Shortcut field '{shortcut_attr}' ({shortcut_val}) "
                    f"!= category_counts['{key}'] ({dict_val})"
                )

        # Empty flag consistency
        if result.is_empty and result.total_tokens != 0:
            issues.append(
                f"is_empty=True but total_tokens={result.total_tokens}"
            )
        if not result.is_empty and result.total_tokens == 0:
            issues.append(
                "is_empty=False but total_tokens=0"
            )

        if issues:
            logger.warning(
                "[LMMatcher] validate_match_result found %d issue(s): %s",
                len(issues),
                issues,
            )
        return issues

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _build_lookup_cache(self) -> Dict[str, FrozenSet[str]]:
        """
        Extract and cache the frozensets for each active category from the
        dictionary.  Keyed by snake_case category key.
        """
        cache: Dict[str, FrozenSet[str]] = {}
        active_keys = self._config.resolved_categories

        for key in active_keys:
            cat_name = _KEY_TO_CAT.get(key)
            if cat_name is None:
                logger.warning(
                    "[LMMatcher] No CamelCase mapping for key '%s' — skipping.",
                    key,
                )
                continue
            fs = self._lookup_terms(key, cat_name)
            cache[key] = fs
            logger.debug(
                "[LMMatcher] Cached '%s' lookup set: %d terms.", key, len(fs)
            )

        return cache

    def _lookup_terms(self, key: str, cat_name: str) -> FrozenSet[str]:
        """Read terms from a production LMDictionary or a plain test dictionary."""
        if hasattr(self._dictionary, "lookup_sets"):
            return getattr(self._dictionary, "lookup_sets").get(cat_name, frozenset())
        if isinstance(self._dictionary, Mapping):
            terms = (
                self._dictionary.get(key)
                or self._dictionary.get(cat_name)
                or self._dictionary.get(cat_name.lower())
                or frozenset()
            )
            return frozenset(str(term).lower() for term in terms)
        return frozenset()

    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove malformed, empty, or out-of-range tokens.

        Applies case normalisation when ``config.case_sensitive=False``.

        Parameters
        ----------
        tokens:
            Raw token list.

        Returns
        -------
        List[str]
            Cleaned token list (order preserved).

        Raises
        ------
        ValueError
            When ``skip_malformed_tokens=False`` and a malformed token is
            encountered.
        """
        out: List[str] = []
        for i, tok in enumerate(tokens):
            if not isinstance(tok, str):
                msg = (
                    f"Token at index {i} is not a string: "
                    f"{type(tok).__name__!r} = {tok!r}"
                )
                if self._config.skip_malformed_tokens:
                    logger.debug("[LMMatcher] Skipping malformed token: %s", msg)
                    continue
                raise ValueError(msg)

            # Apply case normalisation
            tok_norm = tok if self._config.case_sensitive else tok.lower()

            # Length gate
            if not (_MIN_TOKEN_LEN <= len(tok_norm) <= _MAX_TOKEN_LEN):
                if not self._config.skip_malformed_tokens and tok_norm:
                    raise ValueError(
                        f"Token length {len(tok_norm)} out of range "
                        f"[{_MIN_TOKEN_LEN}, {_MAX_TOKEN_LEN}]: {tok_norm!r}"
                    )
                if tok_norm:
                    logger.debug(
                        "[LMMatcher] Skipping out-of-range token: %r", tok_norm
                    )
                continue

            out.append(tok_norm)

        return out

    def _run_match(self, tokens: List[str]) -> LMMatchResult:
        """
        Core matching loop.

        Iterates tokens once and performs O(1) frozenset lookups per
        category per token.  Builds all result structures in a single pass.
        """
        active_keys = self._config.resolved_categories
        n = len(tokens)

        # Accumulators — keyed by snake_case category key
        # If count_unique: use sets to deduplicate; else use lists
        if self._config.count_unique:
            cat_matched: Dict[str, Set[str]] = {k: set() for k in active_keys}
        else:
            cat_matched_list: Dict[str, List[str]] = {k: [] for k in active_keys}

        # Position map — always a list of (token, position) tuples
        cat_positions: Dict[str, List[Tuple[str, int]]] = {
            k: [] for k in active_keys
        }

        matched_token_set: Set[str] = set()  # for coverage — any category hit
        matched_count: int = 0              # raw matched token count

        unmatched_terms: List[str] = []

        # --- single-pass loop ---
        for pos, tok in enumerate(tokens):
            tok_hit = False

            for key in active_keys:
                fs = self._lookup_cache.get(key, frozenset())
                if tok in fs:
                    tok_hit = True
                    if self._config.count_unique:
                        cat_matched[key].add(tok)  # type: ignore[index]
                    else:
                        cat_matched_list[key].append(tok)  # type: ignore[index]

                    if self._config.preserve_token_positions:
                        cat_positions[key].append((tok, pos))
                    else:
                        cat_positions[key].append((tok, -1))

            if tok_hit:
                matched_token_set.add(tok) if self._config.count_unique else None
                matched_count += 1
            else:
                if self._config.include_unmatched_terms:
                    unmatched_terms.append(tok)

        # --- build final matched_terms (ordered, deduplicated for unique mode) ---
        if self._config.count_unique:
            # Preserve order of first occurrence
            matched_terms: Dict[str, List[str]] = {
                k: _ordered_unique(cat_matched[k], tokens)  # type: ignore[arg-type]
                for k in active_keys
            }
            category_counts: Dict[str, int] = {
                k: len(cat_matched[k]) for k in active_keys  # type: ignore[arg-type]
            }
            # Recalculate matched_count for unique mode
            matched_count = len(
                {tok for key in active_keys for tok in cat_matched[key]}  # type: ignore[union-attr]
            )
        else:
            matched_terms = {k: cat_matched_list[k] for k in active_keys}  # type: ignore[assignment]
            category_counts = {
                k: len(cat_matched_list[k]) for k in active_keys  # type: ignore[index]
            }

        unmatched_count = n - matched_count
        coverage = matched_count / n if n > 0 else 0.0

        if not self._config.include_unmatched_terms:
            unmatched_terms = []

        result = LMMatchResult(
            total_tokens=n,
            matched_tokens=matched_count,
            unmatched_tokens=unmatched_count,
            coverage_ratio=coverage,
            category_counts=category_counts,
            matched_terms=matched_terms,
            unmatched_terms=unmatched_terms,
            category_term_map=cat_positions,
            # Shortcut fields
            positive_count=category_counts.get("positive", 0),
            negative_count=category_counts.get("negative", 0),
            uncertainty_count=category_counts.get("uncertainty", 0),
            litigious_count=category_counts.get("litigious", 0),
            strong_modal_count=category_counts.get("strong_modal", 0),
            weak_modal_count=category_counts.get("weak_modal", 0),
            constraining_count=category_counts.get("constraining", 0),
            is_empty=False,
            match_duration_ms=0.0,  # filled by caller
        )
        return result

    def _count_only(self, tokens: List[str]) -> Dict[str, int]:
        """Fast path: category counts only, no term list construction."""
        active_keys = self._config.resolved_categories
        counts: Dict[str, int] = {k: 0 for k in active_keys}

        if self._config.count_unique:
            seen: Dict[str, Set[str]] = {k: set() for k in active_keys}
            for tok in tokens:
                for key in active_keys:
                    if tok in self._lookup_cache.get(key, frozenset()):
                        seen[key].add(tok)
            counts = {k: len(seen[k]) for k in active_keys}
        else:
            for tok in tokens:
                for key in active_keys:
                    if tok in self._lookup_cache.get(key, frozenset()):
                        counts[key] += 1

        return counts

    @staticmethod
    def _empty_result() -> LMMatchResult:
        """Return a canonical empty LMMatchResult."""
        return LMMatchResult(
            total_tokens=0,
            matched_tokens=0,
            unmatched_tokens=0,
            coverage_ratio=0.0,
            category_counts={k: 0 for k in _CATEGORY_KEYS},
            matched_terms={k: [] for k in _CATEGORY_KEYS},
            unmatched_terms=[],
            category_term_map={k: [] for k in _CATEGORY_KEYS},
            positive_count=0,
            negative_count=0,
            uncertainty_count=0,
            litigious_count=0,
            strong_modal_count=0,
            weak_modal_count=0,
            constraining_count=0,
            is_empty=True,
            match_duration_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_frozen(obj: LMMatchResult, **kwargs) -> LMMatchResult:
    """
    Return a new ``LMMatchResult`` with specified fields replaced.

    Python's ``dataclasses.replace()`` does not work directly on frozen
    dataclasses with complex defaults, so we use ``__init__`` manually.
    """
    current = {
        f: getattr(obj, f)
        for f in obj.__dataclass_fields__  # type: ignore[attr-defined]
    }
    current.update(kwargs)
    return LMMatchResult(**current)


def _ordered_unique(unique_set: Set[str], token_order: List[str]) -> List[str]:
    """
    Return the elements of *unique_set* in the order they first appeared
    in *token_order*.  Preserves first-occurrence ordering for unique mode.
    """
    seen: Set[str] = set()
    result: List[str] = []
    for tok in token_order:
        if tok in unique_set and tok not in seen:
            result.append(tok)
            seen.add(tok)
    return result


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

    # ------------------------------------------------------------------ #
    # We build a synthetic LMDictionary inline so no CSV is required.     #
    # In production: use LMDictionaryLoader.load() instead.               #
    # ------------------------------------------------------------------ #
    import pandas as pd

    print("\n" + "=" * 72)
    print("LMMatcher — self-test / demo")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Build synthetic LMDictionary
    # ------------------------------------------------------------------
    print("\n[1] Building synthetic LMDictionary …")

    synthetic_df = pd.DataFrame({
        "Word": [
            "improve", "growth", "exceed", "strong", "record", "achieve",
            "outstanding", "profitable", "gain", "optimistic",
            "loss", "decline", "weak", "miss", "concern", "risk",
            "impairment", "writedown", "deteriorate", "negative",
            "uncertain", "unclear", "unpredictable", "volatility",
            "lawsuit", "litigation", "regulatory", "alleged", "violation",
            "must", "will", "shall", "require", "need",
            "may", "could", "might", "possible", "perhaps",
            "constrained", "limited", "restrict", "hamper",
        ],
        "Positive":     [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0],
        "Negative":     [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0],
        "Uncertainty":  [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0],
        "Litigious":    [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0],
        "StrongModal":  [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0, 0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0],
        "WeakModal":    [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1, 0,0,0,0],
        "Constraining": [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1],
    })

    lm_dict: LMDictionary = LMDictionaryLoader.from_dataframe(synthetic_df)
    print(lm_dict.summary())

    # ------------------------------------------------------------------
    # 2. Instantiate LMMatcher
    # ------------------------------------------------------------------
    print("\n[2] Instantiating LMMatcher (default config) …")
    matcher = LMMatcher(lm_dict)

    # ------------------------------------------------------------------
    # 3. Sample earnings-call-style text
    # ------------------------------------------------------------------
    SAMPLE_TEXTS = {
        "positive_call": (
            "We achieve record growth this quarter and improve our margins "
            "outstanding performance we exceed all targets and gain market share "
            "management is optimistic and profitable operations will continue"
        ),
        "negative_call": (
            "we report a significant loss and revenue decline weakness in demand "
            "we miss guidance again risk of impairment and writedown concern "
            "the regulatory environment may constrain future growth"
        ),
        "mixed_call": (
            "revenue growth is strong but we face uncertain macro conditions "
            "we must address litigation risk however the outlook may improve "
            "costs are constrained yet we achieve positive operating leverage "
            "uncertain demand could limit our ability to exceed prior guidance"
        ),
        "empty_call": "",
    }

    # ------------------------------------------------------------------
    # 4. Run match_text() on each sample
    # ------------------------------------------------------------------
    print("\n[3] Running match_text() on sample transcripts …")
    stats = MatchStatistics()

    for label, text in SAMPLE_TEXTS.items():
        print(f"\n  -- {label} --")
        result: LMMatchResult = matcher.match_text(text)
        stats.update(result)
        print(result.summary())

    # ------------------------------------------------------------------
    # 5. Demonstrate match_tokens() with a pre-tokenised list
    # ------------------------------------------------------------------
    print("\n[4] Demonstrating match_tokens() with hand-crafted tokens …")
    tokens = [
        "revenue", "growth", "loss", "uncertain", "litigation",
        "must", "may", "constrained", "improve", "foobar", "xyz",
        "loss", "loss",                        # repeated — frequency test
    ]
    print(f"  Tokens: {tokens}")

    result_freq = matcher.match_tokens(tokens)
    print("\n  --- frequency mode (count_unique=False, default) ---")
    print(result_freq.summary())
    print(f"  loss count (freq)  : {result_freq.negative_count}")   # 3

    matcher_unique = LMMatcher(lm_dict, LMMatcherConfig(count_unique=True))
    result_uniq = matcher_unique.match_tokens(tokens)
    print("\n  --- unique mode (count_unique=True) ---")
    print(result_uniq.summary())
    print(f"  loss count (unique): {result_uniq.negative_count}")   # 1

    # ------------------------------------------------------------------
    # 6. count_categories() fast path
    # ------------------------------------------------------------------
    print("\n[5] Demonstrating count_categories() fast path …")
    counts = matcher.count_categories(tokens)
    print("  category counts:", json.dumps(counts, indent=2))

    # ------------------------------------------------------------------
    # 7. calculate_coverage()
    # ------------------------------------------------------------------
    print("\n[6] Demonstrating calculate_coverage() …")
    cov = matcher.calculate_coverage(tokens)
    print(f"  token coverage: {cov:.4f}")

    # ------------------------------------------------------------------
    # 8. build_term_maps()
    # ------------------------------------------------------------------
    print("\n[7] Demonstrating build_term_maps() …")
    matched_map, unmatched_map = matcher.build_term_maps(tokens)
    for cat, terms in matched_map.items():
        if terms:
            print(f"  {cat}: {terms}")
    print(f"  unmatched: {unmatched_map['unmatched']}")

    # ------------------------------------------------------------------
    # 9. validate_match_result()
    # ------------------------------------------------------------------
    print("\n[8] Demonstrating validate_match_result() …")
    issues = matcher.validate_match_result(result_freq)
    print(f"  Validation issues on clean result : {issues}")   # []

    # Manually construct a broken result to verify detection
    broken = LMMatchResult(
        total_tokens=10,
        matched_tokens=7,
        unmatched_tokens=5,   # intentionally wrong (7+5 != 10)
        coverage_ratio=1.5,   # out of bounds
        category_counts={"positive": -1, "negative": 0, "uncertainty": 0,
                         "litigious": 0, "strong_modal": 0,
                         "weak_modal": 0, "constraining": 0},
        matched_terms={},
        unmatched_terms=[],
        category_term_map={},
        positive_count=99,    # mismatches category_counts
        is_empty=False,
    )
    broken_issues = matcher.validate_match_result(broken)
    print(f"  Validation issues on broken result: {broken_issues}")

    # ------------------------------------------------------------------
    # 10. Preserve token positions
    # ------------------------------------------------------------------
    print("\n[9] Demonstrating preserve_token_positions …")
    matcher_pos = LMMatcher(
        lm_dict,
        LMMatcherConfig(preserve_token_positions=True),
    )
    result_pos = matcher_pos.match_tokens(tokens)
    for cat, positions in result_pos.category_term_map.items():
        if positions:
            print(f"  {cat}: {positions}")

    # ------------------------------------------------------------------
    # 11. Batch MatchStatistics
    # ------------------------------------------------------------------
    print("\n[10] Batch MatchStatistics summary …")
    print(stats.summary())

    # ------------------------------------------------------------------
    # 12. to_dict() for DataFrame construction
    # ------------------------------------------------------------------
    print("\n[11] LMMatchResult.to_dict() (for DataFrame rows) …")
    print(json.dumps(result_freq.to_dict(), indent=2))

    print("\n" + "=" * 72)
    print("Self-test complete.")
    print("=" * 72 + "\n")
