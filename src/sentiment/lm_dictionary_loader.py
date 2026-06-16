"""
lm_dictionary_loader.py
=======================
Loughran-McDonald Financial Sentiment Dictionary Loader.

Responsibilities
----------------
* Load the official LM master dictionary CSV from disk.
* Validate schema, normalise terms, and build immutable lookup structures.
* Expose per-category frozenset lookup sets and a term→categories mapping.
* Cache processed LMDictionary objects so expensive I/O is never repeated.
* Provide diagnostics and a human-readable summary at every level.

Does NOT implement:
    matching, scoring, aggregation, preprocessing, pipeline orchestration,
    or parquet export — those live in separate modules.

Expected dictionary path
------------------------
    data/dictionaries/loughran_mcdonald/
        lm_master_dictionary.csv   (default filename; configurable)

The official master dictionary can be downloaded from:
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/

Architecture notes
------------------
* LMDictionaryConfig      — frozen dataclass; no mutable state.
* DictionaryValidationResult — collects all validation findings without raising
                               immediately, letting the caller decide severity.
* LMDictionary            — immutable view of loaded/normalised data.
                            All lookup_sets are frozensets; term_to_categories
                            values are frozensets; total_terms is derived at
                            construction time.
* LMDictionaryLoader      — stateful only through an optional in-process cache
                            keyed by resolved Path.  Thread-safe for read
                            access; write (load) path should be called once
                            per process.
"""

from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All sentiment categories supported by the LM master dictionary.
LM_CATEGORIES: Tuple[str, ...] = (
    "Positive",
    "Negative",
    "Uncertainty",
    "Litigious",
    "StrongModal",
    "WeakModal",
    "Constraining",
)

#: Default column names in the official LM master CSV.
#: The actual CSV uses integer-coded columns (non-zero = term belongs to
#: that category), so we look for the *word* column and category columns.
_DEFAULT_WORD_COLUMN: str = "Word"

#: Mapping from LM category name → default CSV column name.
#: These are the column headers in the 2018+ master dictionary release.
_DEFAULT_COLUMN_MAP: Dict[str, str] = {
    "Positive":     "Positive",
    "Negative":     "Negative",
    "Uncertainty":  "Uncertainty",
    "Litigious":    "Litigious",
    "StrongModal":  "StrongModal",
    "WeakModal":    "WeakModal",
    "Constraining": "Constraining",
}

#: Terms shorter than this are flagged as suspicious during validation.
_MIN_TERM_LENGTH: int = 2

#: Terms longer than this are flagged as suspicious (likely malformed rows).
_MAX_TERM_LENGTH: int = 60


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LMDictionaryConfig:
    """
    Immutable configuration for LMDictionaryLoader.

    Parameters
    ----------
    dictionary_dir:
        Directory containing the LM master CSV file.
    filename:
        CSV filename within *dictionary_dir*.
    word_column:
        Name of the column holding the term/word.
    column_map:
        Mapping from canonical LM category name to CSV column name.
        Override if your CSV uses non-standard headers.
    encoding:
        File encoding (default ``'utf-8'``; some releases ship as
        ``'latin-1'``).
    enable_cache:
        When ``True`` (default) the loader reuses a previously built
        ``LMDictionary`` for the same resolved path within the same process.
    strict_validation:
        When ``True`` any validation error raises ``ValueError``.
        When ``False`` (default) errors are logged as warnings and stored
        in ``DictionaryValidationResult``.
    categories:
        Subset of ``LM_CATEGORIES`` to load.  ``None`` loads all seven.
    """

    dictionary_dir: Path = Path("data/dictionaries/loughran_mcdonald")
    filename: str = "lm_master_dictionary.csv"
    word_column: str = _DEFAULT_WORD_COLUMN
    column_map: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_COLUMN_MAP)
    )
    encoding: str = "utf-8"
    enable_cache: bool = True
    strict_validation: bool = False
    categories: Optional[Tuple[str, ...]] = None  # None → all

    @property
    def csv_path(self) -> Path:
        """Resolved absolute path to the CSV file."""
        return Path(self.dictionary_dir) / self.filename

    @property
    def active_categories(self) -> Tuple[str, ...]:
        """Return the subset of categories this config will load."""
        if self.categories is None:
            return LM_CATEGORIES
        unknown = set(self.categories) - set(LM_CATEGORIES)
        if unknown:
            raise ValueError(
                f"Unknown LM categories requested: {unknown}. "
                f"Valid categories: {LM_CATEGORIES}"
            )
        return tuple(self.categories)


# ---------------------------------------------------------------------------
# Validation result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DictionaryValidationResult:
    """
    Collects all findings from dictionary validation without raising.

    Attributes
    ----------
    is_valid:
        ``True`` when no *errors* were found (warnings are still allowed).
    errors:
        Blocking issues that prevent correct operation.
    warnings:
        Non-blocking observations (e.g. suspiciously long terms).
    duplicate_terms:
        Normalised terms that appeared more than once in the raw CSV.
    malformed_rows:
        Zero-based row indices (in the raw CSV) that could not be parsed.
    empty_terms:
        Rows where the word column was null or blank after normalisation.
    missing_category_columns:
        LM category names whose expected CSV column was absent.
    category_term_counts:
        How many terms belong to each category after deduplication.
    total_raw_rows:
        Number of data rows in the raw CSV before any filtering.
    total_valid_terms:
        Number of unique normalised terms retained after validation.
    validated_at:
        UTC timestamp when validation ran.
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duplicate_terms: List[str] = field(default_factory=list)
    malformed_rows: List[int] = field(default_factory=list)
    empty_terms: List[int] = field(default_factory=list)
    missing_category_columns: List[str] = field(default_factory=list)
    category_term_counts: Dict[str, int] = field(default_factory=dict)
    total_raw_rows: int = 0
    total_valid_terms: int = 0
    validated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False
        logger.error("[LMDictionary] Validation ERROR: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("[LMDictionary] Validation WARNING: %s", msg)

    def summary(self) -> str:
        lines = [
            "=== DictionaryValidationResult ===",
            f"  valid              : {self.is_valid}",
            f"  validated_at       : {self.validated_at}",
            f"  total_raw_rows     : {self.total_raw_rows}",
            f"  total_valid_terms  : {self.total_valid_terms}",
            f"  duplicate_terms    : {len(self.duplicate_terms)}",
            f"  malformed_rows     : {len(self.malformed_rows)}",
            f"  empty_terms        : {len(self.empty_terms)}",
            f"  missing_categories : {self.missing_category_columns}",
            "  category_term_counts:",
        ]
        for cat, cnt in sorted(self.category_term_counts.items()):
            lines.append(f"    {cat:<16}: {cnt}")
        if self.errors:
            lines.append(f"  errors  ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:            # cap output
                lines.append(f"    - {w}")
            if len(self.warnings) > 10:
                lines.append(f"    ... +{len(self.warnings) - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LMDictionary — the core immutable data object
# ---------------------------------------------------------------------------

class LMDictionary:
    """
    Immutable, pre-processed view of the Loughran-McDonald dictionary.

    Construction is handled exclusively by ``LMDictionaryLoader``.
    All public attributes are read-only after ``__init__``.

    Attributes
    ----------
    categories : tuple[str, ...]
        Ordered tuple of active category names.
    lookup_sets : dict[str, frozenset[str]]
        Per-category frozensets of normalised terms.  O(1) membership tests.
    term_to_categories : dict[str, frozenset[str]]
        Maps a normalised term to the frozenset of categories it belongs to.
    total_terms : int
        Total number of *unique* normalised terms across all categories
        (a term in multiple categories is counted once).
    validation_result : DictionaryValidationResult
        Validation findings produced during load.
    loaded_from : Path
        Resolved path of the source CSV.
    loaded_at : str
        UTC ISO-8601 timestamp of when the object was built.
    """

    __slots__ = (
        "_categories",
        "_lookup_sets",
        "_term_to_categories",
        "_total_terms",
        "_validation_result",
        "_loaded_from",
        "_loaded_at",
    )

    def __init__(
        self,
        categories: Tuple[str, ...],
        lookup_sets: Dict[str, FrozenSet[str]],
        term_to_categories: Dict[str, FrozenSet[str]],
        validation_result: DictionaryValidationResult,
        loaded_from: Path,
    ) -> None:
        object.__setattr__(self, "_categories", categories)
        object.__setattr__(self, "_lookup_sets", lookup_sets)
        object.__setattr__(self, "_term_to_categories", term_to_categories)
        object.__setattr__(
            self, "_total_terms", len(term_to_categories)
        )
        object.__setattr__(self, "_validation_result", validation_result)
        object.__setattr__(self, "_loaded_from", loaded_from)
        object.__setattr__(
            self,
            "_loaded_at",
            datetime.now(timezone.utc).isoformat(),
        )

    # Prevent any mutation after construction
    def __setattr__(self, *_):  # type: ignore[override]
        raise AttributeError("LMDictionary is immutable after construction.")

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def categories(self) -> Tuple[str, ...]:
        return object.__getattribute__(self, "_categories")

    @property
    def lookup_sets(self) -> Dict[str, FrozenSet[str]]:
        return object.__getattribute__(self, "_lookup_sets")

    @property
    def term_to_categories(self) -> Dict[str, FrozenSet[str]]:
        return object.__getattribute__(self, "_term_to_categories")

    @property
    def total_terms(self) -> int:
        return object.__getattribute__(self, "_total_terms")

    def __len__(self) -> int:
        return self.total_terms

    @property
    def validation_result(self) -> DictionaryValidationResult:
        return object.__getattribute__(self, "_validation_result")

    @property
    def loaded_from(self) -> Path:
        return object.__getattribute__(self, "_loaded_from")

    @property
    def loaded_at(self) -> str:
        return object.__getattribute__(self, "_loaded_at")

    # ------------------------------------------------------------------
    # Convenience helpers used by downstream matchers
    # ------------------------------------------------------------------

    def contains(self, term: str, category: Optional[str] = None) -> bool:
        """
        Return ``True`` if *term* (already normalised) is in the dictionary.

        Parameters
        ----------
        term:
            A pre-normalised (lowercase, stripped) term.
        category:
            When supplied, restrict the check to that specific category.
        """
        t2c: Dict[str, FrozenSet[str]] = object.__getattribute__(
            self, "_term_to_categories"
        )
        if term not in t2c:
            return False
        if category is None:
            return True
        return category in t2c[term]

    def categories_for(self, term: str) -> FrozenSet[str]:
        """Return the frozenset of categories *term* belongs to, or empty."""
        t2c: Dict[str, FrozenSet[str]] = object.__getattribute__(
            self, "_term_to_categories"
        )
        return t2c.get(term, frozenset())

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a concise human-readable summary of the loaded dictionary.
        """
        lookup_sets: Dict[str, FrozenSet[str]] = object.__getattribute__(
            self, "_lookup_sets"
        )
        lines = [
            "=== LMDictionary Summary ===",
            f"  loaded_from  : {object.__getattribute__(self, '_loaded_from')}",
            f"  loaded_at    : {object.__getattribute__(self, '_loaded_at')}",
            f"  total_terms  : {object.__getattribute__(self, '_total_terms')}",
            f"  categories   : {object.__getattribute__(self, '_categories')}",
            "  terms_per_category:",
        ]
        for cat in object.__getattribute__(self, "_categories"):
            lines.append(f"    {cat:<16}: {len(lookup_sets.get(cat, frozenset()))}")
        return "\n".join(lines)

    def export_summary(self) -> Dict:
        """
        Return dictionary summary as a plain dict (JSON-serialisable).

        Useful for logging, parquet metadata, or pipeline diagnostics.
        """
        lookup_sets: Dict[str, FrozenSet[str]] = object.__getattribute__(
            self, "_lookup_sets"
        )
        return {
            "loaded_from": str(object.__getattribute__(self, "_loaded_from")),
            "loaded_at": object.__getattribute__(self, "_loaded_at"),
            "total_terms": object.__getattribute__(self, "_total_terms"),
            "categories": list(object.__getattribute__(self, "_categories")),
            "terms_per_category": {
                cat: len(lookup_sets.get(cat, frozenset()))
                for cat in object.__getattribute__(self, "_categories")
            },
            "validation_valid": object.__getattribute__(
                self, "_validation_result"
            ).is_valid,
        }

    def __repr__(self) -> str:
        return (
            f"LMDictionary("
            f"total_terms={self.total_terms}, "
            f"categories={self.categories}, "
            f"loaded_at='{self.loaded_at}')"
        )


# ---------------------------------------------------------------------------
# LMDictionaryLoader
# ---------------------------------------------------------------------------

class LMDictionaryLoader:
    """
    Loads, validates, and caches the Loughran-McDonald master dictionary.

    Usage
    -----
    ::

        config = LMDictionaryConfig(
            dictionary_dir=Path("data/dictionaries/loughran_mcdonald"),
        )
        loader = LMDictionaryLoader(config)
        lm = loader.load()

        # Fast O(1) lookup
        "improve" in lm.lookup_sets["Positive"]   # True
        lm.contains("loss", "Negative")            # True
        lm.categories_for("bankruptcy")            # frozenset({"Negative", ...})

    Thread safety
    -------------
    The ``load()`` method is safe for concurrent *reads* once the cache is
    populated.  The first call that actually reads the CSV is **not**
    protected by a lock — call ``load()`` once during startup (or wrap in a
    lock externally) if multiple threads start simultaneously.
    """

    # Class-level cache: resolved_csv_path → LMDictionary
    _cache: Dict[Path, "LMDictionary"] = {}

    def __init__(self, config: Optional[LMDictionaryConfig | Path | str] = None) -> None:
        if isinstance(config, (str, Path)):
            csv_path = Path(config)
            config = LMDictionaryConfig(
                dictionary_dir=csv_path.parent,
                filename=csv_path.name,
            )
        self._config: LMDictionaryConfig = config or LMDictionaryConfig()
        logger.debug(
            "[LMDictionaryLoader] Initialised with config: %s", self._config
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> LMDictionary:
        """
        Load (or return cached) an ``LMDictionary``.

        Returns
        -------
        LMDictionary
            Immutable, validated dictionary object.

        Raises
        ------
        FileNotFoundError
            When the CSV does not exist at the configured path.
        ValueError
            When ``strict_validation=True`` and validation errors are found.
        """
        resolved = self._config.csv_path.resolve()
        logger.info("[LMDictionaryLoader] Requested path: %s", resolved)

        if self._config.enable_cache and resolved in LMDictionaryLoader._cache:
            logger.info(
                "[LMDictionaryLoader] Cache HIT for %s — reusing existing "
                "LMDictionary.",
                resolved,
            )
            return LMDictionaryLoader._cache[resolved]

        logger.info(
            "[LMDictionaryLoader] Cache MISS — loading from disk: %s", resolved
        )
        lm_dict = self._load_from_disk(resolved)

        if self._config.enable_cache:
            LMDictionaryLoader._cache[resolved] = lm_dict
            logger.info(
                "[LMDictionaryLoader] Cached LMDictionary for %s.", resolved
            )

        return lm_dict

    def validate(self, df: pd.DataFrame) -> DictionaryValidationResult:
        """
        Run all validation checks on a raw dictionary DataFrame.

        This is exposed publicly so callers can validate a DataFrame they
        have already loaded themselves (useful for testing with synthetic data).

        Parameters
        ----------
        df:
            Raw DataFrame as loaded from CSV (before any normalisation).

        Returns
        -------
        DictionaryValidationResult
        """
        return self._run_validation(df)

    @classmethod
    def clear_cache(cls) -> None:
        """Evict all entries from the in-process cache."""
        cls._cache.clear()
        logger.info("[LMDictionaryLoader] Cache cleared.")

    @classmethod
    def cache_keys(cls) -> List[Path]:
        """Return all resolved paths currently held in cache."""
        return list(cls._cache.keys())

    # ------------------------------------------------------------------
    # Static helpers — public so downstream modules can reuse them
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_term(raw: str) -> str:
        """
        Normalise a single term for dictionary use.

        Steps
        -----
        1. Unicode NFC normalisation.
        2. Strip leading/trailing whitespace.
        3. Collapse internal whitespace to a single space.
        4. Lowercase.

        Parameters
        ----------
        raw:
            The raw term string from the CSV.

        Returns
        -------
        str
            Normalised term string, possibly empty if *raw* was blank.
        """
        if not isinstance(raw, str):
            return ""
        # NFC normalisation — ensures canonical unicode representation
        normed: str = unicodedata.normalize("NFC", raw)
        normed = " ".join(normed.split())   # collapse whitespace
        return normed.lower()

    @staticmethod
    def normalize_terms(terms: List[str]) -> List[str]:
        """
        Bulk-normalise a list of terms.

        Returns a list of the same length; malformed/empty entries become
        empty strings (they will be filtered during loading).
        """
        return [LMDictionaryLoader.normalize_term(t) for t in terms]

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _load_from_disk(self, resolved: Path) -> LMDictionary:
        """Read CSV → validate → normalise → build structures → return."""
        if not resolved.exists():
            raise FileNotFoundError(
                f"LM master dictionary not found at: {resolved}\n"
                "Download from https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
            )

        logger.info("[LMDictionaryLoader] Reading CSV …")
        df: pd.DataFrame = pd.read_csv(
            resolved, encoding=self._config.encoding, low_memory=False
        )
        logger.info(
            "[LMDictionaryLoader] CSV loaded: %d rows × %d columns.",
            len(df),
            len(df.columns),
        )

        # Validate
        validation_result: DictionaryValidationResult = self._run_validation(df)

        if self._config.strict_validation and not validation_result.is_valid:
            raise ValueError(
                "LM dictionary validation failed with strict_validation=True.\n"
                + validation_result.summary()
            )

        # Build lookup structures
        lookup_sets, term_to_categories = self._build_lookup_sets(
            df, validation_result
        )

        return LMDictionary(
            categories=self._config.active_categories,
            lookup_sets=lookup_sets,
            term_to_categories=term_to_categories,
            validation_result=validation_result,
            loaded_from=resolved,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, df: pd.DataFrame) -> DictionaryValidationResult:
        result = DictionaryValidationResult()
        result.total_raw_rows = len(df)

        logger.info("[LMDictionaryLoader] Running validation on %d rows …", len(df))

        # 1. Check word column exists
        if self._config.word_column not in df.columns:
            result.add_error(
                f"Word column '{self._config.word_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
            # Cannot proceed without the word column
            return result

        # 2. Check category columns
        for cat in self._config.active_categories:
            expected_col = self._config.column_map.get(cat, cat)
            if expected_col not in df.columns:
                result.missing_category_columns.append(cat)
                result.add_warning(
                    f"Category column '{expected_col}' (for '{cat}') not found "
                    f"in CSV. This category will be empty."
                )

        if len(result.missing_category_columns) == len(
            self._config.active_categories
        ):
            result.add_error(
                "All category columns are missing. "
                "Check column_map in LMDictionaryConfig."
            )
            return result

        # 3. Check for null/empty terms
        raw_terms: pd.Series = df[self._config.word_column]
        null_mask: pd.Series = raw_terms.isnull()
        null_indices: List[int] = df.index[null_mask].tolist()
        if null_indices:
            result.empty_terms.extend(null_indices)
            result.add_warning(
                f"{len(null_indices)} row(s) have null/empty word values "
                f"(rows: {null_indices[:5]}{'…' if len(null_indices) > 5 else ''})."
            )

        # 4. Check for malformed (non-string) term entries
        non_str_mask: pd.Series = ~raw_terms.apply(
            lambda x: isinstance(x, (str, float, type(None)))
        )
        malformed_indices: List[int] = df.index[non_str_mask].tolist()
        if malformed_indices:
            result.malformed_rows.extend(malformed_indices)
            result.add_warning(
                f"{len(malformed_indices)} row(s) have non-string word values."
            )

        # 5. Normalise terms and detect duplicates
        valid_raw: pd.Series = raw_terms[~null_mask]
        normalised_list: List[str] = [
            LMDictionaryLoader.normalize_term(str(t)) for t in valid_raw
        ]
        seen: Set[str] = set()
        duplicates: Set[str] = set()
        for t in normalised_list:
            if t in seen:
                duplicates.add(t)
            seen.add(t)

        if duplicates:
            result.duplicate_terms.extend(sorted(duplicates))
            result.add_warning(
                f"{len(duplicates)} duplicate term(s) found after normalisation "
                f"(first 5: {sorted(duplicates)[:5]}). "
                "Duplicates will be merged during load."
            )

        # 6. Flag suspiciously short or long terms
        short = [t for t in seen if 0 < len(t) < _MIN_TERM_LENGTH]
        long_ = [t for t in seen if len(t) > _MAX_TERM_LENGTH]
        if short:
            result.add_warning(
                f"{len(short)} term(s) shorter than {_MIN_TERM_LENGTH} characters."
            )
        if long_:
            result.add_warning(
                f"{len(long_)} term(s) longer than {_MAX_TERM_LENGTH} characters "
                f"(first 3: {long_[:3]})."
            )

        # 7. Empty lexicon check
        valid_terms = {t for t in seen if t}
        if not valid_terms:
            result.add_error("Dictionary is empty after normalisation.")
            return result

        result.total_valid_terms = len(valid_terms)

        # 8. Per-category counts (from raw CSV, before merge)
        for cat in self._config.active_categories:
            col = self._config.column_map.get(cat, cat)
            if col in df.columns:
                # LM uses integer coding: non-zero = member of category
                try:
                    col_series: pd.Series = pd.to_numeric(
                        df[col], errors="coerce"
                    ).fillna(0)
                    count = int((col_series != 0).sum())
                except Exception as exc:  # noqa: BLE001
                    count = 0
                    result.add_warning(
                        f"Could not count terms for category '{cat}': {exc}"
                    )
                result.category_term_counts[cat] = count
            else:
                result.category_term_counts[cat] = 0

        logger.info(
            "[LMDictionaryLoader] Validation complete — valid=%s, "
            "total_valid_terms=%d, duplicates=%d.",
            result.is_valid,
            result.total_valid_terms,
            len(result.duplicate_terms),
        )
        return result

    # ------------------------------------------------------------------
    # Lookup structure construction
    # ------------------------------------------------------------------

    def _build_lookup_sets(
        self,
        df: pd.DataFrame,
        validation_result: DictionaryValidationResult,
    ) -> Tuple[Dict[str, FrozenSet[str]], Dict[str, FrozenSet[str]]]:
        """
        Build per-category frozensets and the term→categories mapping.

        Returns
        -------
        lookup_sets : dict[str, frozenset[str]]
        term_to_categories : dict[str, frozenset[str]]
        """
        logger.info("[LMDictionaryLoader] Building lookup structures …")

        word_col: str = self._config.word_column

        # Collect mutable sets first, then freeze
        mutable_sets: Dict[str, Set[str]] = {
            cat: set() for cat in self._config.active_categories
        }
        term_to_cats: Dict[str, Set[str]] = defaultdict(set)

        for _, row in df.iterrows():
            raw_word = row.get(word_col)
            if pd.isna(raw_word) or not isinstance(raw_word, str):
                continue

            term: str = LMDictionaryLoader.normalize_term(raw_word)
            if not term:
                continue

            for cat in self._config.active_categories:
                col = self._config.column_map.get(cat, cat)
                if col not in df.columns:
                    continue
                raw_val = row.get(col, 0)
                try:
                    val = float(raw_val) if not pd.isna(raw_val) else 0.0
                except (TypeError, ValueError):
                    val = 0.0

                if val != 0:
                    mutable_sets[cat].add(term)
                    term_to_cats[term].add(cat)

        # Freeze all structures
        lookup_sets: Dict[str, FrozenSet[str]] = {
            cat: frozenset(terms) for cat, terms in mutable_sets.items()
        }
        term_to_categories: Dict[str, FrozenSet[str]] = {
            term: frozenset(cats) for term, cats in term_to_cats.items()
        }

        # Update validation result with actual counts post-deduplication
        for cat, fs in lookup_sets.items():
            validation_result.category_term_counts[cat] = len(fs)
        validation_result.total_valid_terms = len(term_to_categories)

        logger.info(
            "[LMDictionaryLoader] Lookup structures built: %d unique terms, "
            "category sizes: %s",
            len(term_to_categories),
            {cat: len(fs) for cat, fs in lookup_sets.items()},
        )

        return lookup_sets, term_to_categories

    # ------------------------------------------------------------------
    # Convenience factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        config: Optional[LMDictionaryConfig] = None,
    ) -> LMDictionary:
        """
        Build an ``LMDictionary`` directly from a DataFrame (no CSV I/O).

        Useful in unit tests and when the dictionary is supplied from
        an in-memory source rather than a file.

        Parameters
        ----------
        df:
            DataFrame with the same schema as the official LM master CSV.
        config:
            Optional config (uses defaults if not supplied).
        """
        cfg = config or LMDictionaryConfig()
        loader = cls(cfg)
        result = loader._run_validation(df)

        if cfg.strict_validation and not result.is_valid:
            raise ValueError(
                "LM dictionary validation failed.\n" + result.summary()
            )

        lookup_sets, term_to_categories = loader._build_lookup_sets(df, result)

        return LMDictionary(
            categories=cfg.active_categories,
            lookup_sets=lookup_sets,
            term_to_categories=term_to_categories,
            validation_result=result,
            loaded_from=Path("<in-memory>"),
        )


# ---------------------------------------------------------------------------
# Self-test / Demo  (python -m src.sentiment.lm_dictionary_loader)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    print("\n" + "=" * 70)
    print("LMDictionaryLoader  —  self-test / demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Build a synthetic LM-style DataFrame
    # ------------------------------------------------------------------
    print("\n[1] Building synthetic LM dictionary DataFrame …")

    synthetic_data = {
        "Word": [
            "improve",         # Positive
            "growth",          # Positive
            "exceed",          # Positive
            "loss",            # Negative
            "decline",         # Negative
            "bankruptcy",      # Negative + Litigious
            "uncertain",       # Uncertainty
            "may",             # WeakModal
            "must",            # StrongModal
            "constrained",     # Constraining
            "GROWTH",          # duplicate (case variant of 'growth')
            "  loss  ",        # duplicate (whitespace variant of 'loss')
            "",                # empty — should be filtered
            None,              # null — should be filtered
        ],
        "Positive":     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "Negative":     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        "Uncertainty":  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "Litigious":    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "StrongModal":  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "WeakModal":    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "Constraining": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    }

    df_synthetic = pd.DataFrame(synthetic_data)
    print(f"  Synthetic DataFrame shape: {df_synthetic.shape}")
    print(df_synthetic.to_string(index=False))

    # ------------------------------------------------------------------
    # 2. Load via from_dataframe (no file I/O)
    # ------------------------------------------------------------------
    print("\n[2] Loading LMDictionary from synthetic DataFrame …")

    lm: LMDictionary = LMDictionaryLoader.from_dataframe(df_synthetic)
    print("\n" + lm.summary())

    # ------------------------------------------------------------------
    # 3. Demonstrate lookup_sets
    # ------------------------------------------------------------------
    print("\n[3] Demonstrating lookup_sets …")
    for cat in lm.categories:
        terms = sorted(lm.lookup_sets[cat])
        print(f"  {cat:<16}: {terms}")

    # ------------------------------------------------------------------
    # 4. Demonstrate term_to_categories
    # ------------------------------------------------------------------
    print("\n[4] Demonstrating term_to_categories …")
    for term in ["improve", "loss", "bankruptcy", "uncertain", "xyz"]:
        cats = lm.categories_for(term)
        print(f"  '{term}' → {cats}")

    # ------------------------------------------------------------------
    # 5. Demonstrate .contains()
    # ------------------------------------------------------------------
    print("\n[5] Demonstrating .contains() …")
    tests = [
        ("improve", None),
        ("improve", "Positive"),
        ("improve", "Negative"),
        ("bankruptcy", "Litigious"),
        ("notaword", None),
    ]
    for term, cat in tests:
        result_bool = lm.contains(term, cat)
        cat_label = f"category='{cat}'" if cat else "any category"
        print(f"  contains('{term}', {cat_label}) → {result_bool}")

    # ------------------------------------------------------------------
    # 6. Duplicate + normalisation verification
    # ------------------------------------------------------------------
    print("\n[6] Verifying duplicate & normalisation handling …")
    print(f"  'growth' in Positive : {'growth' in lm.lookup_sets['Positive']}")
    print(f"  'GROWTH' in Positive : {'GROWTH' in lm.lookup_sets['Positive']} (expected False — raw string)")
    print(f"  'loss'   in Negative : {'loss' in lm.lookup_sets['Negative']}")

    # ------------------------------------------------------------------
    # 7. Validation result
    # ------------------------------------------------------------------
    print("\n[7] Validation result …")
    print(lm.validation_result.summary())

    # ------------------------------------------------------------------
    # 8. export_summary
    # ------------------------------------------------------------------
    print("\n[8] export_summary() (dict) …")
    import json
    print(json.dumps(lm.export_summary(), indent=2))

    # ------------------------------------------------------------------
    # 9. Demonstrate cache reuse
    # ------------------------------------------------------------------
    print("\n[9] Demonstrating cache reuse (LMDictionaryLoader) …")

    # Build a minimal config pointing at a non-existent file to verify
    # that a second from_dataframe call creates a new object (no cache
    # for in-memory objects), but a real load() would cache on disk path.
    lm2: LMDictionary = LMDictionaryLoader.from_dataframe(df_synthetic)
    print(f"  lm  id: {id(lm)}")
    print(f"  lm2 id: {id(lm2)}")
    print(
        f"  Different objects (from_dataframe always builds fresh): "
        f"{lm is not lm2}"
    )
    print(f"  Same total_terms: {lm.total_terms == lm2.total_terms}")

    # ------------------------------------------------------------------
    # 10. Malformed DataFrame demo (strict_validation=False)
    # ------------------------------------------------------------------
    print("\n[10] Demonstrating graceful handling of fully-missing categories …")

    df_bad = pd.DataFrame({
        "Word":     ["improve", "loss"],
        # Intentionally omit all LM category columns
    })
    loader_lenient = LMDictionaryLoader(
        LMDictionaryConfig(strict_validation=False)
    )
    val_result: DictionaryValidationResult = loader_lenient.validate(df_bad)
    print(f"  is_valid : {val_result.is_valid}")
    print(f"  errors   : {val_result.errors}")
    print(f"  warnings : {val_result.warnings[:3]}")

    print("\n" + "=" * 70)
    print("Self-test complete.")
    print("=" * 70 + "\n")
