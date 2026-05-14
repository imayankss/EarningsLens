"""
src/extraction/metadata_extractor.py
======================================
Metadata extraction engine for financial earnings call transcripts.

Responsibilities
----------------
Extract and normalize the following fields from raw transcript text
(and optional filename hints):

    ticker          – stock ticker symbol, e.g. "AAPL"
    company         – full company name, e.g. "Apple Inc."
    quarter         – fiscal quarter, e.g. "Q1"
    year            – fiscal year (int), e.g. 2025
    date            – earnings call date, ISO-8601 string, e.g. "2025-01-30"
    transcript_id   – compound primary key, e.g. "AAPL_Q1_2025"

Output
------
Returns a :class:`TranscriptMetadata` dataclass that:
    * is pandas-compatible via ``.to_dict()`` / ``.to_series()``
    * carries per-field confidence levels ("high" | "medium" | "low" | "missing")
    * stores raw matched strings for debugging / audit

Design principles
-----------------
- All regex sourced from ``src/utils/regex_patterns.py`` (no inline hardcoding).
- Extraction-specific compound patterns are module-level constants here
  (they are pure pattern data, not logic, but too narrow for the shared registry).
- Every public method is independently callable for unit testing.
- Graceful fallbacks: no exception propagates; missing fields are ``None``.
- Full ``logging`` integration — INFO for successes, WARNING for fallbacks,
  DEBUG for every candidate match considered.

Python: 3.11+
Dependencies: python-dateutil, pandas (optional, for ``.to_series()``)
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── third-party ───────────────────────────────────────────────────────────────
try:
    from dateutil import parser as _dateutil_parser
    from dateutil.parser import ParserError as _DateParserError
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "python-dateutil is required: pip install python-dateutil"
    ) from _e

# ── internal ──────────────────────────────────────────────────────────────────
# Adjust the import path if your project root is not on sys.path.
# When running as part of the src package: `from src.utils.regex_patterns import …`
# For standalone testing this file falls back to a sibling import guard below.
try:
    from src.utils.regex_patterns import (
        DatePatterns,
        QuarterPatterns,
        TickerPatterns,
        YearPatterns,
        FY_SHORT_YEAR_PIVOT,
        MIN_PLAUSIBLE_YEAR,
        MAX_PLAUSIBLE_YEAR,
        QUARTER_WORD_MAP,
    )
except ModuleNotFoundError:
    # Allow running the file directly from its own directory during development.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.regex_patterns import (  # type: ignore[no-redef]
        DatePatterns,
        QuarterPatterns,
        TickerPatterns,
        YearPatterns,
        FY_SHORT_YEAR_PIVOT,
        MIN_PLAUSIBLE_YEAR,
        MAX_PLAUSIBLE_YEAR,
        QUARTER_WORD_MAP,
    )

# =============================================================================
# Module logger
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# Module-level extraction constants
# =============================================================================

# Number of lines from the top of a transcript treated as the "header region".
# Most metadata (ticker, date, quarter) appears here.
_HEADER_LINE_COUNT: int = 35

# Uppercase token strings that pattern-match as tickers but are NOT tickers.
# Sourced from common earnings call vocabulary.
_TICKER_FALSE_POSITIVES: frozenset[str] = frozenset(
    {
        # Quarter / year tokens
        "Q1", "Q2", "Q3", "Q4", "FY", "YTD", "YOY", "QOQ", "MOM",
        # Role abbreviations
        "CEO", "CFO", "COO", "CTO", "CMO", "CRO", "CIO", "CLO",
        "VP", "SVP", "EVP", "AVP", "MD", "IR", "GM",
        # Financial acronyms
        "EPS", "PE", "VC", "IPO", "EBITDA", "GAAP", "OPEX", "CAPEX",
        "ARR", "MRR", "LTV", "CAC", "FCF", "NOI", "NII", "ROE", "ROA",
        # Exchanges / index names
        "NYSE", "NASDAQ", "AMEX", "OTC", "SEC", "FDIC", "FED",
        "SPX", "DJI", "VIX",
        # Geographic / currency
        "USA", "USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD",
        "US", "UK", "EU", "UN",
        # Technology acronyms common in transcripts
        "AI", "ML", "API", "SaaS", "PaaS", "IaaS", "IOT", "AR", "VR",
        # Time of day (operator scripts)
        "AM", "PM",
    }
)

# ── Extractor-specific compound patterns ─────────────────────────────────────
# These are too narrow for the shared registry but are still pure pattern data.

# "Apple Inc. Q1 2025 Earnings Call"
# "AAPL Q1 FY2025 Results Conference"
_COMPANY_BEFORE_CALL: re.Pattern[str] = re.compile(
    r"^(.+?)"                                     # Company name (non-greedy)
    r"\s*(?:\([A-Z]{1,5}\))?"                      # Optional (TICKER) in parens
    r"\s*(?:Q[1-4]|first|second|third|fourth)"    # Quarter reference starts
    r".*?(?:earnings|results|conference|call)",    # Call type word
    re.IGNORECASE,
)

# "Welcome to the Apple Inc. Q1 2025 Earnings Call"
# "Welcome to Apple's First Quarter Fiscal 2025 Conference Call"
_WELCOME_COMPANY: re.Pattern[str] = re.compile(
    r"welcome\s+to\s+(?:the\s+)?"               # Operator opening
    r"(.+?)"                                      # Company name (non-greedy)
    r"(?:'s\s+)?"                                 # Optional possessive
    r"(?:Q[1-4]|first|second|third|fourth|\d{4}|earnings|results|fiscal)",
    re.IGNORECASE,
)

# "This is the Apple Inc. Earnings Call for Q1 2025"
_THIS_IS_COMPANY: re.Pattern[str] = re.compile(
    r"this\s+is\s+(?:the\s+)?"
    r"(.+?)\s*"
    r"(?:earnings|results|conference|quarterly)\s+(?:call|results)",
    re.IGNORECASE,
)

# Ticker in parentheses next to company name: "Apple Inc. (AAPL)"
_TICKER_IN_PARENS: re.Pattern[str] = re.compile(
    r"\(([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\)",
)

# Confidence levels (ordered worst → best for comparison)
_CONFIDENCE_RANK: dict[str, int] = {
    "missing": 0,
    "low":     1,
    "medium":  2,
    "high":    3,
}

# =============================================================================
# Output dataclass
# =============================================================================


@dataclass
class TranscriptMetadata:
    """
    Structured metadata record for a single earnings call transcript.

    All nullable fields are ``None`` when extraction failed.
    Use :meth:`is_complete` to check whether all required fields are present
    before writing to the structured dataset.

    Attributes
    ----------
    ticker:
        Uppercase ticker symbol, e.g. ``"AAPL"``.
    company:
        Cleaned company name, e.g. ``"Apple Inc."``.
    quarter:
        Fiscal quarter in canonical form, e.g. ``"Q1"``.
    year:
        Fiscal year as integer, e.g. ``2025``.
    date:
        Earnings call date, ISO-8601 string ``"YYYY-MM-DD"``, or ``None``.
    transcript_id:
        Compound primary key: ``"<TICKER>_<QUARTER>_<YEAR>"``,
        e.g. ``"AAPL_Q1_2025"``.
    source_file:
        Original filename / path, preserved for audit trail.
    confidence:
        Per-field extraction confidence:
        ``"high"`` | ``"medium"`` | ``"low"`` | ``"missing"``.
    raw_matches:
        Raw strings that triggered each extraction, for debugging.
    """

    ticker: str | None = None
    company: str | None = None
    quarter: str | None = None
    year: int | None = None
    date: str | None = None
    transcript_id: str | None = None
    source_file: str | None = None

    # Per-field confidence tracking
    confidence: dict[str, str] = field(
        default_factory=lambda: {
            "ticker":  "missing",
            "company": "missing",
            "quarter": "missing",
            "year":    "missing",
            "date":    "missing",
        }
    )

    # Raw extraction evidence for debugging / audit
    raw_matches: dict[str, str | None] = field(
        default_factory=lambda: {
            "ticker":  None,
            "company": None,
            "quarter": None,
            "year":    None,
            "date":    None,
        }
    )

    # ── Convenience helpers ───────────────────────────────────────────────────

    def is_complete(self, require_date: bool = False) -> bool:
        """
        Return ``True`` when all pipeline-critical fields are populated.

        Parameters
        ----------
        require_date:
            If ``True``, ``date`` must also be non-None.
        """
        core_ok = all(
            getattr(self, f) is not None
            for f in ("ticker", "quarter", "year", "transcript_id")
        )
        if require_date:
            return core_ok and self.date is not None
        return core_ok

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for ``pd.DataFrame`` row insertion."""
        d = asdict(self)
        # Flatten nested dicts into prefixed keys for DataFrame compatibility
        for field_name, conf in d.pop("confidence").items():
            d[f"conf_{field_name}"] = conf
        d.pop("raw_matches")  # omit debug data from default flat export
        return d

    def to_series(self) -> "pd.Series":  # type: ignore[name-defined]
        """
        Return a ``pandas.Series`` indexed by the flat schema fields.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pandas is required for to_series(): pip install pandas"
            ) from exc
        return pd.Series(self.to_dict(), name=self.transcript_id)

    def __repr__(self) -> str:  # noqa: D105
        conf_summary = {k: v for k, v in self.confidence.items() if v != "high"}
        return (
            f"TranscriptMetadata("
            f"id={self.transcript_id!r}, "
            f"ticker={self.ticker!r}, "
            f"quarter={self.quarter!r}, "
            f"year={self.year!r}, "
            f"date={self.date!r}, "
            f"low_confidence={conf_summary or 'none'!r}"
            f")"
        )


# =============================================================================
# MetadataExtractor
# =============================================================================


class MetadataExtractor:
    """
    Extract structured metadata from a raw earnings call transcript.

    Each public ``extract_*`` method is independently callable and returns
    its field value plus silently updates internal confidence/evidence state.
    :meth:`extract_all` orchestrates the full pipeline and returns a
    :class:`TranscriptMetadata` instance.

    Parameters
    ----------
    text:
        Full raw transcript text.
    filename:
        Optional source filename or path. Used as the highest-priority
        hint for ticker and quarter/year extraction.
    header_lines:
        How many lines from the top of the transcript to treat as the
        "header region" for metadata scanning. Default: 35.
    extra_ticker_blocklist:
        Additional uppercase strings to suppress from ticker candidates.
        Merged with the built-in ``_TICKER_FALSE_POSITIVES`` set.

    Examples
    --------
    >>> extractor = MetadataExtractor(
    ...     text=transcript_text,
    ...     filename="AAPL_Q1_2025_earnings.txt",
    ... )
    >>> meta = extractor.extract_all()
    >>> meta.transcript_id
    'AAPL_Q1_2025'
    >>> meta.to_dict()
    {'ticker': 'AAPL', 'company': 'Apple Inc.', 'quarter': 'Q1', ...}
    """

    def __init__(
        self,
        text: str,
        filename: str | Path | None = None,
        header_lines: int = _HEADER_LINE_COUNT,
        extra_ticker_blocklist: set[str] | None = None,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        self._text: str = text
        self._filename: str | None = Path(filename).name if filename else None
        self._header_lines: int = max(1, header_lines)

        # Build combined ticker blocklist
        self._ticker_blocklist: frozenset[str] = (
            _TICKER_FALSE_POSITIVES | frozenset(extra_ticker_blocklist or set())
        )

        # Precompute header region once (avoids repeated splitting)
        self._header: str = self._build_header()

        # Internal state updated by each extract_* call
        self._confidence: dict[str, str] = {
            "ticker":  "missing",
            "company": "missing",
            "quarter": "missing",
            "year":    "missing",
            "date":    "missing",
        }
        self._raw_matches: dict[str, str | None] = dict.fromkeys(
            ["ticker", "company", "quarter", "year", "date"], None
        )

        # Cache for intermediate results (quarter extraction may also yield year)
        self._cached_year_from_quarter: int | None = None

        logger.debug(
            "MetadataExtractor initialised | filename=%s | text_len=%d | header_lines=%d",
            self._filename,
            len(text),
            self._header_lines,
        )

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _build_header(self) -> str:
        """Return the first ``_header_lines`` lines of the transcript."""
        lines = self._text.splitlines()
        return "\n".join(lines[: self._header_lines])

    @staticmethod
    def _normalize_fy_year(raw: str) -> int:
        """
        Convert a 2-digit or 4-digit fiscal year string to a full 4-digit integer.

        Uses the pivot defined in ``regex_patterns.FY_SHORT_YEAR_PIVOT``:
            - raw < pivot  →  2000 + raw
            - raw >= pivot →  1900 + raw

        Examples
        --------
        >>> MetadataExtractor._normalize_fy_year("25")
        2025
        >>> MetadataExtractor._normalize_fy_year("99")
        1999
        >>> MetadataExtractor._normalize_fy_year("2024")
        2024
        """
        val = int(raw)
        if val < 100:
            val = 2000 + val if val < FY_SHORT_YEAR_PIVOT else 1900 + val
        return val

    @staticmethod
    def _is_plausible_year(year: int) -> bool:
        """Return True when *year* falls within the pipeline's expected range."""
        return MIN_PLAUSIBLE_YEAR <= year <= MAX_PLAUSIBLE_YEAR

    def _set_confidence(
        self,
        field_name: str,
        level: str,
        raw: str | None = None,
    ) -> None:
        """
        Update confidence for *field_name* only if *level* is better than current.

        Confidence levels (ascending): missing → low → medium → high.
        """
        current_rank = _CONFIDENCE_RANK.get(self._confidence[field_name], 0)
        new_rank = _CONFIDENCE_RANK.get(level, 0)
        if new_rank > current_rank:
            self._confidence[field_name] = level
            if raw is not None:
                self._raw_matches[field_name] = raw

    def _clean_company_name(self, raw: str) -> str:
        """
        Strip artefacts from a raw company name candidate.

        Removes:
        - Leading/trailing whitespace
        - Trailing punctuation (commas, periods, dashes)
        - Common transcript header noise words
        """
        name = raw.strip().strip(",-.")
        # Drop trailing noise fragments
        noise_suffixes = (
            "Inc", "Corp", "Ltd", "LLC", "PLC",
            "Earnings", "Results", "Conference", "Call",
        )
        for suffix in noise_suffixes:
            # Only strip if it appears as a bare trailing word
            name = re.sub(
                rf"\s+{re.escape(suffix)}\s*$", "", name, flags=re.IGNORECASE
            ).strip()
        return name or raw.strip()

    # =========================================================================
    # Public extraction methods
    # =========================================================================

    def extract_ticker(self) -> str | None:
        """
        Extract the stock ticker symbol from the transcript.

        Search order (highest confidence first):
        1. Ticker in parentheses in the header: ``"Apple Inc. (AAPL)"``.
        2. Exchange-prefixed pattern: ``"NASDAQ: AAPL"``.
        3. Filename token (first uppercase token ≤5 chars not on the blocklist).
        4. Class-share pattern: ``"BRK.B"``.
        5. Isolated uppercase word in the header (weak, filtered by blocklist).

        Returns
        -------
        str | None
            Uppercase ticker, e.g. ``"AAPL"``, or ``None`` if not found.
        """
        logger.debug("extract_ticker() — scanning header (%d chars)", len(self._header))

        # ── Strategy 1: (TICKER) in parens ───────────────────────────────────
        for m in _TICKER_IN_PARENS.finditer(self._header):
            candidate = m.group(1).upper()
            if candidate not in self._ticker_blocklist:
                logger.info("Ticker found via parentheses: %r", candidate)
                self._set_confidence("ticker", "high", raw=m.group(0))
                return candidate

        # ── Strategy 2: Exchange-prefixed pattern (NASDAQ: AAPL) ─────────────
        m = TickerPatterns.EXCHANGE_PREFIX.search(self._header)
        if m:
            candidate = m.group(1).upper()
            logger.info("Ticker found via exchange prefix: %r", candidate)
            self._set_confidence("ticker", "high", raw=m.group(0))
            return candidate

        # ── Strategy 3: Filename token ────────────────────────────────────────
        if self._filename:
            for m in TickerPatterns.IN_FILENAME.finditer(self._filename):
                candidate = m.group(1).upper()
                if candidate not in self._ticker_blocklist and len(candidate) >= 1:
                    logger.info("Ticker found via filename: %r", candidate)
                    self._set_confidence("ticker", "high", raw=candidate)
                    return candidate

        # ── Strategy 4: Class-share ticker (BRK.B) ────────────────────────────
        m = TickerPatterns.CLASS_SHARE.search(self._header)
        if m:
            candidate = m.group(1).upper()
            if candidate.split(".")[0] not in self._ticker_blocklist:
                logger.info("Ticker found via class-share pattern: %r", candidate)
                self._set_confidence("ticker", "medium", raw=m.group(0))
                return candidate

        # ── Strategy 5: Isolated uppercase header token (weakest) ────────────
        # Only scan the very first 3 lines to minimise false positives.
        first_lines = "\n".join(self._header.splitlines()[:3])
        for m in TickerPatterns.HEADER_ISOLATED.finditer(first_lines):
            candidate = m.group(1).upper()
            if (
                candidate not in self._ticker_blocklist
                and 2 <= len(candidate) <= 5   # 1-char tickers too risky
                and candidate.isalpha()         # no digits in candidate
            ):
                logger.warning(
                    "Ticker inferred from isolated header token (low confidence): %r",
                    candidate,
                )
                self._set_confidence("ticker", "low", raw=candidate)
                return candidate

        logger.warning("extract_ticker(): no ticker found")
        return None

    # ─────────────────────────────────────────────────────────────────────────

    def extract_company(self) -> str | None:
        """
        Extract the company name from the transcript header.

        Search order:
        1. ``"Welcome to the <Company> [Q1/earnings/…]"`` — high confidence.
        2. ``"<Company> [Q1] [YYYY] Earnings Call"`` on the first line.
        3. ``"This is the <Company> Earnings Call"`` — medium confidence.

        Returns
        -------
        str | None
            Cleaned company name, or ``None`` if not found.
        """
        logger.debug("extract_company() — scanning header")

        first_lines = "\n".join(self._header.splitlines()[:10])

        # ── Strategy 1: "Welcome to the <Company> …" ─────────────────────────
        m = _WELCOME_COMPANY.search(first_lines)
        if m:
            name = self._clean_company_name(m.group(1))
            if len(name) >= 3:
                logger.info("Company found via welcome script: %r", name)
                self._set_confidence("company", "high", raw=m.group(1))
                return name

        # ── Strategy 2: First/second line "<Company> Q1 YYYY Earnings Call" ──
        for line in self._header.splitlines()[:5]:
            m = _COMPANY_BEFORE_CALL.match(line.strip())
            if m:
                name = self._clean_company_name(m.group(1))
                if len(name) >= 3:
                    logger.info("Company found via header line: %r", name)
                    self._set_confidence("company", "medium", raw=m.group(1))
                    return name

        # ── Strategy 3: "This is the <Company> Earnings Call" ────────────────
        m = _THIS_IS_COMPANY.search(first_lines)
        if m:
            name = self._clean_company_name(m.group(1))
            if len(name) >= 3:
                logger.info("Company found via 'this is' pattern: %r", name)
                self._set_confidence("company", "medium", raw=m.group(1))
                return name

        logger.warning("extract_company(): no company name found")
        return None

    # ─────────────────────────────────────────────────────────────────────────

    def extract_quarter(self) -> str | None:
        """
        Extract the fiscal quarter from the transcript.

        Patterns tried in order (most → least specific):

        1. ``Q<n>FY<yy>`` combined pattern (also caches the year).
        2. Bloomberg-style ``1Q25`` (also caches the year).
        3. Standard ``Q1`` notation (with optional "fiscal" prefix).
        4. Written ordinals: "first quarter", "second quarter", etc.

        The header region is scanned first; the full text is used as fallback.

        Returns
        -------
        str | None
            Canonical quarter string ``"Q1"`` … ``"Q4"``, or ``None``.
        """
        logger.debug("extract_quarter() — scanning header then full text")

        def _search(text: str, confidence: str) -> str | None:
            # ── Pattern 1: Q<n>FY<yy> — most specific ────────────────────────
            m = QuarterPatterns.WITH_FY.search(text)
            if m:
                q = f"Q{m.group(1)}"
                yr = self._normalize_fy_year(m.group(2))
                if self._is_plausible_year(yr):
                    self._cached_year_from_quarter = yr
                logger.info("Quarter [WITH_FY] found: %r (year cached: %s)", q, yr)
                self._set_confidence("quarter", confidence, raw=m.group(0))
                return q

            # ── Pattern 2: Bloomberg 1Q25 ─────────────────────────────────────
            m = QuarterPatterns.BLOOMBERG_STYLE.search(text)
            if m:
                q = f"Q{m.group(1)}"
                yr = self._normalize_fy_year(m.group(2))
                if self._is_plausible_year(yr):
                    self._cached_year_from_quarter = yr
                logger.info("Quarter [BLOOMBERG] found: %r (year cached: %s)", q, yr)
                self._set_confidence("quarter", confidence, raw=m.group(0))
                return q

            # ── Pattern 3: Standard Q1 notation ──────────────────────────────
            m = QuarterPatterns.Q_NOTATION.search(text)
            if m:
                q = f"Q{m.group(1)}"
                logger.info("Quarter [Q_NOTATION] found: %r", q)
                self._set_confidence("quarter", confidence, raw=m.group(0))
                return q

            # ── Pattern 4: Written ordinal ────────────────────────────────────
            m = QuarterPatterns.WRITTEN_ORDINAL.search(text)
            if m:
                word = m.group(1).lower()
                q = QUARTER_WORD_MAP.get(word)
                if q:
                    logger.info("Quarter [ORDINAL] found: %r → %r", word, q)
                    self._set_confidence("quarter", confidence, raw=m.group(0))
                    return q

            return None

        # Header first (higher confidence)
        result = _search(self._header, "high")
        if result:
            return result

        # Full text fallback (lower confidence)
        logger.debug("extract_quarter(): header miss — trying full text")
        result = _search(self._text, "low")
        if result:
            return result

        logger.warning("extract_quarter(): no quarter found")
        return None

    # ─────────────────────────────────────────────────────────────────────────

    def extract_year(self) -> int | None:
        """
        Extract the fiscal year from the transcript.

        Sources tried in order:
        1. Year already cached by :meth:`extract_quarter` (free, most reliable).
        2. ``"fiscal year 2025"`` / ``"fiscal 2025"`` pattern.
        3. ``"FY2025"`` / ``"FY25"`` notation.
        4. Standalone 4-digit year in header (with plausibility guard).

        Returns
        -------
        int | None
            4-digit fiscal year integer, e.g. ``2025``, or ``None``.
        """
        logger.debug("extract_year()")

        # ── Source 0: Free from quarter extraction ────────────────────────────
        if self._cached_year_from_quarter is not None:
            yr = self._cached_year_from_quarter
            logger.info("Year supplied by extract_quarter cache: %d", yr)
            self._set_confidence("year", "high", raw=str(yr))
            return yr

        def _search(text: str, confidence: str) -> int | None:
            # ── Fiscal written: "fiscal year 2025" ───────────────────────────
            m = YearPatterns.FISCAL_WRITTEN.search(text)
            if m:
                yr = int(m.group(1))
                if self._is_plausible_year(yr):
                    logger.info("Year [FISCAL_WRITTEN] found: %d", yr)
                    self._set_confidence("year", confidence, raw=m.group(0))
                    return yr

            # ── FY notation: FY2025 / FY25 ────────────────────────────────────
            m = YearPatterns.FY_NOTATION.search(text)
            if m:
                yr = self._normalize_fy_year(m.group(1))
                if self._is_plausible_year(yr):
                    logger.info("Year [FY_NOTATION] found: %d", yr)
                    self._set_confidence("year", confidence, raw=m.group(0))
                    return yr

            # ── Four-digit standalone year ────────────────────────────────────
            for m in YearPatterns.FOUR_DIGIT.finditer(text):
                yr = int(m.group(1))
                if self._is_plausible_year(yr):
                    logger.info("Year [FOUR_DIGIT] found: %d", yr)
                    self._set_confidence("year", confidence, raw=m.group(0))
                    return yr

            return None

        result = _search(self._header, "high")
        if result:
            return result

        logger.debug("extract_year(): header miss — trying full text")
        result = _search(self._text, "low")
        if result:
            return result

        logger.warning("extract_year(): no plausible year found")
        return None

    # ─────────────────────────────────────────────────────────────────────────

    def extract_date(self) -> str | None:
        """
        Extract and normalise the earnings call date.

        Tries each pattern in :attr:`DatePatterns.ALL` against the header,
        then falls back to the full text. The raw string matched is handed to
        ``dateutil.parser.parse()`` for robust normalisation.

        Returns
        -------
        str | None
            ISO-8601 date string ``"YYYY-MM-DD"``, or ``None``.
        """
        logger.debug("extract_date()")

        def _try_parse(raw: str, source: str, confidence: str) -> str | None:
            """Attempt dateutil parsing; return ISO string or None."""
            try:
                parsed = _dateutil_parser.parse(raw, dayfirst=False)
                iso = parsed.strftime("%Y-%m-%d")
                yr = parsed.year
                # Sanity-check: year must be plausible for an earnings call
                if not self._is_plausible_year(yr):
                    logger.debug(
                        "Parsed date %r → year %d outside plausible range; skipping",
                        raw, yr,
                    )
                    return None
                logger.info("Date [%s] parsed: %r → %r", source, raw, iso)
                self._set_confidence("date", confidence, raw=raw)
                return iso
            except (_DateParserError, ValueError, OverflowError):
                logger.debug("dateutil could not parse %r; skipping", raw)
                return None

        def _scan(text: str, confidence: str) -> str | None:
            for pattern in DatePatterns.ALL:
                m = pattern.search(text)
                if not m:
                    continue
                raw = m.group(0)
                result = _try_parse(raw, pattern.pattern[:40], confidence)
                if result:
                    return result
            return None

        # Header first
        result = _scan(self._header, "high")
        if result:
            return result

        # Full text fallback
        logger.debug("extract_date(): header miss — trying full text")
        result = _scan(self._text, "low")
        if result:
            return result

        logger.warning("extract_date(): no date found")
        return None

    # ─────────────────────────────────────────────────────────────────────────

    def generate_transcript_id(
        self,
        ticker: str | None,
        quarter: str | None,
        year: int | None,
    ) -> str | None:
        """
        Compose the pipeline primary key from its three components.

        Format: ``<TICKER>_<QUARTER>_<YEAR>``  →  e.g. ``"AAPL_Q1_2025"``

        Parameters
        ----------
        ticker:
            Uppercase ticker string.
        quarter:
            Canonical quarter string (``"Q1"`` … ``"Q4"``).
        year:
            4-digit integer year.

        Returns
        -------
        str | None
            The transcript ID, or ``None`` if any component is missing.
        """
        if not all([ticker, quarter, year]):
            missing = [
                name
                for name, val in (
                    ("ticker", ticker),
                    ("quarter", quarter),
                    ("year", year),
                )
                if not val
            ]
            logger.warning(
                "generate_transcript_id(): cannot build ID — missing fields: %s",
                missing,
            )
            return None

        tid = f"{ticker.upper()}_{quarter.upper()}_{year}"
        logger.info("Transcript ID generated: %r", tid)
        return tid

    # ─────────────────────────────────────────────────────────────────────────

    def extract_all(self) -> TranscriptMetadata:
        """
        Run the full extraction pipeline and return a :class:`TranscriptMetadata`.

        Extraction order is important:
        1. ``extract_quarter()`` first — it may cache the year as a side-effect.
        2. ``extract_year()`` — benefits from the quarter cache.
        3. ``extract_ticker()``
        4. ``extract_company()``
        5. ``extract_date()``
        6. ``generate_transcript_id()``

        Returns
        -------
        TranscriptMetadata
            Fully populated metadata object (some fields may be ``None``
            if extraction failed; check ``meta.confidence`` for details).
        """
        logger.info(
            "extract_all() starting | source=%s",
            self._filename or "<no file>",
        )

        # Ordered extraction pipeline
        quarter = self.extract_quarter()
        year    = self.extract_year()
        ticker  = self.extract_ticker()
        company = self.extract_company()
        date    = self.extract_date()
        tid     = self.generate_transcript_id(ticker, quarter, year)

        # Build result object
        meta = TranscriptMetadata(
            ticker        = ticker,
            company       = company,
            quarter       = quarter,
            year          = year,
            date          = date,
            transcript_id = tid,
            source_file   = self._filename,
            confidence    = dict(self._confidence),
            raw_matches   = dict(self._raw_matches),
        )

        # Summary log
        missing_fields = [
            f for f in ("ticker", "quarter", "year", "transcript_id")
            if getattr(meta, f) is None
        ]
        if missing_fields:
            logger.warning(
                "extract_all() complete — INCOMPLETE metadata; missing: %s | source=%s",
                missing_fields,
                self._filename or "<no file>",
            )
        else:
            logger.info(
                "extract_all() complete — %r | date=%s | company=%r",
                meta.transcript_id,
                meta.date or "unknown",
                meta.company or "unknown",
            )

        return meta


# =============================================================================
# Module-level convenience function
# =============================================================================


def extract_metadata(
    text: str,
    filename: str | Path | None = None,
    *,
    header_lines: int = _HEADER_LINE_COUNT,
    extra_ticker_blocklist: set[str] | None = None,
) -> TranscriptMetadata:
    """
    Convenience wrapper: construct a :class:`MetadataExtractor` and call
    :meth:`~MetadataExtractor.extract_all` in one step.

    Parameters
    ----------
    text:
        Raw transcript text.
    filename:
        Optional source filename for filename-based ticker hints.
    header_lines:
        Lines to treat as the header region (default: 35).
    extra_ticker_blocklist:
        Extra uppercase strings to exclude from ticker candidates.

    Returns
    -------
    TranscriptMetadata

    Examples
    --------
    >>> from src.extraction.metadata_extractor import extract_metadata
    >>> meta = extract_metadata(text, filename="AAPL_Q1_2025.txt")
    >>> meta.transcript_id
    'AAPL_Q1_2025'
    """
    return MetadataExtractor(
        text=text,
        filename=filename,
        header_lines=header_lines,
        extra_ticker_blocklist=extra_ticker_blocklist,
    ).extract_all()


# =============================================================================
# Standalone self-test  (python metadata_extractor.py)
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import json

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    # ── Synthetic transcript samples ──────────────────────────────────────────
    _SAMPLES: list[dict[str, str | None]] = [
        {
            "label": "Apple Q1 FY2025 — full header",
            "filename": "AAPL_Q1_2025_earnings.txt",
            "expect_id": "AAPL_Q1_2025",
            "text": (
                "Apple Inc. (AAPL) Q1 FY2025 Earnings Call\n"
                "January 30, 2025\n"
                "\n"
                "Operator\n"
                "Welcome to the Apple Inc. first quarter fiscal 2025 earnings conference call.\n"
                "\n"
                "Tim Cook -- CEO\n"
                "Good afternoon everyone. We had a tremendous Q1 FY2025.\n"
                "Revenue for the first quarter of fiscal 2025 was $124.3 billion.\n"
            ),
        },
        {
            "label": "Microsoft Q3 2024 — exchange prefix",
            "filename": None,
            "expect_id": "MSFT_Q3_2024",
            "text": (
                "NASDAQ: MSFT — Third Quarter Fiscal 2024 Earnings Call\n"
                "April 25, 2024\n"
                "\n"
                "Operator\n"
                "Welcome to the Microsoft Corporation Q3 2024 Earnings Conference.\n"
                "\n"
                "Satya Nadella -- CEO\n"
                "Good afternoon. Q3 was a strong quarter driven by cloud.\n"
            ),
        },
        {
            "label": "NVIDIA Q4 FY25 — Bloomberg-style filename",
            "filename": "NVDA_4Q2025_transcript.txt",
            "expect_id": "NVDA_Q4_2025",
            "text": (
                "NVIDIA Corporation — 4Q FY25 Earnings Call\n"
                "February 26, 2025\n"
                "\n"
                "Operator: Ladies and gentlemen, welcome to NVIDIA's fourth quarter fiscal 2025 call.\n"
                "\n"
                "Jensen Huang -- CEO\n"
                "The age of AI has arrived. 4Q FY25 revenue was $39.3 billion.\n"
            ),
        },
        {
            "label": "Tesla Q2 2024 — written ordinal, no filename",
            "filename": None,
            "expect_id": "TSLA_Q2_2024",
            "text": (
                "Tesla, Inc. (TSLA)\n"
                "Second Quarter 2024 Earnings Call\n"
                "July 23, 2024\n"
                "\n"
                "Operator\n"
                "Welcome to Tesla's second quarter 2024 earnings call.\n"
                "\n"
                "Elon Musk -- CEO\n"
                "Q2 was marked by strong delivery numbers.\n"
            ),
        },
    ]

    all_pass = True
    print("\n" + "=" * 70)
    print("MetadataExtractor — Self-Test")
    print("=" * 70 + "\n")

    for sample in _SAMPLES:
        print(f"── {sample['label']} {'─' * (45 - len(str(sample['label'])))}")
        extractor = MetadataExtractor(
            text=str(sample["text"]),
            filename=sample.get("filename"),
        )
        meta = extractor.extract_all()
        passed = meta.transcript_id == sample["expect_id"]
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}  expected={sample['expect_id']!r}  got={meta.transcript_id!r}")
        print(f"         ticker={meta.ticker!r}  quarter={meta.quarter!r}  year={meta.year!r}")
        print(f"         date={meta.date!r}  company={meta.company!r}")
        print(f"         confidence={json.dumps(meta.confidence)}")
        print()

    print("=" * 70)
    print(f"Result: {'ALL PASSED ✓' if all_pass else 'FAILURES DETECTED ✗'}")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)
