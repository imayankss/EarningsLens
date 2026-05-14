"""
src/utils/regex_patterns.py
============================
Centralized regex pattern registry for the financial NLP earnings call pipeline.

This module is the single source of truth for ALL regular expressions used across:
    - metadata_extractor.py
    - speaker_extractor.py
    - role_classifier.py
    - transcript_structurer.py
    - validators.py

Design principles:
    - No extraction logic here — patterns only.
    - All patterns are pre-compiled with re.compile() for performance.
    - Patterns are grouped by semantic category.
    - Each group is documented with examples of what it matches.
    - Constants, keyword lists, and normalization mappings live here too.

Python: 3.11+
Author: Financial NLP Pipeline — Day 4
"""

from __future__ import annotations

import re
from typing import Final

# =============================================================================
# RE FLAGS — shared flag combos used across multiple pattern groups
# =============================================================================

# Case-insensitive + multiline (most common combo)
_CI = re.IGNORECASE
_CI_ML = re.IGNORECASE | re.MULTILINE
_CI_ML_DOT = re.IGNORECASE | re.MULTILINE | re.DOTALL


# =============================================================================
# ── SECTION 1: SPEAKER DETECTION PATTERNS
# =============================================================================
#
# Earnings call transcripts use many different conventions to introduce
# a new speaker. We need to capture all common formats.
#
# Format examples this group handles:
#
#   Tim Cook -- CEO                     (double dash)
#   Luca Maestri - CFO                  (single dash)
#   John Doe: Analyst, Goldman Sachs    (colon)
#   Operator                            (single-word role line)
#   JOHN DOE - GOLDMAN SACHS            (all-caps variant)
#   Tim Cook, Apple Inc - CEO           (company interposed)
#   CEO - Tim Cook                      (role-first variant)
#
# Capture groups (where present):
#   Group 1 → speaker name (or role in role-first patterns)
#   Group 2 → role / affiliation string
# =============================================================================

class SpeakerPatterns:
    """
    Compiled patterns for identifying speaker introduction lines.

    All patterns are anchored to the start of a line so they don't
    accidentally match mid-sentence name mentions.
    """

    # NAME -- ROLE  /  NAME — ROLE  (em-dash, en-dash, double-hyphen)
    # Matches: "Tim Cook -- CEO", "Luca Maestri — CFO"
    NAME_DOUBLE_DASH: Final = re.compile(
        r'^([A-Z][a-záéíóúñ\-\.]+(?:\s[A-Z][a-záéíóúñ\-\.]+){1,4})'   # Full name (2–5 parts)
        r'\s*(?:--|—|–)\s*'                                               # separator: --, —, –
        r'(.+?)\s*$',                                                     # Role/affiliation
        _CI_ML,
    )

    # NAME - ROLE  (single hyphen — must be careful not to match mid-word hyphens)
    # Matches: "John Smith - Chief Financial Officer", "Jane Doe - Morgan Stanley"
    NAME_SINGLE_DASH: Final = re.compile(
        r'^([A-Z][a-záéíóúñ\-\.]+(?:\s[A-Z][a-záéíóúñ\-\.]+){1,4})'   # Full name
        r'\s+-\s+'                                                        # single dash with spaces
        r'([A-Z].+?)\s*$',                                                # Role starts with capital
        _CI_ML,
    )

    # NAME: ROLE  (colon separator)
    # Matches: "Tim Cook: CEO", "John Doe: Analyst"
    NAME_COLON: Final = re.compile(
        r'^([A-Z][a-záéíóúñ\-\.]+(?:\s[A-Z][a-záéíóúñ\-\.]+){1,4})'   # Full name
        r'\s*:\s*'                                                        # colon
        r'([A-Za-z].+?)\s*$',                                            # Role text
        _CI_ML,
    )

    # ROLE - NAME  (role-first variant, less common)
    # Matches: "CEO - Tim Cook", "Analyst - Jane Doe"
    ROLE_FIRST: Final = re.compile(
        r'^(CEO|CFO|COO|CTO|President|Analyst|Operator|Chairman|Director)'
        r'\s*[-–—]\s*'
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\s*$',
        _CI_ML,
    )

    # Single-word role lines (Operator only, on its own line)
    # Matches: "Operator", "OPERATOR"
    OPERATOR_STANDALONE: Final = re.compile(
        r'^\s*Operator\s*$',
        _CI_ML,
    )

    # Analyst with firm affiliation: "John Doe - Goldman Sachs"
    # (no explicit "Analyst" keyword — must be classified by context)
    NAME_WITH_FIRM: Final = re.compile(
        r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})'                        # Analyst name
        r'\s*[-–—]\s*'
        r'([A-Z][A-Za-z\s&,\.]+(?:Sachs|Stanley|Capital|Partners|'      # Known firm tokens
        r'Research|Securities|Group|Advisors|Asset|Bank|Trust|'
        r'Management|Investments|Equity|Markets|Financial|Fund).*?)'
        r'\s*$',
        _CI_ML,
    )

    # Ordered list of all speaker-line patterns for sequential matching.
    # Extractors should iterate this list and use the first match.
    ALL: Final[list[re.Pattern]] = [
        NAME_DOUBLE_DASH,
        NAME_SINGLE_DASH,
        NAME_COLON,
        ROLE_FIRST,
        OPERATOR_STANDALONE,
        NAME_WITH_FIRM,
    ]


# =============================================================================
# ── SECTION 2: EXECUTIVE ROLE DETECTION PATTERNS
# =============================================================================
#
# Used to identify whether a role string belongs to company management.
# These match the *raw* role text that follows the speaker's name.
#
# Examples matched:
#   "Chief Executive Officer"
#   "President and CEO"
#   "Executive Vice President, Finance"
#   "SVP & General Counsel"
# =============================================================================

class ExecutiveRolePatterns:
    """Compiled patterns for detecting C-suite and management roles."""

    CEO: Final = re.compile(
        r'\b(chief\s+executive\s+officer|president\s*(?:&|and)?\s*ceo|ceo)\b',
        _CI,
    )

    CFO: Final = re.compile(
        r'\b(chief\s+financial\s+officer|cfo)\b',
        _CI,
    )

    COO: Final = re.compile(
        r'\b(chief\s+operating\s+officer|coo)\b',
        _CI,
    )

    CTO: Final = re.compile(
        r'\b(chief\s+technology\s+officer|chief\s+technical\s+officer|cto)\b',
        _CI,
    )

    CMO: Final = re.compile(
        r'\b(chief\s+marketing\s+officer|cmo)\b',
        _CI,
    )

    CRO: Final = re.compile(
        r'\b(chief\s+revenue\s+officer|chief\s+risk\s+officer|cro)\b',
        _CI,
    )

    PRESIDENT: Final = re.compile(
        r'\bpresident\b(?!\s*(?:&|and)\s*ceo)',  # President alone, not "President & CEO" (→ CEO)
        _CI,
    )

    VP: Final = re.compile(
        r'\b(vice\s+president|evp|svp|avp|vp)\b',
        _CI,
    )

    CHAIRMAN: Final = re.compile(
        r'\b(chairman|chairwoman|chair(?:man)?\s+of\s+the\s+board)\b',
        _CI,
    )

    DIRECTOR: Final = re.compile(
        r'\b((?:managing\s+)?director|board\s+member|board\s+of\s+directors)\b',
        _CI,
    )

    GENERAL_COUNSEL: Final = re.compile(
        r'\b(general\s+counsel|chief\s+legal\s+officer|clo|clco)\b',
        _CI,
    )

    IR: Final = re.compile(
        r'\b(investor\s+relations?|ir\s+(?:officer|director|manager|contact))\b',
        _CI,
    )

    # Catch-all for any "Chief X Officer" not captured above
    GENERIC_C_SUITE: Final = re.compile(
        r'\bchief\s+\w+(?:\s+\w+)?\s+officer\b',
        _CI,
    )


# =============================================================================
# ── SECTION 3: ANALYST DETECTION PATTERNS
# =============================================================================
#
# Identifies external sell-side / buy-side analysts on Q&A calls.
#
# Examples matched:
#   "Analyst"
#   "John Doe - Goldman Sachs"
#   "Jane Smith, Morgan Stanley"
#   "Research Analyst"
# =============================================================================

class AnalystPatterns:
    """Compiled patterns for identifying analyst speakers."""

    # Explicit "Analyst" keyword in role field
    EXPLICIT_ANALYST: Final = re.compile(
        r'\b((?:senior\s+|equity\s+|research\s+|buy.?side\s+|sell.?side\s+)?analyst)\b',
        _CI,
    )

    # Known investment bank / research firm names
    # Expand this list as new firms appear in your corpus.
    KNOWN_FIRMS: Final = re.compile(
        r'\b('
        r'Goldman\s*Sachs|Morgan\s*Stanley|JP\s*Morgan|JPMorgan|'
        r'Bank\s*of\s*America|BofA|Merrill\s*Lynch|'
        r'Citigroup|Citi(?:bank)?|Wells\s*Fargo|'
        r'Barclays|UBS|Deutsche\s*Bank|Credit\s*Suisse|'
        r'HSBC|RBC|Royal\s*Bank(?:\s*of\s*Canada)?|'
        r'Jefferies|Cowen|Piper\s*Sandler|Stifel|'
        r'Evercore|Lazard|Guggenheim|Raymond\s*James|'
        r'Oppenheimer|Mizuho|BTIG|Needham|Canaccord|'
        r'Bernstein|Sanford\s*(?:C\s*)?Bernstein|'
        r'Wolfe\s*Research|MoffettNathanson|'
        r'KeyBanc|Truist|Baird|D\.A\.\s*Davidson|'
        r'Loop\s*Capital|Wedbush|Susquehanna'
        r')\b',
        _CI,
    )

    # "Research" alone in role context often signals analyst
    RESEARCH_SIGNAL: Final = re.compile(
        r'\b(research|coverage|covering)\b',
        _CI,
    )


# =============================================================================
# ── SECTION 4: OPERATOR DETECTION PATTERNS
# =============================================================================
#
# Conference call operators follow strict scripted language.
# We detect them by role label AND characteristic phrases.
#
# Examples matched:
#   "Operator"
#   "Ladies and gentlemen, thank you for standing by."
#   "Your next question comes from..."
#   "Please go ahead."
# =============================================================================

class OperatorPatterns:
    """Compiled patterns for identifying operator speech."""

    # Role label
    ROLE_LABEL: Final = re.compile(
        r'^\s*operator\s*$',
        _CI_ML,
    )

    # Characteristic scripted phrases
    NEXT_QUESTION: Final = re.compile(
        r'\b(your\s+(?:next\s+)?question\s+(?:comes?\s+from|is\s+from)|'
        r'we\s+(?:will\s+)?now\s+take\s+(?:a\s+)?question|'
        r'(?:our\s+)?next\s+question\s+(?:is\s+from|comes?\s+from))\b',
        _CI,
    )

    OPENING_SCRIPT: Final = re.compile(
        r'\b(ladies\s+and\s+gentlemen|good\s+(?:morning|afternoon|evening),?\s+'
        r'(?:and\s+)?(?:welcome|thank\s+you\s+for\s+(?:joining|standing\s+by)))\b',
        _CI,
    )

    PLEASE_PROCEED: Final = re.compile(
        r'\b(please\s+(?:go\s+ahead|proceed|state\s+your\s+(?:name|company|firm))|'
        r'you\s+may\s+(?:proceed|begin|ask\s+your\s+question))\b',
        _CI,
    )

    HOLD_PLEASE: Final = re.compile(
        r'\b(please\s+hold|one\s+moment\s+please|we\s+will\s+pause|'
        r'this\s+concludes|thank\s+you\s+for\s+participating)\b',
        _CI,
    )


# =============================================================================
# ── SECTION 5: FISCAL QUARTER PATTERNS
# =============================================================================
#
# Earnings transcripts reference fiscal quarters in many forms.
#
# Examples matched:
#   "Q1", "Q2", "Q3", "Q4"
#   "first quarter", "second quarter", "third quarter", "fourth quarter"
#   "1Q25", "2Q2025"                      (Bloomberg-style)
#   "fiscal Q3", "fiscal first quarter"
# =============================================================================

class QuarterPatterns:
    """Compiled patterns for extracting fiscal quarter references."""

    # Standard Q-notation:  Q1, Q2, Q3, Q4  (with optional "fiscal" prefix)
    Q_NOTATION: Final = re.compile(
        r'\b(?:fiscal\s+|FY\s*)?Q([1-4])\b',
        _CI,
    )

    # Written-out ordinals
    WRITTEN_ORDINAL: Final = re.compile(
        r'\b(?:fiscal\s+)?'
        r'(first|second|third|fourth)\s+quarter\b',
        _CI,
    )

    # Bloomberg / buyside format: 1Q25, 2Q2024
    BLOOMBERG_STYLE: Final = re.compile(
        r'\b([1-4])Q(\d{2}|\d{4})\b',
        _CI,
    )

    # Fiscal year end quarter reference: "Q4 FY2025", "Q1FY25"
    WITH_FY: Final = re.compile(
        r'\bQ([1-4])\s*FY\s*(\d{2,4})\b',
        _CI,
    )

    # Ordered by specificity — try more specific patterns first
    ALL: Final[list[re.Pattern]] = [
        WITH_FY,
        BLOOMBERG_STYLE,
        Q_NOTATION,
        WRITTEN_ORDINAL,
    ]


# =============================================================================
# ── SECTION 6: FISCAL YEAR PATTERNS
# =============================================================================
#
# Examples matched:
#   "2025", "2024"
#   "fiscal 2025", "FY2025", "FY25"
#   "fiscal year 2025"
# =============================================================================

class YearPatterns:
    """Compiled patterns for extracting fiscal/calendar year references."""

    # Full 4-digit year in range [2000, 2040]
    FOUR_DIGIT: Final = re.compile(
        r'\b(20[0-3][0-9]|20[4][0])\b',
    )

    # FY notation with 2 or 4 digit year: FY2025, FY25
    FY_NOTATION: Final = re.compile(
        r'\bFY\s*(\d{2,4})\b',
        _CI,
    )

    # Written out: "fiscal 2025", "fiscal year 2025"
    FISCAL_WRITTEN: Final = re.compile(
        r'\bfiscal\s+(?:year\s+)?(20[0-3][0-9])\b',
        _CI,
    )

    ALL: Final[list[re.Pattern]] = [
        FISCAL_WRITTEN,
        FY_NOTATION,
        FOUR_DIGIT,
    ]


# =============================================================================
# ── SECTION 7: TICKER SYMBOL PATTERNS
# =============================================================================
#
# Stock tickers appear in several contexts inside transcripts:
#   - Transcript header lines: "AAPL — Q1 2025 Earnings Call"
#   - Filename: "AAPL_Q1_2025_transcript.txt"
#   - Exchange prefix: "NASDAQ: AAPL", "NYSE: JPM"
#
# Tickers are 1–5 uppercase letters (US equities); BRK.B style also supported.
# =============================================================================

class TickerPatterns:
    """Compiled patterns for extracting stock ticker symbols."""

    # Preceded by exchange name: "NASDAQ: AAPL", "NYSE:MSFT"
    EXCHANGE_PREFIX: Final = re.compile(
        r'\b(?:NYSE|NASDAQ|AMEX|OTC)\s*:\s*([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b',
    )

    # Ticker in filename (word boundary, uppercase-only token 1–5 chars)
    IN_FILENAME: Final = re.compile(
        r'(?:^|[_\-/\\])([A-Z]{1,5})(?:_|\-|\.)',
    )

    # In header: isolated uppercase word 1–5 chars (must be used with known ticker list
    # or strong surrounding context to avoid false positives like "Q1", "CEO")
    HEADER_ISOLATED: Final = re.compile(
        r'(?<![A-Z])([A-Z]{1,5})(?:\.[A-Z]{1,2})?(?!\w)',
    )

    # BRK.A / BRK.B style class-share tickers
    CLASS_SHARE: Final = re.compile(
        r'\b([A-Z]{1,4}\.[A-Z]{1,2})\b',
    )

    # Ordered: most reliable first
    ALL: Final[list[re.Pattern]] = [
        EXCHANGE_PREFIX,
        CLASS_SHARE,
        IN_FILENAME,
        HEADER_ISOLATED,
    ]


# =============================================================================
# ── SECTION 8: DATE PATTERNS
# =============================================================================
#
# Earnings call dates appear in multiple formats across sources.
#
# Examples matched:
#   "January 30, 2025"
#   "Jan 30, 2025"
#   "01/30/2025"
#   "2025-01-30"             (ISO 8601 — preferred output format)
#   "30 January 2025"
#   "January 30th, 2025"
# =============================================================================

class DatePatterns:
    """Compiled patterns for extracting and recognising date strings."""

    # ISO 8601: 2025-01-30
    ISO_8601: Final = re.compile(
        r'\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b',
    )

    # US format: 01/30/2025  or  01-30-2025
    US_NUMERIC: Final = re.compile(
        r'\b(0[1-9]|1[0-2])[/\-](0[1-9]|[12]\d|3[01])[/\-](20\d{2})\b',
    )

    # Long month name: "January 30, 2025" or "January 30th, 2025"
    LONG_MONTH_DAY_YEAR: Final = re.compile(
        r'\b(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)'
        r'\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(20\d{2})\b',
        _CI,
    )

    # Abbreviated month: "Jan 30, 2025"
    SHORT_MONTH_DAY_YEAR: Final = re.compile(
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(20\d{2})\b',
        _CI,
    )

    # Day-first European: "30 January 2025"
    DAY_FIRST: Final = re.compile(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+'
        r'(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+(20\d{2})\b',
        _CI,
    )

    # Ordered: most unambiguous first
    ALL: Final[list[re.Pattern]] = [
        ISO_8601,
        LONG_MONTH_DAY_YEAR,
        SHORT_MONTH_DAY_YEAR,
        DAY_FIRST,
        US_NUMERIC,
    ]


# =============================================================================
# ── SECTION 9: SECTION HEADER PATTERNS
# =============================================================================
#
# Earnings calls have three canonical sections:
#   1. Prepared Remarks   — scripted management presentation
#   2. Q&A               — analyst question and management answer
#   3. Closing Remarks   — brief wrap-up
#
# These patterns detect the transition lines between sections.
# =============================================================================

class SectionPatterns:
    """Compiled patterns for detecting earnings call section transitions."""

    # ── Prepared Remarks ──
    PREPARED_REMARKS: Final = re.compile(
        r'\b(prepared\s+remarks?|opening\s+(?:remarks?|statement|comments?)|'
        r'management\s+(?:remarks?|discussion|commentary)|'
        r'executive\s+(?:remarks?|commentary)|'
        r'presentation\s+(?:section|portion))\b',
        _CI,
    )

    # ── Q&A Section ──
    QA_SECTION: Final = re.compile(
        r'\b(question(?:s)?\s+and\s+answer(?:s)?|'           # "question and answer(s)"
        r'q\s*(?:and|&)\s*a|q/a|'                            # "Q&A", "Q and A", "Q/A"
        r'question(?:s)?\s+session|'                          # "questions session"
        r'(?:open(?:ing)?\s+(?:the\s+)?)?floor\s+(?:for|to)\s+questions?|'
        r'analyst\s+q(?:\s+and|\s*&\s*)a)\b',                # "analyst Q&A"
        _CI,
    )

    # ── Closing Remarks ──
    CLOSING_REMARKS: Final = re.compile(
        r'\b(closing\s+(?:remarks?|statement(?:s)?|comments?)|'
        r'concluding\s+(?:remarks?|comments?|statement(?:s)?)|'
        r'(?:final|wrap.?up)\s+(?:remarks?|comments?)|'
        r'this\s+concludes\s+(?:the\s+)?(?:prepared\s+)?remarks?)\b',
        _CI,
    )

    # Generic "section header" line — short line in all-caps or title case,
    # used as a weak fallback signal when content alone determines section.
    ALL_CAPS_HEADER: Final = re.compile(
        r'^[A-Z][A-Z\s\-&]{8,60}[A-Z]$',
        _CI_ML,
    )


# =============================================================================
# ── SECTION 10: Q&A DETECTION PATTERNS
# =============================================================================
#
# Specifically detects question-asking behaviour within Q&A blocks.
# Useful for separating "question" turns from "answer" turns.
#
# Examples matched:
#   "Can you talk about..."
#   "I was wondering if you could..."
#   "My question is about..."
#   "Could you provide more color on..."
# =============================================================================

class QAPatterns:
    """Compiled patterns for identifying question vs. answer turns within Q&A."""

    # Classic question opener phrases
    QUESTION_OPENER: Final = re.compile(
        r'\b(my\s+(?:first\s+|second\s+|)?question\s+(?:is|relates?\s+to)|'
        r'can\s+you\s+(?:talk|speak|discuss|provide|give|walk)\s+(?:us\s+)?(?:through|about|on)|'
        r'could\s+you\s+(?:elaborate|expand|clarify|provide|give)|'
        r'i\s+(?:was\s+)?wondering\s+(?:if\s+you\s+could|whether)|'
        r'just\s+(?:a\s+)?(?:quick\s+)?(?:follow.?up|question)|'
        r'wanted\s+to\s+(?:ask|get\s+your\s+(?:thoughts|view|take)))\b',
        _CI,
    )

    # Analyst intro before a question: "Thanks for taking my question."
    ANALYST_INTRO: Final = re.compile(
        r'\b(thank(?:s|\s+you)\s+for\s+taking\s+(?:my\s+)?(?:the\s+)?question|'
        r'(?:hi|hey|hello),?\s+(?:good\s+)?(?:morning|afternoon|evening)|'
        r'(?:thanks|thank\s+you)\s+(?:for\s+the\s+)?(?:opportunity|color|detail))\b',
        _CI,
    )

    # Management answer starter phrases
    ANSWER_OPENER: Final = re.compile(
        r'\b(great\s+question|thank(?:s|\s+you)\s+for\s+(?:that\s+)?question|'
        r'so\s+(?:let\s+me|i\s+would\s+like\s+to|i\'ll)\s+(?:address|take|start)|'
        r'(?:yeah|yes|sure|absolutely),?\s+(?:let\s+me|i\'ll|i\s+can))\b',
        _CI,
    )


# =============================================================================
# ── SECTION 11: TRANSCRIPT ID PATTERN
# =============================================================================
#
# The transcript ID format used throughout this pipeline:
#   AAPL_Q1_2025
#   MSFT_Q4_2024
#
# Validation pattern — does NOT generate IDs (that's metadata_extractor's job).
# =============================================================================

class TranscriptIDPatterns:
    """Patterns for validating and parsing transcript ID strings."""

    # Full transcript ID: TICKER_Q#_YYYY
    FULL_ID: Final = re.compile(
        r'^([A-Z]{1,5}(?:\.[A-Z]{1,2})?)_Q([1-4])_(20\d{2})$',
    )

    # Partial match (use when searching within text, not validating)
    PARTIAL_ID: Final = re.compile(
        r'\b([A-Z]{1,5})_Q([1-4])_(20\d{2})\b',
    )


# =============================================================================
# ── SECTION 12: TEXT QUALITY PATTERNS
# =============================================================================
#
# Used by the validation layer to detect low-quality or boilerplate text.
# =============================================================================

class TextQualityPatterns:
    """Patterns for detecting low-signal text that should be filtered or flagged."""

    # Very short responses (used in validators.py — not a regex, but kept here for cohesion)
    # Minimum word threshold enforced in code; pattern used for pre-filter.
    VERY_SHORT: Final = re.compile(
        r'^\W*(\w+\W+){0,5}\w*\W*$',  # 6 words or fewer
    )

    # Pure boilerplate phrases that add no analytical value
    BOILERPLATE: Final = re.compile(
        r'^(thank(?:s|\s+you)[.,!]*|'
        r'please\s+go\s+ahead[.,!]*|'
        r'sure[.,!]*|'
        r'great[.,!]*|'
        r'ok(?:ay)?[.,!]*|'
        r'yes[.,!]*|'
        r'no\s+(?:further\s+)?questions?[.,!]*)\s*$',
        _CI,
    )

    # Filler / transitional phrases (soft signal — log but don't auto-drop)
    FILLER_PHRASE: Final = re.compile(
        r'\b(you\s+know|kind\s+of|sort\s+of|i\s+mean|like\s+i\s+said|'
        r'going\s+forward|at\s+the\s+end\s+of\s+the\s+day|'
        r'to\s+be\s+honest(?:ly)?|to\s+be\s+fair)\b',
        _CI,
    )

    # Repeated whitespace / formatting artefacts
    EXTRA_WHITESPACE: Final = re.compile(r'\s{2,}')

    # Ellipsis / transcription artefacts
    TRANSCRIPTION_ARTEFACT: Final = re.compile(
        r'(\[inaudible\]|\[crosstalk\]|\[laughter\]|\[pause\]|\.{3,})',
        _CI,
    )


# =============================================================================
# ── SECTION 13: HELPER CONSTANTS
# =============================================================================

# ── Quarter word → Q-number mapping ─────────────────────────────────────────
QUARTER_WORD_MAP: Final[dict[str, str]] = {
    "first":  "Q1",
    "second": "Q2",
    "third":  "Q3",
    "fourth": "Q4",
}

# ── 2-digit FY year → 4-digit year (pivot at 50: <50 → 2000s, >=50 → 1900s)
# Used in: metadata_extractor when parsing "FY25" → "2025"
FY_SHORT_YEAR_PIVOT: Final[int] = 50

# ── Minimum word count threshold for a valid speaker text block ──────────────
MIN_SPEAKER_TEXT_WORDS: Final[int] = 10

# ── Minimum character count threshold for a valid speaker text block ─────────
MIN_SPEAKER_TEXT_CHARS: Final[int] = 50

# ── Maximum plausible year value in earnings transcripts ─────────────────────
MAX_PLAUSIBLE_YEAR: Final[int] = 2040
MIN_PLAUSIBLE_YEAR: Final[int] = 2000


# =============================================================================
# ── SECTION 14: ROLE NORMALIZATION MAP
# =============================================================================
#
# Maps raw role strings (lowercased) → normalized short form.
# Used by role_classifier.py — stored here so updates are in one place.
#
# Key   = lowercase raw role text (or substring)
# Value = normalized role label used in the structured schema
# =============================================================================

ROLE_NORMALIZATION_MAP: Final[dict[str, str]] = {
    # CEO variants
    "chief executive officer":            "CEO",
    "president and ceo":                  "CEO",
    "president & ceo":                    "CEO",
    "president/ceo":                      "CEO",
    "ceo":                                "CEO",
    "ceo and president":                  "CEO",

    # CFO variants
    "chief financial officer":            "CFO",
    "cfo":                                "CFO",
    "executive vp and cfo":               "CFO",
    "evp and cfo":                        "CFO",
    "svp and cfo":                        "CFO",

    # COO variants
    "chief operating officer":            "COO",
    "coo":                                "COO",

    # CTO / CIO
    "chief technology officer":           "CTO",
    "chief technical officer":            "CTO",
    "cto":                                "CTO",
    "chief information officer":          "CIO",
    "cio":                                "CIO",

    # CMO
    "chief marketing officer":            "CMO",
    "cmo":                                "CMO",

    # President (standalone)
    "president":                          "President",

    # Chairman
    "chairman":                           "Chairman",
    "chairman of the board":              "Chairman",
    "executive chairman":                 "Chairman",
    "chairwoman":                         "Chairman",

    # VP variants
    "vice president":                     "VP",
    "executive vice president":           "EVP",
    "senior vice president":              "SVP",
    "assistant vice president":           "AVP",
    "vp":                                 "VP",
    "evp":                                "EVP",
    "svp":                                "SVP",

    # Finance / Accounting
    "vp finance":                         "Finance",
    "vp of finance":                      "Finance",
    "svp finance":                        "Finance",
    "head of finance":                    "Finance",
    "controller":                         "Controller",
    "chief accounting officer":           "Controller",

    # IR
    "investor relations":                 "IR",
    "head of investor relations":         "IR",
    "director of investor relations":     "IR",
    "vp investor relations":              "IR",
    "ir officer":                         "IR",

    # Legal
    "general counsel":                    "General Counsel",
    "chief legal officer":                "General Counsel",
    "clo":                                "General Counsel",

    # Sell-side / buy-side
    "analyst":                            "Analyst",
    "equity analyst":                     "Analyst",
    "research analyst":                   "Analyst",
    "senior analyst":                     "Analyst",

    # Conference operator
    "operator":                           "Operator",
    "conference operator":                "Operator",
    "moderator":                          "Operator",

    # Director level
    "director":                           "Director",
    "managing director":                  "Managing Director",
    "md":                                 "Managing Director",
}


# =============================================================================
# ── SECTION 15: SPEAKER TYPE MAP
# =============================================================================
#
# Maps normalized role label → speaker_type category.
# Categories: "management" | "analyst" | "operator" | "unknown"
# =============================================================================

SPEAKER_TYPE_MAP: Final[dict[str, str]] = {
    "CEO":               "management",
    "CFO":               "management",
    "COO":               "management",
    "CTO":               "management",
    "CIO":               "management",
    "CMO":               "management",
    "CRO":               "management",
    "President":         "management",
    "Chairman":          "management",
    "VP":                "management",
    "EVP":               "management",
    "SVP":               "management",
    "AVP":               "management",
    "Finance":           "management",
    "Controller":        "management",
    "IR":                "management",
    "General Counsel":   "management",
    "Director":          "management",
    "Managing Director": "management",   # internal MD; external MD → analyst via firm context

    "Analyst":           "analyst",

    "Operator":          "operator",
}

SPEAKER_TYPE_DEFAULT: Final[str] = "unknown"


# =============================================================================
# ── SECTION 16: SECTION KEYWORD LISTS
# =============================================================================
#
# String keyword lists (non-regex) for fast section detection.
# Used as a pre-filter before compiling regex — avoids regex overhead on
# lines that obviously don't contain section headers.
# =============================================================================

PREPARED_REMARKS_KEYWORDS: Final[list[str]] = [
    "prepared remarks",
    "prepared statement",
    "opening remarks",
    "opening statement",
    "opening comments",
    "management remarks",
    "management discussion",
    "executive remarks",
    "executive commentary",
]

QA_KEYWORDS: Final[list[str]] = [
    "question and answer",
    "questions and answers",
    "q&a",
    "q and a",
    "question session",
    "questions session",
    "open the floor",
    "floor to questions",
    "analyst q&a",
    "analyst questions",
]

CLOSING_KEYWORDS: Final[list[str]] = [
    "closing remarks",
    "closing statement",
    "closing comments",
    "concluding remarks",
    "concluding statement",
    "wrap-up remarks",
    "wrap up remarks",
    "final remarks",
    "this concludes",
]

# Canonical section label values written to structured_transcripts.parquet
SECTION_LABELS: Final[dict[str, str]] = {
    "prepared": "prepared_remarks",
    "qa":       "qa",
    "closing":  "closing_remarks",
    "unknown":  "unknown",
}


# =============================================================================
# MODULE SELF-TEST  (python -m src.utils.regex_patterns)
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import sys

    _SAMPLE_LINES = [
        ("Tim Cook -- CEO",              SpeakerPatterns.NAME_DOUBLE_DASH),
        ("Luca Maestri - CFO",           SpeakerPatterns.NAME_SINGLE_DASH),
        ("John Doe: Analyst",            SpeakerPatterns.NAME_COLON),
        ("Operator",                     SpeakerPatterns.OPERATOR_STANDALONE),
        ("John Doe - Goldman Sachs",     SpeakerPatterns.NAME_WITH_FIRM),
        ("Q1 2025",                      QuarterPatterns.Q_NOTATION),
        ("fiscal first quarter",         QuarterPatterns.WRITTEN_ORDINAL),
        ("NASDAQ: AAPL",                 TickerPatterns.EXCHANGE_PREFIX),
        ("January 30, 2025",             DatePatterns.LONG_MONTH_DAY_YEAR),
        ("2025-01-30",                   DatePatterns.ISO_8601),
        ("AAPL_Q1_2025",                 TranscriptIDPatterns.FULL_ID),
        ("Question and Answer Session",  SectionPatterns.QA_SECTION),
        ("Prepared Remarks",             SectionPatterns.PREPARED_REMARKS),
    ]

    all_pass = True
    for text, pattern in _SAMPLE_LINES:
        m = pattern.search(text)
        status = "✓ PASS" if m else "✗ FAIL"
        if not m:
            all_pass = False
        print(f"  {status}  |  pattern={pattern.pattern[:50]!r:<55} | input={text!r}")

    sys.exit(0 if all_pass else 1)
