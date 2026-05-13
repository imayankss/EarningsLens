from __future__ import annotations

QA_PATTERNS = [
    r"question(?:s)?\\s*(?:and|&)\\s*answer(?:s)?",
    r"q\\s*&\\s*a",
    r"question-and-answer",
    r"questions?\\s+and\\s+answers?",
]


MAX_TOKENS = 420
OVERLAP = 60

from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

"""
src/preprocessing/text_cleaner.py
===================================
Stage 1 — Raw text cleaning and normalisation.

Responsibilities:
    - Unicode normalisation (NFKD)
    - HTML / XBRL tag removal
    - Boilerplate removal (legal disclaimers, operator cues, safe-harbour)
    - Speaker label stripping
    - Whitespace and punctuation normalisation
    - Encoding artefact removal (e.g. â€™ → ')

Design principles:
    - Pure functions only — no state, no side effects
    - Every regex is named and documented
    - All operations are independently testable
    - Returns str → str so stages can be chained via functools.reduce

TODO:
    - [ ] Add language detection (filter non-English transcripts)
    - [ ] Add custom boilerplate pattern loader from config
    - [ ] Handle PDF-extracted text with column-break artefacts
    - [ ] Add redaction of PII (phone numbers, emails)
"""
import re
import unicodedata
from functools import reduce
from typing import Callable

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns — defined at module level for performance
# ---------------------------------------------------------------------------

# HTML / XML tags
_RE_HTML = re.compile(r"<[^>]+>", re.DOTALL)

# XBRL inline tags e.g. <ix:nonNumeric ...>
_RE_XBRL = re.compile(r"</?ix:[^>]+>", re.IGNORECASE | re.DOTALL)

# Boilerplate patterns common across earnings call transcript providers
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)this (?:conference call )?transcript is produced by.*?\."),
    re.compile(r"(?i)this call (?:is being )?recorded.*?\."),
    re.compile(r"(?i)safe.harbor statement.*?(?:forward.looking statements?).*?\."),
    re.compile(r"(?i)forward.looking statements? (?:involve|are subject to).*?\."),
    re.compile(r"(?i)\[operator instructions?\]"),
    re.compile(r"(?i)\[technical difficulty\]"),
    re.compile(r"(?i)operator[:\s]+(?:good (?:morning|afternoon|evening)[^.]*\.)"),
    re.compile(r"(?i)thank you for standing by[^.]*\."),
    re.compile(r"(?i)ladies and gentlemen[^.]*(?:thank you|welcome)[^.]*\."),
    re.compile(r"(?i)this concludes today's (?:conference )?call[^.]*\."),
    re.compile(r"(?i)please go ahead[,.]?"),
    re.compile(r"(?i)your (?:next|first) question comes from[^.]*\."),
    re.compile(r"(?i)we'll now take (?:our )?(?:first|next) question[^.]*\."),
]

# Speaker label patterns: "John Smith - CEO:", "OPERATOR:", "ANALYST:"
_RE_SPEAKER = re.compile(
    r"^[A-Z][A-Za-z \t\-\.]+(?:\s*[-–]\s*[A-Za-z \t,\.&]+)?[: \t]*$",
    re.MULTILINE,
)

# Encoding artefacts from PDF extraction
_ENCODING_FIXES: dict[str, str] = {
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "\u2014": "—",
    "â€¦": "…",
    "\u00a0": " ",  # non-breaking space
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",  # ellipsis
}


class TextCleaner:
    """
    Stateless text cleaning pipeline for earnings call transcripts.

    Each public method is independently usable. The full pipeline
    is run via clean() which chains all stages in the correct order.

    Example:
        cleaner = TextCleaner()
        clean_text = cleaner.clean(raw_text)
    """

    def __init__(self, remove_speaker_labels: bool = True) -> None:
        """
        Args:
            remove_speaker_labels: Strip "SPEAKER NAME:" prefixes if True.
                                   Set False if you need speaker info downstream.
        """
        self.remove_speaker_labels = remove_speaker_labels
        log.debug("TextCleaner initialised")

    # ── Full pipeline ────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Run the full cleaning pipeline on a single text string.

        Pipeline order (order matters — do not rearrange):
            1. Guard against empty / non-string input
            2. Fix encoding artefacts
            3. Remove HTML and XBRL tags
            4. Unicode normalisation
            5. Remove boilerplate
            6. Optionally strip speaker labels
            7. Normalise whitespace

        Args:
            text: Raw transcript text string

        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""

        steps: list[Callable[[str], str]] = [
            self._fix_encoding,
            self._remove_html,
            self._normalise_unicode,
            self._remove_boilerplate,
            self._normalise_whitespace,
        ]

        if self.remove_speaker_labels:
            steps.insert(4, self._remove_speaker_labels)

        return reduce(lambda t, fn: fn(t), steps, text).strip()

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "transcript_text",
        out_col: str = "transcript_text_clean",
    ) -> pd.DataFrame:
        """
        Apply clean() to every row in a DataFrame.

        Args:
            df      : Input DataFrame
            text_col: Column containing raw transcript text
            out_col : Column name for cleaned output
                      (set equal to text_col to overwrite in-place)

        Returns:
            DataFrame with out_col added
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame")

        log.info(f"Cleaning {len(df):,} transcripts (col='{text_col}')...")
        df = df.copy()
        df[out_col] = df[text_col].fillna("").astype(str).apply(self.clean)
        df["char_count_clean"] = df[out_col].str.len()
        df["word_count_clean"] = df[out_col].str.split().str.len()

        reduction = 1 - df["word_count_clean"].sum() / (
            df[text_col].fillna("").str.split().str.len().sum() + 1
        )
        log.info(f"Cleaning complete. Word-count reduction: {reduction:.1%}")
        return df

    # ── Individual cleaning stages ───────────────────────────────

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """Replace known encoding artefacts with correct characters."""
        for bad, good in _ENCODING_FIXES.items():
            text = text.replace(bad, good)
        return text

    @staticmethod
    def _remove_html(text: str) -> str:
        """Strip HTML and XBRL tags."""
        text = _RE_XBRL.sub(" ", text)
        text = _RE_HTML.sub(" ", text)
        return text

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        """NFKD normalisation — decomposes ligatures and special forms."""
        return unicodedata.normalize("NFKD", text)

    @staticmethod
    def _remove_boilerplate(text: str) -> str:
        """Remove legal disclaimers, operator cues, and standard boilerplate."""
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub(" ", text)
        return text

    @staticmethod
    def _remove_speaker_labels(text: str) -> str:
        """
        Strip speaker labels like 'John Smith - CEO:' or 'OPERATOR:'.
        Preserves the spoken content that follows.
        """
        return _RE_SPEAKER.sub("", text)

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Collapse runs of whitespace; normalise line breaks."""
        text = re.sub(r"\r\n|\r", "\n", text)  # normalise line endings
        text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 consecutive newlines
        text = re.sub(r"[ \t]{2,}", " ", text)  # collapse spaces/tabs
        text = re.sub(r" +\n", "\n", text)  # trailing spaces before newline
        return text

    # ── Utility ──────────────────────────────────────────────────

    @staticmethod
    def word_count(text: str) -> int:
        """Return word count of a text string."""
        return len(tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def char_count(text: str) -> int:
        """Return character count of a text string."""
        return len(text)

    # TODO: add detect_language(text) -> str using langdetect
    # TODO: add redact_pii(text) -> str for phone/email removal
    # TODO: add remove_custom_boilerplate(text, patterns: list[str]) -> str


    def separate_sections(self, text: str) -> dict:
        """
        Backward-compatible wrapper for old preprocessing tests.
        Splits transcript into prepared remarks and Q&A sections.
        """

        import re

        qa_patterns = [
            r"question(?:s)?\s*(?:and|&)\s*answer(?:s)?",
            r"q\s*&\s*a",
            r"question-and-answer",
            r"questions?\s+and\s+answers?",
        ]

        split_idx = None

        for pattern in qa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                split_idx = match.start()
                break

        if split_idx is None:
            return {
                "prepared": text.strip(),
                "qa": "",
                "full": text.strip()
            }

        return {
            "prepared": text[:split_idx].strip(),
            "qa": text[split_idx:].strip(),
            "full": text.strip()
        }
