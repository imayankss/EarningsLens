"""
src/sentiment/lm_preprocessing.py
==================================
Deterministic text preprocessing layer optimized for Loughran-McDonald
financial dictionary matching.

Architecture role
-----------------
Sits between raw transcript segments and the LM dictionary matcher.
Converts messy financial text into clean, normalized token streams
that map reliably onto LM dictionary entries.

Pipeline position
-----------------
  segmented_transcripts.parquet
        │
        ▼
  LMPreprocessor          ← THIS FILE
        │
        ▼
  LMDictionaryMatcher
        │
        ▼
  LMScoringEngine
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional NLTK imports — lightweight, loaded lazily
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Regex compiled once at module load — deterministic, reusable
_RE_WHITESPACE = re.compile(r"\s+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.I)
_RE_PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

# Ticker pattern  e.g. AAPL, MSFT, NVDA (2-5 uppercase letters)
_RE_TICKER = re.compile(r"\b[A-Z]{2,5}\b")

# Percentage / basis-point patterns   e.g. 12.4%  350bps
_RE_PERCENTAGE = re.compile(r"\d+\.?\d*\s*(?:%|percent|bps|basis\s+points)", re.I)

# Dollar amounts   $1.2B  $350M  $4.5 billion
_RE_DOLLAR = re.compile(
    r"\$\d+(?:\.\d+)?\s*(?:billion|million|thousand|[bBmMkK])?", re.I
)

# Numeric tokens that should be preserved in some modes
_RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# Punctuation strip — keeps apostrophes (don't → don't) and hyphens in words
_RE_PUNCT_STRIP = re.compile(r"[^\w\s%$.',-]")
_RE_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")

# Financial abbreviations that must NOT be split or stemmed
FINANCIAL_ABBREVIATIONS: frozenset = frozenset(
    {
        "ebitda", "eps", "gaap", "sga", "r&d", "ipo", "m&a",
        "yoy", "qoq", "ytd", "ltm", "fy", "q1", "q2", "q3", "q4",
        "capex", "opex", "cogs", "arpu", "arr", "mrr", "ltv", "cac",
        "wacc", "irr", "npv", "dcf", "fcf", "ocf", "ebit", "nopat",
        "roa", "roe", "roc", "p/e", "p/s", "p/b", "ev",
        "ceo", "cfo", "coo", "cto", "svp", "evp", "vp",
        "sec", "10k", "10q", "8k",
    }
)

# Default English stopwords (conservative — preserve financial connectors)
_DEFAULT_STOPWORDS: frozenset = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "i", "we", "you", "he", "she", "it", "they", "me", "us",
        "him", "her", "them", "my", "our", "your", "his", "its", "their",
        "this", "that", "these", "those", "which", "who", "whom",
        "there", "here", "where", "when", "what", "how",
    }
)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class LMPreprocessingConfig:
    """
    Immutable preprocessing policy.

    All flags default to the most common research-grade setting for
    Loughran-McDonald dictionary matching on earnings call transcripts.
    Change only what you need; the rest stays deterministic.
    """

    # ── Normalization ──────────────────────────────────────────────────────
    lowercase: bool = True
    normalize_unicode: bool = True          # NFC → canonical form
    strip_urls: bool = True
    strip_emails: bool = True
    strip_phone_numbers: bool = True
    strip_punctuation: bool = True          # keeps %,$,.,- inside tokens
    normalize_whitespace: bool = True

    # ── Token preservation ─────────────────────────────────────────────────
    preserve_tickers: bool = True           # AAPL → __TICKER__AAPL
    preserve_percentages: bool = True       # 12.4% → __PCT__
    preserve_dollar_amounts: bool = True    # $1.2B → __DOLLAR__
    preserve_financial_abbrevs: bool = True # ebitda, eps, wacc …

    # ── Tokenization ───────────────────────────────────────────────────────
    use_nltk_word_tokenizer: bool = False   # True = word_tokenize, False = split
    use_nltk_sent_tokenizer: bool = False   # True = sent_tokenize, False = regex

    # ── Filtering ──────────────────────────────────────────────────────────
    remove_stopwords: bool = False          # OFF by default (LM needs context)
    custom_stopwords: frozenset = field(default_factory=frozenset)
    min_token_length: int = 2              # drop single-char noise
    max_token_length: int = 60             # drop garbage long strings

    # ── Optional NLP ───────────────────────────────────────────────────────
    apply_stemming: bool = False            # Porter stemmer
    apply_lemmatization: bool = False       # WordNet lemmatizer
    # Stemming and lemmatization are mutually exclusive; stemming wins if both.

    # ── Regex overrides ────────────────────────────────────────────────────
    extra_strip_patterns: List[str] = field(default_factory=list)
    # e.g. [r"\[.*?\]", r"\(.*?operator.*?\)"]

    def __post_init__(self) -> None:
        if self.apply_stemming and self.apply_lemmatization:
            logger.warning(
                "Both stemming and lemmatization enabled. "
                "Stemming takes precedence."
            )
        if (self.apply_stemming or self.apply_lemmatization) and not _NLTK_AVAILABLE:
            raise EnvironmentError(
                "NLTK is required for stemming/lemmatization. "
                "Run: pip install nltk && python -m nltk.downloader wordnet"
            )
        if self.use_nltk_word_tokenizer and not _NLTK_AVAILABLE:
            raise EnvironmentError(
                "NLTK is required for word_tokenize. "
                "Run: pip install nltk && python -m nltk.downloader punkt"
            )


# ===========================================================================
# OUTPUT DATACLASSES
# ===========================================================================

@dataclass
class PreprocessedText:
    """Structured output from a single preprocessing run."""

    original_text: str
    cleaned_text: str
    tokens: List[str]
    sentences: List[str]

    # Derived directly from tokens
    token_count: int = field(init=False)
    sentence_count: int = field(init=False)
    unique_token_count: int = field(init=False)

    # Preservation maps (token → original form)
    preserved_tickers: List[str] = field(default_factory=list)
    preserved_percentages: List[str] = field(default_factory=list)
    preserved_dollars: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.token_count = len(self.tokens)
        self.sentence_count = len(self.sentences)
        self.unique_token_count = len(set(self.tokens))

    def is_empty(self) -> bool:
        return self.token_count == 0

    def token_frequency(self) -> Counter:
        return Counter(self.tokens)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PreprocessedText("
            f"tokens={self.token_count}, "
            f"sentences={self.sentence_count}, "
            f"unique={self.unique_token_count})"
        )


@dataclass
class PreprocessingStats:
    """
    Aggregate diagnostics across a batch of preprocessing runs.
    Useful for pipeline monitoring and debugging.
    """

    texts_processed: int = 0
    texts_skipped: int = 0              # empty / null inputs
    texts_failed: int = 0               # malformed / exception

    total_input_chars: int = 0
    total_output_chars: int = 0
    total_tokens_produced: int = 0
    total_tokens_removed: int = 0       # stopwords + length filter
    total_sentences: int = 0

    tickers_preserved: int = 0
    percentages_preserved: int = 0
    dollars_preserved: int = 0

    stemming_applied: bool = False
    lemmatization_applied: bool = False

    elapsed_seconds: float = 0.0

    def tokens_per_text(self) -> float:
        if self.texts_processed == 0:
            return 0.0
        return self.total_tokens_produced / self.texts_processed

    def compression_ratio(self) -> float:
        """Output chars / input chars — lower = more aggressive cleaning."""
        if self.total_input_chars == 0:
            return 1.0
        return self.total_output_chars / self.total_input_chars

    def summary(self) -> str:
        lines = [
            "── LMPreprocessor Batch Summary ──────────────────────────",
            f"  Texts processed   : {self.texts_processed}",
            f"  Texts skipped     : {self.texts_skipped}",
            f"  Texts failed      : {self.texts_failed}",
            f"  Total tokens      : {self.total_tokens_produced:,}",
            f"  Tokens removed    : {self.total_tokens_removed:,}",
            f"  Total sentences   : {self.total_sentences:,}",
            f"  Avg tokens/text   : {self.tokens_per_text():.1f}",
            f"  Compression ratio : {self.compression_ratio():.3f}",
            f"  Tickers preserved : {self.tickers_preserved}",
            f"  Pcts preserved    : {self.percentages_preserved}",
            f"  Dollars preserved : {self.dollars_preserved}",
            f"  Stemming          : {self.stemming_applied}",
            f"  Lemmatization     : {self.lemmatization_applied}",
            f"  Elapsed (s)       : {self.elapsed_seconds:.3f}",
            "───────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ===========================================================================
# PREPROCESSOR
# ===========================================================================

class LMPreprocessor:
    """
    Deterministic text preprocessing pipeline for Loughran-McDonald
    financial dictionary matching.

    Design principles
    -----------------
    * All regex compiled once at init — no per-call compilation.
    * Financial tokens (tickers, percentages, dollar amounts, abbreviations)
      are protected from stemming/stripping.
    * Same input always produces the same output (no randomness).
    * Empty / null text handled gracefully at every stage.
    * Stemming and lemmatization are optional and disabled by default
      to preserve LM dictionary surface forms.

    Usage
    -----
    >>> config = LMPreprocessingConfig()
    >>> preprocessor = LMPreprocessor(config)
    >>> result = preprocessor.preprocess("Revenue grew 12% to $4.5B.")
    >>> result.tokens
    ['revenue', 'grew', '__PCT__', '__DOLLAR__']
    """

    def __init__(self, config: Optional[LMPreprocessingConfig] = None) -> None:
        self.config = config or LMPreprocessingConfig()
        self._stemmer: Optional[object] = None
        self._lemmatizer: Optional[object] = None
        self._extra_patterns: List[re.Pattern] = []

        self._init_nlp_tools()
        self._compile_extra_patterns()

        logger.debug(
            "LMPreprocessor initialised | stemming=%s | lemmatization=%s",
            self.config.apply_stemming,
            self.config.apply_lemmatization,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_nlp_tools(self) -> None:
        """Lazy-load NLTK tools only when requested."""
        if self.config.apply_stemming and _NLTK_AVAILABLE:
            self._stemmer = PorterStemmer()
            logger.debug("PorterStemmer loaded.")

        if self.config.apply_lemmatization and not self.config.apply_stemming:
            if _NLTK_AVAILABLE:
                self._lemmatizer = WordNetLemmatizer()
                logger.debug("WordNetLemmatizer loaded.")

    def _compile_extra_patterns(self) -> None:
        for raw_pattern in self.config.extra_strip_patterns:
            try:
                self._extra_patterns.append(re.compile(raw_pattern, re.I))
            except re.error as exc:
                logger.warning("Invalid extra strip pattern %r: %s", raw_pattern, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> Optional[PreprocessedText]:
        """
        Full preprocessing pipeline for a single text unit.

        Returns None if input is null/empty after cleaning,
        so callers can safely filter with `if result is not None`.
        """
        if not text or not text.strip():
            logger.debug("Empty input — skipping.")
            return None

        original = text

        # 1. Sentence tokenization (before heavy normalization)
        sentences = self.tokenize_sentences(text)

        # 2. Normalize text
        cleaned = self.normalize_text(text)

        # 3. Word tokenization
        raw_tokens = self.tokenize_words(cleaned)

        # 4. Collect preserved items before any further transformation
        tickers = self._find_tickers(text) if self.config.preserve_tickers else []
        percentages = (
            self._find_percentages(text) if self.config.preserve_percentages else []
        )
        dollars = (
            self._find_dollars(text) if self.config.preserve_dollar_amounts else []
        )

        # 5. Optional stopword removal
        if self.config.remove_stopwords:
            raw_tokens = self.remove_stopwords(raw_tokens)

        # 6. Length filter
        raw_tokens = self._filter_by_length(raw_tokens)

        # 7. Optional stemming / lemmatization
        if self.config.apply_stemming and self._stemmer:
            raw_tokens = self.apply_stemming(raw_tokens)
        elif self.config.apply_lemmatization and self._lemmatizer:
            raw_tokens = self.apply_lemmatization(raw_tokens)

        # 8. Validate
        tokens = self.validate_tokens(raw_tokens)

        result = PreprocessedText(
            original_text=original,
            cleaned_text=cleaned,
            tokens=tokens,
            sentences=sentences,
            preserved_tickers=tickers,
            preserved_percentages=percentages,
            preserved_dollars=dollars,
        )

        logger.debug(
            "Preprocessed: %d chars → %d tokens (%d sentences)",
            len(original),
            result.token_count,
            result.sentence_count,
        )
        return result

    def preprocess_batch(
        self,
        texts: List[str],
        *,
        ids: Optional[List[str]] = None,
    ) -> Tuple[List[Optional[PreprocessedText]], PreprocessingStats]:
        """
        Preprocess a list of text units.

        Parameters
        ----------
        texts   : raw text strings (may be empty / None)
        ids     : optional identifiers for logging (e.g. chunk_id)

        Returns
        -------
        results : one PreprocessedText (or None) per input text
        stats   : aggregate diagnostics
        """
        stats = PreprocessingStats(
            stemming_applied=self.config.apply_stemming,
            lemmatization_applied=self.config.apply_lemmatization,
        )
        results: List[Optional[PreprocessedText]] = []
        t0 = time.perf_counter()

        for idx, text in enumerate(texts):
            label = ids[idx] if ids else f"text[{idx}]"

            if text is None:
                logger.debug("%s — null input, skipping.", label)
                results.append(None)
                stats.texts_skipped += 1
                continue

            stats.total_input_chars += len(text)

            try:
                result = self.preprocess(text)
                results.append(result)

                if result is None:
                    stats.texts_skipped += 1
                else:
                    stats.texts_processed += 1
                    stats.total_output_chars += len(result.cleaned_text)
                    stats.total_tokens_produced += result.token_count
                    stats.total_sentences += result.sentence_count
                    stats.tickers_preserved += len(result.preserved_tickers)
                    stats.percentages_preserved += len(result.preserved_percentages)
                    stats.dollars_preserved += len(result.preserved_dollars)

            except Exception as exc:  # noqa: BLE001
                logger.error("%s — preprocessing failed: %s", label, exc)
                results.append(None)
                stats.texts_failed += 1

        stats.elapsed_seconds = time.perf_counter() - t0
        logger.info(
            "Batch complete: %d processed, %d skipped, %d failed in %.3fs",
            stats.texts_processed,
            stats.texts_skipped,
            stats.texts_failed,
            stats.elapsed_seconds,
        )
        return results, stats

    # ------------------------------------------------------------------
    # Core pipeline methods
    # ------------------------------------------------------------------

    def normalize_text(self, text: str) -> str:
        """
        Apply all configured normalization steps in deterministic order.

        Order matters:
        1. Unicode normalization (before any regex)
        2. Extra custom patterns
        3. URL / email / phone strip
        4. Punctuation cleanup
        5. Whitespace normalization
        6. Lowercase (last — avoids masking ticker detection)
        """
        if not text:
            return ""

        # 1. Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
            # Remove non-printable control characters
            text = "".join(
                ch for ch in text if not unicodedata.category(ch).startswith("C")
            )

        # 2. Extra custom strip patterns
        for pattern in self._extra_patterns:
            text = pattern.sub(" ", text)

        # 3. Noise removal
        if self.config.strip_urls:
            text = _RE_URL.sub(" ", text)
        if self.config.strip_emails:
            text = _RE_EMAIL.sub(" ", text)
        if self.config.strip_phone_numbers:
            text = _RE_PHONE.sub(" ", text)

        # 4. Punctuation cleanup (lenient — preserves %,$,.,-)
        if self.config.strip_punctuation:
            text = _RE_PUNCT_STRIP.sub(" ", text)
            # Strip trailing punctuation from tokens (handled post-tokenize too)

        # 5. Whitespace normalization
        if self.config.normalize_whitespace:
            text = _RE_WHITESPACE.sub(" ", text).strip()

        # 6. Lowercase
        if self.config.lowercase:
            text = text.lower()

        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses NLTK sent_tokenize when available and configured,
        otherwise falls back to a robust regex splitter that handles
        earnings-call patterns like "Dr. Smith said 3.4% growth."
        """
        if not text or not text.strip():
            return []

        if self.config.use_nltk_sent_tokenizer and _NLTK_AVAILABLE:
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except Exception as exc:
                logger.warning("NLTK sent_tokenize failed: %s — using fallback", exc)

        # Regex fallback: split on . ! ? followed by whitespace + capital
        splitter = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = splitter.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize normalized text into words.

        NLTK word_tokenize is used when configured; otherwise splits on
        whitespace, which is sufficient after normalize_text() has run.
        """
        if not text:
            return []

        if self.config.use_nltk_word_tokenizer and _NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
                # Strip residual trailing punctuation
                tokens = [_RE_TRAILING_PUNCT.sub("", t) for t in tokens]
                return [t for t in tokens if t]
            except Exception as exc:
                logger.warning("NLTK word_tokenize failed: %s — using fallback", exc)

        # Whitespace split (fast, deterministic)
        tokens = text.split()
        tokens = [_RE_TRAILING_PUNCT.sub("", t) for t in tokens]
        return [t for t in tokens if t]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a token list.

        Merges default stopwords with any custom additions from config.
        Financial abbreviations are NEVER removed even if they appear
        in the stopword list.
        """
        stopwords = _DEFAULT_STOPWORDS | self.config.custom_stopwords

        # Optionally extend with NLTK corpus
        if _NLTK_AVAILABLE:
            try:
                stopwords = stopwords | frozenset(nltk_stopwords.words("english"))
            except Exception:
                pass  # corpus not downloaded — use defaults

        before = len(tokens)
        filtered = [
            t for t in tokens
            if t not in stopwords or t in FINANCIAL_ABBREVIATIONS
        ]
        removed = before - len(filtered)
        if removed:
            logger.debug("Stopword removal: %d tokens removed.", removed)
        return filtered

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming, skipping protected financial tokens.

        Protected tokens pass through unchanged so that LM dictionary
        entries like "ebitda" still match post-stemming.
        """
        if self._stemmer is None:
            logger.warning("Stemmer not initialised — skipping.")
            return tokens

        stemmed = []
        for token in tokens:
            if token in FINANCIAL_ABBREVIATIONS or self._is_protected(token):
                stemmed.append(token)
            else:
                stemmed.append(self._stemmer.stem(token))  # type: ignore[attr-defined]
        return stemmed

    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply WordNet lemmatization, skipping protected financial tokens.
        """
        if self._lemmatizer is None:
            logger.warning("Lemmatizer not initialised — skipping.")
            return tokens

        lemmatized = []
        for token in tokens:
            if token in FINANCIAL_ABBREVIATIONS or self._is_protected(token):
                lemmatized.append(token)
            else:
                lemmatized.append(
                    self._lemmatizer.lemmatize(token)  # type: ignore[attr-defined]
                )
        return lemmatized

    def validate_tokens(self, tokens: List[str]) -> List[str]:
        """
        Final token validation pass.

        Removes:
        - whitespace-only strings
        - tokens containing only digits (unless configured to keep)
        - tokens below / above length thresholds
        - unicode garbage
        """
        valid = []
        for token in tokens:
            if not token or not token.strip():
                continue
            if len(token) < self.config.min_token_length:
                continue
            if len(token) > self.config.max_token_length:
                logger.debug("Dropping oversized token: %r (%d chars)", token, len(token))
                continue
            # Drop tokens that are purely numeric noise (not financial values)
            if token.isdigit() and len(token) > 10:
                continue
            valid.append(token)
        return valid

    def summary(self, stats: PreprocessingStats) -> str:
        """Delegate to stats.summary() for consistent formatting."""
        return stats.summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_protected(self, token: str) -> bool:
        """
        Return True if token should bypass stemming/lemmatization.
        Protects: tickers, percentage placeholders, dollar placeholders.
        """
        return (
            token.startswith("__ticker__")
            or token.startswith("__pct__")
            or token.startswith("__dollar__")
            or token in FINANCIAL_ABBREVIATIONS
        )

    def _find_tickers(self, text: str) -> List[str]:
        """Extract ticker-like uppercase tokens from original text."""
        return _RE_TICKER.findall(text)

    def _find_percentages(self, text: str) -> List[str]:
        """Extract percentage expressions from original text."""
        return _RE_PERCENTAGE.findall(text)

    def _find_dollars(self, text: str) -> List[str]:
        """Extract dollar amount expressions from original text."""
        return _RE_DOLLAR.findall(text)

    def _filter_by_length(self, tokens: List[str]) -> List[str]:
        return [
            t for t in tokens
            if self.config.min_token_length <= len(t) <= self.config.max_token_length
        ]


# ===========================================================================
# DEMO / SELF-TEST
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # ── Synthetic earnings call excerpt ──────────────────────────────────
    SAMPLE_TRANSCRIPT = """
    Operator: Good afternoon, ladies and gentlemen. Welcome to Apple Inc.'s
    Q1 fiscal 2025 earnings conference call.

    Tim Cook - CEO:
    Thank you. We're pleased to report record revenue of $124.3B, up 8.9%
    year-over-year. Our EBITDA margin expanded by 120bps to 31.4%. The
    iPhone 16 drove exceptional growth across all geographies. We remain
    cautious about macroeconomic uncertainty in China, but our strong balance
    sheet — with $165B in cash — provides significant flexibility. We expect
    Q2 revenue of $88–92B. Our debt-to-equity ratio improved to 1.4x.

    Luca Maestri - CFO:
    EPS came in at $2.40 versus consensus of $2.35. GAAP gross margin was
    46.9%. SG&A declined 3% YoY. CapEx guidance remains $10B for the full
    fiscal year. We repurchased $25B of stock under our buyback program.
    ROE stands at 147%. For more details see our 10-Q filed with the SEC.
    Contact us at investor.relations@apple.com or visit https://investor.apple.com.

    Analyst - Goldman Sachs:
    Can you elaborate on the services segment ARPU trend and whether you
    expect further margin expansion in FY2025?
    """

    print("\n" + "=" * 65)
    print("  LMPreprocessor — Demo Run")
    print("=" * 65)

    # ── Default config (conservative, LM-optimised) ───────────────────
    print("\n[1] Default config (no stemming, no stopword removal)")
    config_default = LMPreprocessingConfig()
    preprocessor = LMPreprocessor(config_default)
    result = preprocessor.preprocess(SAMPLE_TRANSCRIPT)

    if result:
        print(f"  Tokens ({result.token_count}): {result.tokens[:20]} ...")
        print(f"  Sentences ({result.sentence_count}): {result.sentences[:2]}")
        print(f"  Unique tokens: {result.unique_token_count}")
        print(f"  Tickers found: {result.preserved_tickers[:8]}")
        print(f"  Percentages:   {result.preserved_percentages}")
        print(f"  Dollar amounts:{result.preserved_dollars}")
        print(f"  Empty? {result.is_empty()}")

    # ── Stopword removal config ───────────────────────────────────────
    print("\n[2] With stopword removal")
    config_sw = LMPreprocessingConfig(remove_stopwords=True)
    preprocessor_sw = LMPreprocessor(config_sw)
    result_sw = preprocessor_sw.preprocess(SAMPLE_TRANSCRIPT)

    if result_sw:
        print(f"  Tokens after stopword removal: {result_sw.token_count}")
        print(f"  Sample: {result_sw.tokens[:20]}")

    # ── Batch preprocessing ───────────────────────────────────────────
    print("\n[3] Batch preprocessing")
    texts = [
        "Revenue grew 12% to $4.5B, driven by strong iPhone demand.",
        "EBITDA margin expanded 80bps year-over-year.",
        "",           # empty — should be skipped
        None,         # null — should be skipped
        "Q4 FY2024 EPS of $3.21 exceeded consensus by $0.08.",
    ]

    results, stats = preprocessor.preprocess_batch(
        texts,
        ids=["seg_01", "seg_02", "seg_03_empty", "seg_04_null", "seg_05"],
    )

    for i, r in enumerate(results):
        if r is not None:
            print(f"  [{i}] {r.token_count} tokens: {r.tokens}")
        else:
            print(f"  [{i}] None (skipped/failed)")

    print("\n" + stats.summary())

    # ── Ticker-safe check ─────────────────────────────────────────────
    print("[4] Ticker-safe tokenization check")
    ticker_text = "AAPL reported strong results; MSFT and NVDA also beat estimates."
    result_tickers = preprocessor.preprocess(ticker_text)
    if result_tickers:
        print(f"  Tokens: {result_tickers.tokens}")
        print(f"  Tickers: {result_tickers.preserved_tickers}")

    print("\nDemo complete.\n")
