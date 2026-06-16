"""
sentence_segmenter.py
=====================
Deterministic sentence segmentation engine for long earnings call transcripts.

Part of: Earnings Call Sentiment Analyzer — DAY 8 Pipeline
Stage:   Sentence Segmentation (between Speaker Segmentation and Chunk Generation)

Responsibilities
----------------
* Split transcript speaker-blocks into individual, ordered sentences
* Preserve all upstream metadata (transcript_id, speaker, speaker_role,
  section_type, order_index) on every output sentence
* Assign globally deterministic sentence IDs (stable across re-runs)
* Estimate HuggingFace-compatible token counts for downstream chunking
* Emit a rich, validated SentenceSegmentationResult per transcript
* Support spaCy (preferred), NLTK punkt (fallback), and regex (universal
  fallback) tokenization backends — discovered automatically at runtime

This module does NOT chunk sentences into transformer windows and does NOT
run FinBERT inference.  It is the feed for chunk_generator.py (DAY 8,
Stage 3).

Input Schema (DataFrame rows)
------------------------------
    transcript_id   : str   — unique transcript key, e.g. "AAPL_Q1_2025"
    ticker          : str   — stock symbol (optional, propagated if present)
    speaker         : str   — speaker name from role_classifier
    speaker_role    : str   — normalized role: CEO / CFO / Analyst / Operator
    section_type    : str   — "prepared_remarks" | "qa" | "closing_remarks"
    text            : str   — cleaned transcript block text
    order_index     : int   — global ordering key across all speaker blocks

Output Schema (SentenceSegment)
--------------------------------
    sentence_id          : str   — globally unique, deterministic
    transcript_id        : str
    ticker               : str
    sentence_order       : int   — 0-based, globally ordered across transcript
    speaker              : str
    speaker_role         : str
    section_type         : str
    sentence_text        : str
    token_estimate       : int
    source_segment_index : int   — the input order_index this came from
    chunk_candidate_id   : str   — pre-chunk grouping hint for chunk_generator

Architecture Notes
------------------
* Token estimation uses a calibrated character / word heuristic that closely
  approximates BertTokenizer (WordPiece) outputs without requiring a live
  tokenizer at segmentation time.  The multiplier is intentionally slightly
  generous (conservative) so chunk_generator never over-fills a 512-token
  window.
* Sentence IDs are SHA-256 derived from (transcript_id, sentence_order),
  making them fully reproducible across pipeline re-runs.
* The backend auto-detection order is: spaCy → NLTK punkt → regex.  Each
  tier degrades gracefully; the regex fallback is finance-aware and handles
  common abbreviations (e.g. "vs.", "Mr.", "Corp.", "$1.2B", "Q1").

Usage
-----
    from sentence_segmenter import SentenceSegmenter, SentenceSegmenterConfig

    config = SentenceSegmenterConfig(min_sentence_tokens=4, max_sentence_tokens=200)
    segmenter = SentenceSegmenter(config=config)

    # Single DataFrame (all rows belong to one transcript)
    result = segmenter.segment_transcript(df)
    sentence_df = result.to_dataframe()

    # Multiple transcripts in one DataFrame
    all_results = segmenter.segment_dataframe(full_df)

Python: 3.11+
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Iterator, Optional, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SegmentationBackend(Enum):
    """Sentence-tokenization backend.

    Precedence: SPACY > NLTK > REGEX
    """
    SPACY = auto()
    NLTK  = auto()
    REGEX = auto()


class SectionType(str, Enum):
    """Normalized transcript section labels."""
    PREPARED_REMARKS = "prepared_remarks"
    QA               = "qa"
    CLOSING_REMARKS  = "closing_remarks"
    UNKNOWN          = "unknown"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentenceSegmenterConfig:
    """All tunable parameters for :class:`SentenceSegmenter`.

    Attributes
    ----------
    backend : SegmentationBackend | None
        Force a specific backend.  ``None`` = auto-detect (recommended).
    spacy_model : str
        spaCy model name, e.g. ``"en_core_web_sm"`` or ``"en_core_web_trf"``.
    min_sentence_chars : int
        Drop sentences shorter than this many characters.
        Filters out orphaned punctuation, page markers, etc.
    min_sentence_tokens : int
        Drop sentences whose token estimate is below this threshold.
    max_sentence_tokens : int
        Warn (but keep) sentences whose token estimate exceeds this.
        Downstream chunk_generator must handle these edge cases.
    token_chars_divisor : float
        Primary character-based token estimator: tokens ≈ chars / divisor.
        Calibrated to approximate BERT WordPiece tokenization.
    token_word_multiplier : float
        Secondary word-based estimator: tokens ≈ words × multiplier.
        Final estimate = max(char_estimate, word_estimate) for conservatism.
    strip_boilerplate : bool
        Remove common earnings-call boilerplate phrases from sentence text.
    normalize_whitespace : bool
        Collapse internal whitespace to a single space.
    preserve_section_boundaries : bool
        When True, chunk_candidate_id resets across section_type transitions,
        giving chunk_generator a natural boundary hint.
    chunk_candidate_window : int
        Number of consecutive sentences grouped into one chunk_candidate_id.
        Set to match the expected chunk_size in chunk_generator.py.
        Default 15 ≈ ~300 tokens worth of typical earnings-call sentences.
    """

    backend: Optional[SegmentationBackend] = None
    spacy_model: str                        = "en_core_web_sm"

    min_sentence_chars: int                 = 10
    min_sentence_tokens: int                = 3
    max_sentence_tokens: int                = 200

    token_chars_divisor: float              = 3.8
    token_word_multiplier: float            = 1.35

    strip_boilerplate: bool                 = True
    normalize_whitespace: bool              = True
    preserve_section_boundaries: bool       = True
    chunk_candidate_window: int             = 15


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SentenceSegment:
    """A single sentence extracted from an earnings-call transcript block.

    All fields are required so that downstream modules (chunk_generator,
    finbert_pipeline, hierarchical_aggregation) can operate without
    additional lookups.
    """

    # --- Identity ---
    sentence_id          : str   # globally unique, deterministic

    # --- Transcript linkage ---
    transcript_id        : str
    ticker               : str   # propagated from input ("" if absent)

    # --- Ordering ---
    sentence_order       : int   # 0-based global index within transcript

    # --- Speaker / structural metadata ---
    speaker              : str
    speaker_role         : str
    section_type         : str   # normalized SectionType value

    # --- Content ---
    sentence_text        : str
    token_estimate       : int

    # --- Provenance ---
    source_segment_index : int   # input row's order_index
    chunk_candidate_id   : str   # pre-grouping hint for chunk_generator

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a plain Python dict (JSON-serializable)."""
        return asdict(self)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        preview = self.sentence_text[:60].replace("\n", " ")
        return (
            f"SentenceSegment(id={self.sentence_id!r}, "
            f"order={self.sentence_order}, "
            f"speaker={self.speaker!r}, "
            f"tokens={self.token_estimate}, "
            f"text={preview!r}…)"
        )


@dataclass
class SegmentationStats:
    """Per-transcript segmentation diagnostics."""

    transcript_id          : str
    total_input_segments   : int = 0
    total_sentences        : int = 0
    dropped_too_short      : int = 0
    dropped_empty          : int = 0
    total_token_estimate   : int = 0
    sections_found         : list[str] = field(default_factory=list)
    speakers_found         : list[str] = field(default_factory=list)
    backend_used           : str = ""
    warnings               : list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentenceSegmentationResult:
    """Complete output of segmenting one transcript.

    Attributes
    ----------
    transcript_id : str
        Unique transcript identifier.
    sentences : list[SentenceSegment]
        All retained sentences in deterministic order.
    stats : SegmentationStats
        Diagnostic information useful for pipeline monitoring.
    """

    transcript_id : str
    sentences     : list[SentenceSegment]
    stats         : SegmentationStats

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert sentences to a pandas DataFrame ready for parquet export."""
        if not self.sentences:
            return pd.DataFrame(columns=list(SentenceSegment.__dataclass_fields__))
        rows = [s.to_dict() for s in self.sentences]
        return pd.DataFrame(rows)

    def to_dict_list(self) -> list[dict]:
        """Return list-of-dicts (compatible with JSON / JSONL export)."""
        return [s.to_dict() for s in self.sentences]

    def __len__(self) -> int:
        return len(self.sentences)

    def __repr__(self) -> str:
        return (
            f"SentenceSegmentationResult("
            f"transcript_id={self.transcript_id!r}, "
            f"sentences={len(self.sentences)}, "
            f"backend={self.stats.backend_used!r})"
        )


# ---------------------------------------------------------------------------
# Boilerplate patterns (finance-specific noise)
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Operator instructions
    re.compile(
        r"^(ladies and gentlemen|please stand by|your conference will begin|"
        r"thank you for (standing by|holding)|please (press|dial)|"
        r"this (call|conference) (is being )?recorded|"
        r"at this time (all|please)|to ask a question press|"
        r"please press \*\d)",
        re.I,
    ),
    # Legal / safe-harbour boilerplate
    re.compile(
        r"(forward[- ]looking statements?|safe[- ]harbor|"
        r"actual results (may|could|might) differ|"
        r"cautionary (note|statement)|risk factors|"
        r"securities and exchange commission|"
        r"this (press release|transcript) contains)",
        re.I,
    ),
    # Filler openers
    re.compile(
        r"^(thank you(,? operator)?\.?\s*$|"
        r"good (morning|afternoon|evening)(,?\s+(everyone|all|ladies))?\.?\s*$|"
        r"hello,?\s+everyone\.?\s*$)",
        re.I,
    ),
]

# Finance-aware sentence abbreviations — do NOT split after these
_ABBREV_PATTERN = re.compile(
    r"\b(Mr|Ms|Mrs|Dr|Prof|Sr|Jr|vs|etc|Corp|Inc|Ltd|Co|LLC|LP|LLP|"
    r"No|Vol|pp|ed|rev|approx|est|avg|max|min|fig|dept|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"Q[1-4]|H[12]|FY|YoY|QoQ|YTD|TTM|EPS|EBITDA|GAAP|"
    r"U\.S|U\.K|E\.U)\."
)


# ---------------------------------------------------------------------------
# Regex sentence splitter (universal fallback)
# ---------------------------------------------------------------------------

class _RegexSentenceSplitter:
    """Finance-aware regex sentence splitter.

    Handles common patterns found in earnings call transcripts:
    * Monetary amounts: "$1.2B", "3.5%"
    * Abbreviations: "Corp.", "Q1.", "vs."
    * Ellipses: "..."
    * Trailing newlines within a speaker block
    """

    # Finance-aware sentence boundary: terminal punctuation followed by
    # whitespace and an uppercase letter, with abbreviation protection
    # applied in the split() method (Python lookbehind requires fixed width).
    _SPLIT_RE = re.compile(
        r'(?<=[.!?])'          # must have terminal punctuation
        r'(?:\s*\n+\s*|\s{2,}|\s(?=[A-Z]))'  # followed by whitespace/newline
    )

    # Protect these patterns from being split on their trailing period
    _NO_SPLIT_RE = re.compile(
        r'\b(?:Mr|Ms|Mrs|Dr|Prof|Sr|Jr|vs|etc|Corp|Inc|Ltd|Co|LLC|LP|LLP|'
        r'No|Vol|pp|ed|rev|approx|est|avg|max|min|FY|YoY|QoQ|'
        r'U\.S|U\.K|E\.U|Q[1-4]|H[1-2])\.'
        r'|[A-Z]\.'        # single-letter initials: "J. Smith"
        r'|\d+\.\d'        # decimals: "3.5 billion"
        r'|\$\d+\.'        # dollar amounts: "$1.2"
    )

    def split(self, text: str) -> list[str]:
        """Split *text* into sentences using a two-pass approach.

        Pass 1: Replace protected abbreviation periods with a placeholder so
                the splitter does not break on them.
        Pass 2: Split on terminal punctuation followed by whitespace.
        Pass 3: Restore placeholders.

        Returns an empty list for blank input.
        """
        if not text or not text.strip():
            return []

        # Normalise newlines first: single newlines within a sentence become
        # spaces, double newlines become hard boundaries.
        text = re.sub(r'\n{2,}', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Split on double-newlines as definite sentence boundaries first
        paragraphs = text.split('\n\n')
        sentences: list[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # --- Pass 1: protect abbreviations ---
            # Replace their period with a Unicode placeholder
            _PH = '\u2060'  # WORD JOINER — invisible, safe placeholder
            protected, subs = self._protect_abbreviations(para, _PH)

            # --- Pass 2: split ---
            parts = self._SPLIT_RE.split(protected)

            # --- Pass 3: restore ---
            for part in parts:
                restored = part.replace(_PH, '.')
                cleaned = restored.strip()
                if cleaned:
                    sentences.append(cleaned)

        return sentences

    def _protect_abbreviations(
        self, text: str, placeholder: str
    ) -> tuple[str, list[str]]:
        """Replace periods in known abbreviations with *placeholder*.

        Returns the modified text and a list of original substrings for
        reconstruction (not used directly since we restore all placeholders).
        """
        subs: list[str] = []

        def replacer(m: re.Match) -> str:
            original = m.group(0)
            subs.append(original)
            # Replace only the trailing period with the placeholder
            return original[:-1] + placeholder

        protected = self._NO_SPLIT_RE.sub(replacer, text)
        return protected, subs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SentenceSegmenter:
    """Deterministic sentence segmentation engine for earnings call transcripts.

    Converts structured transcript rows (output of speaker_extractor /
    role_classifier) into ordered, annotated sentence-level units ready for
    transformer chunking.

    Parameters
    ----------
    config : SentenceSegmenterConfig, optional
        If ``None``, defaults are used.

    Examples
    --------
    >>> segmenter = SentenceSegmenter()
    >>> result = segmenter.segment_transcript(transcript_df)
    >>> df_out = result.to_dataframe()
    >>> df_out.to_parquet("data/interim/transcript_sentences.parquet")
    """

    # Required input columns
    REQUIRED_COLUMNS: frozenset[str] = frozenset({
        "transcript_id",
        "speaker",
        "speaker_role",
        "section_type",
        "text",
        "order_index",
    })

    # Output column order (matches architecture spec)
    OUTPUT_COLUMNS: list[str] = [
        "sentence_id",
        "transcript_id",
        "ticker",
        "sentence_order",
        "speaker",
        "speaker_role",
        "section_type",
        "sentence_text",
        "token_estimate",
        "source_segment_index",
        "chunk_candidate_id",
    ]

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, config: Optional[SentenceSegmenterConfig] = None) -> None:
        self.config = config or SentenceSegmenterConfig()
        self._backend: SegmentationBackend = self._resolve_backend()
        self._nlp = None          # spaCy model (lazy-loaded)
        self._nltk_tokenizer = None  # NLTK punkt tokenizer (lazy-loaded)
        self._regex_splitter = _RegexSentenceSplitter()

        logger.info(
            "SentenceSegmenter initialised | backend=%s | "
            "min_tokens=%d | max_tokens=%d",
            self._backend.name,
            self.config.min_sentence_tokens,
            self.config.max_sentence_tokens,
        )

    # -----------------------------------------------------------------------
    # Backend resolution
    # -----------------------------------------------------------------------

    def _resolve_backend(self) -> SegmentationBackend:
        """Auto-detect the best available backend, or honour config override."""
        if self.config.backend is not None:
            logger.debug("Backend forced to %s by config.", self.config.backend.name)
            return self.config.backend

        # --- 1. Try spaCy ---
        try:
            import spacy  # noqa: F401
            # Verify the requested model is installed
            spacy.load(self.config.spacy_model)
            logger.info("Sentence backend: spaCy (%s)", self.config.spacy_model)
            return SegmentationBackend.SPACY
        except Exception:
            pass

        # --- 2. Try NLTK punkt ---
        try:
            import nltk
            # Check if punkt data is available
            nltk.data.find("tokenizers/punkt_tab")
            logger.info("Sentence backend: NLTK punkt_tab")
            return SegmentationBackend.NLTK
        except Exception:
            pass

        try:
            import nltk
            nltk.data.find("tokenizers/punkt")
            logger.info("Sentence backend: NLTK punkt")
            return SegmentationBackend.NLTK
        except Exception:
            pass

        # --- 3. Regex fallback ---
        logger.warning(
            "Neither spaCy (%s) nor NLTK punkt is available. "
            "Falling back to finance-aware regex splitter. "
            "Install 'spacy' + model or run nltk.download('punkt_tab') "
            "for higher-quality sentence boundaries.",
            self.config.spacy_model,
        )
        return SegmentationBackend.REGEX

    def _load_spacy(self):
        """Lazy-load spaCy NLP pipeline (sentence detection only)."""
        if self._nlp is not None:
            return
        import spacy
        self._nlp = spacy.load(
            self.config.spacy_model,
            disable=["ner", "lemmatizer", "tagger", "attribute_ruler"],
        )
        if "senter" not in self._nlp.pipe_names and "sentencizer" not in self._nlp.pipe_names:
            self._nlp.add_pipe("sentencizer")

    def _load_nltk(self):
        """Lazy-load NLTK punkt sentence tokenizer."""
        if self._nltk_tokenizer is not None:
            return
        import nltk
        try:
            self._nltk_tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        except Exception:
            try:
                self._nltk_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            except Exception:
                # If punkt data missing, fall back to regex for this run
                logger.warning(
                    "NLTK punkt model not found at runtime; switching to regex backend."
                )
                self._backend = SegmentationBackend.REGEX

    # -----------------------------------------------------------------------
    # Sentence tokenization (backend dispatch)
    # -----------------------------------------------------------------------

    def _tokenize_sentences(self, text: str) -> list[str]:
        """Tokenize *text* into raw sentence strings using the active backend."""
        if not text or not text.strip():
            return []

        if self._backend == SegmentationBackend.SPACY:
            return self._spacy_tokenize(text)
        if self._backend == SegmentationBackend.NLTK:
            return self._nltk_tokenize(text)
        return self._regex_tokenize(text)

    def _spacy_tokenize(self, text: str) -> list[str]:
        """Split *text* using spaCy's sentence detector."""
        self._load_spacy()
        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _nltk_tokenize(self, text: str) -> list[str]:
        """Split *text* using NLTK's punkt tokenizer."""
        self._load_nltk()
        if self._backend == SegmentationBackend.REGEX:
            # Fell back to regex during _load_nltk
            return self._regex_tokenize(text)
        sentences = self._nltk_tokenizer.tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def _regex_tokenize(self, text: str) -> list[str]:
        """Split *text* using the finance-aware regex splitter."""
        return self._regex_splitter.split(text)

    # -----------------------------------------------------------------------
    # Token estimation
    # -----------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of WordPiece tokens for *text*.

        Uses a calibrated heuristic that is intentionally conservative (slightly
        over-estimates) to ensure chunk_generator never exceeds the 512-token
        FinBERT limit.

        Formula
        -------
            char_est  = ceil(len(text) / chars_divisor)
            word_est  = ceil(word_count * word_multiplier)
            estimate  = max(char_est, word_est)
        """
        if not text:
            return 0
        import math
        char_est = math.ceil(len(text) / self.config.token_chars_divisor)
        word_count = len(text.split())
        word_est = math.ceil(word_count * self.config.token_word_multiplier)
        return max(char_est, word_est)

    # -----------------------------------------------------------------------
    # Text cleaning helpers
    # -----------------------------------------------------------------------

    def _clean_sentence(self, text: str) -> str:
        """Apply lightweight cleaning to an individual sentence."""
        if not text:
            return ""

        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _is_boilerplate(self, text: str) -> bool:
        """Return True if *text* matches a known boilerplate pattern."""
        if not self.config.strip_boilerplate:
            return False
        lowered = text.lower().strip()
        return any(p.search(lowered) for p in _BOILERPLATE_PATTERNS)

    def _is_valid_sentence(
        self,
        text: str,
        stats: SegmentationStats,
    ) -> bool:
        """Return True if the sentence passes all quality gates."""
        cleaned = text.strip()

        if not cleaned:
            stats.dropped_empty += 1
            return False

        if len(cleaned) < self.config.min_sentence_chars:
            stats.dropped_too_short += 1
            return False

        token_est = self._estimate_tokens(cleaned)
        if token_est < self.config.min_sentence_tokens:
            stats.dropped_too_short += 1
            return False

        return True

    # -----------------------------------------------------------------------
    # ID generation
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_sentence_id(transcript_id: str, sentence_order: int) -> str:
        """Generate a globally unique, deterministic sentence identifier.

        Format: ``sent_{transcript_id}_{order:06d}_{hash8}``

        The 8-character SHA-256 suffix prevents collisions if transcript_id
        values are long or contain special characters.
        """
        raw = f"{transcript_id}__sent_{sentence_order:06d}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        safe_tid = re.sub(r'[^A-Za-z0-9_]', '_', transcript_id)
        return f"sent_{safe_tid}_{sentence_order:06d}_{digest}"

    @staticmethod
    def _make_chunk_candidate_id(
        transcript_id: str,
        sentence_order: int,
        section_type: str,
        window: int,
    ) -> str:
        """Generate a chunk-candidate grouping ID.

        Sentences in the same section and the same (sentence_order // window)
        bucket share a chunk_candidate_id, giving chunk_generator natural
        grouping hints without hard coupling.
        """
        bucket = sentence_order // window
        safe_tid = re.sub(r'[^A-Za-z0-9_]', '_', transcript_id)
        safe_sec = re.sub(r'[^A-Za-z0-9_]', '_', section_type)
        return f"cand_{safe_tid}_{safe_sec}_{bucket:04d}"

    # -----------------------------------------------------------------------
    # Normalisation helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalise_section_type(raw: str) -> str:
        """Map raw section text to a canonical SectionType value."""
        if not raw or not isinstance(raw, str):
            return SectionType.UNKNOWN.value
        lowered = raw.strip().lower().replace(" ", "_").replace("-", "_")
        mapping = {
            "prepared_remarks"  : SectionType.PREPARED_REMARKS.value,
            "prepared"          : SectionType.PREPARED_REMARKS.value,
            "management_remarks": SectionType.PREPARED_REMARKS.value,
            "remarks"           : SectionType.PREPARED_REMARKS.value,
            "qa"                : SectionType.QA.value,
            "q_a"               : SectionType.QA.value,
            "q&a"               : SectionType.QA.value,
            "question_and_answer": SectionType.QA.value,
            "question_answer"   : SectionType.QA.value,
            "closing_remarks"   : SectionType.CLOSING_REMARKS.value,
            "closing"           : SectionType.CLOSING_REMARKS.value,
        }
        return mapping.get(lowered, SectionType.UNKNOWN.value)

    # -----------------------------------------------------------------------
    # Core segmentation: single block → list[SentenceSegment]
    # -----------------------------------------------------------------------

    def segment_text_block(
        self,
        *,
        text: str,
        transcript_id: str,
        ticker: str,
        speaker: str,
        speaker_role: str,
        section_type: str,
        source_segment_index: int,
        sentence_order_start: int,
        stats: SegmentationStats,
    ) -> list[SentenceSegment]:
        """Segment a single speaker-block into :class:`SentenceSegment` objects.

        Parameters
        ----------
        text : str
            The raw transcript text block for this speaker turn.
        transcript_id : str
            Transcript-level primary key.
        ticker : str
            Company ticker (propagated for convenience).
        speaker : str
            Speaker name as extracted by speaker_extractor.
        speaker_role : str
            Normalised role from role_classifier.
        section_type : str
            Section label; will be normalised internally.
        source_segment_index : int
            The ``order_index`` of the input row — preserved for provenance.
        sentence_order_start : int
            The global sentence counter offset for this block.  Incremented
            externally so that ordering is consistent across the transcript.
        stats : SegmentationStats
            Mutable stats object updated in place.

        Returns
        -------
        list[SentenceSegment]
            Ordered list of sentences from this block.
        """
        normalised_section = self._normalise_section_type(section_type)
        raw_sentences = self._tokenize_sentences(text)
        segments: list[SentenceSegment] = []

        local_order = sentence_order_start
        for raw in raw_sentences:
            cleaned = self._clean_sentence(raw)

            if not self._is_valid_sentence(cleaned, stats):
                continue

            if self._is_boilerplate(cleaned):
                stats.dropped_empty += 1
                logger.debug(
                    "Dropped boilerplate sentence [%s]: %r",
                    transcript_id, cleaned[:80]
                )
                continue

            token_est = self._estimate_tokens(cleaned)
            if token_est > self.config.max_sentence_tokens:
                stats.warnings.append(
                    f"Long sentence at order={local_order}: "
                    f"{token_est} tokens (>{self.config.max_sentence_tokens})"
                )
                logger.debug(
                    "Oversized sentence [%s|order=%d]: %d tokens",
                    transcript_id, local_order, token_est,
                )

            sentence_id = self._make_sentence_id(transcript_id, local_order)
            chunk_cand  = self._make_chunk_candidate_id(
                transcript_id,
                local_order,
                normalised_section,
                self.config.chunk_candidate_window,
            )

            seg = SentenceSegment(
                sentence_id          = sentence_id,
                transcript_id        = transcript_id,
                ticker               = ticker,
                sentence_order       = local_order,
                speaker              = speaker,
                speaker_role         = speaker_role,
                section_type         = normalised_section,
                sentence_text        = cleaned,
                token_estimate       = token_est,
                source_segment_index = source_segment_index,
                chunk_candidate_id   = chunk_cand,
            )
            segments.append(seg)
            local_order += 1

        stats.total_sentences      += len(segments)
        stats.total_token_estimate += sum(s.token_estimate for s in segments)
        return segments

    # -----------------------------------------------------------------------
    # Transcript-level segmentation
    # -----------------------------------------------------------------------

    def segment_transcript(self, df: pd.DataFrame) -> SentenceSegmentationResult:
        """Segment all blocks of a single transcript DataFrame.

        The DataFrame must contain exactly one distinct ``transcript_id`` and
        all required columns defined in :attr:`REQUIRED_COLUMNS`.

        Parameters
        ----------
        df : pd.DataFrame
            Speaker-segmented transcript rows, sorted by ``order_index``.

        Returns
        -------
        SentenceSegmentationResult
        """
        self._validate_input_schema(df)

        # Normalise: sort by order_index to guarantee deterministic ordering
        df = df.sort_values("order_index").reset_index(drop=True)

        # Allow multiple distinct transcript_ids (warn if > 1; use first for stats)
        unique_ids = df["transcript_id"].unique()
        if len(unique_ids) > 1:
            warnings.warn(
                f"segment_transcript() received {len(unique_ids)} distinct "
                f"transcript_ids. Consider using segment_dataframe() instead. "
                f"All rows will be processed but stats are grouped together.",
                UserWarning,
                stacklevel=2,
            )
        transcript_id = str(unique_ids[0])
        ticker = str(df["ticker"].iloc[0]) if "ticker" in df.columns else ""

        stats = SegmentationStats(
            transcript_id        = transcript_id,
            total_input_segments = len(df),
            backend_used         = self._backend.name,
        )

        all_segments: list[SentenceSegment] = []
        global_order_counter = 0

        for _, row in df.iterrows():
            row_text         = str(row.get("text", "") or "")
            row_speaker      = str(row.get("speaker", "") or "")
            row_role         = str(row.get("speaker_role", "") or "")
            row_section      = str(row.get("section_type", "") or "")
            row_order_index  = int(row.get("order_index", 0))
            row_ticker       = str(row.get("ticker", ticker) or ticker)

            if not row_text.strip():
                stats.dropped_empty += 1
                continue

            block_segments = self.segment_text_block(
                text                 = row_text,
                transcript_id        = str(row.get("transcript_id", transcript_id)),
                ticker               = row_ticker,
                speaker              = row_speaker,
                speaker_role         = row_role,
                section_type         = row_section,
                source_segment_index = row_order_index,
                sentence_order_start = global_order_counter,
                stats                = stats,
            )
            all_segments.extend(block_segments)
            global_order_counter += len(block_segments)

        # Populate stats
        stats.sections_found = sorted(
            {s.section_type for s in all_segments}
        )
        stats.speakers_found = sorted(
            {s.speaker for s in all_segments if s.speaker}
        )

        # Final ordering validation
        all_segments = self._enforce_ordering(all_segments)

        logger.info(
            "[%s] Segmentation complete | "
            "input_rows=%d | sentences=%d | dropped=%d | "
            "total_tokens≈%d | backend=%s",
            transcript_id,
            stats.total_input_segments,
            stats.total_sentences,
            stats.dropped_empty + stats.dropped_too_short,
            stats.total_token_estimate,
            stats.backend_used,
        )

        return SentenceSegmentationResult(
            transcript_id = transcript_id,
            sentences     = all_segments,
            stats         = stats,
        )

    # -----------------------------------------------------------------------
    # Batch segmentation (multiple transcripts in one DataFrame)
    # -----------------------------------------------------------------------

    def segment_dataframe(
        self, df: pd.DataFrame
    ) -> list[SentenceSegmentationResult]:
        """Segment all transcripts present in *df*.

        Groups rows by ``transcript_id`` and calls :meth:`segment_transcript`
        for each group independently, ensuring correct per-transcript ordering.

        Parameters
        ----------
        df : pd.DataFrame
            Combined transcript DataFrame from the preprocessing pipeline.

        Returns
        -------
        list[SentenceSegmentationResult]
            One result per distinct ``transcript_id``, in the order they
            appear after sorting.
        """
        self._validate_input_schema(df)
        results: list[SentenceSegmentationResult] = []

        grouped = df.groupby("transcript_id", sort=True)
        n_transcripts = len(grouped)

        logger.info(
            "Starting batch segmentation | transcripts=%d | total_rows=%d",
            n_transcripts, len(df),
        )

        for i, (tid, group_df) in enumerate(grouped, 1):
            logger.debug(
                "Segmenting transcript %d/%d: %s (%d rows)",
                i, n_transcripts, tid, len(group_df),
            )
            result = self.segment_transcript(group_df)
            results.append(result)

        total_sentences = sum(len(r) for r in results)
        logger.info(
            "Batch segmentation complete | "
            "transcripts=%d | total_sentences=%d",
            n_transcripts, total_sentences,
        )

        return results

    # -----------------------------------------------------------------------
    # Export helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def results_to_dataframe(
        results: list[SentenceSegmentationResult],
    ) -> pd.DataFrame:
        """Concatenate multiple results into a single flat DataFrame.

        This is the primary export method for saving to parquet:

        >>> df_out = SentenceSegmenter.results_to_dataframe(results)
        >>> df_out.to_parquet("data/interim/transcript_sentences.parquet")
        """
        if not results:
            return pd.DataFrame()

        frames = [r.to_dataframe() for r in results if r.sentences]
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # Ensure output column order matches the architecture spec
        available = [c for c in SentenceSegmenter.OUTPUT_COLUMNS if c in combined.columns]
        extra = [c for c in combined.columns if c not in available]
        return combined[available + extra]

    @staticmethod
    def results_to_stats_dataframe(
        results: list[SentenceSegmentationResult],
    ) -> pd.DataFrame:
        """Return a DataFrame of per-transcript segmentation statistics."""
        rows = [r.stats.to_dict() for r in results]
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Sentence reconstruction utility
    # -----------------------------------------------------------------------

    @staticmethod
    def reconstruct_text(
        segments: Sequence[SentenceSegment],
        separator: str = " ",
    ) -> str:
        """Reconstruct full transcript text from a sequence of segments.

        Useful for round-trip validation and debugging.

        Parameters
        ----------
        segments : Sequence[SentenceSegment]
            Must be in sentence_order order.
        separator : str
            Text to insert between consecutive sentences.
        """
        ordered = sorted(segments, key=lambda s: s.sentence_order)
        return separator.join(s.sentence_text for s in ordered)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_input_schema(self, df: pd.DataFrame) -> None:
        """Raise ValueError if required columns are missing."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {sorted(missing)}. "
                f"Required: {sorted(self.REQUIRED_COLUMNS)}"
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

    @staticmethod
    def _enforce_ordering(
        segments: list[SentenceSegment],
    ) -> list[SentenceSegment]:
        """Guarantee sentences are sorted by sentence_order.

        Also re-emits a warning if any duplicate orders are detected.
        """
        segments = sorted(segments, key=lambda s: s.sentence_order)

        # Duplicate detection
        seen_orders: set[int] = set()
        duplicates: list[int] = []
        for seg in segments:
            if seg.sentence_order in seen_orders:
                duplicates.append(seg.sentence_order)
            seen_orders.add(seg.sentence_order)

        if duplicates:
            logger.warning(
                "Duplicate sentence_order values detected: %s — "
                "this may indicate a bug in sentence_order_start management.",
                duplicates[:10],
            )

        return segments


# ---------------------------------------------------------------------------
# Standalone validation helper
# ---------------------------------------------------------------------------

class SegmentationValidator:
    """Post-hoc validation of a :class:`SentenceSegmentationResult`.

    Designed for use in test suites and pipeline quality checks.

    Examples
    --------
    >>> validator = SegmentationValidator(max_token_limit=512)
    >>> issues = validator.validate(result)
    >>> assert not issues, f"Validation failed: {issues}"
    """

    def __init__(self, max_token_limit: int = 512) -> None:
        self.max_token_limit = max_token_limit

    def validate(
        self, result: SentenceSegmentationResult
    ) -> list[str]:
        """Run all checks and return a list of issue strings.

        An empty list means the result passed all checks.
        """
        issues: list[str] = []
        issues += self._check_non_empty(result)
        issues += self._check_ordering(result)
        issues += self._check_no_duplicates(result)
        issues += self._check_token_limits(result)
        issues += self._check_no_empty_sentences(result)
        issues += self._check_metadata_completeness(result)
        issues += self._check_sentence_id_format(result)
        return issues

    def validate_dataframe(self, df: pd.DataFrame) -> list[str]:
        """Validate a flattened sentence DataFrame directly."""
        issues: list[str] = []
        if df.empty:
            issues.append("DataFrame is empty.")
            return issues

        # Required columns
        required = set(SentenceSegmenter.OUTPUT_COLUMNS)
        missing = required - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {sorted(missing)}")

        # Duplicate sentence_id
        dup_ids = df[df.duplicated("sentence_id", keep=False)]
        if not dup_ids.empty:
            issues.append(
                f"Duplicate sentence_id values: {dup_ids['sentence_id'].unique()[:5].tolist()}"
            )

        # Token limit per sentence
        if "token_estimate" in df.columns:
            over_limit = df[df["token_estimate"] > self.max_token_limit]
            if not over_limit.empty:
                issues.append(
                    f"{len(over_limit)} sentences exceed token limit "
                    f"({self.max_token_limit}): max={df['token_estimate'].max()}"
                )

        # Empty sentence text
        if "sentence_text" in df.columns:
            empty = df[df["sentence_text"].str.strip() == ""]
            if not empty.empty:
                issues.append(f"{len(empty)} rows have empty sentence_text.")

        # Per-transcript ordering
        if "sentence_order" in df.columns and "transcript_id" in df.columns:
            for tid, group in df.groupby("transcript_id"):
                orders = group["sentence_order"].tolist()
                if orders != sorted(orders):
                    issues.append(
                        f"[{tid}] sentence_order is not monotonically increasing."
                    )
                if len(orders) != len(set(orders)):
                    issues.append(
                        f"[{tid}] Duplicate sentence_order values detected."
                    )

        return issues

    # --- Private checks ---

    def _check_non_empty(self, r: SentenceSegmentationResult) -> list[str]:
        if not r.sentences:
            return [f"[{r.transcript_id}] No sentences produced."]
        return []

    def _check_ordering(self, r: SentenceSegmentationResult) -> list[str]:
        orders = [s.sentence_order for s in r.sentences]
        if orders != sorted(orders):
            return [f"[{r.transcript_id}] sentence_order is not sorted."]
        return []

    def _check_no_duplicates(self, r: SentenceSegmentationResult) -> list[str]:
        seen: set[str] = set()
        dupes: list[str] = []
        for s in r.sentences:
            if s.sentence_id in seen:
                dupes.append(s.sentence_id)
            seen.add(s.sentence_id)
        if dupes:
            return [f"[{r.transcript_id}] Duplicate sentence_ids: {dupes[:5]}"]
        return []

    def _check_token_limits(self, r: SentenceSegmentationResult) -> list[str]:
        over = [s for s in r.sentences if s.token_estimate > self.max_token_limit]
        if over:
            return [
                f"[{r.transcript_id}] {len(over)} sentences exceed "
                f"{self.max_token_limit} tokens (max={max(s.token_estimate for s in over)})."
            ]
        return []

    def _check_no_empty_sentences(self, r: SentenceSegmentationResult) -> list[str]:
        empty = [s for s in r.sentences if not s.sentence_text.strip()]
        if empty:
            return [f"[{r.transcript_id}] {len(empty)} empty sentence_text values."]
        return []

    def _check_metadata_completeness(self, r: SentenceSegmentationResult) -> list[str]:
        issues: list[str] = []
        for s in r.sentences:
            if not s.transcript_id:
                issues.append(f"Missing transcript_id at order={s.sentence_order}")
            if not s.section_type:
                issues.append(f"Missing section_type at order={s.sentence_order}")
        return issues

    def _check_sentence_id_format(self, r: SentenceSegmentationResult) -> list[str]:
        bad = [
            s.sentence_id for s in r.sentences
            if not s.sentence_id.startswith("sent_")
        ]
        if bad:
            return [
                f"[{r.transcript_id}] {len(bad)} sentence_ids with unexpected format: "
                f"{bad[:3]}"
            ]
        return []


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 72)
    print("  SentenceSegmenter — Self-Test & Demo")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 1. Build a synthetic transcript DataFrame
    # -----------------------------------------------------------------------
    SAMPLE_DATA = [
        {
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "speaker": "Tim Cook",
            "speaker_role": "CEO",
            "section_type": "prepared_remarks",
            "text": (
                "Good afternoon everyone, and thank you for joining us. "
                "I'm Tim Cook, CEO of Apple. "
                "We're reporting revenue of $124.3 billion for Q1 fiscal 2025, "
                "up 4% year-over-year. "
                "iPhone revenue reached $69.7 billion, a new December quarter record. "
                "Our Services segment grew 14%, reaching $26.3 billion. "
                "We are deeply committed to AI across all our product lines. "
                "The integration of Apple Intelligence is proceeding ahead of schedule. "
                "I'm proud of the team's execution in a challenging macroeconomic environment. "
                "Now I'll turn it over to Luca to walk you through the financials."
            ),
            "order_index": 0,
        },
        {
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "speaker": "Luca Maestri",
            "speaker_role": "CFO",
            "section_type": "prepared_remarks",
            "text": (
                "Thank you, Tim. "
                "Revenue grew 4% year-over-year to $124.3 billion. "
                "Gross margin was 46.9%, up 90 basis points from a year ago. "
                "Operating cash flow was $29.9 billion. "
                "We returned over $30 billion to shareholders during the quarter. "
                "EPS grew 11% to $2.40. "
                "For Q2 FY2025, we expect revenue between $93 and $97 billion. "
                "We expect gross margins between 46.5% and 47.5%. "
                "With that, let me open it up to questions."
            ),
            "order_index": 1,
        },
        {
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "speaker": "Wamsi Mohan",
            "speaker_role": "Analyst",
            "section_type": "qa",
            "text": (
                "Hi, this is Wamsi from Bank of America. "
                "Congratulations on the strong results. "
                "My first question is around Services growth. "
                "Can you provide more color on the sustainability of 14% growth? "
                "And how should we think about the contribution from Apple Intelligence going forward?"
            ),
            "order_index": 2,
        },
        {
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "speaker": "Tim Cook",
            "speaker_role": "CEO",
            "section_type": "qa",
            "text": (
                "Thanks, Wamsi. "
                "We feel very good about Services. "
                "The installed base continues to hit all-time highs. "
                "Transacting accounts on the App Store are also at record levels. "
                "Apple Intelligence will open up entirely new monetization vectors. "
                "We're still in early stages but the engagement data is very encouraging."
            ),
            "order_index": 3,
        },
        {
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "speaker": "Operator",
            "speaker_role": "Operator",
            "section_type": "closing_remarks",
            "text": (
                "That concludes today's Apple Q1 FY2025 earnings conference call. "
                "Thank you for your participation. You may disconnect."
            ),
            "order_index": 4,
        },
        # --- Edge cases ---
        {
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "speaker": "Unknown",
            "speaker_role": "",
            "section_type": "",          # blank section — should normalise to 'unknown'
            "text": "   ",              # whitespace-only — should be dropped
            "order_index": 5,
        },
        {
            "transcript_id": "AAPL_Q1_2025",
            "ticker": "AAPL",
            "speaker": "Tim Cook",
            "speaker_role": "CEO",
            "section_type": "prepared_remarks",
            "text": "OK.",              # too short — should be dropped
            "order_index": 6,
        },
    ]

    df_input = pd.DataFrame(SAMPLE_DATA)

    print(f"\n[INPUT] {len(df_input)} transcript rows\n")

    # -----------------------------------------------------------------------
    # 2. Instantiate and run the segmenter
    # -----------------------------------------------------------------------
    config = SentenceSegmenterConfig(
        min_sentence_chars   = 10,
        min_sentence_tokens  = 3,
        max_sentence_tokens  = 200,
        strip_boilerplate    = True,
        chunk_candidate_window = 5,
    )
    segmenter = SentenceSegmenter(config=config)

    print(f"  Backend selected: {segmenter._backend.name}")
    print()

    result = segmenter.segment_transcript(df_input)

    # -----------------------------------------------------------------------
    # 3. Display results
    # -----------------------------------------------------------------------
    print(f"[RESULT] {result}")
    print(f"\n  Total sentences  : {result.stats.total_sentences}")
    print(f"  Dropped (short)  : {result.stats.dropped_too_short}")
    print(f"  Dropped (empty)  : {result.stats.dropped_empty}")
    print(f"  Token estimate   : {result.stats.total_token_estimate}")
    print(f"  Sections found   : {result.stats.sections_found}")
    print(f"  Speakers found   : {result.stats.speakers_found}")
    if result.stats.warnings:
        print(f"  Warnings         : {result.stats.warnings}")

    print("\n" + "-" * 72)
    print("  Sample sentences:")
    print("-" * 72)
    for seg in result.sentences[:6]:
        print(
            f"  [{seg.sentence_order:03d}] "
            f"{seg.speaker:<18s} | {seg.section_type:<20s} | "
            f"~{seg.token_estimate:>3d} tok | {seg.sentence_text[:65]}"
        )
    if len(result.sentences) > 6:
        print(f"  ... ({len(result.sentences) - 6} more sentences)")

    # -----------------------------------------------------------------------
    # 4. DataFrame export check
    # -----------------------------------------------------------------------
    df_out = result.to_dataframe()
    print(f"\n[EXPORT] DataFrame shape: {df_out.shape}")
    print(f"  Columns: {list(df_out.columns)}")

    # -----------------------------------------------------------------------
    # 5. Validation
    # -----------------------------------------------------------------------
    validator = SegmentationValidator(max_token_limit=512)
    issues = validator.validate(result)

    print("\n[VALIDATION]")
    if issues:
        print(f"  ❌ {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"     • {issue}")
        sys.exit(1)
    else:
        print("  ✅ All validation checks passed.")

    # -----------------------------------------------------------------------
    # 6. Reconstruction round-trip
    # -----------------------------------------------------------------------
    reconstructed = SentenceSegmenter.reconstruct_text(result.sentences)
    print(f"\n[RECONSTRUCTION] Reconstructed text length: {len(reconstructed)} chars")
    print(f"  Preview: {reconstructed[:120]}…")

    # -----------------------------------------------------------------------
    # 7. Determinism check
    # -----------------------------------------------------------------------
    result2 = segmenter.segment_transcript(df_input)
    ids1 = [s.sentence_id for s in result.sentences]
    ids2 = [s.sentence_id for s in result2.sentences]
    deterministic = ids1 == ids2
    print(f"\n[DETERMINISM] Identical IDs on re-run: {'✅ yes' if deterministic else '❌ NO — BUG'}")

    # -----------------------------------------------------------------------
    # 8. Batch segmentation demo (simulate two transcripts)
    # -----------------------------------------------------------------------
    MSFT_ROW = {
        "transcript_id": "MSFT_Q2_2025",
        "ticker": "MSFT",
        "speaker": "Satya Nadella",
        "speaker_role": "CEO",
        "section_type": "prepared_remarks",
        "text": (
            "Good afternoon everyone. "
            "Microsoft delivered a strong quarter driven by Azure growth. "
            "Azure and other cloud services revenue grew 31% in constant currency. "
            "Microsoft 365 Commercial cloud revenue grew 15%. "
            "We are well-positioned to capture the AI infrastructure opportunity. "
            "Copilot is gaining significant enterprise adoption across all segments."
        ),
        "order_index": 0,
    }
    df_combined = pd.concat(
        [df_input, pd.DataFrame([MSFT_ROW])],
        ignore_index=True,
    )
    batch_results = segmenter.segment_dataframe(df_combined)
    print(f"\n[BATCH] Processed {len(batch_results)} transcripts:")
    for r in batch_results:
        print(f"  {r.transcript_id}: {len(r)} sentences")

    combined_df = SentenceSegmenter.results_to_dataframe(batch_results)
    print(f"  Combined DataFrame: {combined_df.shape[0]} rows × {combined_df.shape[1]} cols")

    print("\n" + "=" * 72)
    print("  Self-test complete.")
    print("=" * 72)
