"""
src/extraction/speaker_extractor.py

Production-grade speaker segmentation engine for earnings call transcripts.

Parses raw transcript text into a structured sequence of SpeakerBlock records,
each containing:
    - speaker name
    - normalized role (via RoleClassifier)
    - speaker type (management / analyst / operator / unknown)
    - section label (prepared_remarks / qa / closing_remarks)
    - full speech text
    - sequence index (preserves original speaking order)

Pipeline Position:
    TranscriptParser ──▶ SpeakerExtractor ──▶ RoleClassifier
                                         └──▶ TranscriptStructurer

Supported Transcript Formats:
    - Motley Fool / Seeking Alpha  → "Tim Cook -- CEO"
    - Bloomberg / Refinitiv        → "TIMOTHY D. COOK, CEO, APPLE INC.:"
    - SEC EDGAR filing format      → "Timothy Cook:"
    - Generic / no separator       → "Operator" (standalone token)
    - Role-first format            → "CEO Tim Cook"
    - Analyst with firm            → "John Doe -- Goldman Sachs"

Author: Earnings Call Sentiment Analyzer Pipeline
Python: 3.11+
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# NOTE: In production, import pattern constants from src/utils/regex_patterns.py
# The SPEAKER_HEADER_PATTERNS, SECTION_BOUNDARY_PATTERNS, and NOISE_LINE_PATTERNS
# constants below follow the same centralized-registry convention used there.
# ---------------------------------------------------------------------------
from src.preprocessing.role_classifier import RoleClassifier, SpeakerType

logger = logging.getLogger(__name__)


# ===========================================================================
# Enums
# ===========================================================================


class TranscriptSection(str, Enum):
    """
    Canonical section labels for earnings call structure.

    PREPARED_REMARKS : Opening management statements (CEO, CFO monologues).
    QA               : Question-and-answer session with analysts.
    CLOSING_REMARKS  : Closing statements and operator sign-off.
    UNKNOWN          : Section could not be determined.
    """

    PREPARED_REMARKS = "prepared_remarks"
    QA = "qa"
    CLOSING_REMARKS = "closing_remarks"
    UNKNOWN = "unknown"


# ===========================================================================
# Regex Pattern Registry
# (Mirror / import these from src/utils/regex_patterns.py in production)
# ===========================================================================

# ------------------------------------------------------------------
# Speaker header patterns
# Each entry: (pattern_name: str, compiled_pattern: re.Pattern)
# Named capture groups required: `name` (always), `role` (optional).
# Ordered from most-specific to least-specific — first match wins.
# ------------------------------------------------------------------
SPEAKER_HEADER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # ----------------------------------------------------------------
    # Format 1 — NAME -- ROLE  (double dash / em dash / en dash)
    # e.g. "Tim Cook -- CEO"
    # e.g. "Luca Maestri — Chief Financial Officer"
    # ----------------------------------------------------------------
    (
        "double_dash",
        re.compile(
            r"^(?P<name>[A-Z][a-zA-Z.''\-]+(?:\s+[A-Z][a-zA-Z.''\-]+)*)"
            r"\s+(?:--|[-–—])\s+"
            r"(?P<role>[A-Za-z][A-Za-z0-9\s&,.()/'\-]{1,80})\s*$",
        ),
    ),
    # ----------------------------------------------------------------
    # Format 2 — NAME - ROLE  (single dash with spaces)
    # e.g. "John Smith - Goldman Sachs"
    # Requires at least 2 words in name to avoid matching sentences.
    # ----------------------------------------------------------------
    (
        "single_dash",
        re.compile(
            r"^(?P<name>[A-Z][a-zA-Z.''\-]+(?:\s+[A-Z][a-zA-Z.''\-]+)+)"
            r"\s+-\s+"
            r"(?P<role>[A-Za-z][A-Za-z0-9\s&,.()/'\-]{1,80})\s*$",
        ),
    ),
    # ----------------------------------------------------------------
    # Format 3 — NAME, ROLE:  (Bloomberg / Refinitiv terminal style)
    # e.g. "Timothy D. Cook, Chief Executive Officer, Apple Inc.:"
    # e.g. After allcaps normalization: "Luca Maestri, CFO:"
    # ----------------------------------------------------------------
    (
        "comma_role_colon",
        re.compile(
            r"^(?P<name>[A-Z][a-zA-Z.''\-]+(?:\s+[A-Z][a-zA-Z.''\-]+)*)"
            r",\s+"
            r"(?P<role>[A-Za-z][A-Za-z0-9\s&,.()/'\-]{1,80})"
            r":?\s*$",
        ),
    ),
    # ----------------------------------------------------------------
    # Format 4 — NAME:  (SEC filing / plain format, name only)
    # e.g. "Timothy Cook:"
    # Requires minimum 2-word name to avoid matching short labels.
    # ----------------------------------------------------------------
    (
        "name_colon_only",
        re.compile(
            r"^(?P<name>[A-Z][a-zA-Z.''\-]+(?:\s+[A-Z][a-zA-Z.''\-]+)+)"
            r":\s*$",
        ),
    ),
    # ----------------------------------------------------------------
    # Format 5 — ROLE - NAME  (role precedes name)
    # e.g. "Analyst John Doe"
    # Restricted to known short role prefixes to avoid false positives.
    # ----------------------------------------------------------------
    (
        "role_prefix_name",
        re.compile(
            r"^(?P<role>CEO|CFO|COO|CTO|CIO|CMO|SVP|EVP|VP|President"
            r"|Analyst|Operator|Director|Chairman)"
            r"\s+(?P<name>[A-Z][a-zA-Z.''\-]+(?:\s+[A-Z][a-zA-Z.''\-]+)*)\s*$",
        ),
    ),
    # ----------------------------------------------------------------
    # Format 6 — Standalone known tokens  (no role, no separator)
    # e.g. "Operator", "Moderator", "Conference Operator"
    # ----------------------------------------------------------------
    (
        "standalone_token",
        re.compile(
            r"^(?P<name>Operator|Moderator|Coordinator"
            r"|Conference\s+Operator|Conference\s+Moderator"
            r"|Unidentified\s+(?:Analyst|Participant|Speaker))"
            r"\s*:?\s*$",
            re.IGNORECASE,
        ),
    ),
    # ----------------------------------------------------------------
    # Format 7 — ALL-CAPS NAME  (Bloomberg terminal / raw feeds)
    # e.g. "TIMOTHY COOK"  |  "LUCA MAESTRI"
    # Optional trailing comma + role + colon.
    # ----------------------------------------------------------------
    (
        "allcaps_name",
        re.compile(
            r"^(?P<name>[A-Z]{2,}(?:[\s.]+[A-Z]{2,})+)"
            r"(?:\s*[,:-]\s*(?P<role>[A-Za-z][A-Za-z0-9\s&,.()/'\-]{1,80}))?"
            r":?\s*$",
        ),
    ),
]

# ------------------------------------------------------------------
# Inline speaker turns: "Tim Cook: Thank you..."
# Common in Bose345/Hugging Face transcript content. Captures the speaker
# name separately from the same-line speech body. Single-word labels are
# intentionally restricted to Operator-like special speakers.
# ------------------------------------------------------------------
INLINE_SPEAKER_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<name>Operator|Moderator|Coordinator|"
    r"[A-Z][a-zA-Z.''\-]+(?:\s+[A-Z][a-zA-Z.''\-]+){1,4})"
    r"\s*:\s+"
    r"(?P<speech>\S.*)$",
)

# ------------------------------------------------------------------
# Section boundary detection patterns
# Scanned against each speaker's speech text (and Operator text).
# First match in priority order determines the section transition.
# ------------------------------------------------------------------
SECTION_BOUNDARY_PATTERNS: list[tuple[re.Pattern[str], TranscriptSection]] = [
    (
        re.compile(
            r"\b(prepared\s+remarks?|management\s+remarks?|opening\s+(?:remarks?|statement))\b",
            re.IGNORECASE,
        ),
        TranscriptSection.PREPARED_REMARKS,
    ),
    (
        re.compile(
            r"\b("
            r"question[s]?\s+and\s+answer[s]?"
            r"|q\s*(?:and|&)\s*a"
            r"|now\s+(?:take|open|begin|start)\s+(?:your\s+)?questions?"
            r"|open\s+(?:the\s+)?(?:floor|call)\s+(?:for|to)\s+questions?"
            r"|now\s+turn\s+(?:the\s+call\s+)?(?:over\s+)?(?:back\s+)?to\s+questions?"
            r"|we\s+will\s+now\s+(?:begin|conduct|open)\s+(?:the\s+)?question"
            r")\b",
            re.IGNORECASE,
        ),
        TranscriptSection.QA,
    ),
    (
        re.compile(
            r"\b("
            r"closing\s+remarks?"
            r"|concluding\s+remarks?"
            r"|that\s+(?:concludes?|ends?|completes?)\s+(?:today['s]*\s+)?(?:call|conference|presentation|earnings)"
            r"|this\s+(?:concludes?|ends?|completes?)\s+(?:the\s+)?(?:call|conference|earnings\s+call)"
            r"|thank\s+you\s+(?:all\s+)?for\s+(?:joining|participating|attending|your\s+time)"
            r")\b",
            re.IGNORECASE,
        ),
        TranscriptSection.CLOSING_REMARKS,
    ),
]

# ------------------------------------------------------------------
# Noise line patterns — lines to skip entirely during parsing
# ------------------------------------------------------------------
NOISE_LINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-=*]{3,}\s*$"),
    re.compile(r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*(?:applause|laughter|pause|silence|inaudible|crosstalk)\s*\]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\(?\s*(?:applause|laughter|pause|inaudible|crosstalk)\s*\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:end\s+of\s+transcript|transcript\s+end)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*(?:operator\s+instructions?)\s*\]?\s*$", re.IGNORECASE),
]

# Thresholds
MIN_SPEECH_LENGTH: int = 10
MIN_SPEECH_WORDS: int = 3
MAX_HEADER_LINE_LENGTH: int = 120


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class SpeakerHeader:
    """
    Result of parsing a single transcript header line.

    Attributes:
        raw_line:      Original, unmodified header line.
        speaker_name:  Extracted and cleaned speaker name.
        raw_role:      Role string extracted from line (may be empty string).
        parse_pattern: Name of SPEAKER_HEADER_PATTERNS entry that matched.
        confidence:    Parse-level confidence: 1.0 = high-confidence pattern,
                       0.7 = fallback/allcaps, 0.5 = standalone token.
    """

    raw_line: str
    speaker_name: str
    raw_role: str
    parse_pattern: str
    confidence: float = 1.0


@dataclass
class SpeakerBlock:
    """
    A single speaker turn extracted from an earnings call transcript.

    This is the primary atomic unit produced by SpeakerExtractor and
    consumed by TranscriptStructurer and FinBERT sentiment scoring.

    Attributes:
        speaker:          Normalized speaker name.
        raw_role:         Role string as it appeared in the transcript.
        normalized_role:  Standardized role label from RoleClassifier.
        speaker_type:     Canonical category string (management/analyst/operator/unknown).
        section:          Transcript section label string.
        text:             Complete speech text for this turn (whitespace-normalized).
        sequence_index:   Zero-based position in the transcript (ORDER PRESERVED).
        role_confidence:  RoleClassifier confidence for this block's role.
        parse_pattern:    Header regex pattern that produced this block.
        word_count:       Word count of `text` (auto-computed, not in constructor).
    """

    speaker: str
    raw_role: str
    normalized_role: str
    speaker_type: str
    section: str
    text: str
    sequence_index: int
    role_confidence: float = 1.0
    parse_pattern: str = "unknown"
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split()) if self.text else 0

    def to_dict(self) -> dict[str, str | int | float]:
        """Serialize to flat dict — one DataFrame row per block."""
        return {
            "speaker": self.speaker,
            "raw_role": self.raw_role,
            "normalized_role": self.normalized_role,
            "speaker_type": self.speaker_type,
            "section": self.section,
            "text": self.text,
            "sequence_index": self.sequence_index,
            "word_count": self.word_count,
            "role_confidence": self.role_confidence,
            "parse_pattern": self.parse_pattern,
        }


@dataclass
class ExtractionConfig:
    """
    Runtime configuration controlling SpeakerExtractor behavior.

    All thresholds are tunable without touching extraction logic, making
    this the single place for pipeline engineers to adjust behavior.

    Attributes:
        min_speech_length:        Minimum character count for valid speech text.
        min_speech_words:         Minimum word count for valid speech text.
        max_header_line_length:   Lines longer than this are never treated as headers.
        deduplicate_blocks:       Remove exact-duplicate (speaker, text) combos.
        normalize_allcaps:        Title-case ALL-CAPS names (Bloomberg format).
        include_operator_blocks:  Keep Operator turns in the output list.
        default_section:          Label for blocks before any boundary is detected.
        continuation_on_no_match: Append unmatched paragraphs to the previous speaker.
    """

    min_speech_length: int = MIN_SPEECH_LENGTH
    min_speech_words: int = MIN_SPEECH_WORDS
    max_header_line_length: int = MAX_HEADER_LINE_LENGTH
    deduplicate_blocks: bool = True
    normalize_allcaps: bool = True
    include_operator_blocks: bool = True
    default_section: TranscriptSection = TranscriptSection.PREPARED_REMARKS
    continuation_on_no_match: bool = True


# ===========================================================================
# ExtractionResult — output container
# ===========================================================================


@dataclass
class ExtractionResult:
    """
    Complete output of a single SpeakerExtractor.extract() call.

    Attributes:
        blocks:            Ordered list of SpeakerBlock records.
        total_blocks:      Number of blocks before filtering.
        filtered_blocks:   Number of blocks removed by quality/dedup filters.
        section_counts:    Counts per TranscriptSection value.
        parse_pattern_counts: Header pattern frequency — useful for diagnostics.
        warnings:          Non-fatal issues encountered during extraction.
    """

    blocks: list[SpeakerBlock]
    total_blocks: int
    filtered_blocks: int
    section_counts: dict[str, int]
    parse_pattern_counts: dict[str, int]
    warnings: list[str]

    @property
    def management_blocks(self) -> list[SpeakerBlock]:
        return [b for b in self.blocks if b.speaker_type == SpeakerType.MANAGEMENT.value]

    @property
    def analyst_blocks(self) -> list[SpeakerBlock]:
        return [b for b in self.blocks if b.speaker_type == SpeakerType.ANALYST.value]

    @property
    def operator_blocks(self) -> list[SpeakerBlock]:
        return [b for b in self.blocks if b.speaker_type == SpeakerType.OPERATOR.value]

    def to_records(self) -> list[dict[str, str | int | float]]:
        """Return list of flat dicts — directly usable as pd.DataFrame(records)."""
        return [b.to_dict() for b in self.blocks]


# ===========================================================================
# SpeakerExtractor — main class
# ===========================================================================


class SpeakerExtractor:
    """
    Production-grade speaker segmentation engine for earnings call transcripts.

    Converts raw transcript text into an ordered sequence of SpeakerBlock
    records, each representing a single speaker turn with full metadata.

    Core algorithm:
        1. Pre-process:   normalize line endings, strip noise lines
        2. Segment:       split text into paragraph-level chunks
        3. Parse headers: attempt header detection on each chunk's first line
        4. Accumulate:    collect speech lines per speaker turn
        5. Section label: scan accumulated text for section boundary signals
        6. Classify roles: delegate to RoleClassifier
        7. Post-process:  filter short blocks, deduplicate, assign sequence index

    Typical usage::

        extractor = SpeakerExtractor()
        result = extractor.extract(raw_text)
        df = pd.DataFrame(result.to_records())

    Inject custom config::

        config = ExtractionConfig(include_operator_blocks=False)
        extractor = SpeakerExtractor(config=config)

    Inject custom role classifier (e.g. with extra firm mappings)::

        classifier = RoleClassifier(extra_firm_fragments={"citadel"})
        extractor = SpeakerExtractor(role_classifier=classifier)
    """

    def __init__(
        self,
        role_classifier: Optional[RoleClassifier] = None,
        config: Optional[ExtractionConfig] = None,
    ) -> None:
        """
        Initialize the extractor.

        Args:
            role_classifier: Shared RoleClassifier instance. If None, a
                             default instance is created.
            config:          ExtractionConfig controlling thresholds and flags.
                             If None, defaults are used.
        """
        self._classifier: RoleClassifier = role_classifier or RoleClassifier()
        self._config: ExtractionConfig = config or ExtractionConfig()

        logger.info(
            "SpeakerExtractor initialized | include_operator=%s | dedup=%s | "
            "min_words=%d | continuation=%s",
            self._config.include_operator_blocks,
            self._config.deduplicate_blocks,
            self._config.min_speech_words,
            self._config.continuation_on_no_match,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def extract(self, transcript_text: str) -> ExtractionResult:
        """
        Full extraction pipeline for a complete transcript string.

        Args:
            transcript_text: Raw text of a single earnings call transcript.

        Returns:
            ExtractionResult containing ordered SpeakerBlock list and diagnostics.
        """
        if not transcript_text or not transcript_text.strip():
            logger.warning("extract() called with empty transcript_text")
            return ExtractionResult(
                blocks=[],
                total_blocks=0,
                filtered_blocks=0,
                section_counts={},
                parse_pattern_counts={},
                warnings=["Empty transcript provided"],
            )

        warnings: list[str] = []

        # Stage 1 — pre-process
        normalized_text = self._preprocess_text(transcript_text)

        # Stage 2 — segment into paragraphs
        paragraphs = self._split_into_paragraphs(normalized_text)
        logger.debug("Segmented into %d paragraphs", len(paragraphs))

        # Stage 3 & 4 — parse headers and accumulate raw blocks
        raw_blocks = self._parse_paragraphs(paragraphs, warnings)
        logger.debug("Parsed %d raw blocks", len(raw_blocks))

        # Stage 5 — section labeling
        blocks_with_sections = self._assign_sections(raw_blocks)

        # Stage 6 — context inference
        # Bose345-style transcripts often use "Name: speech" without role text.
        # Infer stable management names from prepared remarks, then mark unknown
        # Q&A speakers outside that registry as analysts.
        blocks_with_sections = self._infer_speaker_types_from_context(blocks_with_sections)

        # Stage 7 — post-process
        total_before = len(blocks_with_sections)
        final_blocks = self._postprocess_blocks(blocks_with_sections, warnings)
        filtered_count = total_before - len(final_blocks)

        # Diagnostics
        section_counts = self._count_sections(final_blocks)
        pattern_counts = self._count_patterns(final_blocks)

        logger.info(
            "Extraction complete | blocks=%d | filtered=%d | sections=%s",
            len(final_blocks),
            filtered_count,
            section_counts,
        )

        return ExtractionResult(
            blocks=final_blocks,
            total_blocks=total_before,
            filtered_blocks=filtered_count,
            section_counts=section_counts,
            parse_pattern_counts=pattern_counts,
            warnings=warnings,
        )

    def extract_speaker_blocks(
        self, transcript_text: str
    ) -> list[dict[str, str | int | float]]:
        """
        Convenience wrapper — returns list of plain dicts (DataFrame-ready).

        Args:
            transcript_text: Raw transcript string.

        Returns:
            List of dicts, one per SpeakerBlock.
        """
        return self.extract(transcript_text).to_records()

    # ==================================================================
    # Stage 1 — Pre-processing
    # ==================================================================

    def _preprocess_text(self, text: str) -> str:
        """
        Normalize raw transcript text before parsing.

        Operations:
            - Unify line endings to LF
            - Remove carriage returns
            - Collapse runs of 3+ blank lines to 2
            - Strip trailing whitespace from each line
            - Optionally title-case ALL-CAPS lines (Bloomberg format)

        Args:
            text: Raw transcript string.

        Returns:
            Normalized string.
        """
        # Unify line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines: list[str] = []
        for line in text.split("\n"):
            line = line.rstrip()

            # Skip pure noise lines
            if self._is_noise_line(line):
                logger.debug("Skipping noise line: %r", line)
                continue

            # Normalize ALL-CAPS lines if configured
            if self._config.normalize_allcaps and self._is_allcaps_line(line):
                line = self._normalize_allcaps_line(line)

            lines.append(line)

        # Collapse 3+ consecutive blank lines → 2 blank lines
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized

    # ==================================================================
    # Stage 2 — Paragraph segmentation
    # ==================================================================

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """
        Split normalized text into speaker-aware paragraph chunks.

        Bose345 transcripts commonly store one speaker turn per line using
        inline labels such as ``Tim Cook: Thank you...`` rather than blank-line
        separated header blocks. This splitter starts a new paragraph whenever
        a line looks like either a standalone speaker header or an inline
        speaker turn, while still preserving continuation lines.
        """
        paragraphs: list[str] = []
        current: list[str] = []

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            starts_speaker = (
                self._parse_speaker_header(line) is not None
                or self._parse_inline_speaker_line(line) is not None
            )

            if starts_speaker and current:
                paragraphs.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)

        if current:
            paragraphs.append("\n".join(current).strip())

        return paragraphs

    # ==================================================================
    # Stage 3 & 4 — Header parsing + block accumulation
    # ==================================================================

    def _parse_paragraphs(
        self, paragraphs: list[str], warnings: list[str]
    ) -> list[SpeakerBlock]:
        """
        Iterate over paragraphs, detect speaker headers, accumulate speech.

        Each paragraph is examined:
            - If its first line matches a speaker header pattern → new block.
            - Otherwise, if continuation_on_no_match is True → append to
              the previous block's text.
            - Otherwise → log a warning and skip.

        Args:
            paragraphs: List of paragraph strings.
            warnings:   Mutable list to append non-fatal warnings to.

        Returns:
            Unfiltered list of SpeakerBlock records in transcript order.
        """
        blocks: list[SpeakerBlock] = []
        current_speaker_text_accumulator: list[str] = []
        current_header: Optional[SpeakerHeader] = None
        sequence_counter = 0

        def _flush_current() -> None:
            """Commit accumulated text into a SpeakerBlock and reset state."""
            nonlocal current_header, sequence_counter
            if current_header is None:
                return
            text = self.merge_multiline_speech(current_speaker_text_accumulator)
            block = self._build_speaker_block(
                header=current_header,
                text=text,
                section=TranscriptSection.UNKNOWN,  # assigned later
                sequence_index=sequence_counter,
            )
            blocks.append(block)
            sequence_counter += 1
            current_speaker_text_accumulator.clear()

        for para_idx, paragraph in enumerate(paragraphs):
            lines = paragraph.split("\n")
            first_line = lines[0].strip()
            rest_lines = lines[1:] if len(lines) > 1 else []

            inline_candidate = self._parse_inline_speaker_line(first_line)

            # Skip paragraphs that are too long to be standalone headers AND have no rest.
            # Long inline turns like "Tim Cook: <long speech>" are valid and must
            # still be parsed.
            if (
                len(first_line) > self._config.max_header_line_length
                and not rest_lines
                and inline_candidate is None
            ):
                # Entire paragraph is likely body text — append if we have a speaker
                if current_header and self._config.continuation_on_no_match:
                    current_speaker_text_accumulator.append(paragraph)
                else:
                    warnings.append(
                        f"Para {para_idx}: no active speaker for continuation text "
                        f"({len(first_line)} chars)"
                    )
                continue

            # Attempt header parse on first line. Inline turns such as
            # "Tim Cook: Thank you..." provide both header and speech text.
            inline = inline_candidate
            if inline is not None:
                header, inline_speech = inline
            else:
                header = self._parse_speaker_header(first_line)
                inline_speech = ""

            if header is not None:
                # Flush previous block before starting new one
                _flush_current()
                current_header = header
                # Rest of this paragraph is the start of speech text
                speech_parts = []
                if inline_speech:
                    speech_parts.append(inline_speech)
                if rest_lines:
                    speech_parts.append("\n".join(rest_lines).strip())
                speech_start = "\n".join(p for p in speech_parts if p).strip()
                if speech_start:
                    current_speaker_text_accumulator.append(speech_start)

                logger.debug(
                    "Para %d: header matched [%s] → name=%r role=%r",
                    para_idx,
                    header.parse_pattern,
                    header.speaker_name,
                    header.raw_role,
                )

            else:
                # No header match — handle as continuation or orphan
                if current_header and self._config.continuation_on_no_match:
                    current_speaker_text_accumulator.append(paragraph)
                    logger.debug("Para %d: appended as continuation", para_idx)
                elif current_header is None:
                    # First paragraph, no header yet — likely preamble/metadata
                    logger.debug(
                        "Para %d: no header match and no active speaker (preamble?), skipping",
                        para_idx,
                    )
                else:
                    warnings.append(
                        f"Para {para_idx}: no header match and continuation disabled. "
                        f"Skipping: {first_line[:60]!r}"
                    )

        # Flush the final accumulated block
        _flush_current()

        return blocks

    # ==================================================================
    # Stage 5 — Section assignment
    # ==================================================================

    def _assign_sections(
        self, blocks: list[SpeakerBlock]
    ) -> list[SpeakerBlock]:
        """
        Walk through blocks in sequence and assign TranscriptSection labels.

        Section transitions are triggered by:
            1. Operator speech announcing Q&A / closing
            2. Any block whose text contains a section boundary pattern

        State machine:
            PREPARED_REMARKS (default)
                 └─▶ QA               (on Q&A boundary signal)
                       └─▶ CLOSING_REMARKS  (on closing signal)

        Closing remarks can also be detected directly from PREPARED_REMARKS
        if Q&A is absent (unusual but handled).

        Args:
            blocks: Blocks with section=TranscriptSection.UNKNOWN.

        Returns:
            New list of SpeakerBlock with section fields populated.
        """
        current_section: TranscriptSection = self._config.default_section
        labeled: list[SpeakerBlock] = []

        for block in blocks:
            # Label the current block with the active section first. Boundary
            # phrases usually announce what comes next ("we will now begin Q&A"),
            # so the transition is applied after this speaker turn.
            updated = SpeakerBlock(
                speaker=block.speaker,
                raw_role=block.raw_role,
                normalized_role=block.normalized_role,
                speaker_type=block.speaker_type,
                section=current_section.value,
                text=block.text,
                sequence_index=block.sequence_index,
                role_confidence=block.role_confidence,
                parse_pattern=block.parse_pattern,
            )
            labeled.append(updated)

            detected_section = self._detect_section_from_text(block.text)
            if detected_section is not None and self._is_forward_transition(
                current_section, detected_section
            ):
                logger.debug(
                    "Section transition: %s → %s  (speaker=%r)",
                    current_section.value,
                    detected_section.value,
                    block.speaker,
                )
                current_section = detected_section

        return labeled

    def _infer_speaker_types_from_context(
        self, blocks: list[SpeakerBlock]
    ) -> list[SpeakerBlock]:
        """Infer management/analyst labels when inline speaker turns omit roles."""
        management_names = {
            block.speaker.lower()
            for block in blocks
            if block.section == TranscriptSection.PREPARED_REMARKS.value
            and block.speaker_type not in {SpeakerType.OPERATOR.value, SpeakerType.ANALYST.value}
            and block.speaker.lower() != "operator"
        }

        inferred: list[SpeakerBlock] = []
        for block in blocks:
            speaker_key = block.speaker.lower()
            speaker_type = block.speaker_type
            normalized_role = block.normalized_role
            confidence = block.role_confidence

            if speaker_type == SpeakerType.UNKNOWN.value:
                if speaker_key in management_names:
                    speaker_type = SpeakerType.MANAGEMENT.value
                    normalized_role = "Management" if normalized_role == "Unknown" else normalized_role
                    confidence = max(confidence, 0.40)
                elif (
                    block.section == TranscriptSection.QA.value
                    and speaker_key != "operator"
                ):
                    speaker_type = SpeakerType.ANALYST.value
                    normalized_role = "Analyst"
                    confidence = max(confidence, 0.45)
                elif (
                    block.section == TranscriptSection.PREPARED_REMARKS.value
                    and speaker_key != "operator"
                ):
                    speaker_type = SpeakerType.MANAGEMENT.value
                    normalized_role = "Management" if normalized_role == "Unknown" else normalized_role
                    confidence = max(confidence, 0.35)

            inferred.append(
                SpeakerBlock(
                    speaker=block.speaker,
                    raw_role=block.raw_role,
                    normalized_role=normalized_role,
                    speaker_type=speaker_type,
                    section=block.section,
                    text=block.text,
                    sequence_index=block.sequence_index,
                    role_confidence=confidence,
                    parse_pattern=block.parse_pattern,
                )
            )

        return inferred

    # ==================================================================
    # Stage 7 — Post-processing
    # ==================================================================

    def _postprocess_blocks(
        self, blocks: list[SpeakerBlock], warnings: list[str]
    ) -> list[SpeakerBlock]:
        """
        Apply quality filters and deduplication to the block list.

        Filters applied (in order):
            1. Remove blocks with text below minimum length / word count.
            2. Remove blocks for excluded speaker types (e.g. operator).
            3. Deduplicate exact (speaker, text) pairs.
            4. Re-index sequence_index to be contiguous after filtering.

        Args:
            blocks:   Labeled SpeakerBlock list.
            warnings: Mutable warnings list.

        Returns:
            Cleaned, re-indexed SpeakerBlock list.
        """
        filtered: list[SpeakerBlock] = []

        for block in blocks:
            # Quality: text too short
            if len(block.text) < self._config.min_speech_length:
                logger.debug(
                    "Filtered (too short): speaker=%r len=%d",
                    block.speaker,
                    len(block.text),
                )
                continue

            # Quality: too few words
            if block.word_count < self._config.min_speech_words:
                logger.debug(
                    "Filtered (too few words): speaker=%r words=%d",
                    block.speaker,
                    block.word_count,
                )
                continue

            # Config: operator blocks excluded
            if (
                not self._config.include_operator_blocks
                and block.speaker_type == SpeakerType.OPERATOR.value
            ):
                logger.debug("Filtered (operator excluded): speaker=%r", block.speaker)
                continue

            filtered.append(block)

        # Deduplication
        if self._config.deduplicate_blocks:
            original_count = len(filtered)
            filtered = self._deduplicate(filtered)
            removed = original_count - len(filtered)
            if removed:
                warnings.append(f"Deduplicated {removed} duplicate speaker block(s)")
                logger.debug("Deduplication removed %d block(s)", removed)

        # Re-index sequence numbers to be contiguous after filtering
        reindexed: list[SpeakerBlock] = []
        for idx, block in enumerate(filtered):
            reindexed.append(
                SpeakerBlock(
                    speaker=block.speaker,
                    raw_role=block.raw_role,
                    normalized_role=block.normalized_role,
                    speaker_type=block.speaker_type,
                    section=block.section,
                    text=block.text,
                    sequence_index=idx,
                    role_confidence=block.role_confidence,
                    parse_pattern=block.parse_pattern,
                )
            )

        return reindexed

    # ==================================================================
    # Public utility methods
    # ==================================================================

    def merge_multiline_speech(self, lines: list[str]) -> str:
        """
        Merge a list of raw speech lines into clean, normalized paragraph text.

        Rules:
            - Join non-empty lines with single spaces within a paragraph.
            - Preserve paragraph boundaries (double newline) when present
              in source chunks.
            - Collapse runs of whitespace.
            - Strip leading/trailing whitespace from the result.

        Args:
            lines: List of raw speech text chunks or lines.

        Returns:
            Single normalized string.
        """
        if not lines:
            return ""

        # Re-join all chunks, then re-split by blank line to preserve paragraphs
        full_text = "\n".join(lines)
        paragraphs = re.split(r"\n{2,}", full_text)
        cleaned_paragraphs: list[str] = []

        for para in paragraphs:
            # Collapse internal whitespace within each paragraph
            para_lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
            if para_lines:
                cleaned_paragraphs.append(" ".join(para_lines))

        merged = "\n\n".join(cleaned_paragraphs)
        return merged.strip()

    def identify_special_speakers(
        self, raw_name: str, raw_role: str
    ) -> tuple[str, str]:
        """
        Normalize special speaker names to canonical forms.

        Handles:
            - "Operator" / "Moderator" → canonical "Operator"
            - "Unidentified Analyst" variants → "Unidentified Analyst"
            - "Unidentified Participant" variants → "Unidentified Participant"

        Args:
            raw_name: Extracted speaker name.
            raw_role: Extracted role string.

        Returns:
            Tuple of (canonical_name, canonical_role).
        """
        name_lower = raw_name.strip().lower()

        if re.match(r"^(operator|moderator|conference\s+operator|coordinator)$", name_lower):
            return "Operator", "Operator"

        if re.match(r"^unidentified\s+analyst$", name_lower):
            return "Unidentified Analyst", "Analyst"

        if re.match(r"^unidentified\s+(participant|speaker)$", name_lower):
            return "Unidentified Participant", raw_role

        return raw_name, raw_role

    # ==================================================================
    # Private helpers — header parsing
    # ==================================================================

    def _parse_speaker_header(self, line: str) -> Optional[SpeakerHeader]:
        """
        Attempt to parse a single line as a speaker header.

        Tries each pattern in SPEAKER_HEADER_PATTERNS in order.
        Returns the first successful match, or None if no pattern matches.

        Args:
            line: Single transcript line (already stripped).

        Returns:
            SpeakerHeader if the line is a speaker header, else None.
        """
        if not line or len(line) > self._config.max_header_line_length:
            return None

        for pattern_name, pattern in SPEAKER_HEADER_PATTERNS:
            match = pattern.match(line)
            if match is None:
                continue

            groups = match.groupdict()
            raw_name: str = groups.get("name", "").strip()
            raw_role: str = (groups.get("role") or "").strip()

            if not raw_name:
                continue

            # Normalize ALL-CAPS name if needed (allcaps_name pattern)
            if pattern_name == "allcaps_name" and self._config.normalize_allcaps:
                raw_name = self._normalize_allcaps_line(raw_name)

            # Special-speaker normalization
            raw_name, raw_role = self.identify_special_speakers(raw_name, raw_role)

            # Assign confidence based on pattern reliability
            confidence = self._pattern_confidence(pattern_name)

            logger.debug(
                "_parse_speaker_header: line=%r → pattern=%s name=%r role=%r conf=%.2f",
                line,
                pattern_name,
                raw_name,
                raw_role,
                confidence,
            )

            return SpeakerHeader(
                raw_line=line,
                speaker_name=raw_name,
                raw_role=raw_role,
                parse_pattern=pattern_name,
                confidence=confidence,
            )

        return None

    def _parse_inline_speaker_line(
        self, line: str
    ) -> Optional[tuple[SpeakerHeader, str]]:
        """Parse lines like ``Tim Cook: Thank you...`` into header + speech."""
        if not line or len(line) <= 3:
            return None

        match = INLINE_SPEAKER_PATTERN.match(line)
        if match is None:
            return None

        raw_name = match.group("name").strip()
        speech = match.group("speech").strip()
        if not raw_name or not speech:
            return None

        raw_name, raw_role = self.identify_special_speakers(raw_name, "")
        if raw_name == "Operator":
            raw_role = "Operator"

        header = SpeakerHeader(
            raw_line=line,
            speaker_name=raw_name,
            raw_role=raw_role,
            parse_pattern="inline_colon",
            confidence=self._pattern_confidence("inline_colon"),
        )
        return header, speech

    def _build_speaker_block(
        self,
        header: SpeakerHeader,
        text: str,
        section: TranscriptSection,
        sequence_index: int,
    ) -> SpeakerBlock:
        """
        Combine a parsed SpeakerHeader with its accumulated speech text
        into a fully-classified SpeakerBlock.

        Delegates role normalization and speaker type classification to
        the injected RoleClassifier instance.

        Args:
            header:         Parsed header for this speaker turn.
            text:           Merged speech text.
            section:        Current section label (may be UNKNOWN at this stage).
            sequence_index: Position counter.

        Returns:
            SpeakerBlock with all fields populated.
        """
        classification = self._classifier.classify(header.raw_role)

        return SpeakerBlock(
            speaker=header.speaker_name,
            raw_role=header.raw_role,
            normalized_role=classification.normalized_role,
            speaker_type=classification.speaker_type.value,
            section=section.value,
            text=text,
            sequence_index=sequence_index,
            role_confidence=classification.confidence,
            parse_pattern=header.parse_pattern,
        )

    # ==================================================================
    # Private helpers — section detection
    # ==================================================================

    def _detect_section_from_text(
        self, text: str
    ) -> Optional[TranscriptSection]:
        """
        Scan a block's speech text for section boundary signals.

        Patterns are checked in the order defined in SECTION_BOUNDARY_PATTERNS.
        The first match is returned; subsequent patterns are not checked.

        Args:
            text: Speech text of a single SpeakerBlock.

        Returns:
            TranscriptSection if a boundary is detected, else None.
        """
        lowered = text.lower()

        # Closing detection is intentionally concrete. Opening remarks often say
        # "thank you for joining us", which is not a closing boundary.
        closing_markers = (
            "this concludes",
            "that concludes",
            "wraps up the q&a",
            "wraps up the question",
            "concludes today's conference",
            "disconnect your lines",
            "end of transcript",
        )
        if any(marker in lowered for marker in closing_markers):
            return TranscriptSection.CLOSING_REMARKS

        # Q&A detection is conservative. Opening remarks often say Q&A "will
        # follow", which should not mark prepared remarks as Q&A.
        qa_markers = (
            "we'll now move over to q&a",
            "we will now move over to q&a",
            "now move over to q&a",
            "move over to q&a",
            "begin the question",
            "begin our question",
            "open the line for questions",
            "open the lines for questions",
            "first question",
            "next question",
            "last question",
            "question from",
            "question comes from",
        )
        if any(marker in lowered for marker in qa_markers):
            return TranscriptSection.QA

        return None

    @staticmethod
    def _is_forward_transition(
        current: TranscriptSection, detected: TranscriptSection
    ) -> bool:
        """
        Enforce a monotonically-forward section state machine.

        Valid transitions:
            PREPARED_REMARKS → QA
            PREPARED_REMARKS → CLOSING_REMARKS  (Q&A-less call)
            QA               → CLOSING_REMARKS

        Invalid (ignored):
            CLOSING_REMARKS → anything
            QA              → PREPARED_REMARKS
            Same section    → same section

        Args:
            current:  Active section.
            detected: Newly detected section.

        Returns:
            True if the transition is valid and should be applied.
        """
        if current == detected:
            return False

        order = {
            TranscriptSection.PREPARED_REMARKS: 0,
            TranscriptSection.QA: 1,
            TranscriptSection.CLOSING_REMARKS: 2,
            TranscriptSection.UNKNOWN: -1,
        }
        return order.get(detected, -1) > order.get(current, -1)

    # ==================================================================
    # Private helpers — text utilities
    # ==================================================================

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        """Return True if the line matches any NOISE_LINE_PATTERNS entry."""
        if not line.strip():
            return False  # blank lines are handled by paragraph splitting
        for pattern in NOISE_LINE_PATTERNS:
            if pattern.match(line):
                return True
        return False

    @staticmethod
    def _is_allcaps_line(line: str) -> bool:
        """
        Return True if the line is predominantly ALL-CAPS (Bloomberg style).

        A line is considered ALL-CAPS if it has at least 4 alpha characters
        and ≥ 80% of alphabetic characters are uppercase.
        """
        alpha_chars = [c for c in line if c.isalpha()]
        if len(alpha_chars) < 4:
            return False
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        return upper_ratio >= 0.80

    @staticmethod
    def _normalize_allcaps_line(line: str) -> str:
        """
        Title-case an ALL-CAPS line while preserving known abbreviations.

        Known preserved abbreviations: CEO, CFO, COO, CTO, CIO, EVP, SVP,
        VP, MD, IR, USA, INC, LLC, LP, LTD.

        Args:
            line: ALL-CAPS string.

        Returns:
            Title-cased string with abbreviations preserved.
        """
        _PRESERVED = frozenset(
            ["CEO", "CFO", "COO", "CTO", "CIO", "CMO", "EVP", "SVP",
             "VP", "MD", "IR", "USA", "INC", "LLC", "LP", "LTD", "PLC"]
        )
        words = line.split()
        result: list[str] = []
        for word in words:
            # Strip trailing punctuation for check, re-attach afterward
            stripped = word.rstrip(".,;:")
            punct = word[len(stripped):]
            if stripped.upper() in _PRESERVED:
                result.append(stripped.upper() + punct)
            else:
                result.append(stripped.title() + punct)
        return " ".join(result)

    # ==================================================================
    # Private helpers — deduplication and diagnostics
    # ==================================================================

    @staticmethod
    def _deduplicate(blocks: list[SpeakerBlock]) -> list[SpeakerBlock]:
        """
        Remove SpeakerBlock entries with duplicate (speaker, text) pairs.

        First occurrence is kept; subsequent duplicates are dropped.
        This handles cases where malformed transcripts repeat sections.

        Args:
            blocks: List of SpeakerBlock records.

        Returns:
            Deduplicated list preserving original order.
        """
        seen: set[tuple[str, str]] = set()
        unique: list[SpeakerBlock] = []
        for block in blocks:
            key = (block.speaker.lower(), block.text[:200].strip().lower())
            if key in seen:
                logger.debug(
                    "Duplicate removed: speaker=%r text_prefix=%r",
                    block.speaker,
                    block.text[:60],
                )
                continue
            seen.add(key)
            unique.append(block)
        return unique

    @staticmethod
    def _count_sections(blocks: list[SpeakerBlock]) -> dict[str, int]:
        """Return frequency map of section values."""
        counts: dict[str, int] = {}
        for block in blocks:
            counts[block.section] = counts.get(block.section, 0) + 1
        return counts

    @staticmethod
    def _count_patterns(blocks: list[SpeakerBlock]) -> dict[str, int]:
        """Return frequency map of parse_pattern values — useful for debugging."""
        counts: dict[str, int] = {}
        for block in blocks:
            counts[block.parse_pattern] = counts.get(block.parse_pattern, 0) + 1
        return counts

    @staticmethod
    def _pattern_confidence(pattern_name: str) -> float:
        """
        Return a heuristic parse confidence based on which pattern matched.

        Higher confidence patterns are more specific / less ambiguous.

        double_dash        → 1.0  (canonical earnings transcript format)
        comma_role_colon   → 0.95 (Bloomberg style, reliable)
        single_dash        → 0.90 (less specific than double dash)
        standalone_token   → 0.90 (Operator line — unambiguous)
        name_colon_only    → 0.80 (no role, must infer from context)
        role_prefix_name   → 0.80 (role-first, fairly reliable)
        allcaps_name       → 0.70 (Bloomberg raw — name normalization needed)
        unknown            → 0.50 (fallback)
        """
        _MAP: dict[str, float] = {
            "double_dash": 1.00,
            "comma_role_colon": 0.95,
            "single_dash": 0.90,
            "standalone_token": 0.90,
            "name_colon_only": 0.80,
            "role_prefix_name": 0.80,
            "allcaps_name": 0.70,
            "inline_colon": 0.75,
        }
        return _MAP.get(pattern_name, 0.50)
