"""
src/extraction/role_classifier.py

Role normalization and speaker type classification for earnings call transcripts.

Converts messy, inconsistent role strings from raw transcript text into
standardized normalized roles and speaker type categories.

Pipeline Position:
    SpeakerExtractor → RoleClassifier → TranscriptStructurer

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
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SpeakerType(str, Enum):
    """
    Canonical speaker type categories for the earnings call pipeline.

    Used downstream by:
        - FinBERT sentiment segmentation
        - management tone analysis
        - analyst sentiment analysis
        - speaker-level aggregation
    """

    MANAGEMENT = "management"
    ANALYST = "analyst"
    OPERATOR = "operator"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Output Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleClassification:
    """
    Immutable result of classifying a single raw role string.

    Attributes:
        raw_role:        Original unmodified role string from transcript.
        normalized_role: Standardized role label (e.g. "CEO", "CFO", "Analyst").
        speaker_type:    Broad category as SpeakerType enum value.
        confidence:      Heuristic confidence score in [0.0, 1.0].
                         1.0 = exact dictionary match,
                         0.8 = regex/partial match,
                         0.5 = heuristic/firm-name match,
                         0.0 = fallback unknown.
    """

    raw_role: str
    normalized_role: str
    speaker_type: SpeakerType
    confidence: float = 1.0

    def to_dict(self) -> dict[str, str | float]:
        """Serialize to plain dict for DataFrame integration."""
        return {
            "raw_role": self.raw_role,
            "normalized_role": self.normalized_role,
            "speaker_type": self.speaker_type.value,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Mapping Tables  (centralized — easy to extend without touching logic)
# ---------------------------------------------------------------------------


# Exact and near-exact title → normalized role
# Keys are lowercased for case-insensitive lookup.
EXECUTIVE_ROLE_MAP: dict[str, str] = {
    # CEO variants
    "chief executive officer": "CEO",
    "ceo": "CEO",
    "president and ceo": "CEO",
    "president & ceo": "CEO",
    "president/ceo": "CEO",
    "co-chief executive officer": "Co-CEO",
    "co-ceo": "Co-CEO",
    # CFO variants
    "chief financial officer": "CFO",
    "cfo": "CFO",
    "chief finance officer": "CFO",
    "evp and cfo": "CFO",
    "evp & cfo": "CFO",
    "svp and cfo": "CFO",
    "svp & cfo": "CFO",
    # COO variants
    "chief operating officer": "COO",
    "coo": "COO",
    "president and coo": "COO",
    "president & coo": "COO",
    # CTO / CIO
    "chief technology officer": "CTO",
    "cto": "CTO",
    "chief information officer": "CIO",
    "cio": "CIO",
    "chief digital officer": "CDO",
    # CMO / CSO / CCO
    "chief marketing officer": "CMO",
    "cmo": "CMO",
    "chief strategy officer": "CSO",
    "chief commercial officer": "CCO",
    "chief revenue officer": "CRO",
    "chief accounting officer": "CAO",
    "chief legal officer": "CLO",
    "general counsel": "General Counsel",
    # President
    "president": "President",
    "executive chairman": "Executive Chairman",
    "chairman": "Chairman",
    "chairman and ceo": "CEO",
    "chairman & ceo": "CEO",
    # EVP / SVP / VP
    "executive vice president": "EVP",
    "evp": "EVP",
    "senior vice president": "SVP",
    "svp": "SVP",
    "vice president": "VP",
    "vp": "VP",
    "vp finance": "VP Finance",
    "vp investor relations": "VP IR",
    "vp of investor relations": "VP IR",
    "vice president of finance": "VP Finance",
    "vice president of investor relations": "VP IR",
    # Investor Relations
    "head of investor relations": "Head of IR",
    "director of investor relations": "Director of IR",
    "investor relations": "IR",
    "ir": "IR",
    # Analyst / Sell-side
    "analyst": "Analyst",
    "research analyst": "Analyst",
    "equity research analyst": "Analyst",
    "senior analyst": "Analyst",
    "managing director": "Managing Director",
    "md": "Managing Director",
    # Operator
    "operator": "Operator",
    "conference operator": "Operator",
    "conference call operator": "Operator",
    "moderator": "Operator",
}


# Keyword fragments → normalized role  (checked when exact map misses)
# Order matters: more specific patterns first.
ROLE_KEYWORD_MAP: list[tuple[str, str]] = [
    ("chief executive", "CEO"),
    ("chief financial", "CFO"),
    ("chief operating", "COO"),
    ("chief technology", "CTO"),
    ("chief information", "CIO"),
    ("chief marketing", "CMO"),
    ("chief strategy", "CSO"),
    ("chief revenue", "CRO"),
    ("chief accounting", "CAO"),
    ("chief legal", "CLO"),
    ("chief", "C-Suite"),          # generic fallback for unknown C-suite
    ("executive vice president", "EVP"),
    ("senior vice president", "SVP"),
    ("vice president", "VP"),
    ("president", "President"),
    ("chairman", "Chairman"),
    ("general counsel", "General Counsel"),
    ("investor relations", "IR"),
    ("head of ir", "Head of IR"),
    ("director", "Director"),
    ("treasurer", "Treasurer"),
    ("controller", "Controller"),
    ("analyst", "Analyst"),
    ("operator", "Operator"),
    ("moderator", "Operator"),
]


# Known investment bank / sell-side firm name fragments (lowercase)
# Used to detect analyst speakers when role text is a firm name.
ANALYST_FIRM_FRAGMENTS: frozenset[str] = frozenset(
    [
        "goldman sachs",
        "goldman",
        "jpmorgan",
        "jp morgan",
        "morgan stanley",
        "bank of america",
        "bofa",
        "merrill lynch",
        "citigroup",
        "citi",
        "wells fargo",
        "barclays",
        "ubs",
        "deutsche bank",
        "credit suisse",
        "hsbc",
        "jefferies",
        "piper sandler",
        "raymond james",
        "stifel",
        "baird",
        "cowen",
        "td cowen",
        "needham",
        "oppenheimer",
        "truist",
        "bmo",
        "rbc",
        "susquehanna",
        "bernstein",
        "wolfe research",
        "evercore",
        "mizuho",
        "keybanc",
        "wedbush",
        "canaccord",
        "rosenblatt",
        "new street",
        "atlantic equities",
        "redburn",
        "exane",
    ]
)


# Speaker types for normalized management roles
MANAGEMENT_ROLES: frozenset[str] = frozenset(
    [
        "CEO", "Co-CEO", "CFO", "COO", "CTO", "CIO", "CMO", "CSO",
        "CCO", "CRO", "CAO", "CLO", "CDO", "C-Suite",
        "EVP", "SVP", "VP", "VP Finance", "VP IR",
        "President", "Executive Chairman", "Chairman",
        "General Counsel", "Head of IR", "Director of IR",
        "IR", "Director", "Treasurer", "Controller",
        "Managing Director",
    ]
)

ANALYST_ROLES: frozenset[str] = frozenset(["Analyst"])

OPERATOR_ROLES: frozenset[str] = frozenset(["Operator"])


# ---------------------------------------------------------------------------
# RoleClassifier
# ---------------------------------------------------------------------------


class RoleClassifier:
    """
    Normalizes raw role/title strings from earnings call transcripts and
    assigns each speaker to a canonical SpeakerType category.

    Design principles:
        - All mapping tables live outside the class for easy extension.
        - Classification is a deterministic pipeline: exact → keyword →
          firm-name → operator → unknown.  Each stage reports a confidence.
        - No external dependencies beyond stdlib + re.
        - Thread-safe (stateless after construction).

    Typical usage::

        classifier = RoleClassifier()
        result = classifier.classify("Chief Executive Officer")
        # RoleClassification(raw_role='Chief Executive Officer',
        #                     normalized_role='CEO',
        #                     speaker_type=<SpeakerType.MANAGEMENT: 'management'>,
        #                     confidence=1.0)

    Custom mappings can be injected at construction time::

        classifier = RoleClassifier(
            extra_role_map={"founder": "Founder"},
            extra_firm_fragments={"citadel", "renaissance"},
        )
    """

    def __init__(
        self,
        extra_role_map: Optional[dict[str, str]] = None,
        extra_firm_fragments: Optional[set[str]] = None,
    ) -> None:
        """
        Initialize classifier with optional extension mappings.

        Args:
            extra_role_map:       Additional {raw_lower: normalized} entries
                                  merged on top of EXECUTIVE_ROLE_MAP.
            extra_firm_fragments: Additional analyst firm name fragments
                                  merged into ANALYST_FIRM_FRAGMENTS.
        """
        # Merge base maps with any caller-supplied overrides
        self._role_map: dict[str, str] = {**EXECUTIVE_ROLE_MAP, **(extra_role_map or {})}
        self._firm_fragments: frozenset[str] = (
            ANALYST_FIRM_FRAGMENTS | frozenset(f.lower() for f in (extra_firm_fragments or set()))
        )

        # Pre-compile a single regex for operator detection (fast path)
        self._operator_re: re.Pattern[str] = re.compile(
            r"\b(operator|moderator|conference\s+operator|call\s+operator)\b",
            re.IGNORECASE,
        )

        logger.debug(
            "RoleClassifier initialized | role_map_size=%d | firm_fragments=%d",
            len(self._role_map),
            len(self._firm_fragments),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, raw_role: str) -> RoleClassification:
        """
        Full classification pipeline for a single raw role string.

        Classification stages (first match wins):
            1. Exact dictionary lookup  → confidence 1.0
            2. Keyword substring scan   → confidence 0.8
            3. Analyst firm-name match  → confidence 0.5
            4. Operator regex match     → confidence 0.9
            5. Unknown fallback         → confidence 0.0

        Args:
            raw_role: Raw role string as it appears in the transcript
                      (e.g. "Chief Executive Officer", "Goldman Sachs").

        Returns:
            RoleClassification with normalized role and speaker type.
        """
        if not raw_role or not raw_role.strip():
            logger.debug("Empty raw_role — returning unknown")
            return RoleClassification(
                raw_role=raw_role,
                normalized_role="Unknown",
                speaker_type=SpeakerType.UNKNOWN,
                confidence=0.0,
            )

        cleaned = self._clean_input(raw_role)
        logger.debug("classify | raw='%s' | cleaned='%s'", raw_role, cleaned)

        # Stage 1 — exact match
        if (norm := self._role_map.get(cleaned)) is not None:
            return self._build_result(raw_role, norm, confidence=1.0)

        # Stage 2 — keyword substring
        if (norm := self._match_keywords(cleaned)) is not None:
            return self._build_result(raw_role, norm, confidence=0.8)

        # Stage 3 — analyst firm name
        if self._detect_analyst_firm(cleaned):
            return self._build_result(raw_role, "Analyst", confidence=0.5)

        # Stage 4 — operator regex
        if self._detect_operator(cleaned):
            return self._build_result(raw_role, "Operator", confidence=0.9)

        # Stage 5 — unknown fallback
        logger.debug("No match found for raw_role='%s'", raw_role)
        return RoleClassification(
            raw_role=raw_role,
            normalized_role="Unknown",
            speaker_type=SpeakerType.UNKNOWN,
            confidence=0.0,
        )

    def normalize_role(self, raw_role: str) -> str:
        """
        Return only the normalized role string.

        Convenience wrapper around classify() for callers that do not
        need the full RoleClassification object.

        Args:
            raw_role: Raw role string.

        Returns:
            Normalized role label (e.g. "CEO", "CFO", "Analyst").
        """
        return self.classify(raw_role).normalized_role

    def classify_speaker_type(self, raw_role: str) -> SpeakerType:
        """
        Return only the SpeakerType for a raw role string.

        Args:
            raw_role: Raw role string.

        Returns:
            SpeakerType enum value.
        """
        return self.classify(raw_role).speaker_type

    def classify_batch(self, raw_roles: list[str]) -> list[RoleClassification]:
        """
        Classify a list of raw role strings.

        Useful for vectorized application over a DataFrame column.

        Args:
            raw_roles: List of raw role strings.

        Returns:
            List of RoleClassification in the same order.
        """
        return [self.classify(r) for r in raw_roles]

    def detect_analyst(self, raw_role: str) -> bool:
        """
        Return True if the raw role string suggests a sell-side analyst.

        Checks:
            - Normalized role == "Analyst"
            - Role text contains a known investment firm name fragment

        Args:
            raw_role: Raw role string.

        Returns:
            True if analyst, False otherwise.
        """
        result = self.classify(raw_role)
        return result.speaker_type == SpeakerType.ANALYST

    def detect_operator(self, raw_role: str) -> bool:
        """
        Return True if the raw role string identifies a conference operator.

        Args:
            raw_role: Raw role string.

        Returns:
            True if operator, False otherwise.
        """
        result = self.classify(raw_role)
        return result.speaker_type == SpeakerType.OPERATOR

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_input(raw: str) -> str:
        """
        Normalize input string for dictionary and keyword lookup.

        Steps:
            - Strip leading/trailing whitespace
            - Collapse internal whitespace
            - Lowercase
            - Remove common punctuation noise (periods, commas)
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.lower()
        cleaned = re.sub(r"[.,;]", "", cleaned)
        return cleaned

    def _match_keywords(self, cleaned: str) -> Optional[str]:
        """
        Scan ROLE_KEYWORD_MAP for the first fragment present in cleaned string.

        Args:
            cleaned: Pre-lowercased, whitespace-normalized role string.

        Returns:
            Normalized role if a keyword matches, else None.
        """
        for fragment, normalized in ROLE_KEYWORD_MAP:
            if fragment in cleaned:
                logger.debug("Keyword match: '%s' → '%s'", fragment, normalized)
                return normalized
        return None

    def _detect_analyst_firm(self, cleaned: str) -> bool:
        """
        Check if any known investment firm fragment appears in cleaned string.

        This catches cases where the 'role' column in a transcript contains
        the analyst's firm name rather than a job title, e.g.:
            "Goldman Sachs" → Analyst/management

        Args:
            cleaned: Pre-lowercased role string.

        Returns:
            True if a firm fragment is found.
        """
        for firm in self._firm_fragments:
            if firm in cleaned:
                logger.debug("Analyst firm match: '%s'", firm)
                return True
        return False

    def _detect_operator(self, cleaned: str) -> bool:
        """
        Use pre-compiled regex to detect operator/moderator patterns.

        Args:
            cleaned: Pre-lowercased role string.

        Returns:
            True if operator pattern found.
        """
        return bool(self._operator_re.search(cleaned))

    def _build_result(
        self, raw_role: str, normalized_role: str, confidence: float
    ) -> RoleClassification:
        """
        Construct a RoleClassification, deriving speaker_type from normalized_role.

        Args:
            raw_role:        Original raw string.
            normalized_role: Resolved normalized role label.
            confidence:      Heuristic confidence score.

        Returns:
            RoleClassification instance.
        """
        speaker_type = self._resolve_speaker_type(normalized_role)
        logger.debug(
            "Result: raw='%s' → normalized='%s' | type=%s | conf=%.2f",
            raw_role, normalized_role, speaker_type.value, confidence,
        )
        return RoleClassification(
            raw_role=raw_role,
            normalized_role=normalized_role,
            speaker_type=speaker_type,
            confidence=confidence,
        )

    @staticmethod
    def _resolve_speaker_type(normalized_role: str) -> SpeakerType:
        """
        Map a normalized role label to its SpeakerType category.

        Lookup order:
            1. MANAGEMENT_ROLES set
            2. ANALYST_ROLES set
            3. OPERATOR_ROLES set
            4. Fallback to UNKNOWN

        Args:
            normalized_role: Already-normalized role label.

        Returns:
            Appropriate SpeakerType.
        """
        if normalized_role in MANAGEMENT_ROLES:
            return SpeakerType.MANAGEMENT
        if normalized_role in ANALYST_ROLES:
            return SpeakerType.ANALYST
        if normalized_role in OPERATOR_ROLES:
            return SpeakerType.OPERATOR
        return SpeakerType.UNKNOWN
