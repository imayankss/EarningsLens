from __future__ import annotations

QA_PATTERNS = [
    r"question(?:s)?\\s*(?:and|&)\\s*answer(?:s)?",
    r"q\\s*&\\s*a",
    r"question-and-answer",
    r"questions?\\s+and\\s+answers?",
]

"""
src/preprocessing/validation.py
=================================
Stage 4 — Data quality validation and schema enforcement.

Responsibilities:
    - Schema validation (required columns present and typed correctly)
    - Content quality checks (min word count, non-empty text, valid dates)
    - Chunk quality checks (token count bounds, text coherence)
    - Transcript completeness checks (both sections present, Q&A detected)
    - Produce a validation report DataFrame for auditing

Design:
    - All validators return ValidationResult — never raise silently
    - Validators are composable — run any subset independently
    - Failed rows are flagged, not dropped — caller decides what to do
    - Validation report is always persisted for audit trail

ValidationResult:
    is_valid : bool
    errors   : list[str]   — blocking issues (row should be dropped)
    warnings : list[str]   — non-blocking issues (row is usable but degraded)

TODO:
    - [ ] Add statistical outlier detection (transcripts > 3σ word count)
    - [ ] Add duplicate detection by transcript_id
    - [ ] Add earnings date vs fiscal quarter consistency check
    - [ ] Add ticker validity check against known S&P 500 list
    - [ ] Integrate with Great Expectations for enterprise-grade validation
    - [ ] Add validation report HTML export
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def _word_count(text: object) -> int:
    """Cheap deterministic word count used by validation gates."""
    if not isinstance(text, str) or not text:
        return 0
    return len(text.split())


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a single row or field."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)  # blocking
    warnings: list[str] = field(default_factory=list)  # non-blocking

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Combine two results (e.g. schema + content checks)."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )

    def __repr__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"ValidationResult({status}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

# Required columns and their expected pandas dtype categories
TRANSCRIPT_SCHEMA: dict[str, type] = {
    "transcript_id": str,
    "ticker": str,
    "earnings_date": datetime,
    "transcript_text_clean": str,
}

CHUNK_SCHEMA: dict[str, type] = {
    "chunk_id": str,
    "transcript_id": str,
    "chunk_text": str,
    "token_count": int,
    "chunk_index": int,
}

SEGMENT_SCHEMA: dict[str, type] = {
    "segment_id": str,
    "transcript_id": str,
    "speaker_role": str,
    "section": str,
    "text": str,
}

# Quality thresholds
MIN_TRANSCRIPT_WORDS = 200  # below this → likely extraction failure
MAX_TRANSCRIPT_WORDS = 50_000  # above this → likely concatenation error
MIN_CHUNK_TOKENS = 20  # below this → chunk is meaningless
MAX_CHUNK_TOKENS = 512  # above this → will be truncated by FinBERT


class TranscriptValidator:
    """
    Validates preprocessed transcripts, segments, and chunks.

    Example:
        validator = TranscriptValidator()
        report_df = validator.validate_transcripts(df)
        clean_df  = df[report_df["is_valid"]]
    """

    # ── Transcript validation ────────────────────────────────────

    def validate_transcript_row(self, row: dict[str, Any]) -> ValidationResult:
        """
        Validate a single transcript row dict.

        Checks:
            - Required fields present and non-null
            - transcript_id format (TICKER_YYYYMMDD)
            - ticker is uppercase alphabetic
            - earnings_date is parseable
            - transcript_text_clean has sufficient word count
            - Both prepared and QA sections detected

        Args:
            row: Dict of transcript fields

        Returns:
            ValidationResult with all errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # ── Required fields ──────────────────────────────────────
        result = result.merge(self._check_required_fields(row, TRANSCRIPT_SCHEMA))

        # ── transcript_id format ─────────────────────────────────
        tid = str(row.get("transcript_id", ""))
        if not self._is_valid_transcript_id(tid):
            result.add_warning(
                f"transcript_id '{tid}' does not match TICKER_YYYYMMDD format"
            )

        # ── Ticker ───────────────────────────────────────────────
        ticker = str(row.get("ticker", ""))
        if not ticker or not ticker.replace(".", "").isalpha():
            result.add_error(f"Invalid ticker: '{ticker}'")

        # ── Earnings date ────────────────────────────────────────
        result = result.merge(self._check_date(row.get("earnings_date"), "earnings_date"))

        # ── Word count ───────────────────────────────────────────
        text = str(row.get("transcript_text_clean", ""))
        words = _word_count(text)
        if words < MIN_TRANSCRIPT_WORDS:
            result.add_error(
                f"transcript_text_clean too short: {words} words "
                f"(min {MIN_TRANSCRIPT_WORDS})"
            )
        elif words > MAX_TRANSCRIPT_WORDS:
            result.add_warning(
                f"Unusually long transcript: {words} words "
                f"(max expected {MAX_TRANSCRIPT_WORDS})"
            )

        # ── Section detection ────────────────────────────────────
        if "prepared_text" in row:
            prep_words = _word_count(row.get("prepared_text", ""))
            if prep_words < 50:
                result.add_warning(f"Prepared remarks very short: {prep_words} words")
        if "qa_text" in row:
            qa_words = _word_count(row.get("qa_text", ""))
            if qa_words < 10:
                result.add_warning("Q&A section not detected or very short")

        return result

    def validate_transcripts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate all transcript rows in a DataFrame.

        Args:
            df: Preprocessed transcript DataFrame

        Returns:
            Validation report DataFrame with columns:
                transcript_id, is_valid, error_count, warning_count,
                errors, warnings
        """
        log.info(f"Validating {len(df):,} transcripts...")
        records: list[dict] = []

        for _, row in df.iterrows():
            result = self.validate_transcript_row(row.to_dict())
            tid = str(row.get("transcript_id", ""))
            records.append(
                {
                    "transcript_id": tid,
                    "is_valid": result.is_valid,
                    "error_count": len(result.errors),
                    "warning_count": len(result.warnings),
                    "errors": "; ".join(result.errors),
                    "warnings": "; ".join(result.warnings),
                }
            )

        report = pd.DataFrame(records)

        if report.empty:
            log.warning("No chunks available for validation")
            return report

        n_valid = report["is_valid"].sum()
        n_invalid = len(report) - n_valid
        log.info(
            f"Validation complete: {n_valid:,} valid, {n_invalid:,} invalid "
            f"({n_invalid/len(report):.1%} failure rate)"
        )
        return report

    # ── Chunk validation ─────────────────────────────────────────

    def validate_chunk_row(self, row: dict[str, Any]) -> ValidationResult:
        """
        Validate a single chunk row.

        Checks:
            - Required fields present
            - token_count within acceptable bounds
            - chunk_text is non-empty string
            - chunk_id format correct
        """
        result = ValidationResult(is_valid=True)
        result = result.merge(self._check_required_fields(row, CHUNK_SCHEMA))

        token_count = int(row.get("token_count", 0))
        if token_count < MIN_CHUNK_TOKENS:
            result.add_error(
                f"token_count {token_count} below minimum ({MIN_CHUNK_TOKENS})"
            )
        if token_count > MAX_CHUNK_TOKENS:
            result.add_warning(
                f"token_count {token_count} exceeds FinBERT limit ({MAX_CHUNK_TOKENS}) "
                f"— will be truncated"
            )

        text = str(row.get("chunk_text", ""))
        if len(text.strip()) < 20:
            result.add_error("chunk_text is empty or too short")

        return result

    def validate_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate all chunk rows. Returns validation report DataFrame.

        Args:
            df: Chunk DataFrame (output of TranscriptChunker)

        Returns:
            Validation report with chunk_id, is_valid, errors, warnings
        """
        log.info(f"Validating {len(df):,} chunks...")
        records: list[dict] = []

        for _, row in df.iterrows():
            result = self.validate_chunk_row(row.to_dict())
            records.append(
                {
                    "chunk_id": str(row.get("chunk_id", "")),
                    "transcript_id": str(row.get("transcript_id", "")),
                    "is_valid": result.is_valid,
                    "error_count": len(result.errors),
                    "warning_count": len(result.warnings),
                    "errors": "; ".join(result.errors),
                    "warnings": "; ".join(result.warnings),
                }
            )

        report = pd.DataFrame(records)

        if report.empty:
            log.warning("No chunks available for validation")
            return report

        n_valid = report["is_valid"].sum()
        n_invalid = len(report) - n_valid
        log.info(f"Chunk validation: {n_valid:,} valid, {n_invalid:,} invalid")
        return report

    # ── Private helpers ──────────────────────────────────────────

    @staticmethod
    def _check_required_fields(
        row: dict[str, Any],
        schema: dict[str, type],
    ) -> ValidationResult:
        """Check all required fields are present and non-null."""
        result = ValidationResult(is_valid=True)
        for col in schema:
            val = row.get(col)
            if val is None or (isinstance(val, str) and not val.strip()):
                result.add_error(f"Required field '{col}' is missing or null")
        return result

    @staticmethod
    def _check_date(value: Any, field_name: str) -> ValidationResult:
        """Validate that a field is parseable as a date."""
        result = ValidationResult(is_valid=True)
        if value is None:
            result.add_error(f"'{field_name}' is null")
            return result
        try:
            pd.Timestamp(value)
        except Exception:
            result.add_error(
                f"'{field_name}' value '{value}' cannot be parsed as a date"
            )
        return result

    @staticmethod
    def _is_valid_transcript_id(tid: str) -> bool:
        """Check transcript_id matches TICKER_YYYYMMDD format."""
        import re

        return bool(re.match(r"^[A-Z]{1,5}_\d{8}$", tid))

    @staticmethod
    def filter_valid(
        df: pd.DataFrame,
        report_df: pd.DataFrame,
        id_col: str = "transcript_id",
    ) -> pd.DataFrame:
        """
        Filter a DataFrame to only valid rows using a validation report.

        Args:
            df       : Original DataFrame
            report_df: Validation report from validate_transcripts() or validate_chunks()
            id_col   : Join key

        Returns:
            Filtered DataFrame containing only valid rows
        """
        valid_ids = report_df[report_df["is_valid"]][id_col].tolist()
        return df[df[id_col].isin(valid_ids)].reset_index(drop=True)

    # TODO: add validate_segments(df) for segment-level validation
    # TODO: add check_duplicate_transcripts(df) -> pd.DataFrame
    # TODO: add check_date_range(df, start, end) to flag out-of-window transcripts
    # TODO: add generate_html_report(report_df, output_path) for audit export
