from __future__ import annotations

import pandas as pd

from src.preprocessing.metadata_extractor import MetadataExtractor
from src.preprocessing.role_classifier import RoleClassifier, SpeakerType
from src.preprocessing.speaker_extractor import SpeakerExtractor


def test_day4_modules_importable() -> None:
    assert SpeakerExtractor is not None
    assert RoleClassifier is not None
    assert MetadataExtractor is not None


def test_metadata_extractor_uses_filename_hints() -> None:
    meta = MetadataExtractor(
        text="Apple Q4 fiscal year 2024 earnings conference call",
        filename="AAPL_Q4_2024.txt",
    ).extract_all()
    assert meta.ticker == "AAPL"
    assert meta.quarter == "Q4"
    assert meta.year == 2024
    assert meta.transcript_id == "AAPL_Q4_2024"


def test_role_classifier_detects_analyst_firm() -> None:
    result = RoleClassifier().classify("Goldman Sachs")
    assert result.speaker_type == SpeakerType.ANALYST


def test_speaker_extractor_parses_inline_realistic_turns() -> None:
    text = """
Suhasini Chandramouli: Welcome to the Apple Q4 2024 earnings call. Tim and Luca will speak first.
Tim Cook: Revenue grew and customer demand remained strong across our product portfolio.
Luca Maestri: Gross margin improved and services revenue reached a new record.
Suhasini Chandramouli: Operator, may we have the first question, please?
Operator: Our first question comes from Michael Ng with Goldman Sachs. Please go ahead.
Michael Ng: Thanks. Can you discuss guidance for next quarter?
Tim Cook: Thanks, Michael. We remain confident in demand and long-term execution.
"""
    result = SpeakerExtractor().extract(text)
    df = pd.DataFrame(result.to_records())

    assert len(df) >= 6
    assert "speaker" in df.columns
    assert set(df["section"]) == {"prepared_remarks", "qa"}
    assert (df["speaker_type"] == "management").any()
    assert (df["speaker_type"] == "analyst").any()
    assert (df["speaker_type"] == "operator").any()
