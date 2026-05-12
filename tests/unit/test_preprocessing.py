"""Unit tests for preprocessing modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
from src.preprocessing.cleaner import TranscriptCleaner


class TestTranscriptCleaner:
    def setup_method(self):
        self.cleaner = TranscriptCleaner()

    def test_clean_empty_string(self):
        assert self.cleaner.clean("") == ""

    def test_clean_removes_extra_whitespace(self):
        result = self.cleaner.clean("Hello   world  \n\n\n  test")
        assert "   " not in result

    def test_clean_unicode_normalization(self):
        result = self.cleaner.clean("Revenue\u00a0grew")
        assert "\u00a0" not in result

    def test_separate_sections_returns_keys(self):
        sections = self.cleaner.separate_sections("Some prepared remarks here.")
        assert "prepared" in sections
        assert "qa" in sections
        assert "full" in sections

    def test_separate_sections_splits_qa(self):
        text = ("Management remarks. " * 10 +
                "Question-and-answer session. Analyst: What is guidance?")
        sections = self.cleaner.separate_sections(text)
        assert len(sections["prepared"]) > 0
        assert len(sections["qa"]) > 0
