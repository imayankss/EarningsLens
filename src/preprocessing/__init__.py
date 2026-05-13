QA_PATTERNS = [
    r"question(?:s)?\\s*(?:and|&)\\s*answer(?:s)?",
    r"q\\s*&\\s*a",
    r"question-and-answer",
    r"questions?\\s+and\\s+answers?",
]


MAX_TOKENS = 420
OVERLAP = 60

"""
src/preprocessing/__init__.py
===============================
Preprocessing package for earnings call transcript NLP.

Modules:
    text_cleaner            — unicode, boilerplate, whitespace normalisation
    transcript_segmenter    — speaker detection, Q&A separation, section tagging
    chunking                — sliding-window token chunker for FinBERT (512-token limit)
    validation              — schema, quality, and completeness checks
    preprocessing_pipeline  — single entry-point that chains all stages in order

Typical usage:
    from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    result   = pipeline.run(transcripts_df)
"""

from .chunking import TranscriptChunker
from .preprocessing_pipeline import PreprocessingPipeline
from .text_cleaner import TextCleaner
from .transcript_segmenter import TranscriptSegmenter
from .validation import TranscriptValidator

__all__ = [
    "TextCleaner",
    "TranscriptSegmenter",
    "TranscriptChunker",
    "TranscriptValidator",
    "PreprocessingPipeline",
]
