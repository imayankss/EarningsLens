from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.metadata_extractor import MetadataExtractor
from src.preprocessing.speaker_extractor import SpeakerExtractor


def validate_segments(df: pd.DataFrame) -> None:
    assert len(df) > 0, "No speaker segments produced"
    assert "speaker" in df.columns, "Missing speaker column"
    assert "text" in df.columns, "Missing text column"
    assert df["text"].str.len().mean() > 20, "Average segment text is too short"

    sections = df["section"].tolist() if "section" in df.columns else []
    if "prepared_remarks" in sections and "qa" in sections:
        assert sections.index("prepared_remarks") < sections.index("qa"), (
            "Expected prepared_remarks to appear before qa"
        )


def run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    text = path.read_text(encoding="utf-8")

    meta = MetadataExtractor(text=text, filename=path.name).extract_all()
    extractor = SpeakerExtractor()
    result = extractor.extract(text)

    segments_df = pd.DataFrame(result.to_records())
    validate_segments(segments_df)

    meta_df = pd.DataFrame([meta.to_dict()])
    transcript_id = meta.transcript_id or path.stem
    segments_df.insert(0, "transcript_id", transcript_id)
    segments_df.insert(1, "source_file", path.name)

    metadata_dir = Path("data/interim/metadata")
    segments_dir = Path("data/interim/segmented_transcripts")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    meta_out = metadata_dir / f"{path.stem}_metadata.parquet"
    segments_out = segments_dir / f"{path.stem}_segments.parquet"
    meta_df.to_parquet(meta_out, index=False)
    segments_df.to_parquet(segments_out, index=False)

    print("=" * 72)
    print(f"File: {path}")
    print(f"Transcript ID: {transcript_id}")
    print(f"Metadata output: {meta_out}")
    print(f"Segments output: {segments_out}")
    print("=" * 72)
    print("Metadata:")
    print(meta_df.to_string(index=False))
    print("\nSegment shape:", segments_df.shape)
    print("\nSection counts:")
    print(segments_df["section"].value_counts(dropna=False).to_string())
    print("\nSpeaker types:")
    print(segments_df["speaker_type"].value_counts(dropna=False).to_string())
    print("\nParse patterns:")
    print(pd.Series(result.parse_pattern_counts).to_string())
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings[:10]:
            print(f"- {warning}")
    print("\nSample rows:")
    cols = [
        "sequence_index", "speaker", "speaker_type", "section",
        "normalized_role", "word_count", "parse_pattern", "text",
    ]
    print(segments_df[cols].head(12).to_string(index=False, max_colwidth=100))

    return meta_df, segments_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Day 4 transcript parsing")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/raw/sample_transcripts/AAPL_Q4_2024.txt",
        help="Transcript .txt file to parse",
    )
    args = parser.parse_args()
    run(Path(args.path))


if __name__ == "__main__":
    main()
