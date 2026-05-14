from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.metadata_extractor import MetadataExtractor
from src.preprocessing.speaker_extractor import SpeakerExtractor

log = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    source_file: str
    transcript_id: str
    success: bool
    segment_count: int = 0
    unknown_speaker_pct: float = 0.0
    analyst_count: int = 0
    operator_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "transcript_id": self.transcript_id,
            "success": self.success,
            "segment_count": self.segment_count,
            "unknown_speaker_pct": self.unknown_speaker_pct,
            "analyst_count": self.analyst_count,
            "operator_count": self.operator_count,
            "error": self.error,
        }


def validate_segments(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("No speaker segments produced")
    if "speaker" not in df.columns:
        raise ValueError("Missing speaker column")
    if "text" not in df.columns:
        raise ValueError("Missing text column")
    if df["text"].str.len().mean() <= 20:
        raise ValueError("Average segment text length is too short")

    sections = df["section"].tolist() if "section" in df.columns else []
    if "prepared_remarks" in sections and "qa" in sections:
        if sections.index("prepared_remarks") > sections.index("qa"):
            raise ValueError("prepared_remarks appears after qa")


def process_file(
    path: Path,
    metadata_dir: Path,
    segments_dir: Path,
    extractor: SpeakerExtractor | None = None,
) -> ProcessResult:
    extractor = extractor or SpeakerExtractor()
    try:
        text = path.read_text(encoding="utf-8")
        meta = MetadataExtractor(text=text, filename=path.name).extract_all()
        result = extractor.extract(text)

        segments_df = pd.DataFrame(result.to_records())
        validate_segments(segments_df)

        transcript_id = meta.transcript_id or path.stem
        meta_df = pd.DataFrame([meta.to_dict()])
        segments_df.insert(0, "transcript_id", transcript_id)
        segments_df.insert(1, "source_file", path.name)

        metadata_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)
        meta_df.to_parquet(metadata_dir / f"{path.stem}_metadata.parquet", index=False)
        segments_df.to_parquet(segments_dir / f"{path.stem}_segments.parquet", index=False)

        unknown_pct = float((segments_df["speaker_type"] == "unknown").mean() * 100)
        analyst_count = int((segments_df["speaker_type"] == "analyst").sum())
        operator_count = int((segments_df["speaker_type"] == "operator").sum())

        return ProcessResult(
            source_file=path.name,
            transcript_id=transcript_id,
            success=True,
            segment_count=len(segments_df),
            unknown_speaker_pct=round(unknown_pct, 2),
            analyst_count=analyst_count,
            operator_count=operator_count,
        )
    except Exception as exc:  # noqa: BLE001 - batch processor should continue
        log.exception("Failed to process %s", path)
        return ProcessResult(
            source_file=path.name,
            transcript_id=path.stem,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def build_metrics(results: list[ProcessResult]) -> pd.DataFrame:
    rows = [r.to_dict() for r in results]
    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df

    summary = {
        "total_transcripts": int(len(metrics_df)),
        "failed_transcripts": int((~metrics_df["success"]).sum()),
        "avg_segments": float(metrics_df.loc[metrics_df["success"], "segment_count"].mean() or 0),
        "avg_unknown_speaker_pct": float(
            metrics_df.loc[metrics_df["success"], "unknown_speaker_pct"].mean() or 0
        ),
        "total_analyst_count": int(metrics_df["analyst_count"].sum()),
        "total_operator_count": int(metrics_df["operator_count"].sum()),
    }
    print("\nParsing metrics:")
    print(json.dumps(summary, indent=2))
    return metrics_df


def process_batch(
    input_dir: Path = Path("data/raw/sample_transcripts"),
    metadata_dir: Path = Path("data/interim/metadata"),
    segments_dir: Path = Path("data/interim/segmented_transcripts"),
) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt transcripts found in {input_dir}")

    extractor = SpeakerExtractor()
    results = [process_file(path, metadata_dir, segments_dir, extractor) for path in files]
    metrics_df = build_metrics(results)

    metrics_out = Path("data/interim/parsing_metrics.parquet")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_parquet(metrics_out, index=False)
    print(f"Saved parsing metrics -> {metrics_out}")

    failures = metrics_df[~metrics_df["success"]] if not metrics_df.empty else pd.DataFrame()
    if not failures.empty:
        failure_out = Path("data/interim/parsing_failures.csv")
        failures.to_csv(failure_out, index=False)
        print(f"Saved parser failures -> {failure_out}")

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process Day 4 sample transcripts")
    parser.add_argument("--input-dir", default="data/raw/sample_transcripts")
    parser.add_argument("--metadata-dir", default="data/interim/metadata")
    parser.add_argument("--segments-dir", default="data/interim/segmented_transcripts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    metrics_df = process_batch(
        input_dir=Path(args.input_dir),
        metadata_dir=Path(args.metadata_dir),
        segments_dir=Path(args.segments_dir),
    )
    print("\nPer-transcript metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
