"""
Run the official Loughran-McDonald sentiment pipeline.

Default contract:
    input : data/interim/chunks/chunks.parquet
    output: data/processed/sentiment/lm_scores.parquet

The real Loughran-McDonald Master Dictionary CSV is required by default.
Use --allow-stub-dictionary only for smoke tests and local wiring checks.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sentiment.lm_pipeline import LMPipeline, LMPipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Loughran-McDonald dictionary sentiment scoring."
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/interim/chunks/chunks.parquet"),
        help="Chunk parquet input path.",
    )
    parser.add_argument(
        "--dictionary-path",
        type=Path,
        default=Path("data/dictionaries/loughran_mcdonald/LM_MasterDictionary.csv"),
        help="Path to the Loughran-McDonald Master Dictionary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/sentiment"),
        help="Directory for LM parquet/CSV outputs.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory for LM intermediate cache files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of chunks to score per batch.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing LM outputs/cache.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write parquet outputs only.",
    )
    parser.add_argument(
        "--allow-stub-dictionary",
        action="store_true",
        help="Use the small built-in stub dictionary for tests only.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = parse_args()

    config = LMPipelineConfig(
        chunks_path=args.chunks_path,
        dictionary_path=args.dictionary_path,
        output_dir=args.output_dir,
        interim_dir=args.interim_dir,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        overwrite=args.overwrite,
        export_csv=not args.no_csv,
        export_parquet=True,
        allow_stub_dictionary=args.allow_stub_dictionary,
    )
    result = LMPipeline(config).run()
    print(result.summary())

    contract_path = args.output_dir / "lm_scores.parquet"
    if not result.success or not contract_path.exists():
        print(
            f"LM pipeline failed to create required contract file: {contract_path}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
