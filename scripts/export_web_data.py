"""Export verified EarningsLens artifacts for the static Next.js dashboard.

This script intentionally reads already-generated CSV/JSON artifacts. It does not
download transcripts or market data, load FinBERT, or train a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("web/public/data/dashboard.json")


def _read_csv(path: Path, *, required: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required web export source is missing: {path}")
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _number(value: Any) -> float | None:
    normalized = _text(value)
    if normalized is None:
        return None
    try:
        result = float(normalized)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    result = _number(value)
    return int(result) if result is not None else None


def _boolean(value: Any) -> bool | None:
    normalized = (_text(value) or "").lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def sanitize_json(value: Any) -> Any:
    """Recursively replace unsupported/non-finite values with JSON null."""

    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return sanitize_json(value.item())
    return str(value)


def _index(rows: list[dict[str, str]], key: str = "transcript_id") -> dict[str, dict[str, str]]:
    return {
        identifier: row
        for row in rows
        if (identifier := (_text(row.get(key)) or ""))
    }


def _group(rows: list[dict[str, str]], key: str = "transcript_id") -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        identifier = _text(row.get(key))
        if identifier:
            grouped[identifier].append(row)
    return dict(grouped)


def _modeling_summary(root: Path) -> dict[str, Any]:
    metadata = _read_json(root / "models/predictive_model_metadata.json")
    metrics = _read_csv(root / "reports/tables/prediction_model_metrics.csv")
    trained = metadata.get("models_trained")
    attempted = metadata.get("models_attempted")
    skipped = metadata.get("skipped_reasons")
    row_count = metadata.get("row_count")
    feature_columns = metadata.get("feature_columns")

    return {
        "observationCount": row_count if isinstance(row_count, int) else None,
        "featureCount": len(feature_columns) if isinstance(feature_columns, list) else None,
        "modelsAttempted": len(attempted) if isinstance(attempted, list) else len(metrics),
        "modelsTrained": len(trained) if isinstance(trained, list) else 0,
        "status": "insufficient_sample_size" if skipped else "available",
        "message": (
            "Training is intentionally skipped because the verified dataset has one observation."
            if skipped
            else "Model metadata is available."
        ),
    }


def build_export(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    master_path = root / "data/processed/master_dataset.csv"
    masters = _read_csv(master_path, required=True)
    if not masters:
        raise ValueError(f"Required web export source has no rows: {master_path}")

    advanced = _index(_read_csv(root / "data/processed/nlp/advanced_nlp_features.csv"))
    comparisons = _index(_read_csv(root / "reports/tables/sentiment_model_comparison.csv"))
    topics = _group(_read_csv(root / "reports/tables/topic_frequency.csv"))
    speakers = _group(_read_csv(root / "reports/tables/speaker_sentiment_summary.csv"))

    analyses: list[dict[str, Any]] = []
    for master in masters:
        transcript_id = _text(master.get("transcript_id"))
        if transcript_id is None:
            continue

        nlp = advanced.get(transcript_id, {})
        comparison = comparisons.get(transcript_id, {})
        speaker_rows = speakers.get(transcript_id, [])
        speaker_groups = [
            {
                "label": _text(row.get("dominant_speaker")) or _text(row.get("section_type")),
                "chunkCount": _integer(row.get("chunk_count")),
                "averageSentimentScore": _number(row.get("avg_sentiment_score")),
                "sentimentStdDev": _number(row.get("std_sentiment_score")),
                "positiveChunks": _integer(row.get("positive_count")),
                "negativeChunks": _integer(row.get("negative_count")),
                "neutralChunks": _integer(row.get("neutral_count")),
            }
            for row in speaker_rows
        ]
        labeled_speaker_groups = [row for row in speaker_groups if row["label"]]
        aggregate_speaker = speaker_groups[0] if speaker_groups else None

        market_windows = []
        for days in (1, 2, 3, 5, 10):
            raw_return = _number(master.get(f"return_{days}d"))
            abnormal_return = _number(master.get(f"ar_{days}d"))
            if raw_return is not None or abnormal_return is not None:
                market_windows.append(
                    {
                        "label": f"{days}D",
                        "horizonDays": days,
                        "rawReturn": raw_return,
                        "abnormalReturn": abnormal_return,
                    }
                )

        cumulative_abnormal_returns = []
        for days in (3, 5):
            value = _number(comparison.get(f"car_{days}d"))
            if value is not None:
                cumulative_abnormal_returns.append(
                    {"label": f"{days}D CAR", "horizonDays": days, "value": value}
                )

        topic_rows = sorted(
            (
                {
                    "name": (_text(row.get("topic")) or "unknown").replace("_", " ").title(),
                    "count": _integer(row.get("count")),
                    "ratio": _number(row.get("ratio")),
                }
                for row in topics.get(transcript_id, [])
            ),
            key=lambda row: row["count"] or 0,
            reverse=True,
        )

        keywords = [
            keyword.strip()
            for keyword in (_text(nlp.get("top_5_keywords")) or "").split(",")
            if keyword.strip()
        ]

        analyses.append(
            {
                "transcript": {
                    "id": transcript_id,
                    "ticker": _text(master.get("ticker")),
                    "companyName": _text(master.get("company_name")),
                    "earningsDate": _text(master.get("earnings_date")),
                    "fiscalQuarter": _integer(master.get("fiscal_quarter")),
                    "fiscalYear": _integer(master.get("year")),
                    "cleanWordCount": _integer(master.get("word_count_clean")),
                    "chunkCount": _integer(master.get("chunk_count")),
                },
                "finbert": {
                    "score": _number(master.get("finbert_score")),
                    "positiveProbability": _number(master.get("finbert_positive")),
                    "negativeProbability": _number(master.get("finbert_negative")),
                    "neutralProbability": _number(master.get("finbert_neutral")),
                    "meanConfidence": _number(master.get("mean_confidence")),
                    "direction": _text(master.get("finbert_direction")),
                },
                "loughranMcDonald": {
                    "toneScore": _number(master.get("lm_tone_score")),
                    "positiveCount": _integer(master.get("lm_positive_count")),
                    "negativeCount": _integer(master.get("lm_negative_count")),
                    "scoredWordCount": _integer(master.get("lm_word_count")),
                    "label": _text(master.get("lm_label")),
                },
                "comparison": {
                    "directionalAgreement": _boolean(master.get("directional_agreement")),
                    "scoreDifference": _number(master.get("score_difference")),
                    "absoluteDifference": _number(master.get("absolute_difference")),
                },
                "marketReaction": {
                    "eventDate": _text(comparison.get("event_date")),
                    "windows": market_windows,
                    "cumulativeAbnormalReturns": cumulative_abnormal_returns,
                    "marketWindowStart": _text(master.get("market_window_start")),
                    "marketWindowEnd": _text(master.get("market_window_end")),
                },
                "nlp": {
                    "totalTokens": _integer(nlp.get("total_tokens")),
                    "uncertaintyCount": _integer(nlp.get("uncertainty_count")),
                    "uncertaintyRatio": _number(nlp.get("uncertainty_ratio")),
                    "topKeywords": keywords,
                    "topics": topic_rows,
                },
                "speakerAnalysis": {
                    "available": bool(labeled_speaker_groups),
                    "groups": labeled_speaker_groups,
                    "aggregate": aggregate_speaker,
                    "message": (
                        "Speaker or section labels are available in the exported chunk summary."
                        if labeled_speaker_groups
                        else "The verified artifact contains aggregate chunk sentiment, but no speaker or section labels."
                    ),
                },
            }
        )

    if not analyses:
        raise ValueError("No master dataset rows contained a transcript_id.")

    tickers = {
        analysis["transcript"]["ticker"]
        for analysis in analyses
        if analysis["transcript"]["ticker"]
    }
    return sanitize_json(
        {
            "schemaVersion": 1,
            "dataset": {
                "kind": "demonstration" if len(analyses) < 10 else "research",
                "label": "Verified repository artifact snapshot",
                "observationCount": len(analyses),
                "companyCount": len(tickers),
                "isLimited": len(analyses) < 10,
                "notice": (
                    "Demonstration dataset: one checked transcript/event. Results are descriptive, not investment advice or validated predictive evidence."
                    if len(analyses) == 1
                    else "Research dataset exported from checked pipeline artifacts."
                ),
            },
            "analyses": analyses,
            "modeling": _modeling_summary(root),
            "provenance": [
                {"label": "Master analysis", "path": "data/processed/master_dataset.csv"},
                {"label": "Advanced NLP", "path": "data/processed/nlp/advanced_nlp_features.csv"},
                {"label": "Model comparison", "path": "reports/tables/sentiment_model_comparison.csv"},
                {"label": "Topic frequency", "path": "reports/tables/topic_frequency.csv"},
                {"label": "Speaker summary", "path": "reports/tables/speaker_sentiment_summary.csv"},
                {"label": "Model status", "path": "models/predictive_model_metadata.json"},
            ],
        }
    )


def export_web_data(root: Path = PROJECT_ROOT, output: Path = DEFAULT_OUTPUT) -> Path:
    target = output if output.is_absolute() else root / output
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_export(root)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(serialized, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repository root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON output path")
    args = parser.parse_args()
    output = export_web_data(args.root.resolve(), args.output)
    print(f"Exported deployment-safe dashboard data to {output}")


if __name__ == "__main__":
    main()
