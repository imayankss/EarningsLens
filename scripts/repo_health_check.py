"""Repository health check for the completed analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import read_table  # noqa: E402


KEY_SOURCE_FILES = [
    "app.py",
    "src/analysis/sentiment_model_comparison.py",
    "src/analysis/master_dataset_builder.py",
    "src/analysis/statistical_analysis.py",
    "src/visualization/professional_charts.py",
    "src/nlp/advanced_nlp.py",
    "src/models/predictive_model.py",
]

KEY_OUTPUT_ARTIFACTS = [
    "data/processed/master_dataset.parquet",
    "data/processed/master_dataset.csv",
    "data/processed/analysis/sentiment_model_comparison.parquet",
    "data/processed/nlp/advanced_nlp_features.parquet",
    "models/predictive_model_metadata.json",
]

KEY_REPORTS = [
    "reports/tables/master_dataset_summary.csv",
    "reports/tables/sentiment_model_comparison_summary.csv",
    "reports/tables/statistical_summary.csv",
    "reports/tables/correlation_analysis.csv",
    "reports/tables/regression_results.csv",
    "reports/tables/statistical_warnings.csv",
    "reports/tables/keyword_summary.csv",
    "reports/tables/topic_frequency.csv",
    "reports/tables/bigram_summary.csv",
    "reports/tables/uncertainty_summary.csv",
    "reports/tables/prediction_model_summary.csv",
    "reports/tables/prediction_model_metrics.csv",
    "reports/tables/prediction_model_warnings.csv",
]

KEY_FIGURES = [
    "reports/figures/sentiment_distribution.png",
    "reports/figures/finbert_lm_comparison.png",
    "reports/figures/returns_vs_sentiment.png",
    "reports/figures/event_study_return_horizon.png",
    "reports/figures/correlation_heatmap.png",
    "reports/figures/model_agreement_summary.png",
]

KEY_TESTS = [
    "tests/test_sentiment_model_comparison.py",
    "tests/test_master_dataset_builder.py",
    "tests/test_statistical_analysis.py",
    "tests/test_professional_charts.py",
    "tests/test_advanced_nlp.py",
    "tests/test_predictive_model.py",
    "tests/test_streamlit_app.py",
]


def _check_files(root: Path, label: str, rel_paths: list[str]) -> tuple[list[str], list[str]]:
    passes: list[str] = []
    failures: list[str] = []
    for rel_path in rel_paths:
        path = root / rel_path
        if path.exists():
            passes.append(f"{label}: {rel_path}")
        else:
            failures.append(f"{label}: missing {rel_path}")
    return passes, failures


def check_repo(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Check source files, outputs, reports, figures, tests, and row counts."""

    root = Path(root)
    passes: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []

    for label, paths in [
        ("source", KEY_SOURCE_FILES),
        ("artifact", KEY_OUTPUT_ARTIFACTS),
        ("report", KEY_REPORTS),
        ("figure", KEY_FIGURES),
        ("test", KEY_TESTS),
    ]:
        group_passes, group_failures = _check_files(root, label, paths)
        passes.extend(group_passes)
        failures.extend(group_failures)

    master_stem = root / "data/processed/master_dataset"
    if not (master_stem.with_suffix(".parquet").exists() or master_stem.with_suffix(".csv").exists()):
        failures.append("dataset: missing data/processed/master_dataset parquet/csv")
        row_count = None
    else:
        try:
            master = read_table(master_stem, required=True, label="master_dataset")
            row_count = int(len(master))
            passes.append(f"dataset: master_dataset row_count={row_count}")
            if row_count == 1:
                warnings.append("dataset: master_dataset has only 1 row; inference and ML validation remain limited.")
            elif row_count == 0:
                failures.append("dataset: master_dataset has 0 rows")
        except Exception as exc:
            row_count = None
            failures.append(f"dataset: unable to read master_dataset: {exc}")

    return {
        "passes": passes,
        "warnings": warnings,
        "failures": failures,
        "row_count": row_count,
        "exit_code": 1 if failures else 0,
    }


def print_report(result: dict[str, Any]) -> None:
    print("Repository Health Check")
    print("=======================")
    print(f"PASS: {len(result['passes'])}")
    print(f"WARN: {len(result['warnings'])}")
    print(f"FAIL: {len(result['failures'])}")

    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")

    if result["failures"]:
        print("\nFailures:")
        for failure in result["failures"]:
            print(f"- {failure}")

    if not result["failures"]:
        print("\nStatus: PASS with warnings" if result["warnings"] else "\nStatus: PASS")
    else:
        print("\nStatus: FAIL")


def main() -> int:
    result = check_repo(PROJECT_ROOT)
    print_report(result)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
