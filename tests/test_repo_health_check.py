from pathlib import Path

import pandas as pd

from scripts.repo_health_check import (
    KEY_FIGURES,
    KEY_OUTPUT_ARTIFACTS,
    KEY_REPORTS,
    KEY_SOURCE_FILES,
    KEY_TESTS,
    check_repo,
)


def _write_required_repo_shape(root: Path, row_count: int = 1) -> None:
    all_paths = set(KEY_SOURCE_FILES + KEY_OUTPUT_ARTIFACTS + KEY_REPORTS + KEY_FIGURES + KEY_TESTS)
    for rel_path in all_paths:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == "data/processed/master_dataset.parquet":
            pd.DataFrame({"transcript_id": [f"T{i}" for i in range(row_count)]}).to_parquet(path, index=False)
        elif rel_path == "data/processed/master_dataset.csv":
            pd.DataFrame({"transcript_id": [f"T{i}" for i in range(row_count)]}).to_csv(path, index=False)
        else:
            path.write_bytes(b"placeholder")


def test_health_check_passes_with_n1_warning(tmp_path: Path) -> None:
    _write_required_repo_shape(tmp_path, row_count=1)

    result = check_repo(tmp_path)

    assert result["exit_code"] == 0
    assert not result["failures"]
    assert result["row_count"] == 1
    assert any("only 1 row" in warning for warning in result["warnings"])


def test_health_check_fails_when_required_files_are_missing(tmp_path: Path) -> None:
    _write_required_repo_shape(tmp_path, row_count=2)
    (tmp_path / "reports/tables/statistical_summary.csv").unlink()

    result = check_repo(tmp_path)

    assert result["exit_code"] == 1
    assert any("statistical_summary.csv" in failure for failure in result["failures"])
    assert not any("only 1 row" in warning for warning in result["warnings"])
