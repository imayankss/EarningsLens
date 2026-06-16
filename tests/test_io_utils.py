from pathlib import Path

import pandas as pd
import pytest

from src.utils.io_utils import (
    ensure_dir,
    file_exists,
    format_file_size,
    list_files,
    read_table,
    safe_read_json,
    write_json,
    write_table,
)


def test_ensure_dir_and_file_helpers(tmp_path: Path) -> None:
    directory = ensure_dir(tmp_path / "nested/output")
    file_path = directory / "sample.txt"
    file_path.write_text("abc", encoding="utf-8")

    assert directory.exists()
    assert file_exists(file_path)
    assert format_file_size(file_path) == "3 B"
    assert format_file_size(tmp_path / "missing.txt") == "missing"
    assert list_files(directory, "*.txt") == [file_path]


def test_write_table_and_read_table_with_stem(tmp_path: Path) -> None:
    df = pd.DataFrame({"transcript_id": ["T1"], "score": [0.25]})
    written = write_table(df, tmp_path / "tables/model_output")

    assert {path.suffix for path in written} == {".parquet", ".csv"}
    loaded = read_table(tmp_path / "tables/model_output")
    pd.testing.assert_frame_equal(loaded, df)


def test_read_table_falls_back_from_parquet_to_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"ticker": ["AAPL"]})
    csv_path = tmp_path / "master_dataset.csv"
    df.to_csv(csv_path, index=False)

    loaded = read_table(tmp_path / "master_dataset.parquet")
    pd.testing.assert_frame_equal(loaded, df)


def test_read_table_required_and_optional_missing(tmp_path: Path) -> None:
    assert read_table(tmp_path / "missing_table", required=False).empty
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing_table", required=True)


def test_json_helpers(tmp_path: Path) -> None:
    path = tmp_path / "metadata/model.json"
    write_json({"row_count": 1, "models_trained": []}, path)

    assert safe_read_json(path)["row_count"] == 1
    assert safe_read_json(tmp_path / "missing.json", default={"fallback": True}) == {"fallback": True}
