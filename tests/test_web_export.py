from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_web_data import build_export, export_web_data, sanitize_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_checked_artifacts_export_real_aapl_values(tmp_path: Path) -> None:
    output = export_web_data(PROJECT_ROOT, tmp_path / "dashboard.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schemaVersion"] == 1
    assert payload["dataset"]["kind"] == "demonstration"
    assert payload["dataset"]["observationCount"] == 1
    analysis = payload["analyses"][0]
    assert analysis["transcript"]["id"] == "AAPL_20201029"
    assert analysis["finbert"]["positiveProbability"] == pytest.approx(0.4608808283)
    assert analysis["loughranMcDonald"]["positiveCount"] == 172
    assert analysis["speakerAnalysis"]["available"] is False
    assert payload["modeling"]["modelsTrained"] == 0


def test_export_contains_no_non_finite_json_values() -> None:
    payload = build_export(PROJECT_ROOT)
    json.dumps(payload, allow_nan=False)
    assert sanitize_json({"nan": float("nan"), "infinity": float("inf")}) == {
        "nan": None,
        "infinity": None,
    }


def test_missing_master_artifact_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Required web export source is missing"):
        build_export(tmp_path)
