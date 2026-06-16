from pathlib import Path

import pandas as pd

import app


def test_import_app_without_running_server() -> None:
    assert hasattr(app, "main")
    assert "Overview" in app.NAVIGATION
    assert hasattr(app, "render_metric_card")


def test_safe_loader_reads_csv_and_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"transcript_id": ["T1"], "finbert_score": [0.25]})
    csv_path = tmp_path / "sample.csv"
    parquet_path = tmp_path / "sample.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    loaded_csv = app.safe_load_table(str(csv_path))
    loaded_parquet = app.safe_load_table(str(parquet_path))

    pd.testing.assert_frame_equal(loaded_csv, df)
    pd.testing.assert_frame_equal(loaded_parquet, df)


def test_safe_loader_fallback_and_missing_file(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback.csv"
    expected = pd.DataFrame({"ticker": ["AAPL"]})
    expected.to_csv(fallback, index=False)

    loaded = app.safe_load_table(str(tmp_path / "missing.parquet"), str(fallback))
    missing = app.safe_load_table(str(tmp_path / "still_missing.csv"))

    pd.testing.assert_frame_equal(loaded, expected)
    assert missing.empty


def test_safe_json_loader_handles_valid_and_missing_json(tmp_path: Path) -> None:
    json_path = tmp_path / "metadata.json"
    json_path.write_text('{"row_count": 1, "models_trained": []}', encoding="utf-8")

    assert app.safe_load_json(str(json_path))["row_count"] == 1
    assert app.safe_load_json(str(tmp_path / "missing.json")) == {}


def test_discover_figures_returns_existing_expected_files(tmp_path: Path) -> None:
    figures_dir = tmp_path / "reports/figures"
    figures_dir.mkdir(parents=True)
    expected = figures_dir / "sentiment_distribution.png"
    expected.write_bytes(b"not-a-real-png-but-present")

    figures = app.discover_figures(
        figures_dir=figures_dir,
        figure_files=["sentiment_distribution.png", "missing.png"],
    )

    assert figures == [expected]


def test_file_helpers_format_size_caption_and_counts(tmp_path: Path) -> None:
    small = tmp_path / "prediction_model_metrics.csv"
    small.write_bytes(b"abc")
    missing = tmp_path / "missing.csv"

    assert app.file_exists(small)
    assert not app.file_exists(missing)
    assert app.get_file_size(small) == "3 B"
    assert app.get_file_size(missing) == "missing"
    assert app.format_filename_caption(small) == "Prediction Model Metrics"
    assert app.count_available_files([small, missing]) == 1


def test_extract_overview_metrics_from_temporary_master_dataset(tmp_path: Path) -> None:
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    figures_dir.mkdir()
    tables_dir.mkdir()
    (figures_dir / "sentiment_distribution.png").write_bytes(b"png")
    (tables_dir / "statistical_summary.csv").write_text("metric,value\nrow_count,1\n")

    master = pd.DataFrame(
        [
            {
                "transcript_id": "AAPL_20201029",
                "ticker": "AAPL",
                "finbert_score": 0.432,
                "lm_tone_score": 0.492,
                "directional_agreement": True,
            }
        ]
    )

    metrics = app.extract_overview_metrics(master, figures_dir=figures_dir, tables_dir=tables_dir)

    assert metrics["transcripts"] == 1
    assert metrics["tickers"] == 1
    assert metrics["current_ticker"] == "AAPL"
    assert metrics["finbert_score"] == 0.432
    assert metrics["lm_tone_score"] == 0.492
    assert metrics["directional_agreement"] is True
    assert metrics["figures"] == 1
    assert metrics["reports"] == 1


def test_status_counts_and_filtering_helpers() -> None:
    df = pd.DataFrame({"status": ["trained", "insufficient_sample_size", "trained"], "value": [1, 2, 3]})

    assert app._status_counts(df) == {"trained": 2, "insufficient_sample_size": 1}
    assert len(app._filter_status(df, "trained")) == 2
    assert len(app._filter_status(df, "All")) == 3


def test_extract_overview_metrics_handles_empty_master(tmp_path: Path) -> None:
    metrics = app.extract_overview_metrics(
        pd.DataFrame(),
        figures_dir=tmp_path / "missing_figures",
        tables_dir=tmp_path / "missing_tables",
    )

    assert metrics["transcripts"] == 0
    assert metrics["tickers"] == 0
    assert metrics["current_ticker"] == "-"
