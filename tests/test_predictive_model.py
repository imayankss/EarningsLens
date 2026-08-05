from pathlib import Path

import pandas as pd

from src.models.predictive_model import (
    LogisticRegression,
    RandomForestClassifier,
    prepare_modeling_dataset,
    run_predictive_modeling,
)


def _master_rows(n: int, *, single_class: bool = False) -> pd.DataFrame:
    rows = []
    for i in range(n):
        sign = 1 if single_class or i % 2 == 0 else -1
        rows.append(
            {
                "transcript_id": f"T{i:03d}",
                "ticker": f"T{i:03d}",
                "finbert_score": sign * (0.2 + i * 0.001),
                "finbert_positive": 0.60 if sign > 0 else 0.20,
                "finbert_negative": 0.20 if sign > 0 else 0.60,
                "finbert_neutral": 0.20,
                "lm_tone_score": sign * 0.15,
                "lm_positive_count": 12 + i,
                "lm_negative_count": 4 if sign > 0 else 16,
                "score_difference": sign * 0.05,
                "absolute_difference": 0.05,
                "mean_confidence": 0.75,
                "mean_return": sign * 0.01,
                "std_return": 0.02,
                "mean_abnormal_return": sign * 0.008,
                "mean_volume": 1000 + i,
                "price_min": 90 + i,
                "price_max": 100 + i,
                "market_row_count": 10,
                "word_count_clean": 8000 + i,
                "chunk_count": 20,
                "ar_1d": sign * (0.01 + i * 0.0001),
                "ar_3d": sign * (0.02 + i * 0.0001),
                "return_1d": sign * (0.015 + i * 0.0001),
            }
        )
    return pd.DataFrame(rows)


def _advanced_nlp(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transcript_id": transcript_id,
                "uncertainty_count": index + 1,
                "uncertainty_ratio": 0.01 * (index + 1),
                "top_5_keywords": "revenue,growth,margin",
                "topic_revenue_growth": 5 + index,
                "topic_guidance_outlook": 3 + index,
                "topic_risk_uncertainty": 2 + index,
            }
            for index, transcript_id in enumerate(ids)
        ]
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "summary_path": tmp_path / "reports/tables/prediction_model_summary.csv",
        "metrics_path": tmp_path / "reports/tables/prediction_model_metrics.csv",
        "warnings_path": tmp_path / "reports/tables/prediction_model_warnings.csv",
        "feature_importance_path": tmp_path / "reports/tables/prediction_feature_importance.csv",
        "metadata_path": tmp_path / "models/predictive_model_metadata.json",
    }


def test_n_equals_one_skips_training_and_creates_outputs(tmp_path: Path) -> None:
    master_path = tmp_path / "data/processed/master_dataset.parquet"
    nlp_path = tmp_path / "data/processed/nlp/advanced_nlp_features.parquet"
    master_path.parent.mkdir(parents=True)
    nlp_path.parent.mkdir(parents=True)

    master = _master_rows(1)
    master.to_parquet(master_path, index=False)
    _advanced_nlp(master["transcript_id"].tolist()).to_parquet(nlp_path, index=False)

    summary, metrics, warnings_df, feature_importance, metadata, output_paths = run_predictive_modeling(
        master_parquet_path=master_path,
        master_csv_path=tmp_path / "missing.csv",
        advanced_nlp_path=nlp_path,
        **_paths(tmp_path),
    )

    assert summary.loc[summary["metric"] == "row_count", "value"].iloc[0] == 1
    assert set(metrics["status"]) == {"insufficient_sample_size"}
    assert not warnings_df.empty
    assert feature_importance.empty
    assert metadata["row_count"] == 1
    assert metadata["models_trained"] == []
    assert all(path.exists() for path in output_paths)


def test_insufficient_sample_size_warning_for_under_ten_rows(tmp_path: Path) -> None:
    master_path = tmp_path / "master_dataset.parquet"
    master = _master_rows(9)
    master.to_parquet(master_path, index=False)

    _, metrics, warnings_df, _, metadata, _ = run_predictive_modeling(
        master_parquet_path=master_path,
        master_csv_path=tmp_path / "missing.csv",
        advanced_nlp_path=tmp_path / "missing_nlp.parquet",
        **_paths(tmp_path),
    )

    assert set(metrics["status"]) == {"insufficient_sample_size"}
    assert "insufficient_sample_size" in set(metadata["skipped_reasons"].values())
    assert any("n=9 < 10" in message for message in warnings_df["message"])


def test_single_class_target_skip_when_sample_size_is_large_enough(tmp_path: Path) -> None:
    master_path = tmp_path / "master_dataset.parquet"
    _master_rows(12, single_class=True).to_parquet(master_path, index=False)

    _, metrics, warnings_df, _, metadata, _ = run_predictive_modeling(
        master_parquet_path=master_path,
        master_csv_path=tmp_path / "missing.csv",
        advanced_nlp_path=tmp_path / "missing_nlp.parquet",
        **_paths(tmp_path),
    )

    assert set(metrics["status"]) == {"single_class_target"}
    assert "single_class_target" in set(metadata["skipped_reasons"].values())
    assert any("fewer than 2 classes" in message for message in warnings_df["message"])


def test_prepare_modeling_dataset_merges_advanced_nlp_and_derives_aliases(tmp_path: Path) -> None:
    nlp_path = tmp_path / "advanced_nlp_features.parquet"
    master = _master_rows(2)
    _advanced_nlp(master["transcript_id"].tolist()).to_parquet(nlp_path, index=False)

    warnings_list: list[dict[str, str]] = []
    dataset, feature_columns, target_columns = prepare_modeling_dataset(
        master,
        advanced_nlp_path=nlp_path,
        warnings_list=warnings_list,
    )

    assert "target_positive_ar_1d" in target_columns
    assert "target_positive_ar_3d" in target_columns
    assert "target_positive_return_1d" in target_columns
    assert "uncertainty_ratio" in feature_columns
    assert "top_keyword_count" in feature_columns
    assert "topic_revenue_growth_count" in feature_columns
    assert dataset["top_keyword_count"].iloc[0] == 3


def test_csv_fallback_and_output_files_are_created(tmp_path: Path) -> None:
    master_csv_path = tmp_path / "data/processed/master_dataset.csv"
    master_csv_path.parent.mkdir(parents=True)
    _master_rows(1).to_csv(master_csv_path, index=False)

    _, metrics, warnings_df, _, metadata, output_paths = run_predictive_modeling(
        master_parquet_path=tmp_path / "missing.parquet",
        master_csv_path=master_csv_path,
        advanced_nlp_path=tmp_path / "missing_nlp.parquet",
        **_paths(tmp_path),
    )

    assert len(metrics) == 9
    assert not warnings_df.empty
    assert metadata["row_count"] == 1
    assert all(path.exists() for path in output_paths)
    assert pd.read_csv(output_paths[0]).shape[0] > 0
    assert pd.read_csv(output_paths[1]).shape[0] == len(metrics)
    assert pd.read_csv(output_paths[2]).shape[0] == len(warnings_df)


def test_enough_synthetic_rows_train_sklearn_models_when_available(tmp_path: Path) -> None:
    if LogisticRegression is None or RandomForestClassifier is None:
        return

    master_path = tmp_path / "master_dataset.parquet"
    nlp_path = tmp_path / "advanced_nlp_features.parquet"
    master = _master_rows(40)
    master.to_parquet(master_path, index=False)
    _advanced_nlp(master["transcript_id"].tolist()).to_parquet(nlp_path, index=False)

    _, metrics, warnings_df, feature_importance, metadata, _ = run_predictive_modeling(
        master_parquet_path=master_path,
        master_csv_path=tmp_path / "missing.csv",
        advanced_nlp_path=nlp_path,
        **_paths(tmp_path),
    )

    trained = metrics[metrics["status"] == "trained"]
    assert {"logistic_regression", "random_forest"}.issubset(set(trained["model_name"]))
    assert not feature_importance.empty
    assert metadata["models_trained"]
    accuracies = trained["accuracy"].dropna()
    assert not accuracies.empty
    assert accuracies.between(0.0, 1.0).all()
    assert (accuracies >= 0.75).all()
    assert not any(warnings_df["message"].str.contains("sklearn unavailable", case=False, na=False))
