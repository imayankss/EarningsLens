"""Train guarded prediction models for post-earnings return direction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - depends on local environment
    RandomForestClassifier = None
    LogisticRegression = None
    accuracy_score = None
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None
    train_test_split = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_PARQUET_PATH = PROJECT_ROOT / "data/processed/master_dataset.parquet"
MASTER_CSV_PATH = PROJECT_ROOT / "data/processed/master_dataset.csv"
ADVANCED_NLP_PATH = PROJECT_ROOT / "data/processed/nlp/advanced_nlp_features.parquet"

REPORT_TABLES_DIR = PROJECT_ROOT / "reports/tables"
MODEL_METADATA_PATH = PROJECT_ROOT / "models/predictive_model_metadata.json"

SUMMARY_PATH = REPORT_TABLES_DIR / "prediction_model_summary.csv"
METRICS_PATH = REPORT_TABLES_DIR / "prediction_model_metrics.csv"
WARNINGS_PATH = REPORT_TABLES_DIR / "prediction_model_warnings.csv"
FEATURE_IMPORTANCE_PATH = REPORT_TABLES_DIR / "prediction_feature_importance.csv"

TARGET_SOURCE_COLUMNS = [
    "ar_1d",
    "ar_2d",
    "ar_3d",
    "ar_5d",
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
]

TARGET_OUTPUT_COLUMNS = {
    "ar_1d": "target_positive_ar_1d",
    "ar_3d": "target_positive_ar_3d",
    "return_1d": "target_positive_return_1d",
}

FEATURE_COLUMNS = [
    "finbert_score",
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "lm_tone_score",
    "lm_positive_count",
    "lm_negative_count",
    "score_difference",
    "absolute_difference",
    "mean_confidence",
    "uncertainty_count",
    "uncertainty_ratio",
    "top_keyword_count",
    "topic_revenue_growth_count",
    "topic_guidance_outlook_count",
    "topic_risk_uncertainty_count",
    "mean_return",
    "std_return",
    "mean_abnormal_return",
    "mean_volume",
    "price_min",
    "price_max",
    "market_row_count",
    "word_count_clean",
    "chunk_count",
]

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]

METRICS_COLUMNS = [
    "target_column",
    "model_name",
    "n_observations",
    "n_features",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "status",
]

WARNING_COLUMNS = ["category", "item", "message"]

FEATURE_IMPORTANCE_COLUMNS = [
    "target_column",
    "model_name",
    "feature",
    "importance",
    "importance_type",
]


def _warning(category: str, item: str, message: str) -> dict[str, str]:
    return {"category": category, "item": item, "message": message}


def load_master_dataset(
    master_parquet_path: Path = MASTER_PARQUET_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
) -> pd.DataFrame:
    """Load master_dataset from parquet, falling back to CSV."""

    if master_parquet_path.exists():
        return pd.read_parquet(master_parquet_path)
    if master_csv_path.exists():
        return pd.read_csv(master_csv_path)
    raise FileNotFoundError(
        f"Missing master dataset. Tried parquet={master_parquet_path} and csv={master_csv_path}"
    )


def _merge_advanced_nlp(
    master: pd.DataFrame,
    advanced_nlp_path: Path,
    warnings_list: list[dict[str, str]],
) -> pd.DataFrame:
    if not advanced_nlp_path.exists():
        warnings_list.append(
            _warning("optional_input", "advanced_nlp_features", f"Missing optional file: {advanced_nlp_path}")
        )
        return master.copy()

    nlp_features = pd.read_parquet(advanced_nlp_path)
    if "transcript_id" not in nlp_features.columns:
        warnings_list.append(
            _warning(
                "optional_input",
                "advanced_nlp_features",
                "advanced_nlp_features is missing transcript_id; skipping merge.",
            )
        )
        return master.copy()

    duplicate_columns = [
        column for column in nlp_features.columns if column != "transcript_id" and column in master.columns
    ]
    nlp_features = nlp_features.drop(columns=duplicate_columns)
    return master.merge(nlp_features, on="transcript_id", how="left", validate="one_to_one")


def _derive_advanced_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "top_keyword_count" not in df.columns and "top_5_keywords" in df.columns:
        df["top_keyword_count"] = (
            df["top_5_keywords"]
            .fillna("")
            .astype(str)
            .map(lambda value: 0 if not value.strip() else len([item for item in value.split(",") if item.strip()]))
        )

    alias_map = {
        "topic_revenue_growth": "topic_revenue_growth_count",
        "topic_guidance_outlook": "topic_guidance_outlook_count",
        "topic_risk_uncertainty": "topic_risk_uncertainty_count",
    }
    for source, target in alias_map.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
    return df


def prepare_modeling_dataset(
    master: pd.DataFrame,
    advanced_nlp_path: Path = ADVANCED_NLP_PATH,
    warnings_list: list[dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge optional NLP features, create targets, and choose numeric features."""

    warnings_list = warnings_list if warnings_list is not None else []
    if "transcript_id" not in master.columns:
        raise ValueError("master_dataset is missing required column: transcript_id")

    dataset = _merge_advanced_nlp(master, advanced_nlp_path, warnings_list)
    dataset = _derive_advanced_feature_aliases(dataset)

    for source_column, target_column in TARGET_OUTPUT_COLUMNS.items():
        if source_column in dataset.columns:
            values = pd.to_numeric(dataset[source_column], errors="coerce")
            dataset[target_column] = np.where(values > 0, 1, 0)
            dataset.loc[values.isna(), target_column] = np.nan

    target_columns = [column for column in TARGET_OUTPUT_COLUMNS.values() if column in dataset.columns]
    missing_target_sources = [
        source for source in TARGET_OUTPUT_COLUMNS if TARGET_OUTPUT_COLUMNS[source] not in target_columns
    ]
    for source in missing_target_sources:
        warnings_list.append(
            _warning("missing_target", source, f"Target source column {source} is unavailable.")
        )

    feature_columns: list[str] = []
    for column in FEATURE_COLUMNS:
        if column not in dataset.columns:
            continue
        numeric = pd.to_numeric(dataset[column], errors="coerce")
        if numeric.notna().any():
            dataset[column] = numeric
            feature_columns.append(column)

    missing_features = [column for column in FEATURE_COLUMNS if column not in feature_columns]
    if missing_features:
        warnings_list.append(
            _warning(
                "missing_features",
                "candidate_features",
                f"Missing or non-numeric candidate features: {missing_features}",
            )
        )

    return dataset, feature_columns, target_columns


def _median_fill_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = df[feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    for column in feature_columns:
        median = features[column].median()
        fill_value = 0.0 if pd.isna(median) else float(median)
        features[column] = features[column].fillna(fill_value)
    return features


def _empty_metric_row(
    target_column: str,
    model_name: str,
    n_observations: int,
    n_features: int,
    status: str,
) -> dict[str, Any]:
    return {
        "target_column": target_column,
        "model_name": model_name,
        "n_observations": n_observations,
        "n_features": n_features,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "status": status,
    }


def _safe_train_test_split(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    class_counts = y.value_counts()
    stratify = y if len(class_counts) == 2 and class_counts.min() >= 2 else None
    test_size = 0.3 if len(y) >= 20 else 0.25
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )


def _score_classifier(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": np.nan,
    }

    if len(pd.Series(y_test).unique()) >= 2:
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(x_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        except Exception:
            metrics["roc_auc"] = np.nan
    return metrics


def _importance_rows(
    target_column: str,
    model_name: str,
    model: Any,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
        importance_type = "absolute_coefficient"
    else:
        return []

    return [
        {
            "target_column": target_column,
            "model_name": model_name,
            "feature": feature,
            "importance": float(importance),
            "importance_type": importance_type,
        }
        for feature, importance in sorted(
            zip(feature_columns, values), key=lambda item: abs(float(item[1])), reverse=True
        )
    ]


def train_prediction_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    warnings_list: list[dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train guarded classifiers for each available target column."""

    warnings_list = warnings_list if warnings_list is not None else []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    models_trained: list[str] = []
    skipped_reasons: dict[str, str] = {}

    if not target_columns:
        warnings_list.append(_warning("target", "all", "No target columns were created."))

    for target_column in target_columns:
        target_dataset = dataset[feature_columns + [target_column]].copy()
        target_dataset[target_column] = pd.to_numeric(target_dataset[target_column], errors="coerce")
        target_dataset = target_dataset.dropna(subset=[target_column])
        n_observations = int(len(target_dataset))
        n_features = len(feature_columns)
        y = target_dataset[target_column].astype(int)

        base_status: str | None = None
        if n_observations < 10:
            base_status = "insufficient_sample_size"
            warnings_list.append(
                _warning(
                    "training",
                    target_column,
                    f"Skipped model training for {target_column}; n={n_observations} < 10.",
                )
            )
        elif y.nunique() < 2:
            base_status = "single_class_target"
            warnings_list.append(
                _warning(
                    "training",
                    target_column,
                    f"Skipped model training for {target_column}; target has fewer than 2 classes.",
                )
            )
        elif not feature_columns:
            base_status = "missing_features"
            warnings_list.append(
                _warning("training", target_column, f"Skipped model training for {target_column}; no numeric features.")
            )
        elif LogisticRegression is None or train_test_split is None:
            base_status = "sklearn_unavailable"
            warnings_list.append(
                _warning("training", target_column, f"Skipped sklearn models for {target_column}; sklearn unavailable.")
            )

        if base_status is not None:
            for model_name in MODEL_NAMES:
                status = base_status
                if model_name == "xgboost" and base_status == "sklearn_unavailable":
                    status = "xgboost_unavailable" if XGBClassifier is None else base_status
                metric_rows.append(
                    _empty_metric_row(target_column, model_name, n_observations, n_features, status)
                )
                skipped_reasons[f"{target_column}:{model_name}"] = status
            continue

        x = _median_fill_features(target_dataset, feature_columns)
        x_train, x_test, y_train, y_test = _safe_train_test_split(x, y)

        models: list[tuple[str, Any, str | None]] = [
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                None,
            ),
            (
                "random_forest",
                RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
                None,
            ),
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=42,
                )
                if XGBClassifier is not None
                else None,
                "xgboost_unavailable" if XGBClassifier is None else None,
            ),
        ]

        for model_name, model, unavailable_status in models:
            if unavailable_status is not None:
                metric_rows.append(
                    _empty_metric_row(target_column, model_name, n_observations, n_features, unavailable_status)
                )
                skipped_reasons[f"{target_column}:{model_name}"] = unavailable_status
                warnings_list.append(
                    _warning("training", f"{target_column}:{model_name}", "Skipped XGBoost; xgboost is unavailable.")
                )
                continue

            try:
                model.fit(x_train, y_train)
                scores = _score_classifier(model, x_test, y_test)
                row = _empty_metric_row(target_column, model_name, n_observations, n_features, "trained")
                row.update(scores)
                metric_rows.append(row)
                models_trained.append(f"{target_column}:{model_name}")
                importance_rows.extend(
                    _importance_rows(target_column, model_name, model, feature_columns)
                )
            except Exception as exc:
                status = "training_failed"
                metric_rows.append(
                    _empty_metric_row(target_column, model_name, n_observations, n_features, status)
                )
                skipped_reasons[f"{target_column}:{model_name}"] = status
                warnings_list.append(
                    _warning("training", f"{target_column}:{model_name}", f"Training failed: {exc}")
                )

    metrics = pd.DataFrame(metric_rows, columns=METRICS_COLUMNS)
    feature_importance = pd.DataFrame(importance_rows, columns=FEATURE_IMPORTANCE_COLUMNS)
    training_meta = {
        "models_attempted": [f"{target}:{model}" for target in target_columns for model in MODEL_NAMES],
        "models_trained": models_trained,
        "skipped_reasons": skipped_reasons,
    }
    return metrics, feature_importance, training_meta


def build_summary(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    status_counts = metrics["status"].value_counts(dropna=False).to_dict() if not metrics.empty else {}
    rows = [
        {"metric": "row_count", "value": len(dataset)},
        {"metric": "feature_count", "value": len(feature_columns)},
        {"metric": "target_count", "value": len(target_columns)},
        {"metric": "feature_columns", "value": "|".join(feature_columns)},
        {"metric": "target_columns", "value": "|".join(target_columns)},
    ]
    for status, count in sorted(status_counts.items()):
        rows.append({"metric": f"status_{status}", "value": int(count)})
    return pd.DataFrame(rows)


def run_predictive_modeling(
    master_parquet_path: Path = MASTER_PARQUET_PATH,
    master_csv_path: Path = MASTER_CSV_PATH,
    advanced_nlp_path: Path = ADVANCED_NLP_PATH,
    summary_path: Path = SUMMARY_PATH,
    metrics_path: Path = METRICS_PATH,
    warnings_path: Path = WARNINGS_PATH,
    feature_importance_path: Path = FEATURE_IMPORTANCE_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[Path]]:
    """Run Day 15 modeling and export report tables plus metadata."""

    warnings_list: list[dict[str, str]] = []
    master = load_master_dataset(master_parquet_path, master_csv_path)
    dataset, feature_columns, target_columns = prepare_modeling_dataset(
        master,
        advanced_nlp_path=advanced_nlp_path,
        warnings_list=warnings_list,
    )
    metrics, feature_importance, training_meta = train_prediction_models(
        dataset,
        feature_columns,
        target_columns,
        warnings_list=warnings_list,
    )
    summary = build_summary(dataset, feature_columns, target_columns, metrics)
    warnings_df = pd.DataFrame(warnings_list, columns=WARNING_COLUMNS)

    metadata = {
        "row_count": int(len(dataset)),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "models_attempted": training_meta["models_attempted"],
        "models_trained": training_meta["models_trained"],
        "skipped_reasons": training_meta["skipped_reasons"],
    }

    output_paths = [
        summary_path,
        metrics_path,
        warnings_path,
        feature_importance_path,
        metadata_path,
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    warnings_df.to_csv(warnings_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return summary, metrics, warnings_df, feature_importance, metadata, output_paths


def main() -> None:
    summary, metrics, warnings_df, feature_importance, metadata, output_paths = run_predictive_modeling()

    print(f"Ran predictive modeling on {metadata['row_count']} rows.")
    print("Summary:")
    print(summary.to_string(index=False))
    print("Statuses:")
    if metrics.empty:
        print("no model attempts")
    else:
        print(metrics["status"].value_counts(dropna=False).to_string())
    print(f"Feature importance rows: {len(feature_importance)}")
    print(f"Warnings: {len(warnings_df)}")
    if not warnings_df.empty:
        print(warnings_df.to_string(index=False))
    print("Wrote outputs:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
