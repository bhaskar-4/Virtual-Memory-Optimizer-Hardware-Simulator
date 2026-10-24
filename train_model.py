import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_FEATURE_COLUMNS = ["access_frequency", "is_write", "reuse_distance"]
REQUIRED_COLUMNS = ["page_id", *BASE_FEATURE_COLUMNS, "optimal_tier"]


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    transformed = data.copy()
    transformed["access_frequency"] = pd.to_numeric(transformed["access_frequency"], errors="coerce").fillna(0.0)
    transformed["is_write"] = pd.to_numeric(transformed["is_write"], errors="coerce").fillna(0.0)
    transformed["reuse_distance"] = pd.to_numeric(transformed["reuse_distance"], errors="coerce").fillna(0.0)

    transformed["hotness_score"] = transformed["access_frequency"] / (transformed["reuse_distance"] + 1.0)
    transformed["locality_factor"] = (transformed["reuse_distance"] < 2000).astype(float)
    transformed["write_pressure"] = transformed["is_write"] * 0.5
    transformed["frequency_norm"] = transformed["access_frequency"] / (transformed["access_frequency"].max() + 1.0)
    transformed["hotness_norm"] = transformed["hotness_score"] / (transformed["hotness_score"].max() + 1.0)
    transformed["reuse_strength"] = 1.0 / (1.0 + transformed["reuse_distance"] / 1000.0)
    transformed["scan_penalty"] = (transformed["reuse_distance"] > 4000).astype(float)
    transformed["value_score"] = (
        0.5 * transformed["frequency_norm"]
        + 0.25 * transformed["hotness_norm"]
        + 0.15 * transformed["reuse_strength"]
        + 0.10 * transformed["write_pressure"]
        - 0.20 * transformed["scan_penalty"]
    )
    transformed["value_score"] = transformed["value_score"].clip(lower=0.0)
    return transformed


def _resolve_labels(data: pd.DataFrame) -> pd.Series:
    if "optimal_tier" in data.columns:
        labels = data["optimal_tier"].astype(int)
        if labels.nunique() >= 2:
            return labels

    if "value_score" not in data.columns:
        raise ValueError("No usable value_score or optimal_tier column found in the dataset")

    threshold = data["value_score"].quantile(0.75)
    labels = (data["value_score"] >= threshold).astype(int)
    return labels


def train_page_placement_model(
    dataset_path: str | Path = "telemetry_ml_dataset.csv",
    model_output: str | Path = "page_classifier.joblib",
    predictions_output: str | Path = "page_predictions.csv",
    predict_only: bool = False,
) -> None:
    dataset_path = Path(dataset_path)
    model_output = Path(model_output)
    predictions_output = Path(predictions_output)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Telemetry dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    required = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if required:
        if not {"hotness_score", "value_score"}.issubset(data.columns):
            raise ValueError(
                f"Telemetry dataset is missing required columns: {', '.join(required)}. "
                "It also lacks the engineered features needed for a production-grade label policy."
            )

    data = data.dropna(subset=[column for column in ["page_id", "access_frequency", "is_write", "reuse_distance"] if column in data.columns])
    if data.empty:
        raise ValueError("Telemetry dataset contains no usable page records")

    engineered = _engineer_features(data)
    labels = _resolve_labels(engineered)
    feature_columns = [
        "access_frequency",
        "is_write",
        "reuse_distance",
        "hotness_score",
        "locality_factor",
        "frequency_norm",
        "hotness_norm",
        "reuse_strength",
        "scan_penalty",
        "value_score",
    ]
    features = engineered[feature_columns].astype(float)

    if predict_only:
        if not model_output.exists():
            raise FileNotFoundError(f"Trained model not found: {model_output}")
        classifier = joblib.load(model_output)
        print(f"Loaded trained model from {model_output}")
    else:
        model_settings = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "n_jobs": -1,
        }
        class_counts = labels.value_counts()
        if labels.nunique() == 2 and class_counts.min() >= 2:
            features_train, features_test, labels_train, labels_test = train_test_split(
                features,
                labels,
                test_size=0.25,
                random_state=42,
                stratify=labels,
            )
            evaluation_model = RandomForestClassifier(**model_settings)
            evaluation_model.fit(features_train, labels_train)
            held_out_predictions = evaluation_model.predict(features_test)
            held_out_accuracy = accuracy_score(labels_test, held_out_predictions)
            held_out_precision = precision_score(labels_test, held_out_predictions, zero_division=0)
            held_out_recall = recall_score(labels_test, held_out_predictions, zero_division=0)
            held_out_f1 = f1_score(labels_test, held_out_predictions, zero_division=0)
            print(f"Held-out accuracy (25% stratified test split): {held_out_accuracy * 100:.2f}%")
            print(f"Held-out precision: {held_out_precision * 100:.2f}%")
            print(f"Held-out recall: {held_out_recall * 100:.2f}%")
            print(f"Held-out F1: {held_out_f1 * 100:.2f}%")
        else:
            print("Held-out accuracy unavailable: dataset needs at least two examples of each class.")

        classifier = RandomForestClassifier(**model_settings)
        classifier.fit(features, labels)
        training_predictions = classifier.predict(features)
        training_accuracy = accuracy_score(labels, training_predictions)
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, model_output)
        print(f"Trained production-grade Random Forest on {len(engineered)} pages")
        print(f"Training-set accuracy (in-sample): {training_accuracy * 100:.2f}%")
        print(f"Saved model to {model_output}")

    predicted_tiers = classifier.predict(features).astype(int)
    class_labels = list(classifier.classes_)
    positive_class_index = class_labels.index(1) if 1 in class_labels else None
    if positive_class_index is None:
        dram_scores = np.zeros(len(engineered), dtype=float)
    else:
        dram_scores = classifier.predict_proba(features)[:, positive_class_index]

    output = pd.DataFrame(
        {
            "page_id": engineered["page_id"].astype("int64"),
            "access_frequency": engineered["access_frequency"].astype("int64"),
            "is_write": engineered["is_write"].astype("int64"),
            "reuse_distance": engineered["reuse_distance"].astype("int64"),
            "predicted_tier": predicted_tiers,
            "dram_score": dram_scores,
            "keep_probability": dram_scores,
            "evict_probability": 1.0 - dram_scores,
        }
    )
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_output, index=False)
    print(f"Wrote {len(output)} production-grade page placement predictions to {predictions_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or apply the page placement classifier.")
    parser.add_argument("--dataset", default="telemetry_ml_dataset.csv")
    parser.add_argument("--model", default="page_classifier.joblib")
    parser.add_argument("--predictions", default="page_predictions.csv")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    train_page_placement_model(args.dataset, args.model, args.predictions, args.predict_only)


if __name__ == "__main__":
    main()