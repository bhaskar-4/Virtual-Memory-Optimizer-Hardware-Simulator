import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["access_frequency", "is_write", "reuse_distance"]
REQUIRED_COLUMNS = ["page_id", *FEATURE_COLUMNS, "optimal_tier"]


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
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Telemetry dataset is missing columns: {', '.join(missing_columns)}")
    data = data.dropna(subset=REQUIRED_COLUMNS)
    if data.empty:
        raise ValueError("Telemetry dataset contains no usable page records")

    features = data[FEATURE_COLUMNS].astype(float)
    if predict_only:
        if not model_output.exists():
            raise FileNotFoundError(f"Trained model not found: {model_output}")
        classifier = joblib.load(model_output)
        print(f"Loaded trained model from {model_output}")
    else:
        labels = data["optimal_tier"].astype(int)
        model_settings = {
            "n_estimators": 100,
            "max_depth": 5,
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
            held_out_accuracy = accuracy_score(labels_test, evaluation_model.predict(features_test))
            print(f"Held-out accuracy (25% stratified test split): {held_out_accuracy * 100:.2f}%")
        else:
            print("Held-out accuracy unavailable: dataset needs at least two examples of each class.")

        classifier = RandomForestClassifier(**model_settings)
        classifier.fit(features, labels)
        training_accuracy = accuracy_score(labels, classifier.predict(features))
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, model_output)
        print(f"Trained Random Forest on {len(data)} pages")
        print(f"Training-set accuracy (in-sample): {training_accuracy * 100:.2f}%")
        print(f"Saved model to {model_output}")

    predictions = classifier.predict(features).astype(int)
    class_labels = list(classifier.classes_)
    positive_class_index = class_labels.index(1) if 1 in class_labels else None
    dram_scores = (
        classifier.predict_proba(features)[:, positive_class_index]
        if positive_class_index is not None
        else [0.0] * len(data)
    )
    output = pd.DataFrame(
        {
            "page_id": data["page_id"].astype("int64"),
            "access_frequency": data["access_frequency"].astype("int64"),
            "is_write": data["is_write"].astype("int64"),
            "reuse_distance": data["reuse_distance"].astype("int64"),
            "predicted_tier": predictions,
            "dram_score": dram_scores,
        }
    )
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_output, index=False)
    print(f"Wrote {len(output)} page placement predictions to {predictions_output}")


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