
"""
baseline training

Goal:
Train a machine learning model to classify network traffic as
normal (0) or suspicious (1)

Dataset:
KDD Cup 99 HTTP subset
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "intrusion_model.joblib"


def main():
    print("Loading the KDD Cup 99 HTTP dataset...")

    dataset = fetch_kddcup99(
        subset="http",
        percent10=True,
        as_frame=True,
    )

    X = dataset.data.copy()
    

    X = dataset.data.copy()

    # Convert byte-string feature values into ordinary text.
    for column in X.select_dtypes(include=["object"]).columns:
        X[column] = X[column].map(
            lambda value: (
                value.decode("utf-8")
                if isinstance(value, bytes)
                else value
            )
        )

    raw_labels = dataset.target


    raw_labels = dataset.target

    # Convert byte labels into readable strings.
    labels = pd.Series(raw_labels).map(
        lambda value: (
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
        )
    )

    # normal. becomes 0; every attack label becomes 1.
    y = (labels != "normal.").astype(int)

    print(f"Total records: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Normal records: {(y == 0).sum()}")
    print(f"Suspicious records: {(y == 1).sum()}")

    # Identify categorical columns that need encoding
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Convert categories into numeric features.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        ],
        remainder="passthrough",
    )

    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Preprocessing and model are saved together in one pipeline.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining the model...")
    model.fit(X_train, y_train)

    print("Evaluating on unseen test data...")
    predictions = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Suspicious"],
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
