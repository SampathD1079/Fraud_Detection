"""
================= ML ADDITION: standalone training script =================

Trains the RandomForestClassifier fraud model completely SEPARATELY from
the invoice-processing pipeline (fraud_detect_code_with_db.py).

This script is only ever run manually/explicitly when you have (new)
labeled data -- it is never triggered automatically while processing
invoices.

Usage:
    python train_ml_model.py --data training_data.csv --out fraud_model.pkl

The input CSV must have exactly these columns (see sample_training_data.csv):
    creator, modification_creation, pdf_version, trailer_id_match,
    match_xref, font_score, is_fraud
"""

import argparse
import sys

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Must stay in sync with FEATURE_COLUMNS in fraud_detect_code_with_db.py
FEATURE_COLUMNS = [
    "creator",
    "modification_creation",
    "pdf_version",
    "trailer_id_match",
    "match_xref",
    "font_score",
]
LABEL_COLUMN = "is_fraud"


def train(data_path: str, model_out: str, test_size: float = 0.25, random_state: int = 42):
    df = pd.read_csv(data_path)

    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training CSV '{data_path}' is missing required column(s): {missing}. "
            f"Expected columns: {FEATURE_COLUMNS + [LABEL_COLUMN]}"
        )

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    if y.nunique() < 2:
        raise ValueError(
            "Training data must contain BOTH fraud (1) and non-fraud (0) "
            "examples -- only one class was found in the label column. "
            "A model trained on a single class cannot produce a meaningful "
            "fraud probability."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n── Evaluation on held-out test set ──")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    try:
        print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    except ValueError:
        print("ROC-AUC  : undefined (test set contains only one class -- add more data)")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    print("\nFeature Importance:")
    for col, importance in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {col}: {importance:.4f}")

    joblib.dump(model, model_out)
    print(f"\nModel saved to: {model_out}")
    print(
        "\nNOTE: these metrics reflect performance on YOUR training CSV only. "
        "With a small or synthetic dataset they are not a reliable estimate "
        "of real-world accuracy -- treat this model as a starting point, not "
        "a validated fraud detector, until it has been trained on a "
        "sufficiently large, representative, real labeled dataset."
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the invoice fraud-detection RandomForest model (ML ADDITION)."
    )
    parser.add_argument("--data", default="training_data.csv", help="Path to labeled training CSV")
    parser.add_argument("--out", default="fraud_model.pkl", help="Output path for the trained model")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of data held out for testing")
    args = parser.parse_args()

    try:
        train(args.data, args.out, test_size=args.test_size)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)