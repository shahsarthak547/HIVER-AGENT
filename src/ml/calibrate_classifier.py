from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

TRAIN_PATH = Path("data/processed/apple/classification/training.csv")
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
MODEL_PATH = Path("models/apple_intent_classifier.joblib")

def main():
    train = pd.read_csv(TRAIN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)

    golden = golden[
        golden["gold_label"].notna()
        & (golden["gold_label"] != "")
        & (golden["gold_label"] != "other_unclear")
    ].copy()

    x_train = train["customer_text"].fillna("").astype(str)
    y_train = train["label"]

    x_test = golden["customer_text"].fillna("").astype(str)
    y_test = golden["gold_label"]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=1,
        sublinear_tf=True
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    svm = LinearSVC(
        C=0.5,
        class_weight="balanced"
    )

    calibrated = CalibratedClassifierCV(
        svm,
        method="sigmoid",
        cv=5
    )

    calibrated.fit(
        x_train_tfidf,
        y_train
    )

    predictions = calibrated.predict(
        x_test_tfidf
    )

    probabilities = calibrated.predict_proba(
        x_test_tfidf
    )

    confidence = probabilities.max(axis=1)

    golden["prediction"] = predictions
    golden["confidence"] = confidence
    golden["correct"] = (
        golden["gold_label"] == golden["prediction"]
    )

    print()
    print("=" * 80)
    print("CALIBRATED SVM")
    print("=" * 80)

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Macro F1: "
        f"{f1_score(y_test, predictions, average='macro'):.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print()
    print("Confidence statistics:")
    print(
        golden["confidence"].describe()
    )

    print()
    print("Confidence by correctness:")
    print(
        golden.groupby("correct")["confidence"]
        .agg(["count", "mean", "min", "max"])
    )

    print()
    print("Lowest-confidence predictions:")

    print(
        golden[
            [
                "customer_text",
                "gold_label",
                "prediction",
                "confidence"
            ]
        ]
        .sort_values("confidence")
        .head(20)
        .to_string(index=False)
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": calibrated
        },
        MODEL_PATH
    )

    golden.to_csv(
        "data/processed/apple/classification/calibrated_predictions.csv",
        index=False
    )

    print()
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()