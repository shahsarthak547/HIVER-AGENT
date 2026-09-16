from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

TRAIN_PATH = Path("data/processed/apple/classification/training.csv")
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
OUTPUT_DIR = Path("data/processed/apple/classification")

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
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = LinearSVC(
        C=1.0,
        class_weight="balanced"
    )

    model.fit(
        x_train_tfidf,
        y_train
    )

    predictions = model.predict(x_test_tfidf)

    golden["prediction"] = predictions
    golden["correct"] = (
        golden["gold_label"] == golden["prediction"]
    )

    errors = golden[
        ~golden["correct"]
    ].copy()

    errors = errors[
        [
            "customer_tweet_id",
            "customer_text",
            "support_text",
            "gold_label",
            "prediction"
        ]
    ]

    errors.to_csv(
        OUTPUT_DIR / "classification_errors.csv",
        index=False
    )

    labels = sorted(
        set(y_test) | set(predictions)
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    confusion = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    confusion.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print("Misclassified examples:", len(errors))
    print()

    print("Confusion matrix:")
    print(confusion.to_string())

    print()
    print("Misclassified examples:")
    print()

    for _, row in errors.iterrows():
        print("=" * 100)
        print("CUSTOMER:")
        print(row["customer_text"])
        print()
        print(f"TRUE: {row['gold_label']}")
        print(f"PREDICTED: {row['prediction']}")
        print()

    print()
    print("Saved:")
    print(OUTPUT_DIR / "classification_errors.csv")
    print(OUTPUT_DIR / "confusion_matrix.csv")

if __name__ == "__main__":
    main()