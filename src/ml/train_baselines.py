from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TRAIN_PATH = Path("data/processed/apple/classification/training.csv")
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
OUTPUT_PATH = Path("data/processed/apple/classification/baseline_results.csv")

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

    models = {
        "majority": DummyClassifier(
            strategy="most_frequent"
        ),
        "naive_bayes": MultinomialNB(
            alpha=0.1
        ),
        "logistic_regression": LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced"
        ),
        "linear_svm": LinearSVC(
            C=1.0,
            class_weight="balanced"
        )
    }

    results = []

    for name, model in models.items():
        model.fit(
            x_train_tfidf,
            y_train
        )

        predictions = model.predict(
            x_test_tfidf
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        print()
        print("=" * 80)
        print(name)
        print("=" * 80)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print()
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        })

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Summary:")
    print(
        results_df
        .sort_values("macro_f1", ascending=False)
        .to_string(index=False)
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()