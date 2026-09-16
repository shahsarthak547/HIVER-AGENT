from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TRAIN_PATH = Path("data/processed/apple/classification/training.csv")
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
OUTPUT_PATH = Path("data/processed/apple/classification/tuning_results.csv")

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

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer()
        ),
        (
            "classifier",
            LinearSVC()
        )
    ])

    parameters = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2),
            (1, 3)
        ],
        "tfidf__min_df": [
            1,
            2,
            5
        ],
        "tfidf__sublinear_tf": [
            True,
            False
        ],
        "classifier__C": [
            0.5,
            1.0,
            2.0,
            4.0
        ],
        "classifier__class_weight": [
            None,
            "balanced"
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        pipeline,
        parameters,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    search.fit(
        x_train,
        y_train
    )

    predictions = search.predict(
        x_test
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
    print("BEST PARAMETERS")
    print("=" * 80)
    print(search.best_params_)

    print()
    print(f"CV Macro F1: {search.best_score_:.4f}")
    print(f"Golden Accuracy: {accuracy:.4f}")
    print(f"Golden Macro F1: {macro_f1:.4f}")
    print(f"Golden Weighted F1: {weighted_f1:.4f}")

    results = pd.DataFrame(
        search.cv_results_
    )

    results = results[
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "params"
        ]
    ].sort_values(
        "rank_test_score"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()