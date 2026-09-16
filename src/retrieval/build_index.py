from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_PATH = Path("data/retrieval_apple_cases.csv")
OUTPUT_PATH = Path("models/apple_support_retrieval.joblib")

MAX_FEATURES = 30000


def main():
    cases = pd.read_csv(INPUT_PATH)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=MAX_FEATURES,
        sublinear_tf=True
    )

    embeddings = vectorizer.fit_transform(
        cases["customer_text"].fillna("")
    )

    artifact = {
        "vectorizer": vectorizer,
        "embeddings": embeddings,
        "metadata": cases
    }

    joblib.dump(
        artifact,
        OUTPUT_PATH,
        compress=3
    )

    print(f"Indexed examples: {len(cases)}")
    print(f"Features: {embeddings.shape[1]}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()