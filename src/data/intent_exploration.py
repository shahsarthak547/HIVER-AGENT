from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.data.text_cleaning import normalize_text


INPUT_PATH = Path("data/processed/apple/apple_clean_pairs.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    texts = (
        df["customer_text"]
        .fillna("")
        .map(normalize_text)
    )

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=50,
        max_df=0.95
    )

    matrix = vectorizer.fit_transform(texts)

    counts = matrix.sum(axis=0).A1

    terms = vectorizer.get_feature_names_out()

    frequencies = (
        pd.DataFrame({
            "term": terms,
            "count": counts
        })
        .sort_values("count", ascending=False)
    )

    print("Top customer terms and phrases:")
    print()
    print(frequencies.head(100).to_string(index=False))


if __name__ == "__main__":
    main()