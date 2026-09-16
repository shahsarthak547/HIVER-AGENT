from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    text = df["customer_text"].fillna("").astype(str)

    word_counts = text.str.split().str.len()
    char_counts = text.str.len()

    print(f"Total customer messages: {len(df):,}")
    print()

    print("Message length statistics:")
    print(word_counts.describe())
    print()

    print("Character length statistics:")
    print(char_counts.describe())
    print()

    print("Empty messages:", (text.str.strip() == "").sum())
    print("Unique customer messages:", text.nunique())
    print()

    print("Shortest messages:")
    print(
        df.loc[
            word_counts.nsmallest(10).index,
            ["customer_text", "support_text"]
        ].to_string(index=False)
    )

    print()
    print("Longest messages:")
    print(
        df.loc[
            word_counts.nlargest(10).index,
            ["customer_text", "support_text"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()