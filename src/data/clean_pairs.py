from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")
OUTPUT_PATH = Path("data/processed/apple/apple_clean_pairs.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    text = df["customer_text"].fillna("").astype(str).str.strip()

    at_apple_support_only = text.str.fullmatch(
        r"@AppleSupport(?:\s+)?",
        case=False
    )

    very_short = text.str.split().str.len() <= 2

    has_link = text.str.contains(
        r"https?://|www\.",
        case=False,
        regex=True
    )

    repeated_i = text.str.count("I️") >= 10

    duplicate_text = text.duplicated(keep="first")

    df["remove_reason"] = ""

    df.loc[at_apple_support_only, "remove_reason"] = "mention_only"

    df.loc[
        (df["remove_reason"] == "") & very_short,
        "remove_reason"
    ] = "very_short"

    df.loc[
        (df["remove_reason"] == "") & repeated_i,
        "remove_reason"
    ] = "repetitive"

    df.loc[
        (df["remove_reason"] == "") & duplicate_text,
        "remove_reason"
    ] = "duplicate"

    removed = df[df["remove_reason"] != ""].copy()
    clean = df[df["remove_reason"] == ""].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    clean.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Original pairs: {len(df):,}")
    print(f"Clean pairs: {len(clean):,}")
    print(f"Removed pairs: {len(removed):,}")
    print()

    print("Removal breakdown:")
    print(
        removed["remove_reason"]
        .value_counts()
        .to_string()
    )

    print()
    print("Examples of removed messages:")

    print(
        removed[
            ["customer_text", "remove_reason"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()