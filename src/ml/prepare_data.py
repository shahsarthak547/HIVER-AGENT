from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

def main():
    df = pd.read_csv(INPUT_PATH)

    golden = pd.read_csv(GOLDEN_PATH)

    labels = golden[
        ["customer_tweet_id", "gold_label"]
    ].copy()

    labels = labels[
        labels["gold_label"].notna()
        & (labels["gold_label"] != "")
    ]

    df["customer_tweet_id"] = df["customer_tweet_id"].astype(str)
    labels["customer_tweet_id"] = labels["customer_tweet_id"].astype(str)

    labeled = df.merge(
        labels,
        on="customer_tweet_id",
        how="inner"
    )
    labeled = labeled[labeled["gold_label"] != "other_unclear"].copy()
    train, temp = train_test_split(
        labeled,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        stratify=labeled["gold_label"],
        random_state=RANDOM_STATE
    )

    relative_test_size = TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)

    validation, test = train_test_split(
        temp,
        test_size=relative_test_size,
        stratify=temp["gold_label"],
        random_state=RANDOM_STATE
    )

    output_dir = Path("data/processed/apple/classification")
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(
        output_dir / "train.csv",
        index=False
    )

    validation.to_csv(
        output_dir / "validation.csv",
        index=False
    )

    test.to_csv(
        output_dir / "test.csv",
        index=False
    )

    print(f"Total labeled examples: {len(labeled)}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test: {len(test)}")
    print()

    print("Train distribution:")
    print(train["gold_label"].value_counts().to_string())
    print()

    print("Validation distribution:")
    print(validation["gold_label"].value_counts().to_string())
    print()

    print("Test distribution:")
    print(test["gold_label"].value_counts().to_string())

if __name__ == "__main__":
    main()