from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/apple/label_candidates.csv")
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
OUTPUT_PATH = Path("data/processed/apple/classification/training.csv")

MIN_CONFIDENCE = 0.35

def main():
    candidates = pd.read_csv(INPUT_PATH)
    golden = pd.read_csv(GOLDEN_PATH)

    candidates["customer_tweet_id"] = (
        candidates["customer_tweet_id"].astype(str)
    )

    golden["customer_tweet_id"] = (
        golden["customer_tweet_id"].astype(str)
    )

    golden_ids = set(golden["customer_tweet_id"])

    training = candidates[
        ~candidates["customer_tweet_id"].isin(golden_ids)
    ].copy()

    training = training[
        training["candidate_score"] >= MIN_CONFIDENCE
    ].copy()

    training = training[
        training["candidate_intent"] != "other_unclear"
    ].copy()

    training = training.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    training = training.rename(
        columns={
            "candidate_intent": "label"
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    training.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Candidate examples: {len(candidates):,}")
    print(f"Golden examples excluded: {len(golden_ids):,}")
    print(f"Training examples: {len(training):,}")
    print()
    print("Training distribution:")
    print(
        training["label"]
        .value_counts()
        .to_string()
    )
    print()
    print("Confidence statistics:")
    print(
        training["candidate_score"].describe()
    )
    print()
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()