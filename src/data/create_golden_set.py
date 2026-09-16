from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/apple/label_candidates.csv")
OUTPUT_PATH = Path("data/golden/apple_golden_200.csv")

EXAMPLES_PER_INTENT = 20
RANDOM_STATE = 42

def main():
    df = pd.read_csv(INPUT_PATH)

    selected = (
        df.sort_values(
            ["candidate_intent", "candidate_score"],
            ascending=[True, False]
        )
        .groupby("candidate_intent")
        .head(EXAMPLES_PER_INTENT)
    )

    selected = selected.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    selected["gold_label"] = ""

    selected["gold_label"] = selected["candidate_intent"]

    selected["label_notes"] = ""

    columns = [
        "customer_tweet_id",
        "customer_text",
        "support_text",
        "candidate_intent",
        "candidate_score",
        "gold_label",
        "label_notes"
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    selected[columns].to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Golden examples created: {len(selected)}")
    print()
    print(
        selected["candidate_intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()