from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/apple/apple_tweets.csv")
OUTPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")

BRAND = "AppleSupport"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["tweet_id"] = df["tweet_id"].astype(str)

    parent_map = (
        df.set_index("tweet_id")["in_response_to_tweet_id"]
        .dropna()
        .astype(int)
        .astype(str)
        .to_dict()
    )

    tweet_map = df.set_index("tweet_id")

    pairs = []

    for _, row in df.iterrows():
        if str(row["author_id"]) != BRAND:
            continue

        parent_id = row["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            continue

        parent_id = str(int(parent_id))

        if parent_id not in tweet_map.index:
            continue

        parent = tweet_map.loc[parent_id]

        if bool(parent["inbound"]):
            pairs.append({
                "customer_tweet_id": parent_id,
                "support_tweet_id": str(row["tweet_id"]),
                "customer_author_id": str(parent["author_id"]),
                "customer_text": str(parent["text"]),
                "support_text": str(row["text"]),
                "customer_created_at": parent["created_at"],
                "support_created_at": row["created_at"],
            })

    result = pd.DataFrame(pairs)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Customer-support pairs: {len(result):,}")
    print(f"Unique customer tweets: {result['customer_tweet_id'].nunique():,}")
    print(f"Unique support tweets: {result['support_tweet_id'].nunique():,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()