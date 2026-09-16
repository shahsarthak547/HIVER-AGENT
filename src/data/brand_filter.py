from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed/apple/apple_tweets.csv")

BRAND = "AppleSupport"
CHUNK_SIZE = 100_000

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

brand_tweet_ids = set()

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE,
    usecols=columns
):
    matches = chunk[
        chunk["author_id"].astype(str) == BRAND
    ]

    brand_tweet_ids.update(
        matches["tweet_id"].astype(str)
    )

print(f"Total {BRAND} tweets: {len(brand_tweet_ids):,}")

connected_tweet_ids = set(brand_tweet_ids)

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE,
    usecols=columns
):
    parent_ids = (
        chunk["in_response_to_tweet_id"]
        .dropna()
        .astype(int)
        .astype(str)
    )

    parent_mask = parent_ids.isin(brand_tweet_ids)

    parent_indices = parent_ids.index[parent_mask]

    connected_tweet_ids.update(
        chunk.loc[parent_indices, "tweet_id"].astype(str)
    )

    def contains_brand_id(value):
        if pd.isna(value):
            return False

        ids = str(value).split(",")

        return any(
            tweet_id in brand_tweet_ids
            for tweet_id in ids
        )

    response_mask = chunk["response_tweet_id"].apply(
        contains_brand_id
    )

    connected_tweet_ids.update(
        chunk.loc[response_mask, "tweet_id"].astype(str)
    )

print(
    f"Total directly connected tweets: "
    f"{len(connected_tweet_ids):,}"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

first_write = True

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE
):
    filtered = chunk[
        chunk["tweet_id"]
        .astype(str)
        .isin(connected_tweet_ids)
    ]

    if not filtered.empty:
        filtered.to_csv(
            OUTPUT_PATH,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False

print(f"Saved to: {OUTPUT_PATH}")