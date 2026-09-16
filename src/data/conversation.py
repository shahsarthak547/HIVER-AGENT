import pandas as pd


def build_tweet_map(df: pd.DataFrame) -> dict:
    return {
        str(row["tweet_id"]): row
        for _, row in df.iterrows()
    }


def get_parent_id(row) -> str | None:
    value = row["in_response_to_tweet_id"]

    if pd.isna(value):
        return None

    return str(int(value))


def find_support_root(tweet_id: str, tweet_map: dict) -> str:
    current = tweet_id
    visited = set()

    while current in tweet_map and current not in visited:
        visited.add(current)

        row = tweet_map[current]
        parent_id = get_parent_id(row)

        if parent_id is None:
            return current

        if parent_id not in tweet_map:
            return current

        parent = tweet_map[parent_id]

        if bool(row["inbound"]) and not bool(parent["inbound"]):
            return current

        current = parent_id

    return current


def reconstruct_conversations(df: pd.DataFrame) -> pd.DataFrame:
    tweet_map = build_tweet_map(df)

    result = df.copy()

    result["conversation_id"] = result["tweet_id"].astype(str).apply(
        lambda tweet_id: find_support_root(tweet_id, tweet_map)
    )

    return result