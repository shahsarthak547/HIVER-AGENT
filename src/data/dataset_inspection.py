import json
from pathlib import Path
import pandas as pd
DATA_PATH = Path("data/raw/twcs.csv")
CHUNK_SIZE = 100_000

REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}

def inspect_dataset(path: Path) -> dict:
    total_rows = 0
    inbound = 0
    outbound = 0
    missing_text = 0
    missing_created_at = 0
    duplicate_ids = 0
    unique_authors = set()
    date_min = None
    date_max = None
    first_chunk = True
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        if first_chunk:
            missing = REQUIRED_COLUMNS - set(chunk.columns)
            if missing:
                raise ValueError(f"Missing columns: {sorted(missing)}")
            first_chunk = False
        total_rows += len(chunk)
        inbound += int(chunk["inbound"].sum())
        outbound += int((~chunk["inbound"]).sum())
        missing_text += int(chunk["text"].isna().sum())
        missing_created_at += int(chunk["created_at"].isna().sum())
        duplicate_ids += int(chunk["tweet_id"].duplicated().sum())
        unique_authors.update(chunk["author_id"].astype(str).unique())
        dates = pd.to_datetime(
            chunk["created_at"],
            format="mixed",
            errors="coerce"
        )
        chunk_min = dates.min()
        chunk_max = dates.max()
        if date_min is None or chunk_min < date_min:
            date_min = chunk_min
        if date_max is None or chunk_max > date_max:
            date_max = chunk_max
        print(f"Processed {total_rows:,} rows")
    return {
        "rows": total_rows,
        "unique_authors": len(unique_authors),
        "inbound_tweets": inbound,
        "outbound_tweets": outbound,
        "missing_text": missing_text,
        "missing_created_at": missing_created_at,
        "duplicate_tweet_ids_within_chunks": duplicate_ids,
        "date_min": str(date_min),
        "date_max": str(date_max),
    }

if __name__ == "__main__":
    stats = inspect_dataset(DATA_PATH)
    output_path = Path("data/processed/full_dataset_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))