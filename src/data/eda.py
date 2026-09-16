import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_dataset


def generate_eda(path: str, output_dir: str) -> None:
    df = load_dataset(path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = {
        "rows": len(df),
        "unique_tweets": int(df["tweet_id"].nunique()),
        "unique_authors": int(df["author_id"].nunique()),
        "inbound": int(df["inbound"].sum()),
        "outbound": int((~df["inbound"]).sum()),
        "duplicate_tweet_ids": int(df["tweet_id"].duplicated().sum()),
        "missing_text": int(df["text"].eq("").sum()),
        "missing_dates": int(df["created_at"].isna().sum()),
    }
    with open(output / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    df["text_length"] = df["text"].str.len()
    plt.figure(figsize=(10, 5))
    df["text_length"].plot(kind="hist", bins=50)
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.title("Tweet Text Length Distribution")
    plt.savefig(output / "text_length_distribution.png", bbox_inches="tight")
    plt.close()
    inbound_counts = df["inbound"].value_counts()
    plt.figure(figsize=(6, 4))
    inbound_counts.plot(kind="bar")
    plt.xlabel("Inbound")
    plt.ylabel("Count")
    plt.title("Inbound vs Outbound Tweets")
    plt.savefig(output / "inbound_outbound.png", bbox_inches="tight")
    plt.close()
if __name__ == "__main__":
    generate_eda(
        "data/raw/sample.csv",
        "data/processed/eda"
    )