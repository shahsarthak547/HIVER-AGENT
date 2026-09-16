from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
CHUNK_SIZE = 100_000
stats = {}
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    inbound = chunk[chunk["inbound"] == True]
    outbound = chunk[chunk["inbound"] == False]
    for author in outbound["author_id"].dropna().unique():
        stats.setdefault(
            author,
            {
                "total_tweets": 0,
                "inbound_tweets": 0,
                "outbound_tweets": 0
            }
        )
        stats[author]["outbound_tweets"] += int(
            (outbound["author_id"] == author).sum()
        )
        stats[author]["inbound_tweets"] += int(
            (inbound["author_id"] == author).sum()
        )
        stats[author]["total_tweets"] = (
            stats[author]["inbound_tweets"]
            + stats[author]["outbound_tweets"]
        )
rows = []
for author, values in stats.items():
    if values["outbound_tweets"] > 0:
        rows.append({
            "author_id": author,
            **values
        })
result = pd.DataFrame(rows)
result = result.sort_values(
    "outbound_tweets",
    ascending=False
)
output_path = Path("data/processed/brand_candidates.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output_path, index=False)
print(result.head(50).to_string(index=False))