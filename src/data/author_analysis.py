import pandas as pd
df = pd.read_csv("data/raw/sample.csv")
summary = (
    df.groupby("author_id")
    .agg(
        total_tweets=("tweet_id", "count"),
        inbound_tweets=("inbound", "sum"),
        outbound_tweets=("inbound", lambda x: (~x).sum())
    )
    .sort_values("total_tweets", ascending=False)
)
print(summary.to_string())