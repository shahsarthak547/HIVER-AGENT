from pathlib import Path
import pandas as pd
from src.data.conversation import reconstruct_conversations
INPUT_PATH = Path("data/processed/apple/apple_tweets.csv")
OUTPUT_PATH = Path("data/processed/apple/apple_conversations.csv")

def main():
    df = pd.read_csv(INPUT_PATH)
    df = reconstruct_conversations(df)
    df["created_at"] = pd.to_datetime(df["created_at"],format="mixed",errors="coerce")
    df = df.sort_values(["conversation_id", "created_at"])
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUTPUT_PATH,index=False)
    conversation_sizes = (df.groupby("conversation_id").size())
    print(f"Tweets: {len(df):,}")
    print(f"Conversations: {len(conversation_sizes):,}")
    print(f"Average messages: {conversation_sizes.mean():.2f}")
    print(f"Median messages: {conversation_sizes.median():.0f}")
    print(f"Max messages: {conversation_sizes.max():,}")
    print(f"Multi-turn conversations: {(conversation_sizes > 1).sum():,}")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()