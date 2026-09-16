from pathlib import Path
import pandas as pd
INPUT_PATH = Path("data/processed/apple/apple_conversations.csv")

def main():
    df = pd.read_csv(INPUT_PATH)
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="mixed",
        errors="coerce"
    )
    conversations = (
        df.groupby("conversation_id")
        .size()
        .sort_values(ascending=False)
    )
    print("Conversation size distribution:")
    print(conversations.describe())
    print()
    print("Largest conversations:")
    print(conversations.head(10))
    print()
    for conversation_id in conversations.head(5).index:
        conversation = df[
            df["conversation_id"] == conversation_id
        ].sort_values("created_at")
        print("=" * 80)
        print(f"Conversation ID: {conversation_id}")
        print()
        for _, row in conversation.iterrows():
            direction = (
                "CUSTOMER"
                if row["inbound"]
                else "APPLESUPPORT"
            )
            print(f"[{direction}] "f"{row['created_at']} "f"(tweet {row['tweet_id']})")
            print(row["text"])
            print()

if __name__ == "__main__":
    main()