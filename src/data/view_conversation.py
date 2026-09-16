import pandas as pd
from src.data.conversation import reconstruct_conversations

df = pd.read_csv("data/raw/sample.csv")

df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", errors="coerce")

df = reconstruct_conversations(df)

for conversation_id, group in df.groupby("conversation_id"):
    group = group.sort_values("created_at")

    print("\n" + "=" * 100)
    print(f"CONVERSATION: {conversation_id}")
    print("=" * 100)

    for _, row in group.iterrows():
        role = "CUSTOMER" if row["inbound"] else "SUPPORT"

        print(f"\n[{role}]")
        print(row["text"])