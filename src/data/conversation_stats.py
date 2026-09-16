import json
from pathlib import Path

from src.data.conversation import reconstruct_conversations
from src.data.loader import load_dataset
def generate_conversation_stats(path: str, output_path: str) -> None:
    df = load_dataset(path)
    df = reconstruct_conversations(df)
    grouped = df.groupby("conversation_id")
    stats = {
        "conversations": int(df["conversation_id"].nunique()),
        "average_messages": float(grouped.size().mean()),
        "median_messages": float(grouped.size().median()),
        "max_messages": int(grouped.size().max()),
        "single_message_conversations": int((grouped.size() == 1).sum()),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
if __name__ == "__main__":
    generate_conversation_stats(
        "data/raw/sample.csv",
        "data/processed/conversation_stats.json"
    )