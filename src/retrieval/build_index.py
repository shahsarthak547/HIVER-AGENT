from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
INDEX_PATH = Path("models/apple_support.index")
METADATA_PATH = Path("models/apple_support_metadata.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
def main():
    df = pd.read_csv(INPUT_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    golden_ids = set(golden["customer_tweet_id"].astype(str))
    df["customer_tweet_id"] = (df["customer_tweet_id"].astype(str))
    df = df[~df["customer_tweet_id"].isin(golden_ids)].copy()
    df = df.drop_duplicates(subset=["customer_tweet_id"])
    df["customer_text"] = (df["customer_text"].fillna("").astype(str))
    df["support_text"] = (df["support_text"].fillna("").astype(str))
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["customer_text"].tolist(),batch_size=BATCH_SIZE,show_progress_bar=True,normalize_embeddings=True)
    embeddings = np.asarray(embeddings,dtype="float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    INDEX_PATH.parent.mkdir(parents=True,exist_ok=True)
    faiss.write_index(index,str(INDEX_PATH))
    metadata = df[
        [
            "customer_tweet_id",
            "support_tweet_id",
            "customer_text",
            "support_text",
            "customer_created_at",
            "support_created_at"
        ]
    ].copy()
    metadata.to_csv(METADATA_PATH,index=False)
    print()
    print(f"Indexed examples: {len(df):,}")
    print(f"Embedding dimension: {dimension}")
    print(f"Index size: {index.ntotal:,}")
    print()
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
if __name__ == "__main__":
    main()