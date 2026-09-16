from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

INPUT_PATH = Path("data/processed/apple/apple_clean_pairs.csv")
OUTPUT_PATH = Path("data/processed/apple/intent_clusters.csv")

SAMPLE_SIZE = 20000
N_CLUSTERS = 12
RANDOM_STATE = 42

def main():
    df = pd.read_csv(INPUT_PATH)
    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)
    texts = sample["customer_text"].fillna("").astype(str)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts.tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    clustering = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=1024,
        n_init=10
    )
    labels = clustering.fit_predict(embeddings)
    sample["cluster"] = labels
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    sample.to_csv(
        OUTPUT_PATH,
        index=False
    )
    print()
    print("Cluster sizes:")
    print(
        sample["cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()
    print("Sample messages from each cluster:")
    for cluster_id in sorted(sample["cluster"].unique()):
        print()
        print("=" * 100)
        print(f"CLUSTER {cluster_id}")
        cluster_rows = sample[
            sample["cluster"] == cluster_id
        ].sample(
            n=min(10, (sample["cluster"] == cluster_id).sum()),
            random_state=RANDOM_STATE
        )
        for _, row in cluster_rows.iterrows():
            print(f"- {row['customer_text']}")

if __name__ == "__main__":
    main()