from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from src.data.text_cleaning import normalize_text

INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")
OUTPUT_DIR = Path("data/processed/apple/intent_discovery")

SAMPLE_SIZE = 30000
K_VALUES = [8, 10, 12, 15, 18, 20]

MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
BATCH_SIZE = 128
REPRESENTATIVES_PER_CLUSTER = 8

def prepare_text(text):
    normalized = normalize_text(text)
    if not normalized:
        return ""
    return normalized

def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["normalized_text"] = (df["customer_text"].fillna("").map(prepare_text))
    df = df[df["normalized_text"].str.len() > 0].copy()
    return df.reset_index(drop=True)

def sample_data(df):
    if len(df) <= SAMPLE_SIZE:
        return df.copy()
    return df.sample(n=SAMPLE_SIZE,random_state=RANDOM_STATE).reset_index(drop=True)

def create_embeddings(texts):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts.tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(embeddings)

def evaluate_k_values(embeddings):
    results = []
    for k in K_VALUES:
        clustering = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=2048,
            n_init=10
        )
        labels = clustering.fit_predict(embeddings)
        score = silhouette_score(
            embeddings,
            labels,
            sample_size=min(10000, len(embeddings)),
            random_state=RANDOM_STATE
        )
        sizes = pd.Series(labels).value_counts()
        results.append({
            "k": k,
            "silhouette_score": score,
            "min_cluster_size": sizes.min(),
            "max_cluster_size": sizes.max(),
            "median_cluster_size": sizes.median()
        })
        print(f"k={k} | "f"silhouette={score:.4f} | "f"min={sizes.min()} | "f"max={sizes.max()}")
    return pd.DataFrame(results)

def fit_final_clustering(embeddings, k):
    clustering = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=2048,
        n_init=10
    )
    labels = clustering.fit_predict(embeddings)
    return clustering, labels

def get_representatives(df, embeddings, clustering):
    representatives = []
    for cluster_id in range(clustering.n_clusters):
        indices = np.where(clustering.labels_ == cluster_id)[0]
        cluster_embeddings = embeddings[indices]
        centroid = clustering.cluster_centers_[cluster_id]
        similarities = cosine_similarity(cluster_embeddings,centroid.reshape(1, -1)).ravel()
        top_indices = np.argsort(similarities)[-REPRESENTATIVES_PER_CLUSTER:][::-1]
        for rank, position in enumerate(top_indices, start=1):
            original_index = indices[position]
            representatives.append({
                "cluster": cluster_id,
                "rank": rank,
                "similarity_to_centroid": similarities[position],
                "customer_tweet_id": df.iloc[original_index]["customer_tweet_id"],
                "customer_text": df.iloc[original_index]["customer_text"],
                "support_text": df.iloc[original_index]["support_text"]
            })
    return pd.DataFrame(representatives)

def calculate_cluster_similarity(clustering):
    centers = clustering.cluster_centers_
    similarity_matrix = cosine_similarity(centers)
    rows = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            rows.append({"cluster_a": i,"cluster_b": j,"cosine_similarity": similarity_matrix[i, j]})
    return (pd.DataFrame(rows).sort_values("cosine_similarity",ascending=False).reset_index(drop=True))

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print("Loading data...")
    df = load_data()
    print(f"Usable messages: {len(df):,}")
    sample = sample_data(df)
    print(f"Clustering sample: {len(sample):,}")
    print("Creating embeddings...")
    embeddings = create_embeddings(sample["normalized_text"])
    np.save(OUTPUT_DIR / "embeddings.npy",embeddings)
    print()
    print("Evaluating cluster counts...")
    evaluation = evaluate_k_values(embeddings)
    evaluation.to_csv(OUTPUT_DIR / "k_evaluation.csv",index=False)
    selected_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(),"k"])
    print()
    print(f"Best silhouette k: {selected_k}")
    clustering, labels = fit_final_clustering(embeddings,selected_k)
    sample["cluster"] = labels
    sample.to_csv(OUTPUT_DIR / "clustered_messages.csv",index=False)
    representatives = get_representatives(sample,embeddings,clustering)
    representatives.to_csv(OUTPUT_DIR / "cluster_representatives.csv",index=False)
    similarity = calculate_cluster_similarity(clustering)
    similarity.to_csv(OUTPUT_DIR / "cluster_similarity.csv",index=False)
    print()
    print("Cluster sizes:")
    print(sample["cluster"].value_counts().sort_index().to_string())
    print()
    print("Most similar cluster pairs:")
    print(similarity.head(15).to_string(index=False))
    print()
    print("Representative examples:")
    for cluster_id in range(selected_k):
        print()
        print("=" * 100)
        print(f"CLUSTER {cluster_id}")
        rows = representatives[representatives["cluster"] == cluster_id]
        for _, row in rows.iterrows():
            print(f"- {row['customer_text']}")
    print()
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()