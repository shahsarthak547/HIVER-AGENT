from pathlib import Path
import faiss
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
INDEX_PATH = Path("models/apple_support.index")
METADATA_PATH = Path("models/apple_support_metadata.csv")
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
MODEL_PATH = Path("models/apple_intent_classifier.joblib")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
def load_data():
    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    golden = golden[golden["gold_label"] != "other_unclear"].copy()
    loaded = joblib.load(MODEL_PATH)
    if not isinstance(loaded, dict):
        raise ValueError("Expected saved classifier dictionary")
    vectorizer = loaded["vectorizer"]
    classifier = loaded["model"]
    return (index,metadata,golden,vectorizer,classifier)
def evaluate(index,metadata,golden,vectorizer,classifier,model):
    rows = []
    for _, row in golden.iterrows():
        query = str(row["customer_text"])
        gold_intent = row["gold_label"]
        query_embedding = model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(query_embedding,TOP_K)
        retrieved = metadata.iloc[indices[0]].copy()
        retrieved_features = vectorizer.transform(retrieved["customer_text"].tolist())
        retrieved_intents = classifier.predict(retrieved_features)
        first_match = None
        for rank, intent in enumerate(retrieved_intents,start=1):
            if intent == gold_intent:
                first_match = rank
                break
        rows.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "gold_intent": gold_intent,
                "top_similarity": float(scores[0][0]),
                "retrieved_intents": "|".join(
                    retrieved_intents
                ),
                "first_same_intent_rank": first_match
            }
        )
    return pd.DataFrame(rows)
def main():
    (index,metadata,golden,vectorizer,classifier) = load_data()
    model = SentenceTransformer(MODEL_NAME)
    results = evaluate(index,metadata,golden,vectorizer,classifier,model)
    output_dir = Path("data/processed/apple/retrieval")
    output_dir.mkdir(parents=True,exist_ok=True)
    output_path = (output_dir / "retrieval_evaluation.csv")
    results.to_csv(output_path,index=False)
    print()
    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Queries evaluated: {len(results)}")
    for k in [1, 3, 5]:
        recall = (results["first_same_intent_rank"].fillna(999)<= k).mean()
        print(f"Intent-consistent Recall@{k}: " f"{recall:.4f}")
    reciprocal_ranks = (
        results["first_same_intent_rank"]
        .apply(
            lambda x: 1 / x
            if pd.notna(x)
            else 0
        )
    )
    print(f"MRR: {reciprocal_ranks.mean():.4f}")
    print(f"Mean top similarity: " f"{results['top_similarity'].mean():.4f}")
    print()
    print(f"Saved: {output_path}")
if __name__ == "__main__":
    main()