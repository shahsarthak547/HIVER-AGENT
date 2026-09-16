from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = Path("models/apple_support_retrieval.joblib")
TOP_K = 5


def load_retrieval():
    return joblib.load(MODEL_PATH)


def search(query, artifact):
    vectorizer = artifact["vectorizer"]
    embeddings = artifact["embeddings"]
    metadata = artifact["metadata"]

    query_embedding = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarities.argsort()[-TOP_K:][::-1]

    results = metadata.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results