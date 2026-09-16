from pathlib import Path
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
INDEX_PATH = Path("models/apple_support.index")
METADATA_PATH = Path("models/apple_support_metadata.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

def search(query, model, index, metadata):
    embedding = model.encode([query],normalize_embeddings=True)
    scores, indices = index.search(embedding,TOP_K)
    results = metadata.iloc[indices[0]].copy()
    results["similarity"] = scores[0]
    return results

def main():
    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)
    model = SentenceTransformer(MODEL_NAME)
    query = input("Customer message: ").strip()
    results = search(query,model,index,metadata)
    print()
    print("=" * 100)
    print("RETRIEVED HISTORICAL CASES")
    print("=" * 100)
    for rank, (_, row) in enumerate(results.iterrows(),start=1):
        print()
        print(f"RESULT {rank}")
        print(f"Similarity: {row['similarity']:.4f}")
        print()
        print("CUSTOMER:")
        print(row["customer_text"])
        print()
        print("APPLESUPPORT:")
        print(row["support_text"])
if __name__ == "__main__":
    main()