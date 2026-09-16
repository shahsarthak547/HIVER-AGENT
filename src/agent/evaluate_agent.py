from pathlib import Path
import faiss
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.agent.escalation import decide_escalation
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
INDEX_PATH = Path("models/apple_support.index")
METADATA_PATH = Path("models/apple_support_metadata.csv")
MODEL_PATH = Path("models/apple_intent_classifier.joblib")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
def load_components():
    golden = pd.read_csv(GOLDEN_PATH)
    golden = golden[
        golden["gold_label"] != "other_unclear"
    ].copy()

    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)

    saved_model = joblib.load(MODEL_PATH)
    vectorizer = saved_model["vectorizer"]
    classifier = saved_model["model"]
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return (golden,index,metadata,vectorizer,classifier,embedding_model)
def evaluate_agent(golden,index,metadata,vectorizer,classifier,embedding_model):
    messages = golden["customer_text"].tolist()
    features = vectorizer.transform(messages)
    predicted_intents = classifier.predict(features)

    probabilities = classifier.predict_proba(features)
    confidences = probabilities.max(axis=1)
    embeddings = embedding_model.encode(messages,normalize_embeddings=True,batch_size=64,show_progress_bar=True)
    scores, indices = index.search(embeddings,TOP_K)
    rows = []
    for i in range(len(golden)):
        retrieved_cases = []

        for score, index_id in zip(scores[i],indices[i]):
            retrieved_cases.append(
                {
                    "customer_text": metadata.iloc[
                        index_id
                    ]["customer_text"],
                    "support_text": metadata.iloc[
                        index_id
                    ]["support_text"],
                    "similarity": float(score)
                }
            )
        decision = decide_escalation(
            intent=predicted_intents[i],
            classifier_confidence=float(confidences[i]),
            retrieved_cases=retrieved_cases,
            customer_message=messages[i],
            min_classifier_confidence=0.80,
            min_retrieval_similarity=0.70
        )
        rows.append(
            {
                "customer_text": messages[i],
                "gold_intent": golden.iloc[i]["gold_label"],
                "predicted_intent": predicted_intents[i],
                "classifier_confidence": float(confidences[i]),
                "top_retrieval_similarity": float(scores[i][0]),
                "decision": ("HUMAN" if decision["escalate"]else "AUTO-HANDLE"),
                "decision_reason": decision["reason"],
                "intent_correct": (predicted_intents[i] == golden.iloc[i]["gold_label"])
            }
        )
    return pd.DataFrame(rows)
def main():
    (golden,index,metadata,vectorizer,classifier,embedding_model) = load_components()
    print()
    print("Evaluating complete agent...")
    results = evaluate_agent(golden,index,metadata,vectorizer,classifier,embedding_model)
    output_path = Path("data/processed/apple/""classification/" "agent_evaluation.csv")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    results.to_csv(output_path,index=False)
    total = len(results)
    auto_handled = (results["decision"]== "AUTO-HANDLE").sum()
    escalated = (results["decision"] == "HUMAN").sum()
    correct_auto_handled = (
        results.loc[results["decision"]== "AUTO-HANDLE","intent_correct"]).sum()
    auto_handle_rate = (auto_handled / total)
    escalation_rate = (escalated / total)
    auto_handle_accuracy = (correct_auto_handled / auto_handled if auto_handled > 0 else 0)
    overall_intent_accuracy = (results["intent_correct"].mean())
    print()
    print("=" * 80)
    print("COMPLETE AGENT EVALUATION")
    print("=" * 80)
    print()
    print(f"Total golden examples: " f"{total}")
    print(f"Auto-handled: " f"{auto_handled} " f"({auto_handle_rate * 100:.1f}%)")
    print(f"Human escalations: "f"{escalated} "f"({escalation_rate * 100:.1f}%)")
    print(f"Overall intent accuracy: " f"{overall_intent_accuracy * 100:.1f}%")
    print(f"Auto-handled intent accuracy: " f"{auto_handle_accuracy * 100:.1f}%")
    print()
    print("Decision breakdown:")
    print(results["decision"].value_counts().to_string())
    print()
    print("Escalation reasons:")
    print(
        results.loc[results["decision"] == "HUMAN","decision_reason"].value_counts().to_string())
    print()
    print(f"Saved to: {output_path}")
if __name__ == "__main__":
    main()