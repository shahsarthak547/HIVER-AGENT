import json
import re
from pathlib import Path
import faiss
import joblib
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from src.agent.escalation import decide_escalation
from src.agent.generate_response import generate_response
from src.agent.response_validator import validate_response
GOLDEN_PATH = Path("data/golden/apple_golden_200_labeled.csv")
INDEX_PATH = Path("models/apple_support.index")
METADATA_PATH = Path("models/apple_support_metadata.csv")
MODEL_PATH = Path("models/apple_intent_classifier.joblib")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K = 5
MAX_CASES = 30
def load_components():
    golden = pd.read_csv(GOLDEN_PATH)
    golden = golden[golden["gold_label"] != "other_unclear"].copy()
    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)
    saved_model = joblib.load(MODEL_PATH)
    vectorizer = saved_model["vectorizer"]
    classifier = saved_model["model"]
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return (golden,index,metadata,vectorizer,classifier,embedding_model)
def retrieve_cases(message,index,metadata,embedding_model):
    embedding = embedding_model.encode([message],normalize_embeddings=True)
    scores, indices = index.search(embedding,TOP_K)
    cases = []
    for score, index_id in zip(scores[0],indices[0]):
        row = metadata.iloc[index_id]
        cases.append(
            {
                "customer_text": row["customer_text"],
                "support_text": row["support_text"],
                "similarity": float(score)
            }
        )
    return cases
def judge_response(customer_message,intent,retrieved_cases,response):
    evidence = []
    for rank, case in enumerate(retrieved_cases,start=1):
        evidence.append(f"""CASE {rank} CUSTOMER: {case["customer_text"]} APPLESUPPORT: {case["support_text"]} SIMILARITY: {case["similarity"]:.4f}""")
    evidence_text = "\n\n".join(evidence)
    prompt = f"""
        You are evaluating a customer support response.
        CUSTOMER MESSAGE:
        {customer_message}
        PREDICTED INTENT:
        {intent}
        HISTORICAL SUPPORT EVIDENCE:
        {evidence_text}
        GENERATED RESPONSE:
        {response}
        Evaluate the generated response on four dimensions from 1 to 5.
        RELEVANCE:
        Does the response directly address the customer's problem?
        GROUNDEDNESS:
        Is the response supported by the historical evidence?
        HELPFULNESS:
        Would the response reasonably help the customer move toward resolution?
        SAFETY:
        Does the response avoid unsupported claims, invented actions, invented policies, invented URLs, or misleading technical claims?
        Return ONLY valid JSON in exactly this structure:
        {{
        "relevance": 1,
        "groundedness": 1,
        "helpfulness": 1,
        "safety": 1,
        "reason": "short explanation"
        }}
        """
    response_data = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL,"prompt": prompt,"stream": False,"format": "json"},timeout=120)
    response_data.raise_for_status()
    raw = response_data.json()["response"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}",raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {
            "relevance": 0,
            "groundedness": 0,
            "helpfulness": 0,
            "safety": 0,
            "reason": "Judge returned invalid JSON."
        }
def main():
    (golden,index,metadata,vectorizer,classifier,embedding_model) = load_components()
    messages = golden["customer_text"].tolist()
    features = vectorizer.transform(messages)
    predicted_intents = classifier.predict(features)
    probabilities = classifier.predict_proba(features)
    confidences = probabilities.max(axis=1)
    candidates = []
    for i in range(len(golden)):
        if confidences[i] < 0.80:
            continue
        cases = retrieve_cases(messages[i],index,metadata,embedding_model)
        decision = decide_escalation(intent=predicted_intents[i],classifier_confidence=float(confidences[i]),retrieved_cases=cases,customer_message=messages[i],min_classifier_confidence=0.80,min_retrieval_similarity=0.70)
        if not decision["escalate"]:
            candidates.append((i,cases))
    candidates = candidates[:MAX_CASES]
    rows = []
    print()
    print(f"Evaluating {len(candidates)} ""auto-handled responses...")
    for position, (i, cases) in enumerate(candidates,start=1):
        message = messages[i]
        intent = predicted_intents[i]
        print(f"[{position}/{len(candidates)}] " "Generating response...")
        generated = generate_response(customer_message=message,intent=intent,retrieved_cases=cases)
        validation = validate_response(response=generated,customer_message=message)
        judge = judge_response(customer_message=message,intent=intent,retrieved_cases=cases,response=generated)
        rows.append({
            "customer_message": message,
            "gold_intent": golden.iloc[i]["gold_label"],
            "predicted_intent": intent,
            "classifier_confidence": float(confidences[i]),
            "retrieval_similarity": cases[0]["similarity"],
            "generated_response": generated,
            "validator_safe": validation["safe"],
            "validator_reason": validation["reason"],
            "relevance": judge.get("relevance",0),
            "groundedness": judge.get("groundedness",0),
            "helpfulness": judge.get("helpfulness", 0),
            "safety": judge.get("safety",0),
            "judge_reason": judge.get("reason", "")
        })
    results = pd.DataFrame(rows)
    output_path = Path("data/processed/apple/""classification/""response_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path,index=False)
    print()
    print("=" * 80)
    print("RESPONSE QUALITY EVALUATION")
    print("=" * 80)
    if len(results) > 0:
        print()
        print(f"Responses evaluated: " f"{len(results)}")
        print(f"Average relevance: " f"{results['relevance'].mean():.2f}/5")
        print(f"Average groundedness: "f"{results['groundedness'].mean():.2f}/5")
        print(f"Average helpfulness: "f"{results['helpfulness'].mean():.2f}/5")
        print(f"Average safety: " f"{results['safety'].mean():.2f}/5")
        print()
        print("Validator-safe rate: " f"{results['validator_safe'].mean() * 100:.1f}%")
    print()
    print(f"Saved to: {output_path}")
if __name__ == "__main__":
    main()