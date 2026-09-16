from pathlib import Path

import faiss
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.agent.escalation import decide_escalation
from src.agent.generate_response import generate_response
from src.agent.response_validator import validate_response


INDEX_PATH = Path("models/apple_support.index")
METADATA_PATH = Path("models/apple_support_metadata.csv")
MODEL_PATH = Path("models/apple_intent_classifier.joblib")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


def load_components():
    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)

    saved_model = joblib.load(MODEL_PATH)

    vectorizer = saved_model["vectorizer"]
    classifier = saved_model["model"]

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return (
        index,
        metadata,
        vectorizer,
        classifier,
        embedding_model
    )


def classify_message(
    message,
    vectorizer,
    classifier
):
    features = vectorizer.transform([message])

    intent = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]

    confidence = float(
        probabilities.max()
    )

    return intent, confidence


def retrieve_cases(
    message,
    index,
    metadata,
    embedding_model
):
    embedding = embedding_model.encode(
        [message],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        embedding,
        TOP_K
    )

    cases = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):
        row = metadata.iloc[index_id]

        cases.append(
            {
                "customer_text": row["customer_text"],
                "support_text": row["support_text"],
                "similarity": float(score)
            }
        )

    return cases


def main():
    (
        index,
        metadata,
        vectorizer,
        classifier,
        embedding_model
    ) = load_components()

    message = input(
        "Customer message: "
    ).strip()

    intent, confidence = classify_message(
        message,
        vectorizer,
        classifier
    )

    cases = retrieve_cases(
        message,
        index,
        metadata,
        embedding_model
    )

    escalation = decide_escalation(
        intent=intent,
        classifier_confidence=confidence,
        retrieved_cases=cases,
        customer_message=message
    )

    result = None

    if not escalation["escalate"]:
        result = generate_response(
            customer_message=message,
            intent=intent,
            retrieved_cases=cases
        )

        validation = validate_response(
            response=result,
            customer_message=message
        )

        if not validation["safe"]:
            escalation = {
                "escalate": True,
                "reason": validation["reason"]
            }

            result = None

    print()
    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 80)

    print()
    print("CUSTOMER:")
    print(message)

    print()
    print("INTENT:")
    print(intent)

    print()
    print(
        f"CLASSIFIER CONFIDENCE: "
        f"{confidence:.4f}"
    )

    print()
    print("DECISION:")
    print(
        "HUMAN ESCALATION"
        if escalation["escalate"]
        else "AUTO-HANDLE"
    )

    print()
    print("DECISION REASON:")
    print(escalation["reason"])

    if result is not None:
        print()
        print("RESPONSE:")
        print(result)
    else:
        print()
        print("RESPONSE:")
        print(
            "Response generation skipped because "
            "the case requires human review."
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()