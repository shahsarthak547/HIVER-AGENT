from pathlib import Path

import joblib

from src.agent.escalation import decide_escalation
from src.agent.generate_response import generate_response
from src.agent.response_validator import validate_response
from src.retrieval.search import load_retrieval, search


MODEL_PATH = Path("models/apple_intent_classifier.joblib")


def load_components():
    saved_model = joblib.load(MODEL_PATH)

    vectorizer = saved_model["vectorizer"]
    classifier = saved_model["model"]

    retrieval = load_retrieval()

    return vectorizer, classifier, retrieval


def classify_message(message, vectorizer, classifier):
    features = vectorizer.transform([message])

    intent = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]

    confidence = float(probabilities.max())

    return intent, confidence


def retrieve_cases(message, retrieval):
    results = search(message, retrieval)

    cases = []

    for _, row in results.iterrows():
        cases.append(
            {
                "customer_text": row["customer_text"],
                "support_text": row["support_text"],
                "similarity": float(row["similarity"])
            }
        )

    return cases


def main():
    vectorizer, classifier, retrieval = load_components()

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
        retrieval
    )

    print(
        f"TOP RETRIEVAL SIMILARITY: "
        f"{cases[0]['similarity']:.4f}"
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

    print()
    print("RESPONSE:")

    if result is not None:
        print(result)
    else:
        print(
            "Response generation skipped because "
            "the case requires human review."
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()