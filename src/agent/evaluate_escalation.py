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
    golden = golden[golden["gold_label"] != "other_unclear"].copy()

    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)

    saved_model = joblib.load(MODEL_PATH)

    vectorizer = saved_model["vectorizer"]
    classifier = saved_model["model"]

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return (
        golden,
        index,
        metadata,
        vectorizer,
        classifier,
        embedding_model
    )


def prepare_cases(
    golden,
    index,
    metadata,
    vectorizer,
    classifier,
    embedding_model
):
    prepared = []

    messages = golden["customer_text"].tolist()

    features = vectorizer.transform(messages)

    predicted_intents = classifier.predict(features)
    probabilities = classifier.predict_proba(features)

    confidences = probabilities.max(axis=1)

    embeddings = embedding_model.encode(
        messages,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    )

    scores, indices = index.search(
        embeddings,
        TOP_K
    )

    for i in range(len(golden)):
        retrieved_cases = []

        for score, index_id in zip(
            scores[i],
            indices[i]
        ):
            row = metadata.iloc[index_id]

            retrieved_cases.append(
                {
                    "customer_text": row["customer_text"],
                    "support_text": row["support_text"],
                    "similarity": float(score)
                }
            )

        prepared.append(
            {
                "gold_label": golden.iloc[i]["gold_label"],
                "predicted_intent": predicted_intents[i],
                "confidence": float(confidences[i]),
                "retrieved_cases": retrieved_cases
            }
        )

    return prepared


def evaluate_thresholds(
    prepared_cases,
    classifier_threshold,
    retrieval_threshold
):
    auto_handle = 0
    correct_auto_handle = 0

    for case in prepared_cases:
        decision = decide_escalation(
            intent=case["predicted_intent"],
            classifier_confidence=case["confidence"],
            retrieved_cases=case["retrieved_cases"],
            min_classifier_confidence=classifier_threshold,
            min_retrieval_similarity=retrieval_threshold
        )

        if not decision["escalate"]:
            auto_handle += 1

            if case["predicted_intent"] == case["gold_label"]:
                correct_auto_handle += 1

    total = len(prepared_cases)

    auto_handle_rate = (
        auto_handle / total
    )

    auto_handle_accuracy = (
        correct_auto_handle / auto_handle
        if auto_handle > 0
        else 0
    )

    return {
        "classifier_threshold": classifier_threshold,
        "retrieval_threshold": retrieval_threshold,
        "auto_handle_count": auto_handle,
        "auto_handle_rate": auto_handle_rate,
        "correct_auto_handle": correct_auto_handle,
        "auto_handle_accuracy": auto_handle_accuracy
    }


def main():
    (
        golden,
        index,
        metadata,
        vectorizer,
        classifier,
        embedding_model
    ) = load_components()

    print()
    print("Preparing golden-set predictions and retrievals...")

    prepared_cases = prepare_cases(
        golden,
        index,
        metadata,
        vectorizer,
        classifier,
        embedding_model
    )

    classifier_thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    retrieval_thresholds = [
        0.50,
        0.60,
        0.65,
        0.70,
        0.75
    ]

    rows = []

    for classifier_threshold in classifier_thresholds:
        for retrieval_threshold in retrieval_thresholds:
            result = evaluate_thresholds(
                prepared_cases,
                classifier_threshold,
                retrieval_threshold
            )

            rows.append(result)

    results = pd.DataFrame(rows)

    output_path = Path(
        "data/processed/apple/classification/"
        "escalation_threshold_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        output_path,
        index=False
    )

    display = results.copy()

    display["auto_handle_rate"] = (
        display["auto_handle_rate"] * 100
    ).round(1)

    display["auto_handle_accuracy"] = (
        display["auto_handle_accuracy"] * 100
    ).round(1)

    display = display.sort_values(
        ["auto_handle_accuracy", "auto_handle_rate"],
        ascending=[False, False]
    )

    print()
    print("=" * 90)
    print("ESCALATION THRESHOLD EVALUATION")
    print("=" * 90)
    print()
    print(display.to_string(index=False))
    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()