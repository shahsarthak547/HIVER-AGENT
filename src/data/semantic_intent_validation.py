from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_PATH = Path("data/processed/apple/intent_discovery/clustered_messages.csv")
EMBEDDINGS_PATH = Path("data/processed/apple/intent_discovery/embeddings.npy")
OUTPUT_PATH = Path("data/processed/apple/intent_discovery/semantic_intent_examples.csv")

MODEL_NAME = "all-MiniLM-L6-v2"
EXAMPLES_PER_INTENT = 20

INTENT_DESCRIPTIONS = {
    "battery_charging": "customer has a problem with battery life, battery drain, charging, or the device not charging",
    "keyboard_text_input": "customer has a problem with keyboard, typing, autocorrect, text input, letters, symbols, or characters",
    "connectivity": "customer has a problem with Wi-Fi, Bluetooth, cellular network, internet, or connectivity",
    "apple_id_account": "customer has a problem with Apple ID, account access, login, password, activation lock, or account being disabled",
    "messaging": "customer has a problem with iMessage, SMS, text messages, sending messages, receiving messages, or messaging",
    "apple_services_apps": "customer has a problem with an Apple app or service such as Apple Music, iTunes, App Store, iCloud, FaceTime, Safari, or Apple Pay",
    "photos_media": "customer has a problem with photos, pictures, videos, camera, missing photos, or saving media",
    "purchases_orders": "customer has a problem with purchases, orders, delivery, refunds, subscriptions, AppleCare, or payment for a purchase",
    "software_update": "customer has a problem specifically with installing, downloading, completing, or rolling back an iOS or software update",
    "device_troubleshooting": "customer has a general device problem such as freezing, crashing, restarting, being slow, or the device not working"
}


def load_data():
    df = pd.read_csv(INPUT_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Data rows ({len(df)}) do not match embeddings ({len(embeddings)})"
        )

    return df.reset_index(drop=True), embeddings


def create_intent_embeddings():
    model = SentenceTransformer(MODEL_NAME)

    descriptions = list(INTENT_DESCRIPTIONS.values())

    embeddings = model.encode(
        descriptions,
        normalize_embeddings=True
    )

    return np.asarray(embeddings)


def retrieve_examples(
    df,
    message_embeddings,
    intent_embeddings
):
    results = []

    intent_names = list(INTENT_DESCRIPTIONS.keys())

    similarity_matrix = cosine_similarity(
        message_embeddings,
        intent_embeddings
    )

    for intent_index, intent_name in enumerate(intent_names):
        similarities = similarity_matrix[:, intent_index]

        top_indices = np.argsort(
            similarities
        )[-EXAMPLES_PER_INTENT:][::-1]

        for rank, index in enumerate(top_indices, start=1):
            results.append({
                "candidate_intent": intent_name,
                "rank": rank,
                "similarity": similarities[index],
                "customer_tweet_id": df.iloc[index]["customer_tweet_id"],
                "customer_text": df.iloc[index]["customer_text"],
                "support_text": df.iloc[index]["support_text"]
            })

    return pd.DataFrame(results)


def main():
    print("Loading existing data and embeddings...")

    df, message_embeddings = load_data()

    print(f"Messages: {len(df):,}")
    print(f"Embedding dimensions: {message_embeddings.shape[1]}")

    print()
    print("Creating candidate intent embeddings...")

    intent_embeddings = create_intent_embeddings()

    results = retrieve_examples(
        df,
        message_embeddings,
        intent_embeddings
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Representative examples:")

    for intent in INTENT_DESCRIPTIONS:
        print()
        print("=" * 100)
        print(intent.upper())

        examples = results[
            results["candidate_intent"] == intent
        ]

        for _, row in examples.iterrows():
            print(
                f"{row['rank']}. "
                f"[{row['similarity']:.3f}] "
                f"{row['customer_text']}"
            )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
