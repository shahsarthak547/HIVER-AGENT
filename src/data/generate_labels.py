from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")
OUTPUT_PATH = Path("data/processed/apple/label_candidates.csv")

MODEL_NAME = "all-MiniLM-L6-v2"

INTENTS = {
    "battery_charging": "battery life, battery drain, charging, or device not charging",
    "keyboard_text_input": "keyboard, typing, autocorrect, letters, symbols, or text input",
    "connectivity": "Wi-Fi, Bluetooth, cellular network, internet, or connectivity",
    "apple_id_account": "Apple ID, password, login, account access, activation lock, or disabled account",
    "messaging": "iMessage, SMS, text messages, sending or receiving messages",
    "apple_app_service": "Apple Music, iTunes, App Store, iCloud, FaceTime, Safari, Apple Pay, or another Apple service",
    "photos_media": "photos, pictures, videos, camera, missing photos, or media",
    "purchase_payment_order": "purchases, payments, subscriptions, refunds, orders, delivery, or AppleCare",
    "software_update": "installing, downloading, completing, or failing to install an iOS or software update",
    "device_performance": "freezing, crashing, restarting, lagging, overheating, or a device not working"
}

SAMPLE_SIZE = 30000
RANDOM_STATE = 42

def main():
    df = pd.read_csv(INPUT_PATH)

    df = df.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    texts = (
        sample["customer_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    model = SentenceTransformer(MODEL_NAME)

    text_embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    intent_names = list(INTENTS.keys())
    intent_texts = list(INTENTS.values())

    intent_embeddings = model.encode(
        intent_texts,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        text_embeddings,
        intent_embeddings
    )

    best_indices = np.argmax(scores, axis=1)
    best_scores = scores[
        np.arange(len(scores)),
        best_indices
    ]

    sample["candidate_intent"] = [
        intent_names[i]
        for i in best_indices
    ]

    sample["candidate_score"] = best_scores

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sample.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Generated candidates: {len(sample):,}")
    print()
    print("Candidate distribution:")
    print(
        sample["candidate_intent"]
        .value_counts()
        .to_string()
    )
    print()
    print("Confidence statistics:")
    print(
        sample["candidate_score"].describe()
    )
    print()
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()