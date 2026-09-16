from pathlib import Path
import pandas as pd
from src.data.text_cleaning import normalize_text
INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")
OUTPUT_PATH = Path("data/processed/apple/intent_validation_examples.csv")

EXAMPLES_PER_INTENT = 30
RANDOM_STATE = 42

INTENT_PATTERNS = {
    "battery_charging": [
        "battery",
        "charging",
        "charge",
        "drain",
        "battery life",
        "dies",
        "charging port"
    ],
    "keyboard_text_input": [
        "keyboard",
        "type",
        "typing",
        "autocorrect",
        "letter",
        "question mark",
        "symbol",
        "text input"
    ],
    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "connection",
        "connect",
        "network",
        "internet"
    ],
    "apple_id_account": [
        "apple id",
        "password",
        "sign in",
        "sign into",
        "login",
        "locked out",
        "disabled",
        "activation lock"
    ],
    "messaging": [
        "imessage",
        "i message",
        "message",
        "messages",
        "text message",
        "sms"
    ],
    "apple_services_apps": [
        "apple music",
        "itunes",
        "app store",
        "icloud",
        "facetime",
        "safari",
        "apple pay"
    ],
    "photos_media": [
        "photos",
        "pictures",
        "videos",
        "camera",
        "photo"
    ],
    "purchases_orders": [
        "order",
        "ordered",
        "delivery",
        "delivered",
        "purchase",
        "purchased",
        "refund",
        "applecare",
        "subscription"
    ],
    "software_update": [
        "ios update",
        "ios 11",
        "ios 10",
        "ios 12",
        "software update",
        "updated",
        "update",
        "upgraded"
    ],
    "device_troubleshooting": [
        "iphone",
        "ipad",
        "macbook",
        "phone",
        "device",
        "freezing",
        "freeze",
        "crash",
        "slow",
        "restart",
        "reboot",
        "not working"
    ]
}

def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["normalized_text"] = (df["customer_text"].fillna("").astype(str).map(normalize_text))
    return df

def find_examples(df, patterns):
    mask = pd.Series(False, index=df.index)
    for pattern in patterns:
        mask |= df["normalized_text"].str.contains(pattern,case=False,regex=False,na=False)
    return df[mask].copy()

def select_examples(df, patterns):
    matches = find_examples(df, patterns)
    if len(matches) <= EXAMPLES_PER_INTENT:
        return matches
    return matches.sample(n=EXAMPLES_PER_INTENT,random_state=RANDOM_STATE)

def main():
    df = load_data()
    print(f"Total customer messages: {len(df):,}")
    print()
    all_examples = []
    for intent, patterns in INTENT_PATTERNS.items():
        examples = select_examples(df,patterns).copy()
        examples["candidate_intent"] = intent
        all_examples.append(examples)
        print(f"{intent}: "f"{len(find_examples(df, patterns)):,} matches")
    result = pd.concat(all_examples,ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    result[
        [
            "candidate_intent",
            "customer_tweet_id",
            "customer_text",
            "support_text"
        ]
    ].to_csv(
        OUTPUT_PATH,
        index=False
    )
    print()
    print(f"Validation examples saved to: "f"{OUTPUT_PATH}")

if __name__ == "__main__":
    main()