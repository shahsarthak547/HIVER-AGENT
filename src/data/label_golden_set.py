from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/golden/apple_golden_200.csv")
OUTPUT_PATH = Path("data/golden/apple_golden_200_labeled.csv")

INTENTS = [
    "battery_charging",
    "keyboard_text_input",
    "connectivity",
    "apple_id_account",
    "messaging",
    "apple_app_service",
    "photos_media",
    "purchase_payment_order",
    "software_update",
    "device_performance",
    "other_unclear",
]

def main():
    df = pd.read_csv(INPUT_PATH)

    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)

    if "human_verified" not in df.columns:
        df["human_verified"] = False

    for index, row in df.iterrows():
        if bool(row["human_verified"]):
            continue

        print()
        print("=" * 100)
        print(f"EXAMPLE {index + 1}/{len(df)}")
        print()
        print("CUSTOMER:")
        print(row["customer_text"])
        print()
        print("APPLESUPPORT:")
        print(row["support_text"])
        print()
        print(f"CANDIDATE: {row['candidate_intent']}")
        print(f"SCORE: {row['candidate_score']:.3f}")
        print()

        for number, intent in enumerate(INTENTS, start=1):
            print(f"{number}. {intent}")

        print()
        print("Enter = accept candidate")
        print("q = save and quit")
        print()

        choice = input("Label: ").strip().lower()

        if choice == "q":
            df.to_csv(OUTPUT_PATH, index=False)
            print(f"Saved progress to {OUTPUT_PATH}")
            return

        if choice == "":
            label = row["candidate_intent"]
        else:
            try:
                number = int(choice)

                if number < 1 or number > len(INTENTS):
                    print("Invalid choice")
                    continue

                label = INTENTS[number - 1]

            except ValueError:
                print("Invalid choice")
                continue

        df.at[index, "gold_label"] = label
        df.at[index, "human_verified"] = True

        df.to_csv(
            OUTPUT_PATH,
            index=False
        )

    print()
    print("Labeling complete.")
    print(f"Saved to {OUTPUT_PATH}")

    print()
    print("Final label distribution:")
    print(
        df["gold_label"]
        .value_counts()
        .to_string()
    )

if __name__ == "__main__":
    main()