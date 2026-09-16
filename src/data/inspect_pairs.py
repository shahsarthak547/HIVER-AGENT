from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    print(f"Total pairs: {len(df):,}")
    print()

    for i, row in df.head(20).iterrows():
        print("=" * 100)
        print(f"PAIR {i + 1}")
        print()
        print("CUSTOMER:")
        print(row["customer_text"])
        print()
        print("APPLESUPPORT:")
        print(row["support_text"])
        print()


if __name__ == "__main__":
    main()