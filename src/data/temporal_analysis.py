from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/apple/apple_support_pairs.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    df["customer_created_at"] = pd.to_datetime(
        df["customer_created_at"],
        format="mixed",
        errors="coerce"
    )

    df = df.dropna(subset=["customer_created_at"])

    print(f"Total pairs: {len(df):,}")
    print()

    print("Date range:")
    print(df["customer_created_at"].min())
    print(df["customer_created_at"].max())
    print()

    monthly = (
        df.assign(
            month=df["customer_created_at"].dt.to_period("M")
        )
        .groupby("month")
        .size()
        .reset_index(name="messages")
    )

    print("Monthly message counts:")
    print(monthly.to_string(index=False))

    print()
    print("Yearly message counts:")

    yearly = (
        df.assign(
            year=df["customer_created_at"].dt.year
        )
        .groupby("year")
        .size()
        .reset_index(name="messages")
    )

    print(yearly.to_string(index=False))

if __name__ == "__main__":
    main()