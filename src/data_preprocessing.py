import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Load transaction data
# --------------------------------------------------

print("Loading transaction data...")

transaction = pd.read_csv(
    RAW_DIR / "train_transaction.csv"
)

print(f"Transaction shape: {transaction.shape}")


# --------------------------------------------------
# 2. Load identity data
# --------------------------------------------------

print("Loading identity data...")

identity = pd.read_csv(
    RAW_DIR / "train_identity.csv"
)

print(f"Identity shape: {identity.shape}")


# --------------------------------------------------
# 3. Merge
# --------------------------------------------------

print("Merging transaction + identity data...")

df = transaction.merge(
    identity,
    on="TransactionID",
    how="left"
)

print(f"Merged shape: {df.shape}")


# --------------------------------------------------
# 4. Select useful baseline features
# --------------------------------------------------

features = [
    "TransactionAmt",
    "ProductCD",

    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    "addr1",
    "addr2",

    "dist1",

    "P_emaildomain",
    "R_emaildomain",

    "DeviceType",
    "DeviceInfo",

    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",

    "D1",
    "D2",
    "D3",
    "D4",

    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9"
]

target = "isFraud"


# Keep only columns that exist
features = [
    column
    for column in features
    if column in df.columns
]

df = df[features + [target]]


# --------------------------------------------------
# 5. Missing values
# --------------------------------------------------

print("Handling missing values...")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

numeric_columns = df.select_dtypes(
    exclude=["object"]
).columns


for column in categorical_columns:
    df[column] = df[column].fillna("missing")


for column in numeric_columns:

    if column != target:

        df[column] = df[column].fillna(
            df[column].median()
        )


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

output_file = (
    PROCESSED_DIR / "fraud_baseline.csv"
)

df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

print("\n====================================")
print("PREPROCESSING COMPLETE")
print("====================================")

print(f"Rows       : {len(df)}")
print(f"Features   : {len(features)}")
print(f"Fraud cases: {df[target].sum()}")
print(f"Fraud rate : {df[target].mean():.4%}")

print(f"\nSaved to:")
print(output_file)