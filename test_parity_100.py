import pandas as pd
import numpy as np
import xgboost as xgb

from app.preprocessing import create_features


CSV_PATH = r"data/raw/train_transaction.csv"
MODEL_PATH = r"mlruns/1/models/m-18ad4f3607054746bc16a483bae1467c/artifacts/model.ubj"


# ---------------------------------------------------------
# Load transactions used by API test
# ---------------------------------------------------------

results = pd.read_csv("api_test_results_100.csv")

transaction_ids = results["TransactionID"].astype(int).tolist()

df = pd.read_csv(
    CSV_PATH,
    nrows=500000
)

df = df[df["TransactionID"].isin(transaction_ids)].copy()

print("=" * 70)
print("LOCAL vs DOCKER MODEL PARITY TEST")
print("=" * 70)
print(f"Transactions found : {len(df)}")


# ---------------------------------------------------------
# Load SAME XGBoost model artifact
# ---------------------------------------------------------

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

print(f"Model features      : {booster.num_features()}")


# ---------------------------------------------------------
# Load SAME training categories from model
# ---------------------------------------------------------

xgb_categories = booster.get_categories(export_to_arrow=True)

arrow_categories = xgb_categories.to_arrow()

training_categories = {}

for column_name, category_array in arrow_categories:

    if category_array is not None:
        training_categories[column_name] = category_array.to_pylist()

print(
    f"Categorical columns : {len(training_categories)}"
)


# ---------------------------------------------------------
# Features accepted by API
# ---------------------------------------------------------

FEATURES = [
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
    "M9",
]


# ---------------------------------------------------------
# Calculate local probabilities
# ---------------------------------------------------------

local_results = []

for _, row in df.iterrows():

    payload = {}

    for col in FEATURES:

        value = row[col]

        if pd.isna(value):
            payload[col] = None

        elif isinstance(value, np.integer):
            payload[col] = int(value)

        elif isinstance(value, np.floating):
            payload[col] = float(value)

        else:
            payload[col] = value

    # Same as API test
    payload["DeviceType"] = None
    payload["DeviceInfo"] = None

    # SAME production preprocessing
    features = create_features(
        payload,
        training_categories=training_categories
    )

    # SAME native XGBoost prediction
    probability = float(
        booster.predict(
            xgb.DMatrix(
                features,
                enable_categorical=True
            )
        )[0]
    )

    local_results.append({
        "TransactionID": int(row["TransactionID"]),
        "actual": int(row["isFraud"]),
        "local_probability": probability,
    })


local_df = pd.DataFrame(local_results)


# ---------------------------------------------------------
# Compare API vs local
# ---------------------------------------------------------

comparison = results.merge(
    local_df,
    on=["TransactionID", "actual"],
    how="inner"
)

comparison["difference"] = (
    comparison["probability"]
    - comparison["local_probability"]
).abs()


print()
print("=" * 70)
print("INDIVIDUAL COMPARISON")
print("=" * 70)

print(
    comparison[
        [
            "TransactionID",
            "actual",
            "probability",
            "local_probability",
            "difference",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 70)
print("PARITY SUMMARY")
print("=" * 70)

print(f"Transactions compared : {len(comparison)}")

print(
    f"Maximum difference    : "
    f"{comparison['difference'].max():.12f}"
)

print(
    f"Mean difference       : "
    f"{comparison['difference'].mean():.12f}"
)

print(
    f"Median difference     : "
    f"{comparison['difference'].median():.12f}"
)


# ---------------------------------------------------------
# PASS / FAIL
# ---------------------------------------------------------

TOLERANCE = 1e-6

if comparison["difference"].max() <= TOLERANCE:

    print()
    print("PASS: Docker/API predictions match local XGBoost.")

else:

    print()
    print("FAIL: Docker/API predictions differ from local XGBoost.")

    print()
    print("Largest differences:")

    print(
        comparison.nlargest(
            10,
            "difference"
        )[
            [
                "TransactionID",
                "probability",
                "local_probability",
                "difference",
            ]
        ].to_string(index=False)
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

comparison.to_csv(
    "parity_results_100.csv",
    index=False
)

print()
print("Saved: parity_results_100.csv")
print("=" * 70)