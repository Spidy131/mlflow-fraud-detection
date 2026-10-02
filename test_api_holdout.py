import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/fraud_baseline.csv"
API_URL = "http://localhost:8000/predict"

RANDOM_STATE = 42
SAMPLE_SIZE = 2000
THRESHOLD = 0.78

# Same base features expected by FastAPI
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
    "M9",
]


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """Convert pandas/numpy values into JSON-safe values."""

    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def row_to_payload(row):
    return {
        feature: clean_value(row[feature])
        for feature in FEATURES
    }


# ============================================================
# LOAD DATA
# ============================================================

print("Loading processed dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# ============================================================
# RECREATE EXACT TRAIN / TEST SPLIT
# ============================================================

print("\nRecreating original train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")

print("\nTest-set class distribution:")
print(y_test.value_counts())


# ============================================================
# RANDOM SAMPLE FROM TRUE HOLDOUT
# ============================================================

sample_size = min(SAMPLE_SIZE, len(X_test))

sample_indices = np.random.RandomState(
    RANDOM_STATE
).choice(
    X_test.index,
    size=sample_size,
    replace=False,
)

X_sample = X_test.loc[sample_indices]
y_sample = y_test.loc[sample_indices]

print("\n============================================================")
print("HOLDOUT API TEST")
print("============================================================")

print(f"Sample size: {len(X_sample)}")
print(f"Fraud     : {int(y_sample.sum())}")
print(f"Legitimate: {int((y_sample == 0).sum())}")


# ============================================================
# API TEST
# ============================================================

results = []

success = 0
failed = 0

for count, (idx, row) in enumerate(X_sample.iterrows(), start=1):

    payload = row_to_payload(row)

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            print(
                f"ERROR index={idx}, "
                f"status={response.status_code}, "
                f"response={response.text[:300]}"
            )
            failed += 1
            continue

        result = response.json()

        probability = float(
            result["fraud_probability"]
        )

        prediction = int(
            result["prediction"]
        )

        actual = int(
            y_sample.loc[idx]
        )

        results.append(
            {
                "index": idx,
                "actual": actual,
                "probability": probability,
                "prediction": prediction,
            }
        )

        success += 1

    except Exception as e:

        print(
            f"ERROR index={idx}: {e}"
        )

        failed += 1

    if count % 100 == 0:
        print(
            f"Processed {count}/{sample_size}"
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "api_holdout_results_2000.csv",
    index=False,
)


print("\n============================================================")
print("API EXECUTION")
print("============================================================")

print(f"Successful: {success}")
print(f"Failed    : {failed}")


if len(results_df) == 0:
    raise RuntimeError(
        "No successful API predictions."
    )


# ============================================================
# METRICS
# ============================================================

y_true = results_df["actual"].values
probabilities = results_df["probability"].values
predictions = results_df["prediction"].values


accuracy = accuracy_score(
    y_true,
    predictions,
)

precision = precision_score(
    y_true,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_true,
    probabilities,
)

pr_auc = average_precision_score(
    y_true,
    probabilities,
)

cm = confusion_matrix(
    y_true,
    predictions,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n============================================================")
print("PRODUCTION API HOLDOUT RESULTS")
print("============================================================")

print(f"Sample size : {len(results_df)}")
print(f"Threshold   : {THRESHOLD}")

print(f"\nAccuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")
print(f"PR-AUC      : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nPrediction distribution:")
print(results_df["prediction"].value_counts())

print("\nActual distribution:")
print(results_df["actual"].value_counts())


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\n============================================================")
print("PROBABILITY DISTRIBUTION")
print("============================================================")

print("\nLegitimate transactions:")
print(
    results_df.loc[
        results_df["actual"] == 0,
        "probability"
    ].describe()
)

print("\nFraud transactions:")
print(
    results_df.loc[
        results_df["actual"] == 1,
        "probability"
    ].describe()
)


print("\nResults saved to:")
print("api_holdout_results_2000.csv")