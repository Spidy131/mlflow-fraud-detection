import requests
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/fraud_baseline.csv"

API_URL = "http://localhost:8000/predict"

N_TEST = 1000


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# Separate features and target
# ============================================================

TARGET = "isFraud"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# Select 1000 transactions
# ============================================================

sample = df.sample(
    n=N_TEST,
    random_state=42
).reset_index(drop=True)


print(f"\nSending {N_TEST} transactions to API...")
print(API_URL)


# ============================================================
# Convert values for JSON
# ============================================================

def clean_value(value):

    if pd.isna(value):
        return "missing"

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


# ============================================================
# API prediction
# ============================================================

actual = []
predicted = []
probabilities = []

for i, row in sample.iterrows():

    transaction = {
        column: clean_value(row[column])
        for column in X.columns
    }

    try:

        response = requests.post(
            API_URL,
            json=transaction,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        actual.append(int(row[TARGET]))

        predicted.append(
            int(result["prediction"])
        )

        probabilities.append(
            float(result["fraud_probability"])
        )

    except Exception as e:

        print(
            f"\nERROR at transaction {i + 1}: {e}"
        )

        if "response" in locals():
            print(
                "Response:",
                response.text
            )

        raise

    if (i + 1) % 100 == 0:

        print(
            f"Processed: {i + 1}/{N_TEST}"
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    actual,
    predicted
)

precision = precision_score(
    actual,
    predicted,
    zero_division=0
)

recall = recall_score(
    actual,
    predicted,
    zero_division=0
)

f1 = f1_score(
    actual,
    predicted,
    zero_division=0
)

cm = confusion_matrix(
    actual,
    predicted
)


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 40)
print("DOCKER API VALIDATION RESULTS")
print("=" * 40)

print(
    f"Transactions tested : {N_TEST}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nMetrics:")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# Confusion matrix details
# ============================================================

if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    print("\n")
    print("=" * 40)
    print("CONFUSION MATRIX DETAILS")
    print("=" * 40)

    print(
        f"True Negatives  (TN): {tn}"
    )

    print(
        f"False Positives (FP): {fp}"
    )

    print(
        f"False Negatives (FN): {fn}"
    )

    print(
        f"True Positives  (TP): {tp}"
    )


# ============================================================
# Class distribution
# ============================================================

print("\n")
print("=" * 40)
print("CLASS DISTRIBUTION")
print("=" * 40)

print(
    f"Actual Legitimate : "
    f"{actual.count(0)}"
)

print(
    f"Actual Fraud      : "
    f"{actual.count(1)}"
)

print(
    f"Predicted Legitimate : "
    f"{predicted.count(0)}"
)

print(
    f"Predicted Fraud      : "
    f"{predicted.count(1)}"
)

print("\nValidation completed successfully.")