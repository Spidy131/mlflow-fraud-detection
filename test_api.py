import pandas as pd
import requests
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ==================================================
# Configuration
# ==================================================

DATA_PATH = "data/processed/fraud_baseline.csv"
API_URL = "http://localhost:8000/predict"

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


# ==================================================
# Load dataset
# ==================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Random sample containing both fraud and legitimate
sample = df.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
).copy()

y_true = sample["isFraud"].astype(int).tolist()

X = sample.drop(columns=["isFraud"])


# ==================================================
# Send transactions to API
# ==================================================

print(f"\nSending {SAMPLE_SIZE} transactions to API...")
print(API_URL)

predictions = []
probabilities = []

for i, row in X.iterrows():

    transaction = row.to_dict()

    # Convert NaN to None so JSON accepts it
    transaction = {
        key: None if pd.isna(value) else value
        for key, value in transaction.items()
    }

    response = requests.post(
        API_URL,
        json=transaction,
        timeout=30
    )

    if response.status_code != 200:
        print("\nAPI ERROR")
        print("Status:", response.status_code)
        print("Response:", response.text)
        raise RuntimeError("API prediction failed")

    result = response.json()

    predictions.append(result["prediction"])
    probabilities.append(result["fraud_probability"])

    if (len(predictions) % 100) == 0:
        print(f"Processed: {len(predictions)}/{SAMPLE_SIZE}")


# ==================================================
# Metrics
# ==================================================

y_pred = predictions

cm = confusion_matrix(y_true, y_pred)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ==================================================
# Results
# ==================================================

print("\n========================================")
print("DOCKER API VALIDATION RESULTS")
print("========================================")

print(f"Transactions tested : {SAMPLE_SIZE}")

print("\nConfusion Matrix:")
print(cm)

print("\nMetrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n========================================")
print("CONFUSION MATRIX DETAILS")
print("========================================")

tn, fp, fn, tp = cm.ravel()

print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives  (TP): {tp}")

print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

print(f"Actual Legitimate : {tn + fp}")
print(f"Actual Fraud      : {fn + tp}")

print(f"Predicted Legitimate : {tn + fn}")
print(f"Predicted Fraud      : {fp + tp}")

print("\nValidation completed successfully.")