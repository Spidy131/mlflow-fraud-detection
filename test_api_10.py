import pandas as pd
import requests
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

CSV_PATH = r"data/raw/train_transaction.csv"
API_URL = "http://localhost:8000/predict"
THRESHOLD = 0.78

# ---------------------------------------------------------
# Load real IEEE-CIS transactions
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

df = pd.read_csv(
    CSV_PATH,
    nrows=500000
)

# ---------------------------------------------------------
# Select 50 fraud + 50 legitimate transactions
# ---------------------------------------------------------

fraud = df[df["isFraud"] == 1].head(5)
legit = df[df["isFraud"] == 0].head(5)

test_df = pd.concat([fraud, legit]).reset_index(drop=True)

print("=" * 70)
print("IEEE-CIS API DEPLOYMENT TEST")
print("=" * 70)
print(f"Transactions tested : {len(test_df)}")
print(f"Fraud               : {(test_df.isFraud == 1).sum()}")
print(f"Legitimate          : {(test_df.isFraud == 0).sum()}")
print("=" * 70)

results = []

# ---------------------------------------------------------
# Send transactions to FastAPI
# ---------------------------------------------------------

for i, row in test_df.iterrows():

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

    # Identity fields are not in train_transaction.csv
    payload["DeviceType"] = None
    payload["DeviceInfo"] = None

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        prediction = response.json()

        results.append({
            "TransactionID": row["TransactionID"],
            "actual": int(row["isFraud"]),
            "prediction": int(prediction["prediction"]),
            "probability": float(prediction["fraud_probability"]),
        })

        print(
            f"{i+1:3d}/100 | "
            f"ID={int(row['TransactionID'])} | "
            f"Actual={int(row['isFraud'])} | "
            f"Prob={prediction['fraud_probability']:.4f} | "
            f"Pred={prediction['prediction']}"
        )

    except Exception as e:
        print(
            f"{i+1:3d}/100 | "
            f"ERROR | "
            f"ID={int(row['TransactionID'])} | "
            f"{e}"
        )

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

if len(results_df) == 0:
    raise RuntimeError("No successful API predictions.")

y_true = results_df["actual"]
y_pred = results_df["prediction"]
y_prob = results_df["probability"]

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Successful API calls : {len(results_df)}")

print(
    f"Accuracy             : "
    f"{accuracy_score(y_true, y_pred):.4f}"
)

print(
    f"Precision            : "
    f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
)

print(
    f"Recall               : "
    f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
)

print(
    f"F1                   : "
    f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
)

print(
    f"ROC-AUC              : "
    f"{roc_auc_score(y_true, y_prob):.4f}"
)

print()
print("Confusion Matrix")
print("----------------")
print(confusion_matrix(y_true, y_pred))

print()
print("Prediction counts")
print("-----------------")
print(results_df["prediction"].value_counts().sort_index())

print()
print("Actual counts")
print("-------------")
print(results_df["actual"].value_counts().sort_index())

# ---------------------------------------------------------
# Fraud probability statistics
# ---------------------------------------------------------

print()
print("Probability statistics")
print("----------------------")

print(
    results_df.groupby("actual")["probability"]
    .agg(["count", "min", "mean", "max"])
)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df.to_csv(
    "api_test_results_100.csv",
    index=False
)

print()
print("Saved: api_test_results_100.csv")
print("=" * 70)
