import requests
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/fraud_baseline.csv"
API_URL = "http://localhost:8000/predict"

FEATURES = [
    "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
    "C1", "C2", "C3", "C4", "C5",
    "C6", "C7", "C8", "C9", "C10",
    "D1", "D2", "D3", "D4",
    "M1", "M2", "M3", "M4", "M5",
    "M6", "M7", "M8", "M9"
]

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

fraud_indices = y_test[y_test == 1].index

idx = fraud_indices[0]

row = X_test.loc[idx]

def clean(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value

payload = {
    feature: clean(row[feature])
    for feature in FEATURES
}

print("=" * 60)
print("REAL FRAUD HOLDOUT TRANSACTION")
print("=" * 60)
print("Index:", idx)
print("Actual isFraud:", int(y_test.loc[idx]))
print("TransactionAmt:", row["TransactionAmt"])
print("ProductCD:", row["ProductCD"])

response = requests.post(
    API_URL,
    json=payload,
    timeout=30,
)

print("\nHTTP Status:", response.status_code)

if response.status_code != 200:
    print(response.text)
    raise SystemExit(1)

result = response.json()

print("\nMODEL RESULT")
print("=" * 60)
print("Fraud Probability:", result["fraud_probability"])
print("Prediction:", result["prediction"])
print("Threshold:", result.get("threshold", 0.78))

print("\nActual Label:       1 (FRAUD)")
print(
    "Model Classification:",
    "FRAUD" if int(result["prediction"]) == 1 else "LEGITIMATE"
)

print("=" * 60)
