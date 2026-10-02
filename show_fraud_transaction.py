import pandas as pd
import json

DATA_PATH = "data/processed/fraud_baseline.csv"

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

row = df.loc[72645]

print("=" * 70)
print("REAL FRAUD TRANSACTION FOR DASHBOARD")
print("=" * 70)
print("Dataset index:", 72645)
print("Actual isFraud:", int(row["isFraud"]))
print()

for feature in FEATURES:
    value = row[feature]
    if pd.isna(value):
        value = ""
    print(f"{feature:20} = {value}")

print("=" * 70)
