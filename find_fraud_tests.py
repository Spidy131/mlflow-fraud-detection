import csv
import pandas as pd
import mlflow

MODEL_PATH = r"mlruns/1/models/m-ed5df3fdfab64833acf9a3108595f3c7/artifacts"

THRESHOLD = 0.23

print("Loading model...")

model = mlflow.pyfunc.load_model(MODEL_PATH)

print("Loading dataset...")

# Use csv so we don't depend on pandas/pyarrow for reading the CSV
with open(
    "data/processed/fraud_baseline.csv",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    fraud_rows = []

    for row in reader:

        if row["isFraud"] == "1":

            row_without_target = {
                k: v
                for k, v in row.items()
                if k != "isFraud"
            }

            fraud_rows.append(row_without_target)


print("Fraud rows found:", len(fraud_rows))

# Convert to DataFrame only after loading the rows
df = pd.DataFrame(fraud_rows)

# Convert numeric columns
numeric_columns = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
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
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Handle categorical missing values
categorical_columns = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
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

for column in categorical_columns:
    df[column] = df[column].fillna("missing")


print("Calculating probabilities...")

probabilities = model.predict(df)

# MLflow pyfunc may return predictions rather than probabilities,
# so use the underlying sklearn pipeline if available.
try:
    probabilities = model._model_impl.sklearn_model.predict_proba(df)[:, 1]
except Exception:
    print("Could not access predict_proba through pyfunc model.")
    raise


df["fraud_probability"] = probabilities

# Keep only fraud transactions the model predicts as fraud
detected = df[
    df["fraud_probability"] >= THRESHOLD
].sort_values(
    "fraud_probability",
    ascending=False
)

print("\n================================")
print("FRAUD TEST TRANSACTIONS")
print("================================")

print(
    "Detected fraud transactions:",
    len(detected)
)

print(
    detected[
        ["TransactionAmt", "ProductCD", "fraud_probability"]
    ].head(5).to_string(index=False)
)


# Save the top 5 as JSON
top5 = detected.head(5).drop(
    columns=["fraud_probability"]
)

top5.to_json(
    "fraud_test_transactions.json",
    orient="records",
    indent=2
)

print("\nSaved:")
print("fraud_test_transactions.json")