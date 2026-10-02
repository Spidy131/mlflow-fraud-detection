import requests
import pandas as pd
import time

TEST_PATH = "data/processed/fraud_baseline.csv"
API_URL = "http://localhost:8000/predict"

# Load dataset
df = pd.read_csv(TEST_PATH)

# Reproduce the same production/test split
from sklearn.model_selection import train_test_split

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Select 100 real holdout transactions
sample = X_test.sample(
    n=100,
    random_state=42
)

results = []

for i, (_, row) in enumerate(sample.iterrows(), start=1):

    data = row.to_dict()

    # Convert NaN to None so JSON can handle missing values
    for key, value in data.items():
        if pd.isna(value):
            data[key] = None

    start = time.time()

    try:
        response = requests.post(
            API_URL,
            json=data,
            timeout=30
        )

        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            result = response.json()

            results.append({
                "fraud_probability": result["fraud_probability"],
                "prediction": result["prediction"],
                "threshold": result["threshold"],
                "result": result["result"],
                "latency_ms": latency,
            })

            print(
                f"{i}/100 | "
                f"prob={result['fraud_probability']:.4f} | "
                f"prediction={result['prediction']} | "
                f"{result['result']}"
            )

        else:
            print(
                f"{i}/100 | API ERROR "
                f"{response.status_code}: {response.text}"
            )

    except Exception as e:
        print(f"{i}/100 | ERROR: {e}")

print("\nBatch completed.")
print(f"Successful predictions: {len(results)}")

if results:
    results_df = pd.DataFrame(results)

    print("\nPrediction distribution:")
    print(results_df["prediction"].value_counts())

    print("\nAverage probability:")
    print(results_df["fraud_probability"].mean())

    results_df.to_csv(
        "monitoring_batch_100.csv",
        index=False
    )

    print("\nSaved: monitoring_batch_100.csv")