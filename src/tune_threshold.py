import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

import mlflow.sklearn


MODEL_NAME = "fraud-detection-model"

DATA_PATH = "data/processed/fraud_baseline.csv"

MODEL_URI = f"models:/{MODEL_NAME}@production"


# ==================================================
# 1. Load data
# ==================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# ==================================================
# 2. Same test split used during training
# ==================================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# 3. Load production model
# ==================================================

print("Loading production model...")

model = mlflow.sklearn.load_model(
    MODEL_URI
)


# ==================================================
# 4. Get fraud probabilities
# ==================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ==================================================
# 5. Check different thresholds
# ==================================================

results = []

for threshold in [i / 100 for i in range(10, 91)]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )


results_df = pd.DataFrame(results)


# ==================================================
# 6. Find best threshold by F1
# ==================================================

best = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n================================")
print("THRESHOLD OPTIMIZATION")
print("================================")

print(
    f"Best Threshold : {best['threshold']:.2f}"
)

print(
    f"Precision      : {best['precision']:.4f}"
)

print(
    f"Recall         : {best['recall']:.4f}"
)

print(
    f"F1 Score       : {best['f1']:.4f}"
)


# ==================================================
# 7. Show top thresholds
# ==================================================

print("\nTop 10 thresholds:")

print(
    results_df
    .sort_values("f1", ascending=False)
    .head(10)
    .to_string(index=False)
)


# ==================================================
# 8. Log threshold to MLflow
# ==================================================

run = mlflow.search_runs(
    experiment_names=["fraud-detection"],
    filter_string="tags.mlflow.runName = 'XGBoost'",
    order_by=["start_time DESC"],
    max_results=1
)

if not run.empty:

    run_id = run.iloc[0]["run_id"]

    with mlflow.start_run(
        run_id=run_id
    ):

        mlflow.log_param(
            "optimized_threshold",
            float(best["threshold"])
        )

        mlflow.log_metric(
            "optimized_precision",
            float(best["precision"])
        )

        mlflow.log_metric(
            "optimized_recall",
            float(best["recall"])
        )

        mlflow.log_metric(
            "optimized_f1",
            float(best["f1"])
        )

    print(
        f"\nThreshold logged to MLflow run: {run_id}"
    )

else:

    print(
        "\nXGBoost MLflow run not found."
    )
