import sqlite3
import mlflow
from datetime import datetime, timezone
import os
DB_PATH = "data/monitoring.db"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)
EXPERIMENT_NAME = "Fraud_Detection_Monitoring"

MODEL_NAME = "XGBoost_Fraud_Production"
MODEL_ALIAS = "production"


def get_monitoring_metrics():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        COUNT(*) AS total_predictions,
        SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS fraud_predictions,
        SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) AS legitimate_predictions,
        AVG(probability) AS average_probability,
        MAX(probability) AS maximum_probability,
        AVG(latency_ms) AS average_latency_ms
    FROM predictions
    """

    row = conn.execute(query).fetchone()
    conn.close()

    total = row[0] or 0
    fraud = row[1] or 0
    legitimate = row[2] or 0
    avg_probability = row[3] or 0.0
    max_probability = row[4] or 0.0
    avg_latency = row[5] or 0.0

    fraud_rate = fraud / total if total > 0 else 0.0

    return {
        "total_predictions": total,
        "fraud_predictions": fraud,
        "legitimate_predictions": legitimate,
        "fraud_rate": fraud_rate,
        "average_probability": avg_probability,
        "maximum_probability": max_probability,
        "average_latency_ms": avg_latency,
    }


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)

metrics = get_monitoring_metrics()

with mlflow.start_run(run_name="production_monitoring"):

    mlflow.log_metrics({
        "total_predictions": metrics["total_predictions"],
        "fraud_predictions": metrics["fraud_predictions"],
        "legitimate_predictions": metrics["legitimate_predictions"],
        "fraud_rate": metrics["fraud_rate"],
        "average_probability": metrics["average_probability"],
        "maximum_probability": metrics["maximum_probability"],
        "average_latency_ms": metrics["average_latency_ms"],
    })

    mlflow.log_params({
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "monitoring_database": DB_PATH,
    })

    mlflow.set_tag(
        "monitoring_timestamp",
        datetime.now(timezone.utc).isoformat()
    )

print("\nMLflow monitoring run completed successfully.")
print("\nMonitoring metrics:")

for key, value in metrics.items():
    print(f"{key}: {value}")