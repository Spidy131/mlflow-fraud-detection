import sqlite3
import json
import numpy as np
import pandas as pd
import mlflow
import os

# ============================================================
# CONFIGURATION
# ============================================================

MONITORING_DB = "data/monitoring.db"
TRAIN_PATH = "data/processed/fraud_baseline.csv"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)
EXPERIMENT_NAME = "Fraud_Detection_Alerts"

MONITORING_START_ID = int(
    os.getenv("MONITORING_START_ID", "0")
)

MONITORING_END_ID = int(
    os.getenv("MONITORING_END_ID", "0")
)

# Rolling window used specifically for PSI
PSI_ROLLING_WINDOW = 100

# Alert thresholds
MAX_PSI_WARNING = 0.10
MAX_PSI_CRITICAL = 0.25

MISSING_RATE_WARNING = 0.05
MISSING_RATE_CRITICAL = 0.10

LATENCY_WARNING_MS = 500
LATENCY_CRITICAL_MS = 1000


# ============================================================
# LOAD INCREMENTAL PRODUCTION MONITORING DATA
# ============================================================

print("===================================")
print("PRODUCTION MONITORING ALERTS")
print("===================================")

print(f"Monitoring range: {MONITORING_START_ID} to {MONITORING_END_ID}")

print()
print("Loading production monitoring data...")

conn = sqlite3.connect(MONITORING_DB)

# ------------------------------------------------------------
# Incremental monitoring data
# ------------------------------------------------------------

if MONITORING_END_ID > 0:

    rows = conn.execute(
        """
        SELECT
            id,
            probability,
            latency_ms,
            input_features
        FROM predictions
        WHERE input_features IS NOT NULL
          AND id > ?
          AND id <= ?
        ORDER BY id
        """,
        (
            MONITORING_START_ID,
            MONITORING_END_ID
        )
    ).fetchall()

else:

    rows = conn.execute(
        """
        SELECT
            id,
            probability,
            latency_ms,
            input_features
        FROM predictions
        WHERE input_features IS NOT NULL
        ORDER BY id
        """
    ).fetchall()

print(f"Database rows found: {len(rows)}")

if not rows:
    conn.close()
    raise ValueError(
        "No production monitoring data found for the selected range."
    )


# ============================================================
# PARSE INCREMENTAL RECORDS
# ============================================================

records = []
invalid_json_count = 0

for row in rows:

    try:

        data = json.loads(row[3])

        data["_prediction_id"] = row[0]
        data["_probability"] = row[1]
        data["_latency_ms"] = row[2]

        records.append(data)

    except Exception:

        invalid_json_count += 1


conn.close()

df = pd.DataFrame(records)

print(f"Production rows monitored: {len(df)}")
print(f"Invalid JSON rows: {invalid_json_count}")


if df.empty:
    raise ValueError(
        "No valid production monitoring records found."
    )


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

features = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

available_features = [
    f
    for f in features
    if f in df.columns
]

if available_features:

    missing_rate = (
        df[available_features]
        .isna()
        .mean()
        .mean()
    )

else:

    missing_rate = 1.0


# ============================================================
# PROBABILITY VALIDATION
# ============================================================

probability = pd.to_numeric(
    df["_probability"],
    errors="coerce"
)

invalid_probability_count = int(
    (
        probability.isna()
        | (probability < 0)
        | (probability > 1)
    ).sum()
)


# ============================================================
# LATENCY VALIDATION
# ============================================================

latency = pd.to_numeric(
    df["_latency_ms"],
    errors="coerce"
)

invalid_latency_count = int(
    (
        latency.isna()
        | (latency < 0)
    ).sum()
)

average_latency_ms = latency.mean()


# ============================================================
# DATA QUALITY STATUS
# ============================================================

if (
    missing_rate >= MISSING_RATE_CRITICAL
    or invalid_probability_count > 0
    or invalid_latency_count > 0
):

    quality_status = "CRITICAL"

elif missing_rate >= MISSING_RATE_WARNING:

    quality_status = "WARNING"

else:

    quality_status = "PASS"


# ============================================================
# LOAD ROLLING PRODUCTION DATA FOR PSI
# ============================================================

print()
print("Loading rolling production window for PSI...")

conn = sqlite3.connect(MONITORING_DB)

rolling_rows = conn.execute(
    """
    SELECT
        id,
        input_features
    FROM predictions
    WHERE input_features IS NOT NULL
    ORDER BY id DESC
    LIMIT ?
    """,
    (PSI_ROLLING_WINDOW,)
).fetchall()

conn.close()

rolling_records = []

rolling_invalid_json = 0

for row in rolling_rows:

    try:

        data = json.loads(row[1])

        data["_prediction_id"] = row[0]

        rolling_records.append(data)

    except Exception:

        rolling_invalid_json += 1


psi_df = pd.DataFrame(
    rolling_records
)

print(
    f"PSI production rows: {len(psi_df)}"
)

print(
    f"PSI invalid JSON rows: {rolling_invalid_json}"
)


# ============================================================
# PSI DRIFT CALCULATION
# ============================================================

print()
print("Calculating feature drift...")

drift_features = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
]

train_df = pd.read_csv(
    TRAIN_PATH,
    usecols=drift_features
)


def calculate_psi(
    reference,
    current,
    bins=10
):

    reference = pd.to_numeric(
        reference,
        errors="coerce"
    ).dropna().values

    current = pd.to_numeric(
        current,
        errors="coerce"
    ).dropna().values

    if (
        len(reference) == 0
        or len(current) == 0
    ):
        return 0.0

    breakpoints = np.percentile(
        reference,
        np.linspace(
            0,
            100,
            bins + 1
        )
    )

    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 3:
        return 0.0

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    reference_pct = np.clip(
        reference_counts / len(reference),
        0.0001,
        None
    )

    current_pct = np.clip(
        current_counts / len(current),
        0.0001,
        None
    )

    psi = np.sum(
        (
            current_pct - reference_pct
        )
        * np.log(
            current_pct
            / reference_pct
        )
    )

    return float(psi)


psi_results = {}

for feature in drift_features:

    if (
        feature in psi_df.columns
        and feature in train_df.columns
    ):

        psi_results[feature] = calculate_psi(
            train_df[feature],
            psi_df[feature]
        )

    else:

        psi_results[feature] = 0.0


max_psi_feature = max(
    psi_results,
    key=psi_results.get
)

max_psi = psi_results[
    max_psi_feature
]


# ============================================================
# DRIFT STATUS
# ============================================================

if max_psi >= MAX_PSI_CRITICAL:

    drift_status = "CRITICAL"

elif max_psi >= MAX_PSI_WARNING:

    drift_status = "WARNING"

else:

    drift_status = "PASS"


# ============================================================
# LATENCY STATUS
# ============================================================

if average_latency_ms >= LATENCY_CRITICAL_MS:

    latency_status = "CRITICAL"

elif average_latency_ms >= LATENCY_WARNING_MS:

    latency_status = "WARNING"

else:

    latency_status = "PASS"


# ============================================================
# OVERALL STATUS
# ============================================================

statuses = [
    quality_status,
    drift_status,
    latency_status
]

if "CRITICAL" in statuses:

    overall_status = "CRITICAL"

elif "WARNING" in statuses:

    overall_status = "WARNING"

else:

    overall_status = "PASS"


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("===================================")
print("PRODUCTION MONITORING ALERTS")
print("===================================")

print()
print(
    f"Incremental production rows: "
    f"{len(df)}"
)

print(
    f"PSI rolling window rows: "
    f"{len(psi_df)}"
)

print()
print("Data Quality")
print("---------------------")

print(
    f"Missing rate: "
    f"{missing_rate:.4f}"
)

print(
    f"Invalid probabilities: "
    f"{invalid_probability_count}"
)

print(
    f"Invalid latency values: "
    f"{invalid_latency_count}"
)

print(
    f"Invalid JSON rows: "
    f"{invalid_json_count}"
)

print(
    f"Status: {quality_status}"
)


print()
print("Feature Drift")
print("---------------------")

for feature, psi in psi_results.items():

    print(
        f"{feature:<20}"
        f"PSI: {psi:.6f}"
    )

print()
print(
    f"Maximum PSI: "
    f"{max_psi:.6f}"
)

print(
    f"Maximum drift feature: "
    f"{max_psi_feature}"
)

print(
    f"Drift status: "
    f"{drift_status}"
)


print()
print("Latency")
print("---------------------")

print(
    f"Average latency: "
    f"{average_latency_ms:.2f} ms"
)

print(
    f"Status: {latency_status}"
)


print()
print("===================================")
print(
    f"OVERALL STATUS: "
    f"{overall_status}"
)
print("===================================")


# ============================================================
# LOG ALERTS TO MLFLOW
# ============================================================

print()
print("Logging alert results to MLflow...")

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)

with mlflow.start_run(
    run_name="production_monitoring_alert"
):

    mlflow.log_metric(
        "production_rows",
        len(df)
    )

    mlflow.log_metric(
        "psi_rolling_window_rows",
        len(psi_df)
    )

    mlflow.log_metric(
        "missing_rate",
        float(missing_rate)
    )

    mlflow.log_metric(
        "invalid_probability_count",
        invalid_probability_count
    )

    mlflow.log_metric(
        "invalid_latency_count",
        invalid_latency_count
    )

    mlflow.log_metric(
        "average_latency_ms",
        float(average_latency_ms)
    )

    mlflow.log_metric(
        "max_psi",
        float(max_psi)
    )

    for feature, psi in psi_results.items():

        mlflow.log_metric(
            f"psi_{feature}",
            float(psi)
        )

    # --------------------------------------------------------
    # Status tags
    # --------------------------------------------------------

    mlflow.set_tag(
        "data_quality_status",
        quality_status
    )

    mlflow.set_tag(
        "drift_status",
        drift_status
    )

    mlflow.set_tag(
        "latency_status",
        latency_status
    )

    mlflow.set_tag(
        "overall_status",
        overall_status
    )

    mlflow.set_tag(
        "max_drift_feature",
        max_psi_feature
    )

    # --------------------------------------------------------
    # Monitoring range
    # --------------------------------------------------------

    mlflow.set_tag(
        "monitoring_start_id",
        str(MONITORING_START_ID)
    )

    mlflow.set_tag(
        "monitoring_end_id",
        str(MONITORING_END_ID)
    )

    mlflow.set_tag(
        "monitoring_type",
        "incremental_alerts_rolling_psi"
    )

print()
print(
    "MLflow monitoring alert run "
    "completed successfully."
)

print(
    f"Experiment: {EXPERIMENT_NAME}"
)