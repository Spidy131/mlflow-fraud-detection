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

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Fraud_Detection_Drift_Monitoring"

MONITORING_START_ID = int(
    os.getenv("MONITORING_START_ID", "0")
)

MONITORING_END_ID = int(
    os.getenv("MONITORING_END_ID", "0")
)

# Rolling production window used for PSI
PSI_ROLLING_WINDOW = 100

# PSI thresholds
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25


# ============================================================
# HEADER
# ============================================================

print("===================================")
print("DRIFT MONITORING")
print("===================================")

print(
    f"Monitoring range: "
    f"{MONITORING_START_ID} to {MONITORING_END_ID}"
)


# ============================================================
# LOAD TRAINING / REFERENCE DATA
# ============================================================

print()
print("Loading training/reference data...")

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

print(
    f"Reference rows: {len(train_df)}"
)


# ============================================================
# LOAD INCREMENTAL PRODUCTION DATA
# ============================================================

print()
print("Loading production monitoring data...")

conn = sqlite3.connect(
    MONITORING_DB
)

if MONITORING_END_ID > 0:

    rows = conn.execute(
        """
        SELECT
            id,
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
            input_features
        FROM predictions
        WHERE input_features IS NOT NULL
        ORDER BY id
        """
    ).fetchall()

print(
    f"Database rows found: {len(rows)}"
)

if not rows:

    conn.close()

    raise ValueError(
        "No production monitoring data found "
        "for the selected monitoring range."
    )


# ============================================================
# PARSE INCREMENTAL RECORDS
# ============================================================

incremental_records = []
invalid_json_rows = 0

for row in rows:

    try:

        data = json.loads(row[1])

        data["_prediction_id"] = row[0]

        incremental_records.append(data)

    except Exception:

        invalid_json_rows += 1


conn.close()

incremental_df = pd.DataFrame(
    incremental_records
)

print(
    f"Production rows: "
    f"{len(incremental_df)}"
)

print(
    f"Invalid JSON rows: "
    f"{invalid_json_rows}"
)


if incremental_df.empty:

    raise ValueError(
        "No valid production monitoring "
        "records found."
    )


# ============================================================
# LOAD ROLLING PRODUCTION WINDOW FOR PSI
# ============================================================

print()
print("Loading rolling production window for PSI...")

conn = sqlite3.connect(
    MONITORING_DB
)

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
    (
        PSI_ROLLING_WINDOW,
    )
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


production_df = pd.DataFrame(
    rolling_records
)

print(
    f"PSI rolling window rows: "
    f"{len(production_df)}"
)

print(
    f"PSI invalid JSON rows: "
    f"{rolling_invalid_json}"
)


if production_df.empty:

    raise ValueError(
        "No valid production data available "
        "for PSI calculation."
    )


# ============================================================
# PSI FUNCTION
# ============================================================

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
            current_pct / reference_pct
        )
    )

    return float(psi)


# ============================================================
# CALCULATE PSI
# ============================================================

print()
print("Feature Drift Results")
print("---------------------")

psi_results = {}

for feature in drift_features:

    if (
        feature in train_df.columns
        and feature in production_df.columns
    ):

        psi = calculate_psi(
            train_df[feature],
            production_df[feature]
        )

    else:

        psi = 0.0

    psi_results[feature] = psi

    if psi >= PSI_CRITICAL:

        status = "CRITICAL"

    elif psi >= PSI_WARNING:

        status = "WARNING"

    else:

        status = "LOW"

    print(
        f"{feature:<20} "
        f"PSI={psi:.6f} "
        f"STATUS={status}"
    )


# ============================================================
# MAXIMUM PSI
# ============================================================

max_psi_feature = max(
    psi_results,
    key=psi_results.get
)

max_psi = psi_results[
    max_psi_feature
]


if max_psi >= PSI_CRITICAL:

    overall_status = "CRITICAL"

elif max_psi >= PSI_WARNING:

    overall_status = "WARNING"

else:

    overall_status = "LOW"


# ============================================================
# SUMMARY
# ============================================================

print()
print("===================================")
print("DRIFT MONITORING COMPLETED")
print("===================================")

print(
    f"Monitoring range: "
    f"{MONITORING_START_ID} to "
    f"{MONITORING_END_ID}"
)

print(
    f"Incremental production rows: "
    f"{len(incremental_df)}"
)

print(
    f"PSI rolling window rows: "
    f"{len(production_df)}"
)

print(
    f"Features monitored: "
    f"{len(drift_features)}"
)

print(
    f"Maximum PSI: "
    f"{max_psi:.6f}"
)

print(
    f"Feature with maximum drift: "
    f"{max_psi_feature}"
)

print(
    f"Overall drift status: "
    f"{overall_status}"
)


# ============================================================
# LOG TO MLFLOW
# ============================================================

print()
print("Logging drift results to MLflow...")

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)

with mlflow.start_run(
    run_name="production_drift_monitoring"
):

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mlflow.log_metric(
        "production_rows",
        len(incremental_df)
    )

    mlflow.log_metric(
        "psi_rolling_window_rows",
        len(production_df)
    )

    mlflow.log_metric(
        "features_monitored",
        len(drift_features)
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
    # Tags
    # --------------------------------------------------------

    mlflow.set_tag(
        "drift_status",
        overall_status
    )

    mlflow.set_tag(
        "max_drift_feature",
        max_psi_feature
    )

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
        "incremental_with_rolling_psi"
    )

print()
print(
    "MLflow drift monitoring run "
    "completed successfully."
)

print(
    f"Experiment: {EXPERIMENT_NAME}"
)