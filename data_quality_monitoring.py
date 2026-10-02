import sqlite3
import json
import os
import sys

import pandas as pd
import mlflow


# ============================================================
# CONFIGURATION
# ============================================================

MONITORING_START_ID = int(
    os.getenv("MONITORING_START_ID", "0")
)

MONITORING_END_ID = int(
    os.getenv("MONITORING_END_ID", "0")
)

MONITORING_DB = "data/monitoring.db"

MLFLOW_TRACKING_URI = "http://localhost:5000"

EXPERIMENT_NAME = "Fraud_Detection_Data_Quality"


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
]

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FRAUD DETECTION - DATA QUALITY MONITORING")
print("=" * 70)

print(
    f"Monitoring ID range: "
    f"{MONITORING_START_ID + 1} - {MONITORING_END_ID}"
)


# ============================================================
# CHECK DATABASE
# ============================================================

if not os.path.exists(MONITORING_DB):

    print(
        f"ERROR: Monitoring database not found: "
        f"{MONITORING_DB}"
    )

    sys.exit(1)


# ============================================================
# LOAD PRODUCTION DATA
# ============================================================

print()
print("Loading production monitoring data...")


conn = sqlite3.connect(MONITORING_DB)

try:

    rows = conn.execute(
        """
        SELECT
            id,
            probability,
            prediction,
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
            MONITORING_END_ID,
        ),
    ).fetchall()

finally:

    conn.close()


print(
    f"Database rows found: {len(rows)}"
)


# ============================================================
# PARSE INPUT FEATURES
# ============================================================

records = []

invalid_json_count = 0


for row in rows:

    prediction_id = row[0]
    probability_value = row[1]
    prediction_value = row[2]
    latency_value = row[3]
    input_features_json = row[4]

    try:

        data = json.loads(
            input_features_json
        )

        if not isinstance(data, dict):

            print(
                f"Skipping prediction ID "
                f"{prediction_id}: "
                f"input_features is not an object"
            )

            invalid_json_count += 1

            continue


        # Monitoring metadata

        data["_prediction_id"] = (
            prediction_id
        )

        data["_probability"] = (
            probability_value
        )

        data["_prediction"] = (
            prediction_value
        )

        data["_latency_ms"] = (
            latency_value
        )


        records.append(data)


    except Exception as e:

        print(
            f"Skipping prediction ID "
            f"{prediction_id}: {e}"
        )

        invalid_json_count += 1


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)


print(
    f"Production rows: {len(df)}"
)


# ============================================================
# EMPTY DATA HANDLING
# ============================================================

if df.empty:

    print()
    print(
        "No monitorable production predictions "
        "found in this ID range."
    )

    print(
        f"Invalid JSON records: "
        f"{invalid_json_count}"
    )

    print(
        "Data Quality status: NO_DATA"
    )


    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    try:

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            EXPERIMENT_NAME
        )

        with mlflow.start_run(
            run_name="production_data_quality_empty"
        ):

            mlflow.log_param(
                "monitoring_start_id",
                MONITORING_START_ID
            )

            mlflow.log_param(
                "monitoring_end_id",
                MONITORING_END_ID
            )

            mlflow.log_metric(
                "production_rows",
                0
            )

            mlflow.log_metric(
                "invalid_json_records",
                invalid_json_count
            )

            mlflow.set_tag(
                "quality_status",
                "NO_DATA"
            )

            mlflow.set_tag(
                "monitoring_status",
                "NO_DATA"
            )

        print(
            "MLflow empty-window run completed."
        )

    except Exception as e:

        print(
            f"WARNING: MLflow logging failed: {e}"
        )


    sys.exit(0)


# ============================================================
# AVAILABLE FEATURES
# ============================================================

available_features = [
    feature
    for feature in ALL_FEATURES
    if feature in df.columns
]


missing_features = [
    feature
    for feature in ALL_FEATURES
    if feature not in df.columns
]


# ============================================================
# 1. MISSING VALUE RATE
# ============================================================

missing_rates = {}


for feature in available_features:

    missing_rate = float(
        df[feature].isna().mean()
    )

    missing_rates[feature] = (
        missing_rate
    )


if available_features:

    overall_missing_rate = float(
        df[available_features]
        .isna()
        .mean()
        .mean()
    )

else:

    overall_missing_rate = 1.0


# ============================================================
# 2. INVALID NUMERIC VALUES
# ============================================================

invalid_numeric = {}


for feature in NUMERIC_FEATURES:

    if feature not in df.columns:

        continue


    original = df[feature]


    converted = pd.to_numeric(
        original,
        errors="coerce"
    )


    invalid_count = int(
        (
            converted.isna()
            & original.notna()
        ).sum()
    )


    invalid_numeric[feature] = (
        invalid_count
    )


total_invalid_numeric = sum(
    invalid_numeric.values()
)


# ============================================================
# 3. NEGATIVE TRANSACTION AMOUNT
# ============================================================

if "TransactionAmt" in df.columns:

    transaction_amount = pd.to_numeric(
        df["TransactionAmt"],
        errors="coerce"
    )

    negative_transaction_amounts = int(
        (
            transaction_amount < 0
        ).sum()
    )

else:

    negative_transaction_amounts = 0


# ============================================================
# 4. PROBABILITY VALIDATION
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
# 5. LATENCY VALIDATION
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


# ============================================================
# 6. DUPLICATE PREDICTION IDS
# ============================================================

duplicate_prediction_ids = int(
    df["_prediction_id"]
    .duplicated()
    .sum()
)


# ============================================================
# 7. QUALITY ISSUES
# ============================================================

quality_issues = (
    total_invalid_numeric
    + negative_transaction_amounts
    + invalid_probability_count
    + invalid_latency_count
    + duplicate_prediction_ids
    + invalid_json_count
)


if quality_issues == 0:

    quality_status = "PASS"

else:

    quality_status = "FAIL"


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 50)
print("DATA QUALITY RESULTS")
print("=" * 50)

print(
    f"Monitoring ID range: "
    f"{MONITORING_START_ID + 1} - "
    f"{MONITORING_END_ID}"
)

print(
    f"Production rows: "
    f"{len(df)}"
)

print(
    f"Overall missing rate: "
    f"{overall_missing_rate:.4f}"
)

print(
    f"Invalid numeric values: "
    f"{total_invalid_numeric}"
)

print(
    f"Negative TransactionAmt: "
    f"{negative_transaction_amounts}"
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
    f"Duplicate prediction IDs: "
    f"{duplicate_prediction_ids}"
)

print(
    f"Invalid JSON records: "
    f"{invalid_json_count}"
)

print(
    f"Overall quality status: "
    f"{quality_status}"
)


# ============================================================
# FEATURE MISSING RATES
# ============================================================

print()
print("Feature Missing Rates")
print("-" * 40)


for feature in ALL_FEATURES:

    rate = missing_rates.get(
        feature,
        1.0
    )

    print(
        f"{feature:18s} {rate:.4f}"
    )


# ============================================================
# MLFLOW
# ============================================================

print()
print(
    "Logging data quality results to MLflow..."
)


try:

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="production_data_quality"
    ):

        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        mlflow.log_param(
            "monitoring_start_id",
            MONITORING_START_ID
        )

        mlflow.log_param(
            "monitoring_end_id",
            MONITORING_END_ID
        )

        mlflow.log_param(
            "available_features",
            len(available_features)
        )

        mlflow.log_param(
            "missing_features",
            len(missing_features)
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        mlflow.log_metric(
            "production_rows",
            len(df)
        )

        mlflow.log_metric(
            "overall_missing_rate",
            overall_missing_rate
        )

        mlflow.log_metric(
            "invalid_numeric_values",
            total_invalid_numeric
        )

        mlflow.log_metric(
            "negative_transaction_amounts",
            negative_transaction_amounts
        )

        mlflow.log_metric(
            "invalid_probabilities",
            invalid_probability_count
        )

        mlflow.log_metric(
            "invalid_latency_values",
            invalid_latency_count
        )

        mlflow.log_metric(
            "duplicate_prediction_ids",
            duplicate_prediction_ids
        )

        mlflow.log_metric(
            "invalid_json_records",
            invalid_json_count
        )

        mlflow.log_metric(
            "quality_issues",
            quality_issues
        )


        # ----------------------------------------------------
        # FEATURE MISSING METRICS
        # ----------------------------------------------------

        for feature, rate in missing_rates.items():

            mlflow.log_metric(
                f"missing_rate_{feature}",
                rate
            )


        # ----------------------------------------------------
        # TAGS
        # ----------------------------------------------------

        mlflow.set_tag(
            "quality_status",
            quality_status
        )

        mlflow.set_tag(
            "monitoring_status",
            "COMPLETED"
        )


    print(
        "MLflow data quality run completed."
    )


except Exception as e:

    print(
        f"ERROR: MLflow logging failed: {e}"
    )

    sys.exit(1)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 50)

if quality_status == "PASS":

    print(
        "DATA QUALITY MONITORING PASSED"
    )

else:

    print(
        "DATA QUALITY MONITORING FAILED"
    )

print("=" * 50)


# ============================================================
# EXIT
# ============================================================

if quality_status == "FAIL":

    sys.exit(1)


sys.exit(0)