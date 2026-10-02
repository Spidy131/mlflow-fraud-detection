from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from datetime import datetime, timezone

import os
import mlflow
import mlflow.pyfunc
import xgboost as xgb

from app.preprocessing import create_features
from app.monitoring import (
    init_db,
    log_prediction,
    get_monitoring_summary,
    get_recent_predictions,
)

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost Fraud Detection API",
    version="1.0.0",
)

# ============================================================
# INITIALIZE MONITORING DATABASE
# ============================================================

init_db()
# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "/models"
THRESHOLD_PATH = "/app/models/threshold.txt"


# ============================================================
# THRESHOLD
# ============================================================

try:
    with open(THRESHOLD_PATH, "r") as f:
        THRESHOLD = float(f.read().strip())

except Exception:
    THRESHOLD = 0.78


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000"
)

MODEL_NAME = "XGBoost_Fraud_Production"
MODEL_ALIAS = "production"
# ============================================================
# PRODUCTION MONITORING
# ============================================================

# ============================================================
# MONITORING
# ============================================================

@app.get("/monitoring")
def monitoring():

    metrics = get_monitoring_summary()

    return {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": {
            "name": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": model_version.version,
            "model_id": model_id,
        },

        "threshold": THRESHOLD,

        "predictions": {
            "total": metrics["total"],
            "fraud": metrics["fraud"],
            "legitimate": metrics["legitimate"],
            "fraud_rate": metrics["fraud_rate"],
        },

        "probability": {
            "average": metrics[
                "average_probability"
            ],
            "maximum": metrics[
                "maximum_probability"
            ],
        },

        "performance": {
            "average_latency_ms": metrics[
                "average_latency_ms"
            ],
        },
    }
# ============================================================
# RECENT PREDICTIONS
# ============================================================

@app.get("/monitoring/recent")
def recent_predictions(limit: int = 20):

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100"
        )

    return {
        "count": limit,
        "predictions": get_recent_predictions(limit)
    }

# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("Loading production model from MLflow...")
print(f"MLflow URI: {MLFLOW_TRACKING_URI}")


# ------------------------------------------------------------
# MLflow client
# ------------------------------------------------------------

client = mlflow.MlflowClient()


# ------------------------------------------------------------
# Resolve production alias
# ------------------------------------------------------------

model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS
)

print(
    f"Found model: {model_version.name} "
    f"version={model_version.version}"
)


# ------------------------------------------------------------
# Model ID
# ------------------------------------------------------------

model_id = model_version.source.split("/")[-1]

print(f"Model ID: {model_id}")


# ============================================================
# LOAD XGBOOST TRAINING CATEGORIES
# ============================================================

XGB_MODEL_FILE = os.path.join(
    MODEL_PATH,
    "model.ubj"
)

print(
    f"Loading XGBoost model categories from: "
    f"{XGB_MODEL_FILE}"
)


# ------------------------------------------------------------
# Load underlying XGBoost Booster
# ------------------------------------------------------------

booster = xgb.Booster()

booster.load_model(
    XGB_MODEL_FILE
)


# ------------------------------------------------------------
# Get categories saved during training
# ------------------------------------------------------------

xgb_categories = booster.get_categories(
    export_to_arrow=True
)

arrow_categories = xgb_categories.to_arrow()


# ------------------------------------------------------------
# Convert Arrow category arrays into Python lists
#
# Only categorical columns have a category array.
# Numerical columns return None.
# ------------------------------------------------------------

training_categories = {}

for column_name, category_array in arrow_categories:

    if category_array is not None:
        training_categories[column_name] = (
            category_array.to_pylist()
        )


print(
    f"Loaded training categories for "
    f"{len(training_categories)} categorical features."
)


# ============================================================
# LOAD MLflow PYFUNC MODEL
# ============================================================

print(
    f"Loading MLflow model from: {MODEL_PATH}"
)

model = mlflow.pyfunc.load_model(
    MODEL_PATH
)

print("Production model loaded successfully.")


# ============================================================
# REQUEST SCHEMA
# ============================================================

class Transaction(BaseModel):

    # --------------------------------------------------------
    # Transaction information
    # --------------------------------------------------------

    TransactionAmt: float
    ProductCD: str

    # --------------------------------------------------------
    # Card information
    # --------------------------------------------------------

    card1: int
    card2: float | None = None
    card3: float | None = None
    card4: str
    card5: float | None = None
    card6: str

    # --------------------------------------------------------
    # Address information
    # --------------------------------------------------------

    addr1: float | None = None
    addr2: float | None = None

    dist1: float | None = None

    # --------------------------------------------------------
    # Email information
    # --------------------------------------------------------

    P_emaildomain: str | None = None
    R_emaildomain: str | None = None

    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    DeviceType: str | None = None
    DeviceInfo: str | None = None

    # --------------------------------------------------------
    # C features
    # --------------------------------------------------------

    C1: float | None = None
    C2: float | None = None
    C3: float | None = None
    C4: float | None = None
    C5: float | None = None
    C6: float | None = None
    C7: float | None = None
    C8: float | None = None
    C9: float | None = None
    C10: float | None = None

    # --------------------------------------------------------
    # D features
    # --------------------------------------------------------

    D1: float | None = None
    D2: float | None = None
    D3: float | None = None
    D4: float | None = None
    # --------------------------------------------------------
    # M features
    # --------------------------------------------------------

    M1: str | None = None
    M2: str | None = None
    M3: str | None = None
    M4: str | None = None
    M5: str | None = None
    M6: str | None = None
    M7: str | None = None
    M8: str | None = None
    M9: str | None = None


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return {
        "status": "healthy",
        "model": "XGBoost Fraud Detection",
        "environment": "production",
        "threshold": THRESHOLD,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "categorical_features_loaded": len(
            training_categories
        ),
    }

# ============================================================
# MONITORING
# ============================================================

# ============================================================
# MONITORING
# ============================================================

@app.get("/monitoring")
def monitoring():

    summary = get_monitoring_summary()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "model": {
            "name": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": model_version.version,
            "model_id": model_id,
        },

        "threshold": THRESHOLD,

        "predictions": {
            "total": summary["total"],
            "fraud": summary["fraud"],
            "legitimate": summary["legitimate"],
            "fraud_rate": summary["fraud_rate"],
        },

        "probability": {
            "average": summary["average_probability"],
            "maximum": summary["maximum_probability"],
        },

        "performance": {
            "average_latency_ms": summary["average_latency_ms"],
        },
    }


@app.get("/monitoring/recent")
def recent_predictions(limit: int = 20):

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100"
        )

    return {
        "count": limit,
        "predictions": get_recent_predictions(limit)
    }
# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):
    

    start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Convert Pydantic object to dictionary
        # ----------------------------------------------------

        data = transaction.model_dump()

        # ----------------------------------------------------
        # Create the exact 74 model features
        #
        # IMPORTANT:
        # Pass the training-time XGBoost categories.
        # ----------------------------------------------------

        features = create_features(
            data,
            training_categories=training_categories
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if features.shape[1] != 74:

            raise ValueError(
                f"Expected 74 features, "
                f"received {features.shape[1]}"
            )

        # ----------------------------------------------------
        # Predict fraud probability
        # ----------------------------------------------------

        probability = float(
            booster.predict(
                xgb.DMatrix(
                    features,
                    enable_categorical=True
                )
            )[0]
        )

        # ----------------------------------------------------
        # Apply production threshold
        # ----------------------------------------------------

        prediction = int(
            probability >= THRESHOLD
        )

        # ----------------------------------------------------
        # Human-readable result
        # ----------------------------------------------------

        result = (
            "FRAUD"
            if prediction == 1
            else "LEGITIMATE"
        )
        # ----------------------------------------------------
        # Calculate latency
        # ----------------------------------------------------

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # ----------------------------------------------------
        # Persist prediction
        # ----------------------------------------------------

        log_prediction(
        probability=probability,
        prediction=prediction,
        threshold=THRESHOLD,
        result=result,
        model_name=MODEL_NAME,
        model_version=model_version.version,
        model_alias=MODEL_ALIAS,
        model_id=model_id,
        latency_ms=latency_ms,
        input_features=data,
    )
        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {
            "prediction": prediction,
            "fraud_probability": round(
                probability,
                4
            ),
            "threshold": THRESHOLD,
            "result": result,
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )