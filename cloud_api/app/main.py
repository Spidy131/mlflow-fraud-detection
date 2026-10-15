from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hashlib
import os
import time
import xgboost as xgb

from app.preprocessing import create_features


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/model/model.ubj"
)

MODEL_SHA256 = os.getenv(
    "MODEL_SHA256",
    "D97CC8A3AE85501B7356B527B86FBB7DBFDD544F014714569C9B84E3C7E0043E"
)

THRESHOLD = float(
    os.getenv("FRAUD_THRESHOLD", "0.78")
)

MODEL_NAME = "XGBoost_Fraud_Production"
MODEL_VERSION = "1"
MODEL_ALIAS = "production"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Fraud Detection API",
    description="Standalone production XGBoost Fraud Detection API",
    version="1.0.0",
)


# ============================================================
# MODEL VALIDATION
# ============================================================

def verify_model_checksum():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    sha256 = hashlib.sha256()

    with open(MODEL_PATH, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest().upper()

    expected_hash = MODEL_SHA256.upper()

    if actual_hash != expected_hash:
        raise ValueError(
            "Model SHA256 verification failed. "
            f"Expected={expected_hash}, "
            f"Actual={actual_hash}"
        )

    print("Model SHA256 verification: PASSED")
    print(f"SHA256: {actual_hash}")


# ============================================================
# LOAD MODEL
# ============================================================

print("==============================================")
print("AI FRAUD DETECTION CLOUD API")
print("==============================================")

print(f"Model path: {MODEL_PATH}")
print(f"Model version: {MODEL_VERSION}")
print(f"Threshold: {THRESHOLD}")

verify_model_checksum()

booster = xgb.Booster()

booster.load_model(MODEL_PATH)

print("XGBoost model loaded successfully.")


# ============================================================
# LOAD TRAINING CATEGORIES
# ============================================================

xgb_categories = booster.get_categories(
    export_to_arrow=True
)

arrow_categories = xgb_categories.to_arrow()

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
# REQUEST SCHEMA
# ============================================================

class Transaction(BaseModel):

    TransactionAmt: float
    ProductCD: str

    card1: int
    card2: float | None = None
    card3: float | None = None
    card4: str
    card5: float | None = None
    card6: str

    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None

    P_emaildomain: str | None = None
    R_emaildomain: str | None = None

    DeviceType: str | None = None
    DeviceInfo: str | None = None

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

    D1: float | None = None
    D2: float | None = None
    D3: float | None = None
    D4: float | None = None

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

@app.get("/healthz")
def healthz():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
    }


@app.get("/")
def health_check():

    return {
        "status": "healthy",
        "service": "AI Fraud Detection API",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_alias": MODEL_ALIAS,
        "threshold": THRESHOLD,
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model")
def model_info():

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_alias": MODEL_ALIAS,
        "framework": "xgboost",
        "xgboost_version": "3.4.1",
        "features": 74,
        "threshold": THRESHOLD,
        "model_sha256": MODEL_SHA256,
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    start_time = time.perf_counter()

    try:

        data = transaction.model_dump()

        features = create_features(
            data,
            training_categories=training_categories
        )

        if features.shape[1] != 74:

            raise ValueError(
                f"Expected 74 features, "
                f"received {features.shape[1]}"
            )

        probability = float(
            booster.predict(
                xgb.DMatrix(
                    features,
                    enable_categorical=True
                )
            )[0]
        )

        prediction = int(
            probability >= THRESHOLD
        )

        result = (
            "FRAUD"
            if prediction == 1
            else "LEGITIMATE"
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "prediction": prediction,
            "fraud_probability": round(
                probability,
                4
            ),
            "threshold": THRESHOLD,
            "result": result,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_alias": MODEL_ALIAS,
            "latency_ms": round(
                latency_ms,
                2
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# STARTUP
# ============================================================

print("Cloud API initialization completed.")
#'@ | Set-Content -Encoding utf8 cloud_api\app\main.py