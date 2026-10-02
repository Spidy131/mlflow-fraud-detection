import sqlite3
from datetime import datetime

import mlflow
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "data/monitoring.db"

MLFLOW_TRACKING_URI = "http://localhost:5000"

ALERT_EXPERIMENT = "Fraud_Detection_Alerts"

DRIFT_EXPERIMENT = "Fraud_Detection_Drift_Monitoring"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Monitoring",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

@st.cache_data(ttl=10)
def load_predictions():

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            id,
            timestamp,
            probability,
            prediction,
            threshold,
            result,
            model_name,
            model_version,
            model_alias,
            model_id,
            latency_ms
        FROM predictions
        ORDER BY id
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    if not df.empty:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df["probability"] = pd.to_numeric(
            df["probability"],
            errors="coerce"
        )

        df["latency_ms"] = pd.to_numeric(
            df["latency_ms"],
            errors="coerce"
        )

        df["prediction"] = pd.to_numeric(
            df["prediction"],
            errors="coerce"
        )

    return df


# ============================================================
# MLFLOW
# ============================================================

@st.cache_data(ttl=30)
def load_mlflow_alert_history():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    experiment = mlflow.get_experiment_by_name(
        ALERT_EXPERIMENT
    )

    if experiment is None:

        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        order_by=[
            "start_time DESC"
        ],
        max_results=100,
    )

    return runs


@st.cache_data(ttl=30)
def load_mlflow_drift_history():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    experiment = mlflow.get_experiment_by_name(
        DRIFT_EXPERIMENT
    )

    if experiment is None:

        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        order_by=[
            "start_time DESC"
        ],
        max_results=100,
    )

    return runs


# ============================================================
# STATUS HELPERS
# ============================================================

def status_icon(status):

    if status == "PASS":
        return "🟢"

    if status in ["WARNING", "MODERATE"]:
        return "🟡"

    if status in ["FAIL", "CRITICAL"]:
        return "🔴"

    return "⚪"


def get_latest_status(alert_runs):

    if alert_runs.empty:

        return {
            "overall": "NO_DATA",
            "quality": "NO_DATA",
            "drift": "NO_DATA",
            "latency": "NO_DATA",
        }


    latest = alert_runs.iloc[0]

    overall = latest.get(
        "tags.overall_status",
        "UNKNOWN"
    )

    quality = latest.get(
        "tags.data_quality_status",
        "UNKNOWN"
    )

    drift = latest.get(
        "tags.drift_status",
        "UNKNOWN"
    )

    latency = latest.get(
        "tags.latency_status",
        "UNKNOWN"
    )


    return {
        "overall": overall,
        "quality": quality,
        "drift": drift,
        "latency": latency,
    }


# ============================================================
# LOAD DATA
# ============================================================

try:

    predictions = load_predictions()

except Exception as e:

    st.error(
        f"Unable to load monitoring database: {e}"
    )

    st.stop()


try:

    alert_runs = load_mlflow_alert_history()

except Exception as e:

    alert_runs = pd.DataFrame()

    st.warning(
        f"Unable to load MLflow alert history: {e}"
    )


try:

    drift_runs = load_mlflow_drift_history()

except Exception as e:

    drift_runs = pd.DataFrame()

# ============================================================
# TEST TRANSACTION
# ============================================================

import requests

API_URL = "http://localhost:8000/predict"

st.markdown("## 🧪 Test Transaction")

st.info(
    "Enter transaction details below to test the production "
    "XGBoost fraud detection model."
)

with st.expander("Transaction Input", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        transaction_amt = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )

        product_cd = st.selectbox(
            "Product CD",
            ["W", "C", "R", "S", "H"],
        )

        card1 = st.number_input(
            "Card 1",
            min_value=0,
            value=13926,
            step=1,
        )

        card2 = st.number_input(
            "Card 2",
            min_value=0,
            value=123,
            step=1,
        )

        card3 = st.number_input(
            "Card 3",
            min_value=0,
            value=150,
            step=1,
        )

        card4 = st.selectbox(
            "Card 4",
            ["visa", "mastercard", "american express", "discover"],
        )

        card5 = st.number_input(
            "Card 5",
            min_value=0,
            value=226,
            step=1,
        )

        card6 = st.selectbox(
            "Card 6",
            ["credit", "debit"],
        )

    with col2:

        addr1 = st.number_input(
            "Address 1",
            min_value=0,
            value=315,
            step=1,
        )

        addr2 = st.number_input(
            "Address 2",
            min_value=0,
            value=87,
            step=1,
        )

        dist1 = st.number_input(
            "Distance 1",
            min_value=0.0,
            value=20.0,
            step=1.0,
        )

        p_email = st.text_input(
            "P Email Domain",
            value="gmail.com",
        )

        r_email = st.text_input(
            "R Email Domain",
            value="",
        )

        device_type = st.selectbox(
            "Device Type",
            ["desktop", "mobile"],
        )

        device_info = st.text_input(
            "Device Info",
            value="Windows",
        )

    with col3:

        st.markdown("**C Features**")

        c1 = st.number_input("C1", min_value=0.0, value=1.0)
        c2 = st.number_input("C2", min_value=0.0, value=1.0)
        c3 = st.number_input("C3", min_value=0.0, value=0.0)
        c4 = st.number_input("C4", min_value=0.0, value=0.0)
        c5 = st.number_input("C5", min_value=0.0, value=0.0)
        c6 = st.number_input("C6", min_value=0.0, value=0.0)
        c7 = st.number_input("C7", min_value=0.0, value=0.0)
        c8 = st.number_input("C8", min_value=0.0, value=0.0)
        c9 = st.number_input("C9", min_value=0.0, value=0.0)
        c10 = st.number_input("C10", min_value=0.0, value=0.0)

    st.markdown("**D Features**")

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)

    with dcol1:
        d1 = st.number_input("D1", min_value=0.0, value=1.0)

    with dcol2:
        d2 = st.number_input("D2", min_value=0.0, value=1.0)

    with dcol3:
        d3 = st.number_input("D3", min_value=0.0, value=1.0)

    with dcol4:
        d4 = st.number_input("D4", min_value=0.0, value=1.0)

    st.markdown("**M Features**")

    mcol1, mcol2, mcol3 = st.columns(3)

    with mcol1:
        m1 = st.selectbox("M1", ["T", "F", ""])

    with mcol2:
        m2 = st.selectbox("M2", ["T", "F", ""])

    with mcol3:
        m3 = st.selectbox("M3", ["T", "F", ""])

    mcol4, mcol5, mcol6 = st.columns(3)

    with mcol4:
        m4 = st.selectbox("M4", ["M0", "M1", "M2", ""])

    with mcol5:
        m5 = st.selectbox("M5", ["T", "F", ""])

    with mcol6:
        m6 = st.selectbox("M6", ["T", "F", ""])

    mcol7, mcol8, mcol9 = st.columns(3)

    with mcol7:
        m7 = st.selectbox("M7", ["T", "F", ""])

    with mcol8:
        m8 = st.selectbox("M8", ["T", "F", ""])

    with mcol9:
        m9 = st.selectbox("M9", ["T", "F", ""])

    st.markdown("---")

    check_transaction = st.button(
        "🔍 Check Transaction",
        type="primary",
        use_container_width=True,
    )

    if check_transaction:

        payload = {
            "TransactionAmt": transaction_amt,
            "ProductCD": product_cd,
            "card1": card1,
            "card2": card2,
            "card3": card3,
            "card4": card4,
            "card5": card5,
            "card6": card6,
            "addr1": addr1,
            "addr2": addr2,
            "dist1": dist1,
            "P_emaildomain": p_email or None,
            "R_emaildomain": r_email or None,
            "DeviceType": device_type,
            "DeviceInfo": device_info,
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": c4,
            "C5": c5,
            "C6": c6,
            "C7": c7,
            "C8": c8,
            "C9": c9,
            "C10": c10,
            "D1": d1,
            "D2": d2,
            "D3": d3,
            "D4": d4,
            "M1": m1 or None,
            "M2": m2 or None,
            "M3": m3 or None,
            "M4": m4 or None,
            "M5": m5 or None,
            "M6": m6 or None,
            "M7": m7 or None,
            "M8": m8 or None,
            "M9": m9 or None,
        }

        try:

            with st.spinner("Calling production fraud model..."):

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=30,
                )

            if response.status_code == 200:

                result = response.json()

                probability = float(
                    result["fraud_probability"]
                )

                prediction = int(
                    result["prediction"]
                )

                threshold = float(
                    result.get("threshold", 0.78)
                )

                st.markdown("### Prediction Result")

                r1, r2, r3 = st.columns(3)

                with r1:

                    if prediction == 1:
                        st.error("🚨 FRAUD DETECTED")
                    else:
                        st.success("✅ LEGITIMATE")

                with r2:

                    st.metric(
                        "Fraud Probability",
                        f"{probability * 100:.2f}%",
                    )

                with r3:

                    st.metric(
                        "Decision Threshold",
                        f"{threshold * 100:.0f}%",
                    )

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

                st.caption(
                    "Prediction generated by the production "
                    "XGBoost model through FastAPI."
                )

            else:

                st.error(
                    f"API returned HTTP {response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Make sure the fraud-api Docker container is running."
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )
# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Fraud Detection Monitoring Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Production ML monitoring • FastAPI • SQLite • MLflow • XGBoost'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# CURRENT MODEL INFORMATION
# ============================================================

if not predictions.empty:

    latest_prediction = predictions.iloc[-1]

    model_name = latest_prediction.get(
        "model_name",
        "Unknown"
    )

    model_version = latest_prediction.get(
        "model_version",
        "Unknown"
    )

    model_alias = latest_prediction.get(
        "model_alias",
        "Unknown"
    )

else:

    model_name = "Unknown"
    model_version = "Unknown"
    model_alias = "Unknown"


st.info(
    f"Production Model: **{model_name}**  |  "
    f"Version: **{model_version}**  |  "
    f"Alias: **{model_alias}**"
)


# ============================================================
# CURRENT METRICS
# ============================================================

if predictions.empty:

    st.warning(
        "No prediction records available."
    )

else:

    total_predictions = len(
        predictions
    )

    fraud_predictions = int(
        (
            predictions["prediction"] == 1
        ).sum()
    )

    legitimate_predictions = int(
        (
            predictions["prediction"] == 0
        ).sum()
    )

    fraud_rate = (
        fraud_predictions
        / total_predictions
        if total_predictions > 0
        else 0
    )

    avg_probability = predictions[
        "probability"
    ].mean()

    max_probability = predictions[
        "probability"
    ].max()

    avg_latency = predictions[
        "latency_ms"
    ].mean()

    max_latency = predictions[
        "latency_ms"
    ].max()


    st.subheader(
        "Current Production Metrics"
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Predictions",
            f"{total_predictions:,}"
        )

    with col2:

        st.metric(
            "Fraud Predictions",
            f"{fraud_predictions:,}"
        )

    with col3:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2%}"
        )

    with col4:

        st.metric(
            "Average Latency",
            f"{avg_latency:.2f} ms"
        )


    col5, col6, col7, col8 = st.columns(4)

    with col5:

        st.metric(
            "Legitimate",
            f"{legitimate_predictions:,}"
        )

    with col6:

        st.metric(
            "Average Probability",
            f"{avg_probability:.4f}"
        )

    with col7:

        st.metric(
            "Maximum Probability",
            f"{max_probability:.4f}"
        )

    with col8:

        st.metric(
            "Maximum Latency",
            f"{max_latency:.2f} ms"
        )


# ============================================================
# MONITORING STATUS
# ============================================================

st.subheader(
    "Monitoring Status"
)


status = get_latest_status(
    alert_runs
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Overall",
        f"{status_icon(status['overall'])} "
        f"{status['overall']}"
    )


with col2:

    st.metric(
        "Data Quality",
        f"{status_icon(status['quality'])} "
        f"{status['quality']}"
    )


with col3:

    st.metric(
        "Feature Drift",
        f"{status_icon(status['drift'])} "
        f"{status['drift']}"
    )


with col4:

    st.metric(
        "Latency",
        f"{status_icon(status['latency'])} "
        f"{status['latency']}"
    )


# ============================================================
# PREDICTION TRENDS
# ============================================================

if not predictions.empty:

    st.subheader(
        "Prediction Trends"
    )


    trend_df = predictions.copy()

    trend_df["time"] = trend_df[
        "timestamp"
    ].dt.floor("h")


    hourly = (
        trend_df
        .groupby("time")
        .agg(
            predictions=("id", "count"),
            fraud=("prediction", "sum"),
            avg_probability=("probability", "mean"),
            avg_latency=("latency_ms", "mean"),
        )
        .reset_index()
    )


    st.markdown(
        "**Prediction Volume**"
    )

    st.line_chart(
        hourly.set_index("time")[
            ["predictions", "fraud"]
        ]
    )


    st.markdown(
        "**Average Fraud Probability**"
    )

    st.line_chart(
        hourly.set_index("time")[
            ["avg_probability"]
        ]
    )


    st.markdown(
        "**Average API Latency**"
    )

    st.line_chart(
        hourly.set_index("time")[
            ["avg_latency"]
        ]
    )


# ============================================================
# PSI / DRIFT HISTORY
# ============================================================

# ============================================================
# PSI / DRIFT HISTORY
# ============================================================

st.subheader("Feature Drift History")

# Only these are actual monitored production features.
PSI_FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
]

if drift_runs.empty:

    st.info("No MLflow drift history available.")

else:

    drift_history = []

    for _, run in drift_runs.iterrows():

        start_time = run.get("start_time")

        for feature in PSI_FEATURES:

            metric_key = f"metrics.psi_{feature}"

            # Current naming convention
            psi = run.get(metric_key)

            # Support older runs that may use lowercase feature names
            if pd.isna(psi):

                psi = run.get(
                    f"metrics.psi_{feature.lower()}"
                )

            if pd.notna(psi):

                drift_history.append(
                    {
                        "timestamp": start_time,
                        "feature": feature,
                        "psi": float(psi),
                    }
                )

    if not drift_history:

        st.info("No valid PSI feature history available.")

    else:

        drift_df = pd.DataFrame(drift_history)

        drift_df["timestamp"] = pd.to_datetime(
            drift_df["timestamp"],
            errors="coerce",
        )

        drift_df = drift_df.dropna(
            subset=["timestamp", "psi"]
        )

        # Remove duplicate timestamp/feature combinations
        drift_df = drift_df.drop_duplicates(
            subset=["timestamp", "feature"],
            keep="last",
        )

        pivot_df = (
            drift_df
            .pivot(
                index="timestamp",
                columns="feature",
                values="psi",
            )
            .sort_index()
        )

        st.line_chart(pivot_df)

        st.markdown("**Latest PSI values**")

        latest_drift = (
            drift_df
            .sort_values("timestamp")
            .groupby("feature")
            .tail(1)
            .sort_values(
                "psi",
                ascending=False,
            )
        )

        st.dataframe(
            latest_drift[
                [
                    "feature",
                    "psi",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
# ============================================================
# ALERT HISTORY
# ============================================================

st.subheader(
    "Alert History"
)


if alert_runs.empty:

    st.info(
        "No MLflow alert history available."
    )

else:

    alert_history = []

    for _, run in alert_runs.iterrows():

        overall = run.get(
            "tags.overall_status",
            "UNKNOWN"
        )

        quality = run.get(
            "tags.data_quality_status",
            "UNKNOWN"
        )

        drift = run.get(
            "tags.drift_status",
            "UNKNOWN"
        )

        latency_status = run.get(
            "tags.latency_status",
            "UNKNOWN"
        )

        max_psi = run.get(
            "metrics.max_psi",
            None
        )

        avg_latency_value = run.get(
        "metrics.avg_latency_ms"
    )

    if pd.isna(avg_latency_value):

        avg_latency_value = run.get(
            "metrics.average_latency_ms"
        )

    if pd.isna(avg_latency_value):

        avg_latency_value = run.get(
            "avg_latency_ms"
        )

    if pd.notna(avg_latency_value):

        avg_latency_value = float(
            avg_latency_value
        )


        alert_history.append(
            {
                "timestamp": run.get(
                    "start_time"
                ),
                "overall": overall,
                "data_quality": quality,
                "drift": drift,
                "latency": latency_status,
                "max_psi": max_psi,
                "avg_latency_ms": avg_latency_value,
            }
        )


    alert_df = pd.DataFrame(
        alert_history
    )


    if not alert_df.empty:

        alert_df["timestamp"] = pd.to_datetime(
            alert_df["timestamp"],
            errors="coerce"
        )


        st.dataframe(
            alert_df[
                [
                    "timestamp",
                    "overall",
                    "data_quality",
                    "drift",
                    "latency",
                    "max_psi",
                    "avg_latency_ms",
                ]
            ].head(20),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# LATEST PREDICTIONS
# ============================================================

st.subheader(
    "Recent Predictions"
)


if not predictions.empty:

    recent = (
        predictions
        .sort_values(
            "id",
            ascending=False
        )
        .head(20)
        .copy()
    )


    display_columns = [
        "id",
        "timestamp",
        "probability",
        "prediction",
        "threshold",
        "result",
        "latency_ms",
    ]


    st.dataframe(
        recent[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.subheader(
    "System Information"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.write(
        "**Database**"
    )

    st.code(
        DB_PATH
    )


with col2:

    st.write(
        "**MLflow**"
    )

    st.code(
        MLFLOW_TRACKING_URI
    )


with col3:

    st.write(
        "**Dashboard Refresh**"
    )

    st.code(
        "30 seconds"
    )


# ============================================================
# LAST UPDATE
# ============================================================

st.caption(
    f"Dashboard generated: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)


# ============================================================
# AUTO REFRESH
# ============================================================

try:

    @st.fragment(
        run_every="30s"
    )
    def refresh_indicator():

        st.caption(
            "🔄 Dashboard automatically refreshes every 30 seconds."
        )


    refresh_indicator()

except Exception:

    pass