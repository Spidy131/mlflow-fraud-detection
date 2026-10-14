import os

import requests
import streamlit as st

st.set_page_config(
    page_title="AI Fraud Detection",
    layout="wide",
)

st.title("AI-Powered Fraud Detection")
st.caption(
    "Production XGBoost model | "
    "74 engineered features | "
    "Production threshold: 0.78"
)

st.info(
    "Demo application: enter a transaction to request a prediction "
    "from the configured FastAPI service."
)

api_url = os.getenv("API_BASE_URL", "").strip()

if not api_url:
    try:
        api_url = str(st.secrets["API_BASE_URL"]).strip()
    except Exception:
        api_url = ""

api_url = api_url.rstrip("/")

with st.sidebar:
    st.header("API Configuration")
    st.write(
        "Configure API_BASE_URL in Streamlit Cloud Secrets "
        "after deploying your API."
    )

    if api_url:
        st.code(api_url)
        if st.button("Check API health", use_container_width=True):
            try:
                response = requests.get(
                    f"{api_url}/",
                    timeout=15,
                )
                response.raise_for_status()
                st.success("API is reachable.")
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"API health check failed: {exc}")
    else:
        st.warning("Prediction API URL is not configured yet.")

with st.form("transaction_form"):
    st.subheader("Transaction details")

    st.markdown("#### Transaction and card")

    c1, c2, c3 = st.columns(3)

    with c1:
        transaction_amt = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )
        product_cd = st.selectbox(
            "Product code",
            ["W", "C", "R", "H", "S"],
        )
        card1 = st.number_input(
            "Card 1",
            min_value=0,
            value=10000,
            step=1,
        )

    with c2:
        card2 = st.number_input(
            "Card 2 (0 means missing)",
            min_value=0.0,
            value=0.0,
        )
        card3 = st.number_input(
            "Card 3 (0 means missing)",
            min_value=0.0,
            value=150.0,
        )
        card4 = st.selectbox(
            "Card brand",
            ["visa", "mastercard", "discover", "american express"],
        )

    with c3:
        card5 = st.number_input(
            "Card 5 (0 means missing)",
            min_value=0.0,
            value=226.0,
        )
        card6 = st.selectbox(
            "Card type",
            ["debit", "credit", "debit or credit", "charge card"],
        )

    st.markdown("#### Address and email")

    c1, c2, c3 = st.columns(3)

    with c1:
        addr1 = st.number_input(
            "Address 1 (0 means missing)",
            min_value=0.0,
            value=315.0,
        )
        addr2 = st.number_input(
            "Address 2 (0 means missing)",
            min_value=0.0,
            value=87.0,
        )

    with c2:
        dist1 = st.number_input(
            "Distance (0 means missing)",
            min_value=0.0,
            value=0.0,
        )
        p_email = st.text_input(
            "Purchaser email domain",
            value="gmail.com",
        )

    with c3:
        r_email = st.text_input(
            "Recipient email domain",
            value="gmail.com",
        )
        device_type = st.selectbox(
            "Device type",
            ["", "desktop", "mobile"],
        )

    device_info = st.text_input(
        "Device information (optional)",
        value="",
    )

    st.markdown("#### C and D features")

    c_cols = [
        "C1", "C2", "C3", "C4", "C5",
        "C6", "C7", "C8", "C9", "C10",
    ]
    d_cols = ["D1", "D2", "D3", "D4"]

    c_values = {}
    d_values = {}

    columns = st.columns(5)
    for index, name in enumerate(c_cols):
        with columns[index % 5]:
            c_values[name] = st.number_input(
                name,
                min_value=0.0,
                value=0.0,
                key=f"input_{name}",
            )

    columns = st.columns(4)
    for index, name in enumerate(d_cols):
        with columns[index]:
            d_values[name] = st.number_input(
                name,
                min_value=0.0,
                value=0.0,
                key=f"input_{name}",
            )

    st.markdown("#### M features")

    m_cols = [
        "M1", "M2", "M3", "M4", "M5",
        "M6", "M7", "M8", "M9",
    ]

    m_values = {}
    columns = st.columns(3)

    for index, name in enumerate(m_cols):
        with columns[index % 3]:
            m_values[name] = st.selectbox(
                name,
                ["", "T", "F", "M0"],
                key=f"input_{name}",
            )

    submitted = st.form_submit_button(
        "Analyze transaction",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not api_url:
        st.error(
            "The API is not configured. Deploy the FastAPI service "
            "and set API_BASE_URL in Streamlit Cloud Secrets first."
        )
    else:
        payload = {
            "TransactionAmt": transaction_amt,
            "ProductCD": product_cd,
            "card1": int(card1),
            "card2": card2 if card2 > 0 else None,
            "card3": card3 if card3 > 0 else None,
            "card4": card4,
            "card5": card5 if card5 > 0 else None,
            "card6": card6,
            "addr1": addr1 if addr1 > 0 else None,
            "addr2": addr2 if addr2 > 0 else None,
            "dist1": dist1 if dist1 > 0 else None,
            "P_emaildomain": p_email.strip() or None,
            "R_emaildomain": r_email.strip() or None,
            "DeviceType": device_type or None,
            "DeviceInfo": device_info.strip() or None,
        }

        payload.update(c_values)
        payload.update(d_values)

        for name, value in m_values.items():
            payload[name] = value or None

        with st.spinner("Requesting prediction from the production API..."):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

                st.subheader("Prediction result")

                fraud_probability = float(
                    result["fraud_probability"]
                )
                threshold = float(result["threshold"])
                prediction = int(result["prediction"])

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Fraud probability",
                        f"{fraud_probability:.2%}",
                    )

                with col2:
                    st.metric(
                        "Decision threshold",
                        f"{threshold:.0%}",
                    )

                with col3:
                    st.metric(
                        "Prediction",
                        "Fraud" if prediction == 1 else "Legitimate",
                    )

                if prediction == 1:
                    st.error(
                        "⚠️ The model classified this transaction as FRAUD."
                    )
                else:
                    st.success(
                        "✅ The model classified this transaction as LEGITIMATE."
                    )

                st.caption(
                    "This is a model prediction, not a guarantee "
                    "that a transaction is fraudulent or legitimate."
                )

                with st.expander("View API response"):
                    st.json(result)

            except requests.HTTPError:
                st.error(
                    f"API returned HTTP {response.status_code}."
                )
                st.code(response.text[:2000])

            except requests.RequestException as exc:
                st.error(f"Could not contact the prediction API: {exc}")

            except (KeyError, ValueError, TypeError) as exc:
                st.error(f"Unexpected prediction response: {exc}")
