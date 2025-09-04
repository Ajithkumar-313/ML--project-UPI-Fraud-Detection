 # app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
from io import BytesIO

st.set_page_config(page_title="UPI Fraud Detection", page_icon="🛡️", layout="centered")
st.title("UPI Fraud Detection GUI")

# --------- Utilities ---------
EXPECTED_COLUMNS = ["date", "payment_gateway", "transaction_type", "state", "amount"]

@st.cache_resource  # cache shared resources like ML models
def load_model():
    try:
        return joblib.load("models/model.pkl")
    except Exception:
        return None  # fallback to simple rule-based logic if no model file

def sample_template_df():
    return pd.DataFrame({
        "date": [date.today().isoformat()],
        "payment_gateway": ["UPI"],
        "transaction_type": ["Pay"],
        "state": ["Maharashtra"],
        "amount": ,   # <-- FIXED: provide a value
    })

def validate_and_coerce(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist; add missing with defaults
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # Coerce types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["payment_gateway"] = df["payment_gateway"].astype(str)
    df["transaction_type"] = df["transaction_type"].astype(str)
    df["state"] = df["state"].astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df[EXPECTED_COLUMNS]

def predict_df(model, df: pd.DataFrame) -> np.ndarray:
    # Fallback heuristic if no model is present
    if model is None:
        high_amt = (df["amount"].fillna(0) > 300000)
        failed_state = df["state"].str.lower().str.contains("failed", na=False)
        labels = np.where(high_amt | failed_state, "Fraud", "Not Fraud")
        return labels
    # If a model exists, adapt preprocessing to match training pipeline
    X = df.copy()
    preds = model.predict(X)
    if isinstance(preds, (list, tuple)):
        preds = np.array(preds)
    # Convert numeric labels into human-readable
    if getattr(preds, "dtype", None) is not None and preds.dtype.kind in {"i", "u", "f"}:
        preds = np.where(preds == 1, "Fraud", "Not Fraud")
    return preds

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# --------- UI: Tabs ---------
tab_single, tab_bulk = st.tabs(["Single Transaction", "Bulk CSV"])

with tab_single:
    st.subheader("Check a single transaction")
    c1, c2 = st.columns(2)
    tx_date = c1.date_input("Transaction date", value=date.today())
    gateway = c2.selectbox("Payment gateway", ["UPI", "NetBanking", "Card", "Wallet"])
    tx_type = c1.selectbox("Transaction type", ["Pay", "Collect", "Refund", "P2P", "Merchant"])
    state = c2.selectbox(
        "State",
        ["Maharashtra", "Karnataka", "Delhi", "Gujarat", "Tamil Nadu", "Goa", "West Bengal", "Uttar Pradesh"],
    )
    amount = st.number_input(
        "Amount (₹)",
        min_value=1.0,
        max_value=500000.0,  # practical demo cap
        step=1.0,
        help="Demo cap: up to ₹5,00,000",
    )

    if st.button("Check transaction"):
        one = pd.DataFrame({
            "date": [tx_date],
            "payment_gateway": [gateway],
            "transaction_type": [tx_type],
            "state": [state],
            "amount": [amount],
        })
        one = validate_and_coerce(one)
        model = load_model()
        label = predict_df(model, one)  # <-- FIXED: take first row's prediction
        if label == "Fraud":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

with tab_bulk:
    st.subheader("Bulk prediction from CSV")
    st.write("1) Download the template, 2) Fill rows, 3) Upload the CSV to get predictions, 4) Download results.")
    # Download template
    template_df = sample_template_df()
    st.download_button(
        label="Download template CSV",
        data=to_csv_bytes(template_df),
        file_name="upi_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            df = validate_and_coerce(df)
            model = load_model()
            preds = predict_df(model, df)
            out = df.copy()
            out["prediction"] = preds
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button(
                label="Download results CSV",
                data=to_csv_bytes(out),
                file_name="upi_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not process the uploaded file: {e}")
