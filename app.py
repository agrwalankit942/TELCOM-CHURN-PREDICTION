import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction (RF)",
    page_icon="🌳",
    layout="centered"
)

st.title("🌳 Customer Churn Prediction App")
st.write("Random Forest based Machine Learning system")

# =========================
# Load Model & Assets
# =========================
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🧾 Customer Details")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 70.0)
avg_spend = st.sidebar.slider("Avg Monthly Spend", 0.0, 120.0, 65.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# =========================
# Encoding Maps
# =========================
contract_map = {
    "Month-to-month": [1, 0],
    "One year": [0, 1],
    "Two year": [0, 0]
}

internet_map = {
    "Fiber optic": [1, 0],
    "DSL": [0, 1],
    "No": [0, 0]
}

# =========================
# Create Input Data
# =========================
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "AvgMonthlySpend": [avg_spend],
    "Contract_Month-to-month": [contract_map[contract][0]],
    "Contract_One year": [contract_map[contract][1]],
    "InternetService_Fiber optic": [internet_map[internet][0]],
    "InternetService_DSL": [internet_map[internet][1]]
})

# 🔥 Align features with training
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# =========================
# Prediction
# =========================
if st.button("🔮 Predict Churn"):
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    st.progress(int(probability * 100))

    if probability < 0.30:
        st.success(f"✅ Low Churn Risk ({probability:.2%})")
    elif probability < 0.60:
        st.warning(f"⚠️ Medium Churn Risk ({probability:.2%})")
    else:
        st.error(f"❌ High Churn Risk ({probability:.2%})")

    # =========================
    # Feature Importance (UNIQUE)
    # =========================
    st.subheader("🧠 Important Factors")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).head(10)

    st.bar_chart(importance)

# =========================
# Bulk CSV Prediction
# =========================
st.markdown("---")
st.subheader("📂 Bulk Customer Prediction")

uploaded_file = st.file_uploader(
    "Upload cleaned customer CSV",
    type=["csv"]
)

if uploaded_file:
    bulk_df = pd.read_csv(uploaded_file)

    bulk_df = bulk_df.reindex(columns=feature_names, fill_value=0)
    bulk_scaled = scaler.transform(bulk_df)

    bulk_df["Churn_Probability"] = model.predict_proba(bulk_scaled)[:, 1]
    bulk_df["Churn_Prediction"] = (
        bulk_df["Churn_Probability"] > 0.5
    ).map({True: "Yes", False: "No"})

    st.dataframe(bulk_df.head())

    st.download_button(
        "⬇️ Download Predictions",
        bulk_df.to_csv(index=False),
        "rf_churn_predictions.csv"
    )

st.markdown("---")
st.caption("🌳 Random Forest Customer Churn Prediction System")
