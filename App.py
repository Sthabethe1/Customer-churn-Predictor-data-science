import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

st.title("📊 Customer Churn Prediction App")
st.write("Enter the customer details below to predict the likelihood of churn.")
st.markdown("---")

st.subheader("Customer Information")
age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
gender = st.selectbox("Gender", options=["Male", "Female"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12, step=1)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)

gender_encoded = 1 if gender == "Female" else 0

st.markdown("---")
if st.button("Predict Churn Status", type="primary"):
    input_data = pd.DataFrame(
        [[age, gender_encoded, tenure, monthly_charges]],
        columns=["Age", "Gender", "Tenure", "MonthlyCharges"]
    )
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠ **High Risk of Churn!** (Probability: {probability * 100:.1f}%)")
    else:
        st.success(f"✅ **Low Risk / Retention Likely** (Churn Probability: {probability * 100:.1f}%)")
