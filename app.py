import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="E-Commerce Churn Predictor", page_icon="🛍️")

st.title("🛍️ Customer Churn Risk Assessment")
st.write("Enter customer engagement metrics to predict churn probability.")

# Load Model
try:
    model = joblib.load('churn_model.pkl')
except:
    st.error("Model file 'churn_model.pkl' not found!")
    st.stop()

# User Inputs
col1, col2 = st.columns(2)
with col1:
    recency = st.number_input("Days Since Last Purchase (Recency)", min_value=1, max_value=365, value=30)
    frequency = st.number_input("Total Orders Placed (Frequency)", min_value=1, max_value=100, value=10)
with col2:
    monetary = st.number_input("Total Spend ($)", min_value=10.0, max_value=10000.0, value=500.0)
    tickets = st.slider("Support Tickets Raised", min_value=0, max_value=10, value=1)

if st.button("Predict Churn Risk", type="primary"):
    input_df = pd.DataFrame([[recency, frequency, monetary, tickets]], 
                            columns=['Recency_Days', 'Frequency_Orders', 'Monetary_Spend', 'Support_Tickets'])
    prob = model.predict_proba(input_df)[0][1] * 100
    pred = model.predict(input_df)[0]

    st.write("---")
    if pred == 1:
        st.error(f"⚠️ **High Churn Risk!** Probability: {prob:.1f}%")
        st.info("Action: Send retention discount coupon immediately.")
    else:
        st.success(f"✅ **Customer is Active.** Churn Probability: {prob:.1f}%")