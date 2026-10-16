import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="E-Commerce Churn Predictor", page_icon="🛍️")

st.title("🛍️ Customer Churn Risk Assessment")
st.write("Enter customer engagement metrics to predict churn probability.")

# Auto-train and cache model if file doesn't exist
@st.cache_resource
def load_or_train_model():
    model_file = 'churn_model.pkl'
    if not os.path.exists(model_file):
        df = pd.read_csv("ecommerce_churn_dataset.csv")
        X = df.drop(['Customer_ID', 'Churn_Target'], axis=1)
        y = df['Churn_Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_file)
    return joblib.load(model_file)

try:
    model = load_or_train_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
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
