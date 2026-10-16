# 🛍️ E-Commerce Customer Churn Prediction System

An end-to-end Data Science project built to predict customer churn risk using Machine Learning (XGBoost) and serve real-time predictions via a Streamlit web application.

## 📌 Business Problem
Customer retention is vital for e-commerce growth. Acquiring new customers costs 5x more than retaining existing ones. This system identifies high-risk customers based on behavioral features like recency of purchase and support ticket complaints before they leave the platform.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deployment:** Streamlit Community Cloud

## 📊 Project Workflow
1. **Data Extraction & EDA:** Analyzed 10,000 customer records to find drivers of customer churn.
2. **Feature Engineering:** Extracted RFM (Recency, Frequency, Monetary) metrics and complaints frequency.
3. **Model Building:** Trained Baseline Logistic Regression & XGBoost Classifier.
4. **Deployment:** Built an interactive web app for non-technical business users.
