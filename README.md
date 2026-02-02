# TELCOM-CHURN-PREDICTION
Customer Churn Prediction using Random Forest & Streamlit
📌 Project Overview

Customer churn refers to customers leaving a service.
This project predicts whether a customer is likely to churn using Machine Learning and provides an interactive Streamlit UI for real-time and bulk predictions.

The system helps businesses identify high-risk customers early and take retention actions.

🎯 Objectives

Clean and preprocess customer churn data

Perform feature engineering and EDA

Train a Random Forest classifier

Save and load the trained ML model

Build an interactive Streamlit UI

Support single prediction + bulk CSV prediction

🧠 Machine Learning Model

Algorithm: Random Forest Classifier

Why Random Forest?

Handles non-linear data

Robust to noise

High accuracy

Provides feature importance

Target Variable: Churn (0 = No, 1 = Yes)


🧹 Data Preprocessing Steps

Converted TotalCharges to numeric

Removed missing and invalid values

Dropped irrelevant column (customerID)

Feature engineering:

Tenure groups

Average monthly spend

Encoded categorical variables

Removed infinite values

Generated final clean dataset

⚙️ Feature Engineering

TenureGroup: Customer loyalty buckets

AvgMonthlySpend: Spending behavior indicator

One-Hot Encoding: Contract, Internet Service, etc.

Binary Encoding: Yes/No → 1/0

🖥️ Streamlit UI Features (UNIQUE)

✔ Real-time churn probability
✔ Risk level indicator (Low / Medium / High)
✔ Feature importance visualization
✔ Bulk CSV prediction & download
✔ Safe feature alignment (no errors)
