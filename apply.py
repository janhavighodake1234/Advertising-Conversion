import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and the scaler
model = pickle.load(open("advertisement_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Advertising Conversion Predictor")
st.write("Enter the campaign details to predict the number of **Approved Conversions**.")

# Input fields based on your dataset structure
col1, col2 = st.columns(2)

with col1:
    interest = st.number_input("Interest (Interest ID)", min_value=0, value=15)
    impressions = st.number_input("Impressions", min_value=0, value=7350)
    clicks = st.number_input("Clicks", min_value=0, value=1)
    spent = st.number_input("Spent ($)", min_value=0.0, value=1.43)
    total_conv = st.number_input("Total Conversions", min_value=0, value=2)

with col2:
    age = st.selectbox("Age Group", ["30-34", "35-39", "40-44", "45-49"])
    gender = st.selectbox("Gender", ["M", "F"])

# Derived features as defined in your notebook
ctr = (clicks / impressions * 100) if impressions > 0 else 0
conv_rate = (total_conv / clicks * 100) if clicks > 0 else 0

# One-hot encoding logic
age_35_39 = (age == "35-39")
age_40_44 = (age == "40-44")
age_45_49 = (age == "45-49")
gender_m = (gender == "M")

# Prepare feature vector (must match training input columns)
# Ensure columns match: ['ad_id', 'xyz_campaign_id', 'fb_campaign_id', 'interest', 
# 'Impressions', 'Clicks', 'Spent', 'Total_Conversion', 'CTR', 'Conversion_Rate', 
# 'age_35-39', 'age_40-44', 'age_45-49', 'gender_M']
# Note: Since ad_id/campaign_ids are low-importance in your model, we use default placeholders
input_data = pd.DataFrame({
    'ad_id': [0], 'xyz_campaign_id': [0], 'fb_campaign_id': [0],
    'interest': [interest], 'Impressions': [impressions], 'Clicks': [clicks],
    'Spent': [spent], 'Total_Conversion': [total_conv],
    'CTR': [ctr], 'Conversion_Rate': [conv_rate],
    'age_35-39': [age_35_39], 'age_40-44': [age_40_44], 'age_45-49': [age_45_49],
    'gender_M': [gender_m]
})

if st.button("Predict"):
    # Note: Your notebook used StandardScaler for Linear/KNN/SVR 
    # but not necessarily for Random Forest. 
    # If the model was trained on scaled data, uncomment the next line:
    # input_data = scaler.transform(input_data)
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Approved Conversions: {max(0, round(prediction[0], 2))}")