import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("heart_disease_scaler.pkl", "rb"))

st.title("🫀 Cardiovascular Disease Prediction")
st.markdown("Enter patient information below to check heart disease risk.")

# Input fields
age = st.slider("Age (years)", 18, 100, 50)
gender = st.selectbox("Gender", ["Female", "Male"])
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=80, max_value=250, value=120)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=50, max_value=150, value=80)
cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
smoke = st.selectbox("Smokes?", ["No", "Yes"])
alco = st.selectbox("Alcohol Intake?", ["No", "Yes"])
active = st.selectbox("Physically Active?", ["Yes", "No"])

# Convert inputs to model format
gender_val = 1 if gender == "Female" else 2
chol_val = ["Normal", "Above Normal", "Well Above Normal"].index(cholesterol) + 1
gluc_val = ["Normal", "Above Normal", "Well Above Normal"].index(gluc) + 1
smoke_val = 1 if smoke == "Yes" else 0
alco_val = 1 if alco == "Yes" else 0
active_val = 1 if active == "Yes" else 0

input_data = np.array([[age, gender_val, height, weight, ap_hi, ap_lo, chol_val, gluc_val, smoke_val, alco_val, active_val]])
scaled_data = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(scaled_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk: This person is likely to have cardiovascular disease.")
    else:
        st.success("✅ Low Risk: This person is unlikely to have cardiovascular disease.")
