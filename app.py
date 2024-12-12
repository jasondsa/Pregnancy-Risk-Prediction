import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# Load the preprocessing pipeline
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load the trained deep learning model
model = load_model("pregnancy_risk_model.h5")

# Prediction function
def predict_pregnancy_risk(
    age, weight, height, gestational_age, systolic_bp, diastolic_bp, 
    heart_rate, glucose_level, pre_existing_conditions, 
    diet_quality, exercise_level, smoking_status, mothers_education):
    
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        "Age": [age],
        "Weight": [weight],
        "Height": [height],
        "Gestational Age": [gestational_age],
        "systolic blood pressure": [systolic_bp],
        "diastolic blood pressure": [diastolic_bp],
        "Heart Rate": [heart_rate],
        "Glucose Level": [glucose_level],
        "Pre-existing Conditions": [pre_existing_conditions],
        "Diet Quality": [diet_quality],
        "Exercise Level": [exercise_level],
        "Smoking Status": [smoking_status],
        "Mother’s Education": [mothers_education]
    })

    # Preprocess the input data
    input_transformed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_transformed)
    risk = "High Risk" if prediction[0][0] > 0.5 else "Low Risk"
    
    return f"Prediction: {risk} ({prediction[0][0]:.2f})"

# Streamlit UI
st.title("Pregnancy Risk Prediction")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=30)
weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=70)
height = st.number_input("Height (cm)", min_value=0, max_value=300, value=165)
gestational_age = st.number_input("Gestational Age (weeks)", min_value=0, max_value=40, value=20)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, max_value=300, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=300, value=80)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=200, value=72)
glucose_level = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=90)

# Dropdowns for categorical inputs
pre_existing_conditions = st.selectbox("Pre-existing Conditions", ["None", "Diabetes", "Hypertension", "Other"])
diet_quality = st.selectbox("Diet Quality", ["Poor", "Moderate", "Good"])
exercise_level = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])
smoking_status = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"])
mothers_education = st.selectbox("Mother’s Education", ["No Education", "Primary School", "High School", "College Graduate"])

# Predict button
if st.button("Predict"):
    result = predict_pregnancy_risk(
        age, weight, height, gestational_age, systolic_bp, diastolic_bp, 
        heart_rate, glucose_level, pre_existing_conditions, 
        diet_quality, exercise_level, smoking_status, mothers_education)
    
    st.success(result)
