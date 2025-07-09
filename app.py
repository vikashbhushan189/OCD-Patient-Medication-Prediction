import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
# Make sure these files (voting_classifier_model.joblib, min_max_scaler.joblib) are in the same directory
try:
    model = joblib.load('voting_classifier_model.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    st.success("Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or Scaler files not found. Please ensure 'voting_classifier_model.joblib' and 'min_max_scaler.joblib' are in the same directory as this app.py file. You need to run your main training script once to generate these files.")
    st.stop() # Stop the app if files are missing

# Define the feature names in the exact order as your X_data_feature from training
# This is crucial for correct input mapping
feature_names = [
    'Age', 'Gender', 'Ethnicity', 'Marital Status', 'Education Level',
    'Duration of Symptoms (months)', 'Previous Diagnoses', 'Family History of OCD',
    'Obsession Type', 'Compulsion Type', 'Y-BOCS Score (Obsessions)',
    'Y-BOCS Score (Compulsions)', 'Depression Diagnosis', 'Anxiety Diagnosis'
]

# Define mappings for displaying categorical inputs
# This should match your preprocessing step's encoding
gender_map = {'Male': 2, 'Female': 1}
ethnicity_map = {'African': 1, 'Hispanic': 2, 'Asian': 3, 'Caucasian': 4}
marital_status_map = {'Single': 1, 'Divorced': 2, 'Married': 3}
education_map = {'Some College': 1, 'College Degree': 2, 'High School': 3, 'Graduate Degree': 4}
previous_diagnoses_map = {'MDD': 1, 'PTSD': 2, 'GAD': 3, 'Panic Disorder': 4, 'None/Missing': 0} # Assume 0 if original none/unknown
family_history_map = {'Yes': 2, 'No': 1}
obsession_map = {'Harm-related': 1, 'Contamination': 2, 'Symmetry': 3, 'Hoarding': 4, 'Religious': 5}
compulsion_map = {'Checking': 1, 'Washing': 2, 'Ordering': 3, 'Praying': 4, 'Counting': 5}
diagnosis_map = {'Yes': 2, 'No': 1} # For Depression and Anxiety

medication_classes = {0: 'SNRI', 1: 'SSRI', 2: 'Benzodiazepine', 3: 'Unknown'} # Maps back from model's integer prediction


# --- Streamlit App Layout ---
st.set_page_config(page_title="OCD Medication Predictor", layout="centered")

st.title("💊 OCD Medication Predictor")
st.markdown("Use this app to predict the likely medication for an OCD patient based on their demographic and clinical data.")

st.sidebar.header("Patient Input Data")

# Collect user input through sidebar widgets
def get_user_input():
    age = st.sidebar.slider("Age", 18, 75, 45)
    gender = st.sidebar.selectbox("Gender", options=list(gender_map.keys()))
    ethnicity = st.sidebar.selectbox("Ethnicity", options=list(ethnicity_map.keys()))
    marital_status = st.sidebar.selectbox("Marital Status", options=list(marital_status_map.keys()))
    education_level = st.sidebar.selectbox("Education Level", options=list(education_map.keys()))
    
    duration_symptoms = st.sidebar.slider("Duration of Symptoms (months)", 6, 240, 120)
    
    # Adjusted to allow None/Missing input for Previous Diagnoses
    previous_diagnoses = st.sidebar.selectbox("Previous Diagnoses (before OCD)", options=list(previous_diagnoses_map.keys()))
    
    family_history_ocd = st.sidebar.selectbox("Family History of OCD", options=list(family_history_map.keys()))
    obsession_type = st.sidebar.selectbox("Obsession Type", options=list(obsession_map.keys()))
    compulsion_type = st.sidebar.selectbox("Compulsion Type", options=list(compulsion_map.keys()))
    
    y_bocs_obsessions = st.sidebar.slider("Y-BOCS Score (Obsessions)", 0, 40, 20)
    y_bocs_compulsions = st.sidebar.slider("Y-BOCS Score (Compulsions)", 0, 40, 20)
    
    depression_diagnosis = st.sidebar.selectbox("Depression Diagnosis", options=list(diagnosis_map.keys()))
    anxiety_diagnosis = st.sidebar.selectbox("Anxiety Diagnosis", options=list(diagnosis_map.keys()))

    # Store all inputs in a dictionary, applying encoding
    data = {
        'Age': age,
        'Gender': gender_map[gender],
        'Ethnicity': ethnicity_map[ethnicity],
        'Marital Status': marital_status_map[marital_status],
        'Education Level': education_map[education_level],
        'Duration of Symptoms (months)': duration_symptoms,
        'Previous Diagnoses': previous_diagnoses_map[previous_diagnoses],
        'Family History of OCD': family_history_map[family_history_ocd],
        'Obsession Type': obsession_map[obsession_type],
        'Compulsion Type': compulsion_map[compulsion_type],
        'Y-BOCS Score (Obsessions)': y_bocs_obsessions,
        'Y-BOCS Score (Compulsions)': y_bocs_compulsions,
        'Depression Diagnosis': diagnosis_map[depression_diagnosis],
        'Anxiety Diagnosis': diagnosis_map[anxiety_diagnosis]
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

st.subheader("📋 Patient Input Summary")
st.write(input_df) # Show the raw encoded input


# --- Prediction ---
if st.button("Predict Medication"):
    # Ensure input_df has correct feature order and names (should from `get_user_input` itself)
    # The `input_df` is already structured with correct columns based on `feature_names` implicitly.
    
    # Convert input_df to NumPy array before scaling (scaler expects NumPy array or similar)
    input_np = input_df.values 

    # Scale the input data using the trained scaler
    scaled_input = scaler.transform(input_np) # Only transform, not fit_transform
    
    # Make prediction
    prediction_raw = model.predict(scaled_input)
    predicted_class_id = int(prediction_raw[0]) # Get the integer class ID from the prediction
    
    predicted_medication = medication_classes.get(predicted_class_id, "Unknown Class")

    st.subheader("🔮 Predicted Medication:")
    st.success(f"**The likely medication is: {predicted_medication}**")
    st.info(f"Model predicted class ID: {predicted_class_id}")
    st.balloons() # Little celebration animation