import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Page Config
st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="❤️", layout="wide")

# 2. Load Model

@st.cache_resource
def load_model():
    # Get the directory where app.py is located
    base_path = os.path.dirname(__file__)
    # Combine it with the filename
    model_path = os.path.join(base_path, 'heart_attack_risk_model.pkl')
    
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Header
st.title("❤️ Heart Attack Risk Prediction System")
st.markdown("Enter the patient's clinical and lifestyle data below to receive a risk assessment.")
st.divider()

# --- DATA ENTRY FORM ---
# We use an empty dict to store inputs as we go
inputs = {}
bin_map = {"Yes": 1, "No": 0}

# Section 1: Basic & Physical
st.header("👤 Section 1: Basic & Physical Info")
col1, col2, col3, col4 = st.columns(4)
with col1:
    inputs['age'] = st.slider("Age", 1, 100, 45)
with col2:
    inputs['gender'] = st.selectbox("Gender", options=['Male', 'Female'])
with col3:
    inputs['waist_circumference'] = st.number_input("Waist Circumference (cm)", 40, 200, 90)
with col4:
    obesity_choice = st.selectbox("Obesity Status", options=['Yes', 'No'])
    inputs['obesity'] = bin_map[obesity_choice]

st.divider()

# Section 2: Vitals & Lab Results
st.header("🩸 Section 2: Vitals & Lab Results")
vcol1, vcol2, vcol3 = st.columns(3)
with vcol1:
    inputs['blood_pressure_systolic'] = st.number_input("Systolic BP", 80, 250, 120)
    inputs['blood_pressure_diastolic'] = st.number_input("Diastolic BP", 40, 150, 80)
with vcol2:
    inputs['cholesterol_level'] = st.number_input("Total Cholesterol", 100, 600, 200)
    inputs['cholesterol_ldl'] = st.number_input("LDL Cholesterol", 20, 400, 100)
with vcol3:
    inputs['cholesterol_hdl'] = st.number_input("HDL Cholesterol", 10, 150, 50)
    inputs['triglycerides'] = st.number_input("Triglycerides", 50, 1000, 150)

st.divider()

# Section 3: Medical History
st.header("🏥 Section 3: Medical History")
mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    inputs['diabetes'] = bin_map[st.selectbox("Diabetes", options=['No', 'Yes'])]
    inputs['hypertension'] = bin_map[st.selectbox("Hypertension", options=['No', 'Yes'])]
with mcol2:
    inputs['family_history'] = bin_map[st.selectbox("Family History", options=['No', 'Yes'])]
    inputs['previous_heart_disease'] = bin_map[st.selectbox("Previous Heart Disease", options=['No', 'Yes'])]
with mcol3:
    inputs['fasting_blood_sugar'] = bin_map[st.selectbox("FBS > 120 mg/dl", options=['No', 'Yes'])]
    inputs['medication_usage'] = bin_map[st.selectbox("Currently on Medication", options=['No', 'Yes'])]
    inputs['EKG_results'] = st.selectbox("EKG Results", options=['Normal', 'Abnormal', 'ST-T wave abnormality'])

st.divider()

# Section 4: Lifestyle & Environment
st.header("🥗 Section 4: Lifestyle & Environment")
lcol1, lcol2, lcol3 = st.columns(3)
with lcol1:
    inputs['dietary_habits'] = st.selectbox("Dietary Habits", options=['Healthy', 'Average', 'Unhealthy'])
    inputs['physical_activity'] = st.selectbox("Physical Activity Level", options=['Low', 'Moderate', 'High'])
    inputs['smoking_status'] = st.selectbox("Smoking Status", options=['Never', 'Past', 'Current'])
with lcol2:
    inputs['sleep_hours'] = st.slider("Sleep Hours", 1, 15, 7)
    inputs['stress_level'] = st.slider("Stress Level (1-10)", 1, 10, 5)
    inputs['alcohol_consumption'] = st.selectbox("Alcohol Consumption", options=['Non-drinker', 'Low', 'Moderate', 'High'])
with lcol3:
    inputs['air_pollution_exposure'] = st.selectbox("Air Pollution Exposure", options=['Low', 'Medium', 'High'])
    inputs['income_level'] = st.selectbox("Income Level", options=['Low', 'Medium', 'High'])
    inputs['region'] = st.selectbox("Region", options=['North', 'South', 'East', 'West', 'Central'])
    inputs['participated_in_free_screening'] = bin_map[st.selectbox("Participated in Screening", options=['No', 'Yes'])]

st.divider()

# --- PREDICTION LOGIC ---
st.header("🔍 Prediction Results")

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs])

# Layout for button and result
res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    if st.button("Calculate Heart Attack Risk", type="primary", use_container_width=True):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        risk_score = float(probability[0][1]) # Fix float32 error

        with res_col2:
            if prediction[0] == 1:
                st.error(f"### Assessment: AT RISK")
                st.metric("Risk Probability", f"{risk_score:.2%}", delta="High Risk", delta_color="inverse")
                st.progress(risk_score)
                st.warning("Immediate consultation with a healthcare provider is recommended.")
            else:
                st.success(f"### Assessment: LOW RISK")
                st.metric("Risk Probability", f"{risk_score:.2%}", delta="Low Risk")
                st.progress(risk_score)
                st.info("The patient is in a low-risk category. Maintaining a healthy lifestyle is encouraged.")

with st.expander("📄 View Submitted Data Summary"):
    # Fix Arrow error by casting to string for display
    st.dataframe(input_df.T.rename(columns={0: 'Input Value'}).astype(str), use_container_width=True)

st.markdown("---")
st.caption("Machine Learning Prediction Demo | Tuned XGBoost Pipeline | Includes Smoking Status")