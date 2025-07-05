import streamlit as st
import pandas as pd
import joblib

model = joblib.load('heart_disease_model.pkl')
features = joblib.load('model_features.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Risk Predictor")
st.markdown("Dr. Mendoza's simple tool to support early risk detection for adult patients")
st.markdown("---")

with st.form("risk_form"):
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    st.caption(" BMI Categories: Underweight <18.5 | Normal 18.5–24.9 | Overweight 25–29.9 | Obese ≥30")
    
    age = st.selectbox("Age Category", [
        '18-24', '25-29', '30-34', '35-39', '40-44',
        '45-49', '50-54', '55-59', '60-64', '65-69',
        '70-74', '75-79', '80 or older'
    ])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    physical_health = st.slider("Days Physically Unwell", 0, 30, 5)
    sleep_time = st.slider("Sleep Time (hrs/day)", 0, 24, 7)
    gen_health = st.selectbox("General Health", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
    st.caption("""
               **General Health Scale**  
               - **Excellent** – No known health issues, active, no limitations  
               - **Very good** – Minor issues, feels healthy most of the time  
               - **Good** – Occasional symptoms or mild limitations  
               - **Fair** – Noticeable health problems, lower energy  
               - **Poor** – Serious health issues, frequent medical attention
               """)
    diabetic = st.selectbox("Diabetic", ['Yes', 'No'])
    smoking = st.selectbox("Smoking", ['Yes', 'No'])
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'SleepTime': sleep_time,
        'Smoking_Yes': 1 if smoking == 'Yes' else 0,
        'Sex_Male': 1 if sex == 'Male' else 0,
        'Diabetic_Yes': 1 if diabetic == 'Yes' else 0,
        'GenHealth_Excellent': 1 if gen_health == 'Excellent' else 0,
        'GenHealth_Fair': 1 if gen_health == 'Fair' else 0,
        'GenHealth_Good': 1 if gen_health == 'Good' else 0,
        'GenHealth_Poor': 1 if gen_health == 'Poor' else 0,
        'GenHealth_Very good': 1 if gen_health == 'Very good' else 0,
        'AgeCategory_' + age: 1,
    }

    for col in features:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[features]
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("**AT RISK**")
    else:
        st.success("**NOT AT RISK**")

    st.markdown(f"**Confidence Score:** `{confidence:.2f}`")
