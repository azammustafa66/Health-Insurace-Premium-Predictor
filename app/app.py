import streamlit as st
from helper import predict

st.set_page_config(page_title="Health Insurance Cost Predictor", layout="centered")
st.title("Health Insurance Cost Predictor")
st.write(
    "Fill in the details below and click Predict to estimate the insurance premium."
)

categorical_options = {
    "Gender": ["Male", "Female"],
    "Marital Status": ["Unmarried", "Married"],
    "BMI Category": ["Normal", "Obesity", "Overweight", "Underweight"],
    "Smoking Status": ["No Smoking", "Regular", "Occasional"],
    "Employment Status": ["Salaried", "Self-Employed", "Freelancer", ""],
    "Region": ["Northwest", "Southeast", "Northeast", "Southwest"],
    "Medical History": [
        "No Disease",
        "Diabetes",
        "High blood pressure",
        "Diabetes & High blood pressure",
        "Thyroid",
        "Heart disease",
        "High blood pressure & Heart disease",
        "Diabetes & Thyroid",
        "Diabetes & Heart disease",
    ],
    "Insurance Plan": ["Bronze", "Silver", "Gold"],
}

# Layout with columns
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    number_of_dependants = st.number_input(
        "Number of Dependants", min_value=0, max_value=20, value=0, step=1
    )
with col2:
    income_lakhs = st.number_input(
        "Income (in lakhs)", min_value=0.0, max_value=1000.0, value=10.0, step=1.0
    )
    genetical_risk = st.number_input(
        "Genetical Risk (0-5)", min_value=0, max_value=5, value=0, step=1
    )
with col3:
    insurance_plan = st.selectbox(
        "Insurance Plan", categorical_options["Insurance Plan"]
    )
    employment_status = st.selectbox(
        "Employment Status", categorical_options["Employment Status"]
    )

col4, col5, col6 = st.columns(3)
with col4:
    gender = st.selectbox("Gender", categorical_options["Gender"])
with col5:
    marital_status = st.selectbox(
        "Marital Status", categorical_options["Marital Status"]
    )
with col6:
    bmi_category = st.selectbox("BMI Category", categorical_options["BMI Category"])

col7, col8, col9 = st.columns(3)
with col7:
    smoking_status = st.selectbox(
        "Smoking Status", categorical_options["Smoking Status"]
    )
with col8:
    region = st.selectbox("Region", categorical_options["Region"])
with col9:
    medical_history = st.selectbox(
        "Medical History", categorical_options["Medical History"]
    )

input_dict = {
    "Age": age,
    "Number of Dependants": number_of_dependants,
    "Income in Lakhs": income_lakhs,
    "Genetical Risk": genetical_risk,
    "Insurance Plan": insurance_plan,
    "Employment Status": employment_status,
    "Gender": gender,
    "Marital Status": marital_status,
    "BMI Category": bmi_category,
    "Smoking Status": smoking_status,
    "Region": region,
    "Medical History": medical_history,
}

if st.button("Predict"):
    try:
        with st.spinner("Running model..."):
            pred = predict(input_dict)
        st.success(f"Predicted Health Insurance Cost: ₹{pred:,d}")
        st.caption(
            "Prediction is rounded to nearest integer and clipped to non-negative values."
        )
    except Exception as e:
        st.error("Prediction failed — see details below.")
        st.exception(e)
