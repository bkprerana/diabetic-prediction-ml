import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🩺 Diabetes Prediction App")

st.markdown("Enter the patient's details below to predict the likelihood of diabetes:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

# Button to trigger prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")

# Query section
st.markdown("### 🤔 Ask a question about your prediction")
user_query = st.text_input("Type your query here (e.g., Why was I predicted as diabetic?)")

if user_query:
    # For simplicity, we provide canned responses here.
    # Advanced: Use LLM integration (OpenAI) for dynamic answers
    st.info("This feature is under development. Currently, please consult a medical professional for detailed explanations.")
