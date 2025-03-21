import streamlit as st
import pandas as pd
import joblib

# Load the saved model data (ensure 'salary_model.joblib' is in your working directory)
model_data = joblib.load('salary_model.joblib')
model = model_data['model']
features = model_data['features']

st.title("Salary Prediction App")
st.write("Enter the values for each feature:")

# Create input fields for each feature (assuming they are numeric)
user_input = {}
for feature in features:
    # Adjust default values if needed; here we use 0.0 as default for simplicity
    user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

if st.button("Predict Salary"):
    # Convert user input into a DataFrame with a single row
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)
    st.write(f"Predicted Salary: {prediction[0]}")
