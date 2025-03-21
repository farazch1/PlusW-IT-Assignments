import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model data
model_data = joblib.load('stock_model.joblib')
model = model_data['model']
features = model_data['features']

st.title('Stock Price Prediction App')
st.write("Enter the values for the following features to predict the stock price:")

# Create input fields for each feature (assumed to be numeric)
# Adjust default values as needed
day_input = st.number_input("Day", min_value=0, value=100)
volume_input = st.number_input("Volume", min_value=0, value=1000000)
ma5_input = st.number_input("5-Day Moving Average (MA5)", value=150.0)

if st.button("Predict"):
    # Arrange user input to match the model's expected feature order
    input_data = np.array([day_input, volume_input, ma5_input]).reshape(1, -1)
    prediction = model.predict(input_data)
    st.write(f"Predicted Stock Price: ${prediction[0]:.2f}")
