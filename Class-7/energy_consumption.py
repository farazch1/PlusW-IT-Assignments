# Task 1: Install the dependencies
# Run the following command in your terminal:
# pip install pandas numpy matplotlib seaborn scikit-learn streamlit

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Task 2: Collect energy usage dataset from Open Energy Data sources.
# Check if the dataset exists locally; if not, download it.
file_path = "owid-energy-data.csv"
if not os.path.exists(file_path):
    st.write("Dataset not found locally. Downloading dataset...")
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    df = pd.read_csv(url)
    df.to_csv(file_path, index=False)
else:
    df = pd.read_csv(file_path)

# Display dataset columns and the first few rows as part of EDA.
st.write("Dataset Columns:", df.columns.tolist())
st.write("First 5 rows of the dataset:")
st.dataframe(df.head())

# Task 3: Perform exploratory data analysis (EDA) to understand trends.
st.write("### Summary Statistics")
st.write(df.describe())

# Plot a correlation heatmap for numerical features.
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.write("### Correlation Heatmap")
st.pyplot(fig_corr)

# Task 4: Prepare features including temperature and time-based features.
# Convert 'date' column (if available) to datetime and extract year/month.
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

# If no temperature column exists, create a synthetic temperature feature for demonstration.
if 'temperature' not in df.columns:
    np.random.seed(42)
    df['temperature'] = np.random.uniform(low=15, high=35, size=len(df))

# Identify the energy consumption column (case insensitive search).
energy_columns = [col for col in df.columns if "consumption" in col.lower()]
if not energy_columns:
    raise KeyError("No column related to energy consumption found in the dataset.")
energy_column = energy_columns[0]
st.write(f"Using '{energy_column}' as the target variable.")

# Drop rows with missing values.
df.dropna(inplace=True)

# One-hot encode categorical variables if present.
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target.
# For demonstration, we'll focus on temperature and time-based features (year and month) if available.
X = df.drop(columns=[energy_column])
y = df[energy_column]

# Restrict features to 'temperature', 'year', and 'month' if they exist.
relevant_features = []
for feat in ['temperature', 'year', 'month']:
    if feat in X.columns:
        relevant_features.append(feat)
if relevant_features:
    X = X[relevant_features]

# Split the dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using sklearn's LinearRegression (closed-form solution).
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Task 5: Optimize the model with Gradient Descent (custom implementation).
def gradient_descent(X, y, lr=0.0001, n_iterations=10000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for i in range(n_iterations):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
        dw = (1/m) * np.dot(X.T, error)
        db = (1/m) * np.sum(error)
        weights -= lr * dw
        bias -= lr * db
    return weights, bias

# Train using Gradient Descent on training data.
weights_gd, bias_gd = gradient_descent(X_train.values, y_train.values, lr=0.0001, n_iterations=10000)
y_pred_gd = np.dot(X_test.values, weights_gd) + bias_gd
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

# Task 6: Deploy a simple dashboard to display energy usage trends.
st.title('Energy Consumption Prediction Dashboard')

st.write("### Model Performance")
st.write(f"**Sklearn Linear Regression**: MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")
st.write(f"**Gradient Descent**: MSE: {mse_gd:.2f}, R²: {r2_gd:.2f}")

# Visualization: Scatter plot for sklearn model predictions.
fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(y_test, y_pred_lr, alpha=0.5, color='blue')
ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
ax_scatter.set_xlabel('Actual Energy Consumption')
ax_scatter.set_ylabel('Predicted Energy Consumption')
ax_scatter.set_title('Sklearn Linear Regression Predictions')
st.pyplot(fig_scatter)

# Visualization: Trend of energy consumption over time if time-based data exists.
if 'year' in df.columns:
    consumption_trend = df.groupby('year')[energy_column].mean().reset_index()
    st.write("### Energy Consumption Trend Over Years")
    fig_trend, ax_trend = plt.subplots()
    ax_trend.plot(consumption_trend['year'], consumption_trend[energy_column], marker='o')
    ax_trend.set_xlabel('Year')
    ax_trend.set_ylabel('Average Energy Consumption')
    st.pyplot(fig_trend)

# Task 7 & 8: The Streamlit app compiles, runs, and prints the output.
# (Run the app using: streamlit run your_script_name.py)

# Sidebar for user input predictions.
st.sidebar.header('Predict Energy Consumption')
model_choice = st.sidebar.selectbox("Choose Model", options=["Sklearn Linear Regression", "Gradient Descent"])
features_input = {}
for col in X.columns:
    features_input[col] = st.sidebar.number_input(f'Enter {col}:', float(X[col].min()), float(X[col].max()))

if st.sidebar.button('Predict'):
    input_data = np.array([features_input[col] for col in X.columns]).reshape(1, -1)
    if model_choice == "Sklearn Linear Regression":
        prediction = model_lr.predict(input_data)[0]
    else:
        prediction = np.dot(input_data, weights_gd) + bias_gd
        prediction = prediction[0] if hasattr(prediction, '__iter__') else prediction
    st.sidebar.write(f'Predicted Energy Consumption: {prediction:.2f}')
