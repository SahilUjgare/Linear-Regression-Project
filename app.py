import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# Load Model and Data
# =========================
MODEL_PATH = "/mnt/data/Best_model (2).pkl"
DATA_PATH = "/mnt/data/auto-mpg.csv"

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load dataset for default ranges
df = pd.read_csv(DATA_PATH)

st.title("🚗 Auto MPG Prediction App")
st.write("Enter vehicle specifications below to predict fuel efficiency (MPG).")

# =========================
# User Inputs
# =========================
col1, col2 = st.columns(2)

with col1:
    cylinders = st.number_input("Cylinders", min_value=3, max_value=12, value=4)
    displacement = st.number_input("Displacement", min_value=60.0, max_value=500.0, value=150.0)
    horsepower = st.number_input("Horsepower", min_value=40.0, max_value=250.0, value=90.0)

with col2:
    weight = st.number_input("Weight (lbs)", min_value=1000.0, max_value=6000.0, value=2500.0)
    acceleration = st.number_input("Acceleration (0–60 mph time)", min_value=5.0, max_value=30.0, value=15.0)
    model_year = st.number_input("Model Year", min_value=70, max_value=82, value=76)

origin = st.selectbox("Origin", ["usa", "europe", "japan"])

# =========================
# Prepare Input
# =========================
input_data = pd.DataFrame({
    "cylinders": [cylinders],
    "displacement": [displacement],
    "horsepower": [horsepower],
    "weight": [weight],
    "acceleration": [acceleration],
    "model year": [model_year],
    "origin": [origin]
})

# =========================
# Prediction Button
# =========================
if st.button("Predict MPG"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted MPG: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
