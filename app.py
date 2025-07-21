import streamlit as st
import numpy as np
import pickle

# Title and Description
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("🌾 Smart Crop Recommendation System")
st.markdown("Provide soil and weather conditions to get the best crop suggestions.")

# Load Models
dt_model = pickle.load(open("dt_crop_model.pkl", "rb"))
rf_model = pickle.load(open("rf_crop_model.pkl", "rb"))

# Labels used in training (update based on your dataset!)
crop_labels = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
               'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 
               'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 
               'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 
               'coffee']  # example list

# User Inputs
st.header("Enter Soil and Weather Conditions:")
n = st.number_input("Nitrogen (N)", min_value=0)
p = st.number_input("Phosphorous (P)", min_value=0)
k = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)", format="%.2f")
humidity = st.number_input("Humidity (%)", format="%.2f")
ph = st.number_input("Soil pH", format="%.2f")
rainfall = st.number_input("Rainfall (mm)", format="%.2f")

# Predict Button
if st.button("Predict Top 3 Crops"):
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])

    # Decision Tree Prediction
    st.subheader("🌱 Decision Tree Recommendations:")
    if hasattr(dt_model, "predict_proba"):
        probs_dt = dt_model.predict_proba(input_data)[0]
        top3_dt = np.argsort(probs_dt)[-3:][::-1]
        for i, idx in enumerate(top3_dt, 1):
            st.markdown(f"**{i}. {crop_labels[idx].title()}** – Score: {round(probs_dt[idx]*100, 2)}%")

    # Random Forest Prediction
    st.subheader("🌳 Random Forest Recommendations:")
    if hasattr(rf_model, "predict_proba"):
        probs_rf = rf_model.predict_proba(input_data)[0]
        top3_rf = np.argsort(probs_rf)[-3:][::-1]
        for i, idx in enumerate(top3_rf, 1):
            st.markdown(f"**{i}. {crop_labels[idx].title()}** – Score: {round(probs_rf[idx]*100, 2)}%")

# Footer
st.markdown("---")
st.markdown("🔬 Built with ❤️ by Sanya Gupta")
