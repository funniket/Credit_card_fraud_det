# app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler from pickle files
try:
    with open('fraud_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error("Error loading model or scaler. Make sure they are in the project directory.")
    st.stop()

# App Title and Description
st.title("Card Transaction Fraud Detection")
st.markdown("""
This app predicts whether a card transaction is fraudulent or not.
Please fill in the details of the transaction below.
""")

# Create a form for user input
with st.form(key='prediction_form'):
    st.header("Transaction Details")
    
    # Using columns to group related inputs for better layout
    col1, col2 = st.columns(2)
    with col1:
        distance_from_home = st.number_input("Distance from Home (in km)", value=0.0, min_value=0.0)
        ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", value=0.0, min_value=0.0)
        repeat_retailer = st.selectbox("Repeat Retailer", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        distance_from_last_transaction = st.number_input("Distance from Last Transaction (in km)", value=0.0, min_value=0.0)
        used_chip = st.selectbox("Used Chip", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        used_pin_number = st.selectbox("Used PIN Number", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Single element input below the columns
    online_order = st.selectbox("Online Order", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Combine inputs into an array in the same order as expected by the model
    features = [
        distance_from_home,
        distance_from_last_transaction,
        ratio_to_median_purchase_price,
        repeat_retailer,
        used_chip,
        used_pin_number,
        online_order
    ]
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the input features
    features_scaled = scaler.transform(features_array)
    
    # Predict using the trained model
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)
    
    # Display results with enhanced formatting
    st.markdown("## Prediction Results")
    if prediction[0] == 1:
        st.error("**Result:** The transaction is predicted as **FRAUDULENT**!")
    else:
        st.success("**Result:** The transaction is predicted as **NON-FRAUDULENT**!")
    
    st.markdown("### Prediction Probabilities")
    st.write(f"**Non-Fraudulent (Class 0):** {prediction_proba[0][0]*100:.2f}%")
    st.write(f"**Fraudulent (Class 1):** {prediction_proba[0][1]*100:.2f}%")
