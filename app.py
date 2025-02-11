# app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler from the pickle files
with open('fraud_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Card Transaction Fraud Detection")

st.write("Enter the transaction details below to predict if it is fraudulent.")

# Create a form for user input
with st.form(key='prediction_form'):
    distance_from_home = st.number_input("Distance from Home", value=0.0)
    distance_from_last_transaction = st.number_input("Distance from Last Transaction", value=0.0)
    ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", value=0.0)
    repeat_retailer = st.number_input("Repeat Retailer (0 for No, 1 for Yes)", value=0.0)
    used_chip = st.number_input("Used Card (0 for No, 1 for Yes)", value=0.0)
    used_pin_number = st.number_input("Used PIN Number (0 for No, 1 for Yes)", value=0.0)
    online_order = st.number_input("Online Order (0 for No, 1 for Yes)", value=0.0)
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Combine user inputs into an array in the same order as the training features
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
    
    # Scale the input features using the saved StandardScaler
    features_scaled = scaler.transform(features_array)
    
    # Predict the fraud class and get the probabilities
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)
    
    # Display the prediction results
    if prediction[0] == 1:
        st.error("The transaction is predicted as FRAUDULENT!")
    else:
        st.success("The transaction is predicted as NON-FRAUDULENT!")
    
    st.write("Prediction Probabilities:")
    st.write(f"Non-Fraudulent (Class 0): {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Fraudulent (Class 1): {prediction_proba[0][1]*100:.2f}%")
