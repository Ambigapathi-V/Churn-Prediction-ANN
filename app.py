import streamlit as st
import requests
import pandas as pd
import time

# Streamlit App Title
st.title("Customer Churn Prediction")

# Create input fields for the required columns in 3 columns
with st.form("customer_form"):
    # Create 3 columns
    col1, col2, col3 = st.columns(3)
    
    # Inputs for column 1
    with col1:
        CreditScore = st.slider("Credit Score", min_value=300, max_value=900, value=600, step=1)
        Geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
        Gender = st.selectbox("Gender", ['Male', 'Female'])

    # Inputs for column 2
    with col2:
        Age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
        Tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=2, step=1)
        Balance = st.number_input("Balance", min_value=0.0, value=0.0, step=0.01)

    # Inputs for column 3
    with col3:
        NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
        HasCrCard = st.selectbox("Has Credit Card", ["No", "Yes"])  # No for 0, Yes for 1
        IsActiveMember = st.selectbox("Is Active Member", ["No", "Yes"])  # No for 0, Yes for 1
        EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=0.0, step=0.01)
    
    # Submit button inside the form
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        # Prepare payload for API call
        payload = {
            "CreditScore": CreditScore,
            "Geography": Geography,
            "Gender": Gender,
            "Age": Age,
            "Tenure": Tenure,
            "Balance": Balance,
            "NumOfProducts": NumOfProducts,
            "HasCrCard": 1 if HasCrCard == "Yes" else 0,
            "IsActiveMember": 1 if IsActiveMember == "Yes" else 0,
            "EstimatedSalary": EstimatedSalary
        }

        # Show processing message and make prediction
        with st.spinner("Processing data..."):
            time.sleep(1)  # Simulate processing time
            # Call the FastAPI endpoint
            response = requests.post("http://127.0.0.1:8000/predict/", json=payload)
            result = response.json()

        # Display result after processing
        if result["likely_to_churn"]:
            st.write("The customer is likely to churn.")
        else:
            st.write("The customer is not likely to churn.")
        
        st.write(f"Churn Probability: {result['churn_probability']:.2f}")
