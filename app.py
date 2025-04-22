import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("LoanApprovalPrediction.csv")
    return data

# Preprocess data
def preprocess_data(data):
    data = data.drop("Loan_ID", axis=1)
    data.fillna(method='ffill', inplace=True)

    le_dict = {}
    label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']
    for col in label_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le
    
    return data, le_dict
from sklearn.ensemble import HistGradientBoostingClassifier

def train_model(X, y):
    model = HistGradientBoostingClassifier()
    model.fit(X, y)
    return model   

# Train model
def train_model(data):
    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"]
    model = HistGradientBoostingClassifier()
    model.fit(X, y)
    return model


# Predict using user input
def predict(model, le_dict, user_input):
    input_df = pd.DataFrame([user_input])
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
        input_df[col] = le_dict[col].transform([user_input[col]])
    return model.predict(input_df)[0]

# Streamlit UI
def main():
    st.title("🏦 Loan Approval Prediction App")

    data = load_data()
    processed_data, le_dict = preprocess_data(data)
    model = train_model(processed_data)

    st.header("Enter Applicant Details:")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_term = st.number_input("Loan Term (in days)", min_value=0)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict Loan Status"):
        user_input = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }

        prediction = predict(model, le_dict, user_input)
        result = le_dict['Loan_Status'].inverse_transform([prediction])[0]

        if result == 'Y':
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Rejected.")

if __name__ == '__main__':
    main()