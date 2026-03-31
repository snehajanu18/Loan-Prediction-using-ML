import streamlit as st
import numpy as np
import joblib

st.title("Loan Prediction App (3 Models)")
# Load from local paths
lr_model = joblib.load("logistic_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Inputs
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [1, 0])

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
# Encoding
input_dict = {
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Gender_Male": 1 if Gender == "Male" else 0,
    "Married_Yes": 1 if Married == "Yes" else 0,
    "Dependents_1": 1 if Dependents == "1" else 0,
    "Dependents_2": 1 if Dependents == "2" else 0,
    "Dependents_3+": 1 if Dependents == "3+" else 0,
    "Education_Not Graduate": 1 if Education == "Not Graduate" else 0,
    "Self_Employed_Yes": 1 if Self_Employed == "Yes" else 0,
    "Property_Area_Semiurban": 1 if Property_Area == "Semiurban" else 0,
    "Property_Area_Urban": 1 if Property_Area == "Urban" else 0,
}

# Align features
input_data = [input_dict.get(col, 0) for col in feature_columns]
input_array = np.array(input_data).reshape(1, -1)

# Scale
input_scaled = scaler.transform(input_array)

# Predict
if st.button("Predict with All Models"):

    lr_pred = lr_model.predict(input_scaled)[0]
    rf_pred = rf_model.predict(input_scaled)[0]
    dt_pred = dt_model.predict(input_scaled)[0]

    st.subheader("Results")

    st.write("Logistic Regression:", "Approved" if lr_pred == 1 else "Not Approved")
    st.write("Random Forest:", "Approved" if rf_pred == 1 else "Not Approved")
    st.write("Decision Tree:", "Approved" if dt_pred == 1 else "Not Approved")
