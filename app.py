import streamlit as st
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Credit Score Prediction")

Age = st.number_input("Age", value=35.0)
Occupation = st.text_input("Occupation", value="Engineer")

Annual_Income = st.number_input("Annual Income", value=20000.0)
Monthly_Inhand_Salary = st.number_input("Monthly Salary", value=3000.0)

Num_Bank_Accounts = st.number_input("Bank Accounts", value=5.0)
Num_Credit_Card = st.number_input("Credit Cards", value=5.0)
Interest_Rate = st.number_input("Interest Rate", value=15.0)
Num_of_Loan = st.number_input("Number of Loans", value=3.0)

Delay_from_due_date = st.number_input("Delay From Due Date", value=15.0)
Num_of_Delayed_Payment = st.number_input("Delayed Payments", value=10.0)
Changed_Credit_Limit = st.number_input("Credit Limit Change", value=10.0)
Num_Credit_Inquiries = st.number_input("Credit Inquiries", value=5.0)

Credit_Mix = st.text_input("Credit Mix", value="Standard")
Outstanding_Debt = st.number_input("Debt", value=1500.0)
Credit_Utilization_Ratio = st.number_input("Utilization Ratio", value=30.0)
Credit_History_Age = st.number_input("Credit History Age", value=200.0)

Payment_of_Min_Amount = st.text_input("Min Payment", value="Yes")
Total_EMI_per_month = st.number_input("EMI", value=50.0)
Amount_invested_monthly = st.number_input("Investment", value=100.0)

Payment_Behaviour = st.text_input("Payment Behaviour", value="Low_spent_Small_value_payments")
Monthly_Balance = st.number_input("Monthly Balance", value=300.0)

if st.button("Predict"):

    data = {
        "Age": Age,
        "Occupation": Occupation,
        "Annual_Income": Annual_Income,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_Bank_Accounts": Num_Bank_Accounts,
        "Num_Credit_Card": Num_Credit_Card,
        "Interest_Rate": Interest_Rate,
        "Num_of_Loan": Num_of_Loan,
        "Delay_from_due_date": Delay_from_due_date,
        "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
        "Changed_Credit_Limit": Changed_Credit_Limit,
        "Num_Credit_Inquiries": Num_Credit_Inquiries,
        "Credit_Mix": Credit_Mix,
        "Outstanding_Debt": Outstanding_Debt,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
        "Credit_History_Age": Credit_History_Age,
        "Payment_of_Min_Amount": Payment_of_Min_Amount,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Amount_invested_monthly": Amount_invested_monthly,
        "Payment_Behaviour": Payment_Behaviour,
        "Monthly_Balance": Monthly_Balance
    }

    df = pd.DataFrame([data])

    pred = model.predict(df)[0]
    result = label_encoder.inverse_transform([pred])[0]

    st.success(f"Prediction: {result}")
