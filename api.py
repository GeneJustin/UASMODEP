from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("artifacts/best_model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")

class CreditData(BaseModel):
    Age: float
    Occupation: str
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Num_Bank_Accounts: float
    Num_Credit_Card: float
    Interest_Rate: float
    Num_of_Loan: float
    Delay_from_due_date: float
    Num_of_Delayed_Payment: float
    Changed_Credit_Limit: float
    Num_Credit_Inquiries: float
    Credit_Mix: str
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float
    Credit_History_Age: float
    Payment_of_Min_Amount: str
    Total_EMI_per_month: float
    Amount_invested_monthly: float
    Payment_Behaviour: str
    Monthly_Balance: float

@app.get("/")
def root():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: CreditData):
    df = pd.DataFrame([data.model_dump()])

    pred = model.predict(df)[0]

    try:
        pred = label_encoder.inverse_transform([pred])[0]
    except:
        pass

    return {
        "credit_score_prediction": str(pred)
    }