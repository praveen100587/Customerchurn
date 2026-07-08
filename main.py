import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
model = joblib.load("Customerchurnprediction.pkl")
app = FastAPI(title="Customer Churn Prediction API")
class Customerchurn(BaseModel):
    AccountWeeks: int
    ContractRenewal: int
    DataPlan: int
    DataUsage: float
    CustServCalls:int
    DayMins:float
    DayCalls:int
    MonthlyCharge:float
    OverageFee:float
    RoamMins:float

@app.get("/")
def home():
    return {"message": "Welcome to Customer churn Prediction Site"}

@app.post("/predict")
def predict(data: Customerchurn):

    features = [[
        data.AccountWeeks,
        data.ContractRenewal,
        data.DataPlan,
        data.DataUsage,
        data.CustServCalls,
        data.DayMins,
        data.DayCalls,
        data.MonthlyCharge,
        data.OverageFee,
        data.RoamMins
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }