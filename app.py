# app.py
# FastAPI service that serves real-time churn predictions from the saved model

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Bank Churn Prediction API")

# Load the trained pipeline once when the server starts
model = joblib.load("models/churn_model.pkl")

# Feature columns in the exact order the model was trained on (Section 8)
features = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_Germany", "Geography_Spain", "Gender_Male",
]


# Raw customer fields a caller sends; Geography and Gender stay as readable text
class Customer(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def home():
    return {"message": "Bank Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: Customer):
    # Rebuild the one-hot columns from training; France and Female are the baselines
    geo_germany = 0
    geo_spain = 0
    if customer.Geography == "Germany":
        geo_germany = 1
    elif customer.Geography == "Spain":
        geo_spain = 1

    gender_male = 0
    if customer.Gender == "Male":
        gender_male = 1

    row = pd.DataFrame([[
        customer.CreditScore, customer.Age, customer.Tenure, customer.Balance,
        customer.NumOfProducts, customer.HasCrCard, customer.IsActiveMember,
        customer.EstimatedSalary, geo_germany, geo_spain, gender_male,
    ]], columns=features)

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {"churn_prediction": prediction, "churn_probability": round(probability, 3)}