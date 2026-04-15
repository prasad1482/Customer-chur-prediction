from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("models/churn_model.pkl")

app = FastAPI(
    title="Customer Churn Predictor API",
    description="Predicts customer churn probability using XGBoost (0.95 ROC-AUC)",
    version="1.0.0"
)

# Input schema — must match your exact feature names
class CustomerData(BaseModel):
    Age: float
    Gender: int                # 0 = Female, 1 = Male
    Tenure: float
    Support_Calls: float
    Payment_Delay: float
    Subscription_Type: int     # encoded: 0/1/2
    Contract_Length: int       # encoded: 0/1/2
    Total_Spend: float
    Last_Interaction: float
    Usage_Frequency: float

@app.get("/")
def root():
    return {
        "message": "Churn Predictor API is live",
        "model": "XGBoost",
        "accuracy": "93%",
        "roc_auc": "0.95"
    }

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to dataframe
    input_df = pd.DataFrame([data.dict()])
    
    # Rename to match training feature names exactly
    input_df.columns = [
        'Age', 'Gender', 'Tenure', 'Support Calls',
        'Payment Delay', 'Subscription Type', 'Contract Length',
        'Total Spend', 'Last Interaction', 'Usage Frequency'
    ]
    
    # Predict
    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= 0.4)  # tuned threshold
    
    # Risk level
    if proba > 0.7:
        risk = "High"
    elif proba > 0.4:
        risk = "Medium"
    else:
        risk = "Low"
    
    # Top reason (based on your SHAP findings)
    reasons = []
    if data.Support_Calls > 5:
        reasons.append("High support calls — customer likely frustrated")
    if data.Payment_Delay > 20:
        reasons.append("Payment delay over 20 days — disengagement signal")
    if data.Contract_Length == 0:
        reasons.append("Month-to-month contract — easy to leave")
    if not reasons:
        reasons.append("Low individual risk factors")

    return {
        "churn_probability": round(float(proba), 3),
        "will_churn": bool(prediction),
        "risk_level": risk,
        "top_reasons": reasons
    }

@app.get("/health")
def health():
    return {"status": "healthy"}