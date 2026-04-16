from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the saved RandomForest model
model = joblib.load("models/churn_model.pkl")

app = FastAPI(
    title="Customer Churn Predictor API",
    description="Predicts customer churn probability and returns business-friendly explanations.",
    version="1.0.0",
)

GENDER_MAP = {"Female": 0, "Male": 1}
SUBSCRIPTION_MAP = {"Basic": 0, "Premium": 1, "Standard": 2}
CONTRACT_MAP = {"Annual": 0, "Monthly": 1, "Quarterly": 2}
THRESHOLD = 0.4

# Input schema for API payloads
class CustomerData(BaseModel):
    Age: float
    Gender: str
    Tenure: float
    Support_Calls: float
    Payment_Delay: float
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: float
    Usage_Frequency: float

    class Config:
        schema_extra = {
            "example": {
                "Age": 35.0,
                "Gender": "Female",
                "Tenure": 18.0,
                "Support_Calls": 3.0,
                "Payment_Delay": 10.0,
                "Subscription_Type": "Standard",
                "Contract_Length": "Monthly",
                "Total_Spend": 645.0,
                "Last_Interaction": 12.0,
                "Usage_Frequency": 8.0,
            }
        }


def encode_category(value: str, mapping: Dict[str, int], field_name: str) -> int:
    try:
        return mapping[value]
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name}: {value}. Valid values: {list(mapping.keys())}",
        )


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Customer Churn Predictor API is live.",
        "model": "RandomForestClassifier",
        "risk_threshold": str(THRESHOLD),
    }

@app.post("/predict")
def predict(data: CustomerData) -> Dict[str, object]:
    encoded_input = {
        "Age": data.Age,
        "Gender": encode_category(data.Gender, GENDER_MAP, "Gender"),
        "Tenure": data.Tenure,
        "Usage Frequency": data.Usage_Frequency,
        "Support Calls": data.Support_Calls,
        "Payment Delay": data.Payment_Delay,
        "Subscription Type": encode_category(
            data.Subscription_Type, SUBSCRIPTION_MAP, "Subscription_Type"
        ),
        "Contract Length": encode_category(
            data.Contract_Length, CONTRACT_MAP, "Contract_Length"
        ),
        "Total Spend": data.Total_Spend,
        "Last Interaction": data.Last_Interaction,
    }

    input_df = pd.DataFrame([encoded_input])
    probability = float(model.predict_proba(input_df)[0][1])
    will_churn = probability >= THRESHOLD

    if probability > 0.7:
        risk = "High"
    elif probability > THRESHOLD:
        risk = "Medium"
    else:
        risk = "Low"

    reasons = []
    if data.Support_Calls > 5:
        reasons.append("High support calls — customer may be frustrated")
    if data.Payment_Delay > 20:
        reasons.append("Long payment delay — likely disengagement")
    if data.Contract_Length == "Monthly":
        reasons.append("Month-to-month contract — easier to churn")
    if not reasons:
        reasons.append("No dominant churn signals in the input values")

    return {
        "churn_probability": round(probability, 3),
        "will_churn": bool(will_churn),
        "risk_level": risk,
        "top_reasons": reasons,
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}