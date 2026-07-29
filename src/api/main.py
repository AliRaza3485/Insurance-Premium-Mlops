from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.build_features import build_features

app = FastAPI(title="Insurance Charges Predictor")

model = joblib.load("models/xgboost_model.pkl")


class InsuranceInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age must be between 0 and 120")
    sex: Literal["male", "female"]
    bmi: float = Field(
        ..., gt=0, le=80, description="BMI must be a realistic positive value"
    )
    children: int = Field(..., ge=0, le=20)
    smoker: Literal["yes", "no"]
    region: Literal["southwest", "southeast", "northwest", "northeast"]


@app.get("/")
def read_root():
    return {"message": "Insurance Charges Prediction API is running"}


@app.post("/predict")
def predict(data: InsuranceInput):
    input_df = pd.DataFrame([data.dict()])
    input_df["charges"] = 0

    features = build_features(input_df)
    X = features.drop(columns=["charges"])

    prediction = model.predict(X)[0]

    return {"predicted_charge": round(float(prediction), 2)}
