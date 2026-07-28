from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.build_features import build_features

app = FastAPI(title="Insurance Charges Predictor")

# Load model once when API starts (not on every request)
model = joblib.load("models/xgboost_model.pkl")


class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


@app.get("/")
def read_root():
    return {"message": "Insurance Charges Prediction API is running"}


@app.post("/predict")
def predict(data: InsuranceInput):
    input_df = pd.DataFrame([data.dict()])
    input_df["charges"] = 0  # dummy, required by build_features but not used

    features = build_features(input_df)
    X = features.drop(columns=["charges"])

    prediction = model.predict(X)[0]

    return {"predicted_charge": round(float(prediction), 2)}
