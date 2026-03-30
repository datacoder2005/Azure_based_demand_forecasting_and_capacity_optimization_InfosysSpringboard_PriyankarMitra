from fastapi import FastAPI
import joblib
import pandas as pd
import json

app = FastAPI()

# Load model
model = joblib.load(
    "E:/College/infosys springboard project/dashboard/xgb_model.pkl")

# Load feature columns
with open("E:/College/infosys springboard project/dashboard/feature_cols.json", "r") as f:
    feature_cols = json.load(f)


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # One-hot encode region & service
    df = pd.get_dummies(df)

    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    pred = model.predict(df)

    return {
        "region": data.get("region"),
        "service_type": data.get("service_type"),
        "prediction": float(pred[0])
    }
