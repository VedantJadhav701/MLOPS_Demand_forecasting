import mlflow.pyfunc
import pandas as pd
import yaml
from fastapi import FastAPI

from src.features.feature_engineering import build_features

MODEL_NAME = "demand_forecasting_model"

app = FastAPI()

# load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# load model from registry
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")


@app.get("/")
def home():
    return {"message": "Demand Forecasting API is running"}


@app.post("/predict")
def predict(data: dict):
    """
    Accept RAW input → apply feature engineering → predict
    """

    # convert input to dataframe
    df = pd.DataFrame([data])

    # apply SAME feature pipeline as training
    df = build_features(df, config)

    # drop target if accidentally passed
    target = config["data"]["target"]
    if target in df.columns:
        df = df.drop(columns=[target])

    prediction = model.predict(df)

    return {"prediction": float(prediction[0])}