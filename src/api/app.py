import mlflow.pyfunc
import pandas as pd
import yaml
from fastapi import FastAPI

from src.features.feature_engineering import build_features

import os
import mlflow
from pathlib import Path

# Set tracking URI to local mlruns (inside container)
# Must point to the /app/mlruns directory where the artifacts are copied
MODEL_NAME = "demand_forecasting_model"
mlruns_dir = Path("/app/mlruns").resolve()
tracking_uri = f"file://{mlruns_dir}"
mlflow.set_tracking_uri(tracking_uri)

print(f"DEBUG: fs check -> mlruns exists: {mlruns_dir.exists()}")
if mlruns_dir.exists():
    print(f"DEBUG: contents of mlruns: {list(mlruns_dir.iterdir())}")

app = FastAPI()

# load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# load model from registry
try:
    print(f"Attempting to load model '{MODEL_NAME}' from Production stage...")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL: Failed to load model. Error: {str(e)}")
    # Fallback or initialization indicator
    model = None


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