import json
import joblib
import pandas as pd
import torch

from fastapi import FastAPI
from app.model import FraudANN
from app.schema import Transaction

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

device = torch.device("cpu")

preprcoessing_pipeline = joblib.load("scripts/artifacts/preprocessing_pipeline.joblib")

with open("scripts/artifacts/threshold.json", "r") as file:
    threshold = json.load(file)["threshold"]

checkpoint = torch.load("scripts/artifacts/fraud_model.pth", map_location=device)

model_config = checkpoint["model_config"]

model = FraudANN(model_config["input_size"],model_config["hidden1"],model_config["hidden2"],model_config["dropout1"],model_config["dropout2"])

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

@app.get("/")
def home():
    return {
        "message" : "Credit Card Fraud Detection API is running"
    }

@app.post("/predict")
def prediction(transaction: Transaction):
    data = transaction.model_dump()
    input_df = pd.DataFrame([data])

    processed_data = preprcoessing_pipeline.transform(input_df)

    input_tensor = torch.tensor(processed_data,dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    prediction = int(probability >= threshold)

    return {
        "fraud": bool(prediction),
        "fraud_probability": round(probability,6),
        "threshold": threshold,
    }