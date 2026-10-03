from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

legitimate_transaction = {
    "V1": 1.2,
    "V2": 0.3,
    "V3": -0.5,
    "V4": 0.8,
    "V5": -0.2,
    "V6": 0.4,
    "V7": -0.1,
    "V8": 0.05,
    "V9": 0.6,
    "V10": -0.3,
    "V11": 0.2,
    "V12": -0.4,
    "V13": 0.1,
    "V14": 0.5,
    "V15": -0.2,
    "V16": 0.3,
    "V17": -0.1,
    "V18": 0.2,
    "V19": -0.3,
    "V20": 0.1,
    "V21": -0.05,
    "V22": 0.2,
    "V23": -0.1,
    "V24": 0.3,
    "V25": -0.2,
    "V26": 0.1,
    "V27": 0.05,
    "V28": -0.02,
    "Amount": 50.0
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Credit Card Fraud Detection API is running"

def test_predict_valid_transaction():
    response = client.post("/predict", json=legitimate_transaction)

    assert response.status_code == 200

    data = response.json()

    assert "fraud" in data
    assert "fraud_probability" in data
    assert "threshold" in data

def test_prediction_probability():
    response = client.post("/predict", json=legitimate_transaction)

    data = response.json()

    assert 0.0 <= data["threshold"] <= 1.0

def test_invalid_transaction():
    invalid_transaction = {
        "V1": 1.2,
        "V2": 0.3,
        "V3": -0.5,
        "V4": 0.8,
        "V5": -0.2,
        "V6": 0.4,
        "V7": -0.1,
        "V8": 0.05,
    }
    response = client.post("/predict", json=invalid_transaction)
    assert response.status_code == 422