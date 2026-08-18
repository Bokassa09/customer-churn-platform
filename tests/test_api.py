import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Données de test
ahmed = {
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 55,
    "Tenure": 2,
    "Balance": 125000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 80000.0
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_accueil():
    response = client.get("/")
    assert response.status_code == 200
    assert "bienvenue" in response.json()

def test_predict_churn():
    response = client.post("/predict", json=ahmed)
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["probabilite"] > 90
    assert "message" in response.json()

def test_explainability_churn():
    response = client.post("/explain", json=ahmed)
    assert response.status_code == 200
    assert "top3_facteurs" in response.json()
    assert len(response.json()["top3_facteurs"]) == 3