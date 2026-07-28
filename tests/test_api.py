from fastapi.testclient import TestClient
import sys

sys.path.append(".")

from src.api.main import app

client = TestClient(app)


def test_read_root():
    """Test that the health-check endpoint works"""
    response = client.get("/")
    assert response.status_code == 200


def test_predict_returns_valid_response():
    """Test that /predict endpoint returns a prediction for valid input"""
    payload = {
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "smoker": "no",
        "region": "southwest",
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_charge" in response.json()
    assert response.json()["predicted_charge"] > 0


def test_predict_smoker_higher_than_nonsmoker():
    """Sanity check: smokers should have higher predicted charges than non-smokers, all else equal"""
    base_payload = {
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "region": "southwest",
    }

    non_smoker = client.post("/predict", json={**base_payload, "smoker": "no"})
    smoker = client.post("/predict", json={**base_payload, "smoker": "yes"})

    assert smoker.json()["predicted_charge"] > non_smoker.json()["predicted_charge"]
