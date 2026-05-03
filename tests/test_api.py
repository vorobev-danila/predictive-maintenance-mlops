import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def build_prediction_payload():
    payload = {f"sensor{i}": 1.0 for i in range(1, 22)}
    payload.update(
        {
            "setting1": 0.0,
            "setting2": 0.0,
            "setting3": 100.0,
        }
    )
    return payload


def test_openapi_schema_available():
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Predictive Maintenance API"


def test_model_metrics_endpoint_returns_saved_metrics():
    client = TestClient(app)

    response = client.get("/model_metrics")

    assert response.status_code == 200
    metrics = response.json()
    assert "test_mae" in metrics
    assert "test_r2" in metrics


def test_health_endpoint_loads_model_artifacts():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_rul_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", json=build_prediction_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert isinstance(body["rul"], float)


def test_feature_file_matches_api_payload_fields():
    feature_path = Path("models/features.json")
    feature_names = set(json.loads(feature_path.read_text()))
    payload_fields = set(build_prediction_payload())

    assert feature_names.issubset(payload_fields)
