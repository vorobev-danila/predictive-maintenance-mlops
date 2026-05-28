import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_DB_PATH", str(tmp_path / "predictions.db"))
    with TestClient(app) as client:
        yield client


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


def test_health_endpoint_loads_model_artifacts(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_rul_prediction(api_client):
    response = api_client.post("/predict", json=build_prediction_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert isinstance(body["prediction_id"], int)
    assert isinstance(body["rul"], float)


def test_predict_endpoint_persists_prediction_history(api_client):
    prediction_response = api_client.post(
        "/predict",
        json={**build_prediction_payload(), "actual_rul": 12.0},
    )
    history_response = api_client.get("/predictions/recent?limit=1")

    assert prediction_response.status_code == 200
    assert history_response.status_code == 200

    prediction = prediction_response.json()
    history = history_response.json()

    assert len(history) == 1
    assert history[0]["id"] == prediction["prediction_id"]
    assert history[0]["predicted_rul"] == prediction["rul"]
    assert history[0]["actual_rul"] == 12.0
    assert history[0]["anomaly_flag"] is False
    assert history[0]["model_version"] == "predictive-maintenance-random-forest"


def test_feature_file_matches_api_payload_fields():
    feature_path = Path("models/features.json")
    feature_names = set(json.loads(feature_path.read_text()))
    payload_fields = set(build_prediction_payload())

    assert feature_names.issubset(payload_fields)
