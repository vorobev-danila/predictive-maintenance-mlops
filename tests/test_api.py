import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    drift_data_path = tmp_path / "data" / "raw"
    write_cmapss_drift_fixture(drift_data_path)

    monkeypatch.setenv("PREDICTION_DB_PATH", str(tmp_path / "predictions.db"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DRIFT_DATA_PATH", str(drift_data_path))
    with TestClient(app) as client:
        yield client


def write_cmapss_drift_fixture(data_path):
    data_path.mkdir(parents=True, exist_ok=True)
    dataset_base_values = {
        "FD001": 1.0,
        "FD002": 2.0,
        "FD003": 3.0,
        "FD004": 4.0,
    }

    for dataset_id, base_value in dataset_base_values.items():
        train_rows = [
            build_cmapss_row(unit=1, cycle=1, base_value=base_value),
            build_cmapss_row(unit=1, cycle=2, base_value=base_value + 0.2),
            build_cmapss_row(unit=2, cycle=1, base_value=base_value + 0.4),
            build_cmapss_row(unit=2, cycle=2, base_value=base_value + 0.6),
        ]
        test_rows = [
            build_cmapss_row(unit=1, cycle=1, base_value=base_value + 1.0),
            build_cmapss_row(unit=1, cycle=2, base_value=base_value + 1.2),
            build_cmapss_row(unit=2, cycle=1, base_value=base_value + 1.4),
            build_cmapss_row(unit=2, cycle=2, base_value=base_value + 1.6),
        ]

        write_rows(data_path / f"train_{dataset_id}.txt", train_rows)
        write_rows(data_path / f"test_{dataset_id}.txt", test_rows)
        (data_path / f"RUL_{dataset_id}.txt").write_text("5\n7\n", encoding="utf-8")


def build_cmapss_row(unit, cycle, base_value):
    settings = [0.0, 0.0, 100.0]
    sensors = [base_value + index * 0.1 for index in range(1, 22)]
    return [unit, cycle, *settings, *sensors]


def write_rows(path, rows):
    content = "\n".join(" ".join(str(value) for value in row) for row in rows)
    path.write_text(f"{content}\n", encoding="utf-8")


def build_prediction_payload():
    payload = {f"sensor{i}": 1.0 for i in range(1, 22)}
    payload.update(
        {
            "cycle": 10.0,
            "setting1": 0.0,
            "setting2": 0.0,
            "setting3": 100.0,
        }
    )
    return payload


def seed_prediction_window(client, rows=5):
    for index in range(rows):
        payload = {
            **build_prediction_payload(),
            "unit": 1.0,
            "cycle": float(1 + index % 2),
            "actual_rul": float(30 - index),
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


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
    assert "official_test_mae" in metrics
    assert "official_test_r2" in metrics


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
    assert history[0]["model_version"] == "predictive-maintenance-gradient-boosting"


def test_predict_endpoint_updates_error_metrics(api_client):
    response = api_client.post(
        "/predict",
        json={**build_prediction_payload(), "actual_rul": 12.0},
    )
    metrics_response = api_client.get("/metrics")

    assert response.status_code == 200
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    assert "model_prediction_error" in metrics_text
    assert "model_absolute_error" in metrics_text
    assert "model_predictions_total" in metrics_text


def test_random_sample_endpoint_returns_train_fd001_payload(api_client):
    response = api_client.get("/samples/random?dataset_id=FD001")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "FD001"
    assert body["source"] == "train_FD001"
    assert set(build_prediction_payload()).issubset(body["payload"])
    assert "unit" in body["payload"]
    assert "actual_rul" in body["payload"]


def test_drift_run_creates_latest_report(api_client):
    seed_prediction_window(api_client, rows=10)

    run_response = api_client.post(
        "/drift/run",
        json={"dataset_id": "FD001", "scenario": "all", "intensity": 0.5},
    )
    latest_response = api_client.get("/drift/latest")

    assert run_response.status_code == 200
    assert latest_response.status_code == 200

    report = run_response.json()["report"]
    latest = latest_response.json()

    assert report["reference_dataset"] == "clean_train_matching_prediction_window"
    assert report["current_dataset"] == "simulated_FD001_window_last_10"
    assert report["scenario"] == "all"
    assert report["intensity"] == 0.5
    assert report["threshold_source"] == "colleague_default_thresholds"
    assert "data_drift" in latest
    assert "target_drift" in latest
    assert "concept_drift" in latest
    assert latest["data_drift"]["status"] != "skipped"
    assert latest["target_drift"]["status"] != "skipped"
    assert latest["concept_drift"]["status"] == "calculated"
    assert latest["prediction_summary"]["window_rows"] == 10
    assert latest["prediction_summary"]["labeled_rows"] == 10
    assert latest["prediction_summary"]["predicted_rul_mean"] is not None
    assert latest["prediction_summary"]["actual_rul_mean"] is not None


def test_drift_run_updates_feature_drift_metrics(api_client):
    seed_prediction_window(api_client, rows=10)

    response = api_client.post(
        "/drift/run",
        json={"dataset_id": "FD001", "scenario": "data_drift", "intensity": 1.0},
    )
    metrics_response = api_client.get("/metrics")

    assert response.status_code == 200
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    assert 'feature_drift_score{feature="cycle"}' in metrics_text
    assert 'feature_drift_detected{feature="cycle"}' in metrics_text
    assert 'feature_reference_mean{feature="cycle"}' in metrics_text
    assert 'feature_current_mean{feature="cycle"}' in metrics_text
    assert "prediction_window_predicted_rul_mean" in metrics_text
    assert "prediction_window_actual_rul_mean" in metrics_text
    assert "prediction_window_absolute_error_mae" in metrics_text
    assert "prediction_window_absolute_error_p95" in metrics_text


def test_drift_run_keeps_simulation_fields_for_request_compatibility(api_client):
    seed_prediction_window(api_client, rows=10)

    response = api_client.post(
        "/drift/run",
        json={"dataset_id": "FD001", "scenario": "all", "intensity": 1.0},
    )

    assert response.status_code == 200
    report = response.json()["report"]
    assert report["current_dataset"] == "simulated_FD001_window_last_10"
    assert report["intensity"] == 1.0


def test_drift_reports_endpoint_lists_saved_reports(api_client):
    seed_prediction_window(api_client, rows=10)

    run_response = api_client.post(
        "/drift/run",
        json={"dataset_id": "FD001", "scenario": "all", "intensity": 1.0},
    )
    reports_response = api_client.get("/drift/reports?limit=5")

    assert run_response.status_code == 200
    assert reports_response.status_code == 200
    reports = reports_response.json()
    assert len(reports) >= 1
    assert reports[0]["current_dataset"] == "simulated_FD001_window_last_10"
    assert reports[0]["report"]["threshold_source"] == "colleague_default_thresholds"


def test_drift_simulation_updates_report(api_client):
    response = api_client.post(
        "/drift/simulate",
        json={"scenario": "data_drift", "dataset_id": "FD001", "windows": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["report"]["scenario"] == "data_drift"
    assert body["report"]["latest_window"]["data_drift"]["drift_detected"] is True


def test_feature_file_matches_api_payload_fields():
    feature_path = Path("models/features.json")
    feature_names = json.loads(feature_path.read_text())
    payload_fields = set(build_prediction_payload())

    assert len(feature_names) == 25
    assert feature_names[:4] == ["cycle", "setting1", "setting2", "setting3"]
    assert set(feature_names).issubset(payload_fields)
