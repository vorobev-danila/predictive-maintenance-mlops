from storage.prediction_repository import PredictionRepository


def test_prediction_repository_creates_and_lists_recent_predictions(tmp_path):
    repository = PredictionRepository(tmp_path / "predictions.db")
    repository.initialize()

    first = repository.create_prediction(
        input_payload={"sensor1": 1.0, "setting1": 0.0},
        predicted_rul=42.5,
        actual_rul=40.0,
        anomaly_flag=True,
        model_version="test-model",
    )
    second = repository.create_prediction(
        input_payload={"sensor1": 2.0, "setting1": 0.0},
        predicted_rul=21.0,
    )

    recent = repository.list_recent(limit=10)

    assert first["id"] == 1
    assert second["id"] == 2
    assert recent[0]["id"] == 2
    assert recent[1]["id"] == 1
    assert recent[1]["input"] == {"sensor1": 1.0, "setting1": 0.0}
    assert recent[1]["predicted_rul"] == 42.5
    assert recent[1]["actual_rul"] == 40.0
    assert recent[1]["anomaly_flag"] is True
    assert recent[1]["model_version"] == "test-model"


def test_prediction_repository_bounds_recent_limit(tmp_path):
    repository = PredictionRepository(tmp_path / "predictions.db")
    repository.initialize()

    for index in range(3):
        repository.create_prediction(
            input_payload={"sensor1": float(index)},
            predicted_rul=float(index),
        )

    assert len(repository.list_recent(limit=0)) == 1
    assert len(repository.list_recent(limit=200)) == 3
