import json

import joblib
import numpy as np

from evaluation.evaluate import evaluate_on_test
from models.save_model import save_model
from models.train_model import train_gradient_boosting


def make_regression_frame(n_samples=80):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_samples, 5))
    y = data[:, 0] * 4.0 - data[:, 1] * 2.0 + 3.0
    return data, y


def test_train_gradient_boosting_returns_pipeline_and_numeric_metrics():
    X, y = make_regression_frame()

    model, train_metrics, validation_metrics = train_gradient_boosting(
        X[:60],
        y[:60],
        X[60:],
        y[60:],
        model_params={"n_estimators": 10},
    )

    assert hasattr(model, "predict")
    assert model.named_steps["imputer"].strategy == "median"
    assert train_metrics["mae"] >= 0
    assert validation_metrics["mae"] >= 0
    assert validation_metrics["rmse"] >= 0
    assert isinstance(validation_metrics["r2"], float)


def test_evaluate_on_test_returns_expected_metric_tuple():
    X, y = make_regression_frame()
    model, *_ = train_gradient_boosting(
        X[:60],
        y[:60],
        X[60:70],
        y[60:70],
        model_params={"n_estimators": 10},
    )

    test_mae, test_rmse, test_r2 = evaluate_on_test(model, X[70:], y[70:])

    assert test_mae >= 0
    assert test_rmse >= 0
    assert isinstance(test_r2, float)


def test_save_model_writes_pipeline_features_and_metrics(tmp_path):
    X, y = make_regression_frame()
    model, *_ = train_gradient_boosting(
        X[:60],
        y[:60],
        X[60:],
        y[60:],
        model_params={"n_estimators": 10},
    )
    metrics = {
        "official_test_mae": 1.23,
        "official_test_r2": 0.45,
    }
    features = ["cycle", "sensor1", "setting1"]

    save_model(
        model=model,
        base_features=features,
        metrics=metrics,
        models_path=tmp_path,
    )

    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "pipeline.pkl").exists()
    assert (tmp_path / "random_forest_model.pkl").exists()
    assert not (tmp_path / "scaler.pkl").exists()
    assert json.loads((tmp_path / "features.json").read_text()) == features
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    assert hasattr(joblib.load(tmp_path / "pipeline.pkl"), "predict")
