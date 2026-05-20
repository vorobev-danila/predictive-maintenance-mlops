import json

import joblib
import numpy as np

from evaluation.evaluate import evaluate_on_test
from models.save_model import save_model
from models.train_model import train_random_forest


def make_regression_arrays(n_samples=80):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 5))
    y = X[:, 0] * 4.0 - X[:, 1] * 2.0 + 3.0
    return X, y


def test_train_random_forest_returns_model_and_numeric_metrics():
    X, y = make_regression_arrays()

    result = train_random_forest(X[:60], y[:60], X[60:], y[60:])
    model, train_mae, val_mae, train_rmse, val_rmse, train_r2, val_r2 = result

    assert hasattr(model, "predict")
    assert train_mae >= 0
    assert val_mae >= 0
    assert train_rmse >= 0
    assert val_rmse >= 0
    assert isinstance(train_r2, float)
    assert isinstance(val_r2, float)


def test_evaluate_on_test_returns_expected_metric_tuple():
    X, y = make_regression_arrays()
    model, *_ = train_random_forest(X[:60], y[:60], X[60:70], y[60:70])

    test_mae, test_rmse, test_r2 = evaluate_on_test(model, X[70:], y[70:])

    assert test_mae >= 0
    assert test_rmse >= 0
    assert isinstance(test_r2, float)


def test_save_model_writes_model_scaler_features_and_metrics(tmp_path):
    X, y = make_regression_arrays()
    model, *_ = train_random_forest(X[:60], y[:60], X[60:], y[60:])
    metrics = {
        "test_mae": 1.23,
        "test_r2": 0.45,
    }
    features = ["sensor1", "setting1"]

    save_model(
        model=model,
        scaler={"kind": "fake-scaler"},
        base_features=features,
        metrics=metrics,
        models_path=tmp_path,
    )

    assert (tmp_path / "random_forest_model.pkl").exists()
    assert (tmp_path / "scaler.pkl").exists()
    assert json.loads((tmp_path / "features.json").read_text()) == features
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    assert hasattr(joblib.load(tmp_path / "random_forest_model.pkl"), "predict")
    assert joblib.load(tmp_path / "scaler.pkl") == {"kind": "fake-scaler"}
