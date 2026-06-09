import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PARAMS = {
    "n_estimators": 240,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": 42,
}


def build_model_pipeline(model_params=None):
    params = MODEL_PARAMS | (model_params or {})
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(**params)),
        ]
    )


def calculate_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_gradient_boosting(
    X_train, y_train, X_val=None, y_val=None, model_params=None
):
    pipeline = build_model_pipeline(model_params)
    pipeline.fit(X_train, y_train)

    train_metrics = calculate_metrics(y_train, pipeline.predict(X_train))
    if X_val is None or y_val is None:
        return pipeline, train_metrics, None

    validation_metrics = calculate_metrics(y_val, pipeline.predict(X_val))
    return pipeline, train_metrics, validation_metrics
