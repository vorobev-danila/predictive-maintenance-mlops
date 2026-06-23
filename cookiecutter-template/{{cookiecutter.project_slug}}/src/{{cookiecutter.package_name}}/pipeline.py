import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "{{ cookiecutter.mlflow_tracking_uri }}",
    )
    registered_model_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "{{ cookiecutter.registered_model_name }}",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("{{ cookiecutter.project_slug }}")

    x, y = make_regression(n_samples=500, n_features=8, noise=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    }
    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    with mlflow.start_run(run_name="baseline-random-forest"):
        mlflow.log_params(params)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

    print("Training run logged to MLflow")


if __name__ == "__main__":
    main()
