import os

import boto3
import mlflow
import mlflow.sklearn
from botocore.exceptions import ClientError

from data.data_loader import load_and_prepare_data
from data.analysis import (
    analyze_engine_lifetime,
    plot_engine_lifetime,
    plot_rul_distribution,
    plot_sensors_dynamics,
    print_basic_info,
    print_statistics,
)
from evaluation.evaluate import evaluate_on_test
from features.feature_engineering import prepare_data, select_all_sensors
from models.save_model import save_model
from models.train_model import train_random_forest


MODEL_PARAMS = {
    "n_estimators": 30,
    "max_depth": 5,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
}


def ensure_minio_bucket(bucket_name="mlflow"):
    endpoint_url = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")
        if error_code not in {"404", "NoSuchBucket"}:
            raise
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"Created MinIO bucket: {bucket_name}")


def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive-maintenance")

    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio123")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    ensure_minio_bucket("mlflow")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow S3 endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")


def log_training_run(model, scaler, base_features, metrics, data_path):
    registered_model_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "predictive-maintenance-random-forest",
    )

    with mlflow.start_run(run_name="random-forest-rul-training"):
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "dataset": "NASA CMAPSS FD001",
                "data_path": data_path,
                "n_features": len(base_features),
                "features": ",".join(base_features),
            }
        )

        mlflow.log_metrics(
            {
                "train_mae": float(metrics["train_mae"]),
                "val_mae": float(metrics["val_mae"]),
                "test_mae": float(metrics["test_mae"]),
                "train_rmse": float(metrics["train_rmse"]),
                "val_rmse": float(metrics["val_rmse"]),
                "test_rmse": float(metrics["test_rmse"]),
                "train_r2": float(metrics["train_r2"]),
                "val_r2": float(metrics["val_r2"]),
                "test_r2": float(metrics["test_r2"]),
            }
        )

        mlflow.log_artifact("models/metrics.json", artifact_path="model_artifacts")
        mlflow.log_artifact("models/features.json", artifact_path="model_artifacts")
        mlflow.log_artifact("models/scaler.pkl", artifact_path="model_artifacts")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        print(f"MLflow run logged and model registered as: {registered_model_name}")


def main():
    print("Starting model training pipeline")
    data_path = "data/raw"
    configure_mlflow()

    train_with_rul, test_original, rul_original = load_and_prepare_data(
        data_path=data_path
    )

    print_basic_info(train_with_rul, test_original)
    print_statistics(train_with_rul)
    engine_lifetime = analyze_engine_lifetime(train_with_rul)
    plot_engine_lifetime(engine_lifetime)
    plot_sensors_dynamics(train_with_rul)
    plot_rul_distribution(train_with_rul)

    top_sensors = select_all_sensors(train_with_rul)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        base_features,
    ) = prepare_data(train_with_rul, top_sensors)

    model, train_mae, val_mae, train_rmse, val_rmse, train_r2, val_r2 = (
        train_random_forest(X_train, y_train, X_val, y_val)
    )

    test_mae, test_rmse, test_r2 = evaluate_on_test(model, X_test, y_test)

    metrics = {
        "train_mae": train_mae,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "test_r2": test_r2,
        "features": base_features,
        **MODEL_PARAMS,
    }

    save_model(model, scaler, base_features, metrics)
    log_training_run(model, scaler, base_features, metrics, data_path)

    print("Training results")
    print("Model: RandomForestRegressor")
    print(f"Features: {len(base_features)}")
    print(
        "Params: "
        "n_estimators=30, max_depth=5, "
        "min_samples_split=20, min_samples_leaf=10"
    )
    print(f"Validation MAE: {val_mae:.2f}")
    print(f"Validation R2: {val_r2:.3f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test R2: {test_r2:.3f}")
    print(f"Average engine lifetime: {engine_lifetime.mean():.1f} cycles")
    print(f"Relative test error: {(test_mae / engine_lifetime.mean()) * 100:.1f}%")

    if test_r2 > 0.5:
        print("Model quality is acceptable: R2 > 0.5")
    else:
        print("Model quality is below expected and needs improvement")

    print("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
