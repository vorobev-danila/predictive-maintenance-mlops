import json
import os
import sys
from pathlib import Path

import boto3
import mlflow
import mlflow.sklearn
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import (  # noqa: E402
    RAW_FEATURES,
    get_official_test_last_rows,
    load_cmapss_fd001,
)
from src.models.save_model import save_model  # noqa: E402
from src.models.train_model import (  # noqa: E402
    MODEL_PARAMS,
    calculate_metrics,
    train_gradient_boosting,
)

DATASET_ID = "FD001"
FEATURE_SET = "all_raw_features_no_feature_engineering"
MODEL_NAME = "GradientBoostingRegressor"
REQUIRED_MLFLOW_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_S3_ENDPOINT_URL",
)


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

    missing_env_vars = [
        variable for variable in REQUIRED_MLFLOW_ENV_VARS if not os.getenv(variable)
    ]
    if missing_env_vars:
        missing = ", ".join(missing_env_vars)
        raise OSError(
            f"Missing MLflow artifact storage environment variables: {missing}"
        )

    ensure_minio_bucket("mlflow")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow S3 endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")


def should_log_mlflow():
    return os.getenv("ENABLE_MLFLOW_LOGGING", "false").lower() in {"1", "true", "yes"}


def log_training_run(model, metrics, data_path):
    registered_model_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "predictive-maintenance-gradient-boosting",
    )

    try:
        configure_mlflow()
        with mlflow.start_run(run_name="gradient-boosting-raw-fd001-training"):
            mlflow.log_params(MODEL_PARAMS)
            mlflow.log_params(
                {
                    "model_type": MODEL_NAME,
                    "dataset": "NASA CMAPSS FD001",
                    "data_path": data_path,
                    "dataset_id": DATASET_ID,
                    "feature_set": FEATURE_SET,
                    "n_features": len(RAW_FEATURES),
                    "features": ",".join(RAW_FEATURES),
                }
            )
            numeric_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            mlflow.log_metrics(numeric_metrics)
            mlflow.log_artifact("models/metrics.json", artifact_path="model_artifacts")
            mlflow.log_artifact("models/features.json", artifact_path="model_artifacts")
            mlflow.log_artifact(
                "models/official_test_predictions.csv", artifact_path="model_artifacts"
            )
            mlflow.log_artifact(
                "models/evaluation_summary.csv", artifact_path="model_artifacts"
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
        print(f"MLflow run logged and model registered as: {registered_model_name}")
    except (BotoCoreError, ClientError, OSError, mlflow.MlflowException) as error:
        print(f"MLflow logging skipped: {error}")


def split_train_validation_by_unit(train_df):
    units = train_df["unit"].drop_duplicates().to_numpy()
    train_units, validation_units = train_test_split(
        units,
        test_size=0.2,
        random_state=42,
    )
    train_part = train_df[train_df["unit"].isin(train_units)].copy()
    validation_part = train_df[train_df["unit"].isin(validation_units)].copy()
    validation_last = (
        validation_part.sort_values(["unit", "cycle"])
        .groupby("unit", as_index=False)
        .tail(1)
        .sort_values("unit")
    )
    return train_part, validation_part, validation_last


def add_metric_prefix(metrics, prefix):
    return {
        f"{prefix}_mae": metrics["mae"],
        f"{prefix}_rmse": metrics["rmse"],
        f"{prefix}_r2": metrics["r2"],
    }


def build_training_metrics(train_df, test_df, official_rul):
    train_part, validation_part, validation_last = split_train_validation_by_unit(
        train_df
    )

    diagnostic_model, train_metrics, _ = train_gradient_boosting(
        train_part[RAW_FEATURES],
        train_part["RUL"],
    )
    validation_full_metrics = calculate_metrics(
        validation_part["RUL"],
        diagnostic_model.predict(validation_part[RAW_FEATURES]),
    )
    validation_last_metrics = calculate_metrics(
        validation_last["RUL"],
        diagnostic_model.predict(validation_last[RAW_FEATURES]),
    )

    final_model, final_train_metrics, _ = train_gradient_boosting(
        train_df[RAW_FEATURES],
        train_df["RUL"],
    )

    official_last_rows, y_official = get_official_test_last_rows(test_df, official_rul)
    official_predictions = final_model.predict(official_last_rows[RAW_FEATURES])
    official_metrics = calculate_metrics(y_official, official_predictions)

    metrics = {
        **add_metric_prefix(final_train_metrics, "train"),
        **add_metric_prefix(validation_full_metrics, "validation_full"),
        **add_metric_prefix(validation_last_metrics, "validation_last"),
        **add_metric_prefix(official_metrics, "official_test"),
        "train_full_rows": int(len(train_part)),
        "validation_full_rows": int(len(validation_part)),
        "validation_last_rows": int(len(validation_last)),
        "model_name": MODEL_NAME,
        "dataset_id": DATASET_ID,
        "feature_set": FEATURE_SET,
        **MODEL_PARAMS,
    }

    prediction_frame = official_last_rows[["unit", "cycle"]].copy()
    prediction_frame["actual_rul"] = y_official.astype(float)
    prediction_frame["predicted_rul"] = official_predictions
    prediction_frame["absolute_error"] = (
        prediction_frame["actual_rul"] - prediction_frame["predicted_rul"]
    ).abs()

    return final_model, metrics, prediction_frame


def save_evaluation_outputs(metrics, prediction_frame, models_path="models"):
    models_dir = Path(models_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame.to_csv(models_dir / "official_test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(models_dir / "evaluation_summary.csv", index=False)
    with (models_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main():
    print("Starting FD001 raw-features GradientBoosting training pipeline")
    data_path = os.getenv("CMAPSS_DATA_PATH", "data/raw")

    train_df, test_df, official_rul = load_cmapss_fd001(data_path=data_path)
    model, metrics, prediction_frame = build_training_metrics(
        train_df=train_df,
        test_df=test_df,
        official_rul=official_rul,
    )

    save_model(model=model, base_features=RAW_FEATURES, metrics=metrics)
    save_evaluation_outputs(metrics, prediction_frame)

    if should_log_mlflow():
        log_training_run(model, metrics, data_path)
    else:
        print("MLflow logging disabled. Set ENABLE_MLFLOW_LOGGING=true to enable it.")

    print("Training pipeline completed successfully")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Features ({len(RAW_FEATURES)}): {', '.join(RAW_FEATURES)}")
    print(f"Official test MAE: {metrics['official_test_mae']:.3f}")
    print(f"Official test RMSE: {metrics['official_test_rmse']:.3f}")
    print(f"Official test R2: {metrics['official_test_r2']:.3f}")


if __name__ == "__main__":
    main()
