# MLflow, MinIO and DVC

[← Back to README](../../README.md)

## Contents

- [MLflow tracking](#mlflow-tracking)
- [MinIO artifact storage](#minio-artifact-storage)
- [DVC data versioning](#dvc-data-versioning)
- [Credentials](#credentials)

## MLflow Tracking

The training pipeline logs:

- model parameters;
- metrics: `train_mae`, `validation_full_mae`, `validation_last_mae`,
  `official_test_mae`, and matching `rmse`/`r2` values;
- artifacts: `metrics.json`, `features.json`, `pipeline.pkl`,
  `official_test_predictions.csv`, `evaluation_summary.csv`;
- registered sklearn model: `predictive-maintenance-gradient-boosting`.

Run training:

```bash
uv run python src/pipeline.py
```

In Docker Compose, the API service enables MLflow logging for `/retrain` with
`ENABLE_MLFLOW_LOGGING=true`. The MLflow service also allows internal compose
requests from `mlflow:5000`, so the API can log metrics, artifacts, and the
registered model during retraining.

Open MLflow UI:

```text
http://localhost:5000
```

## MinIO Artifact Storage

MLflow metrics and parameters are stored in the MLflow backend store. Model
artifacts are saved to MinIO bucket:

```text
mlflow
```

Local MinIO UI:

```text
http://localhost:9001
```

Default local credentials:

```text
login: minio
password: minio123
```

## DVC Data Versioning

Raw data under `data/raw` is managed by DVC.

Common commands:

```bash
dvc status
dvc pull
dvc push
```

DVC remote is configured for MinIO bucket:

```text
dvc
```

## Credentials

For local shell access to MinIO-backed DVC:

```bash
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
```

PowerShell:

```powershell
$env:AWS_ACCESS_KEY_ID="minio"
$env:AWS_SECRET_ACCESS_KEY="minio123"
```

Do not store secrets in `.dvc/config`. In Kubernetes, credentials are provided
through `Secret`.
