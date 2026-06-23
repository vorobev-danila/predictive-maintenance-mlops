# API Guide

[← Back to README](../../README.md)

## Contents

- [OpenAPI](#openapi)
- [Endpoints](#endpoints)
- [Prediction history](#prediction-history)
- [Prometheus metrics](#prometheus-metrics)

## OpenAPI

Start the API:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open interactive documentation:

```text
http://localhost:8080/docs
```

Fetch raw OpenAPI schema:

```bash
curl http://localhost:8080/openapi.json
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check that model artifacts are loaded |
| `POST` | `/predict` | Predict RUL from sensor and setting payload |
| `GET` | `/predictions/recent` | Return recent predictions from SQLite |
| `GET` | `/model_metrics` | Return metrics from `models/metrics.json` |
| `POST` | `/retrain` | Run training pipeline and reload the model |
| `POST` | `/drift/run` | Calculate drift and save JSON report |
| `POST` | `/drift/simulate` | Run controlled drift simulation for demo metrics |
| `GET` | `/drift/latest` | Return latest drift report |
| `GET` | `/metrics` | Prometheus metrics endpoint |
| `POST` | `/reset_metrics` | Reset custom Prometheus gauges |

## Prediction History

The API records every successful prediction in SQLite through the repository
layer in `src/storage/prediction_repository.py`.

Default local database path:

```text
state/predictions.db
```

Kubernetes path configured through `PREDICTION_DB_PATH`:

```text
/app/state/predictions.db
```

Read recent predictions:

```bash
curl "http://localhost:8080/predictions/recent?limit=20"
```

Stored fields:

| Field | Description |
| --- | --- |
| `id` | Prediction identifier |
| `created_at` | UTC timestamp |
| `input` | Original request payload |
| `predicted_rul` | Predicted remaining useful life |
| `actual_rul` | Optional ground truth |
| `anomaly_flag` | Reserved for anomaly/drift workflows |
| `model_version` | Model name/version marker |

## Prometheus Metrics

The API exposes:

- standard FastAPI request metrics through `prometheus-fastapi-instrumentator`;
- custom gauges `model_predicted_rul` and `model_actual_rul`.

Prometheus scrapes:

```text
http://predictive-maintenance-api:8080/metrics
```
