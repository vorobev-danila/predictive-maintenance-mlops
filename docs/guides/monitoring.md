# Monitoring and Drift

[← Back to README](../../README.md)

## Contents

- [Drift workflow](#drift-workflow)
- [API endpoints](#api-endpoints)
- [Report format](#report-format)
- [Prometheus metrics](#prometheus-metrics)
- [Runtime paths](#runtime-paths)

## Drift Workflow

The project compares a reference CMAPSS dataset with a current CMAPSS dataset:

```text
reference = train_FD001
current   = test_FD001
```

The drift module calculates:

- data drift for model features;
- target drift for `RUL`;
- concept drift from model error growth.

The implementation lives in:

```text
src/monitoring/drift.py
```

## API Endpoints

Run drift calculation:

```bash
curl -X POST http://localhost:8080/drift/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "FD001"}'
```

Read latest report:

```bash
curl http://localhost:8080/drift/latest
```

## Report Format

Reports are written as JSON:

```text
reports/drift/latest.json
reports/drift/drift_report_<timestamp>.json
```

In Docker Compose and Kubernetes, reports are stored under API state:

```text
/app/state/reports/drift/latest.json
```

Main report sections:

| Section | Meaning |
| --- | --- |
| `data_drift` | Feature distribution shifts |
| `target_drift` | RUL distribution shift |
| `concept_drift` | Model error growth |

## Prometheus Metrics

After `/drift/run`, the API updates these gauges:

| Metric | Meaning |
| --- | --- |
| `data_drift_score` | Latest data drift score |
| `data_drift_detected` | `1` when data drift is detected |
| `target_drift_score` | Latest target drift score |
| `target_drift_detected` | `1` when target drift is detected |
| `concept_drift_score` | Latest concept drift score |
| `concept_drift_detected` | `1` when concept drift is detected |
| `drifted_features_count` | Number of drifted features |

Prometheus reads them from:

```text
http://localhost:8080/metrics
```

## Runtime Paths

| Variable | Default |
| --- | --- |
| `DRIFT_DATA_PATH` | `data/raw` |
| `REPORTS_DIR` | `reports` |
| `PREDICTION_DB_PATH` | `state/predictions.db` |

Docker Compose mounts local `data/raw` into the API container as read-only data
for drift calculation.
