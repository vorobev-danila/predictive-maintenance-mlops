# Monitoring and Drift

[← Back to README](../../README.md)

## Contents

- [Drift workflow](#drift-workflow)
- [API endpoints](#api-endpoints)
- [Report format](#report-format)
- [Prometheus metrics](#prometheus-metrics)
- [Prometheus alerts](#prometheus-alerts)
- [Streamlit UI alerts](#streamlit-ui-alerts)
- [Runtime paths](#runtime-paths)

## Drift Workflow

The current demo drift workflow compares clean FD001 train data with the latest
simulated FD001 prediction window received through `/predict`:

```text
reference = clean train_FD001 rows matching current unit/cycle
current   = simulated_FD001_window_last_N
```

By default, `N` is controlled by `DRIFT_WINDOW_SIZE=10`. This keeps Grafana and
Prometheus focused on the recent incoming stream instead of recalculating drift
over the full simulated dataset.

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

Read saved drift reports:

```bash
curl http://localhost:8080/drift/reports?limit=10
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

## Prometheus Alerts

Prometheus loads drift alerting rules from:

```text
prometheus_alert_rules.yml
```

Configured alerts:

| Alert | Expression |
| --- | --- |
| `DataDriftDetected` | `data_drift_detected == 1` |
| `TargetDriftDetected` | `target_drift_detected == 1` |
| `ConceptDriftDetected` | `concept_drift_detected == 1` |
| `AnyDriftDetected` | any drift flag is active |

Check active alerts:

```bash
curl http://localhost:9090/api/v1/alerts
```

## Streamlit UI Alerts

The Streamlit UI displays real Prometheus alerts from:

```text
http://prometheus:9090/api/v1/alerts
```

Detailed drift flags and scores are still available in Grafana panels and in
the latest JSON report from `/drift/latest`.

Open the UI locally:

```text
http://localhost:8501
```

## Runtime Paths

| Variable | Default |
| --- | --- |
| `DRIFT_DATA_PATH` | `data/raw` |
| `REPORTS_DIR` | `reports` |
| `PREDICTION_DB_PATH` | `state/predictions.db` |

Docker Compose mounts local `data/raw` into the API container as read-only data
for drift calculation.
