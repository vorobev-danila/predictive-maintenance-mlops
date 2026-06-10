# Monitoring and Drift

[Back to README](../../README.md)

## Contents

- [Drift workflow](#drift-workflow)
- [API endpoints](#api-endpoints)
- [Drift simulations](#drift-simulations)
- [Plotly debug dashboard](#plotly-debug-dashboard)
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

Run a demo simulation and update Prometheus metrics window by window:

```bash
curl -X POST http://localhost:8080/drift/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "all", "dataset_id": "FD001", "windows": 8, "sleep_seconds": 6}'
```

Use `sleep_seconds` greater than the Prometheus `scrape_interval` so Grafana can
capture intermediate windows instead of only the final gauge values.

## Drift Simulations

The simulation endpoint creates controlled drift scenarios from the CMAPSS
current data. The baseline is the unmodified `test_FD001` dataset, and each
window is a synthetic modification of that same dataset. This isolates the
demonstration from natural train/test distribution differences. The API sends
each synthetic window through the loaded model, updates Prometheus gauges, and
writes JSON, CSV, PNG and Plotly HTML artifacts for the demo.

Single-drift scenarios are intentionally focused: only the selected drift flag is
reported as active, even when the synthetic change has secondary effects on raw
model error. This keeps the demo readable and lets each dashboard show one
clear signal at a time.

Supported scenarios:

| Scenario | What changes | Main signal |
| --- | --- | --- |
| `data_drift` | Selected sensor distributions are shifted | Only `data_drift_detected` |
| `target_drift` | RUL distribution is shifted | Only `target_drift_detected` |
| `concept_drift` | Feature-to-target relation is broken while features and target distribution stay similar | Only `concept_drift_detected` |
| `all` | Combines data, target and concept drift effects | All three drift flags |

Run a single scenario:

```bash
curl -X POST http://localhost:8080/drift/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "data_drift", "dataset_id": "FD001", "windows": 6}'
```

PowerShell example:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8080/drift/simulate" `
  -ContentType "application/json" `
  -Body '{"scenario":"all","dataset_id":"FD001","windows":8,"sleep_seconds":6}'
```

Local report-only mode without FastAPI:

```bash
uv run python -m monitoring.drift_simulation --scenario all --windows 6
```

## Plotly Debug Dashboard

Each simulation also exports a standalone Plotly HTML dashboard. It is useful
when Grafana looks flat and you need to inspect the raw simulation results
without Prometheus scraping delay.

Docker Compose path on the host:

```text
state/reports/drift/simulations/<scenario>_<timestamp>_dashboard.html
```

Local path:

```text
reports/drift/simulations/<scenario>_<timestamp>_dashboard.html
```

The dashboard contains:

| Panel | Meaning |
| --- | --- |
| Drift scores | Data, target and concept drift scores by window |
| Drift flags | Binary drift indicators by window |
| Prediction error | MAE and p95 absolute error by window |
| RUL means | Actual vs predicted RUL mean |
| Feature distribution | Reference/current histogram for the most shifted feature |
| RUL distribution | Reference/current RUL histogram |
| Error distribution | Final-window prediction error histogram |
| Drifted features | Features marked as drifted in the final window |

## Report Format

Reports are written as JSON/CSV/PNG artifacts:

```text
reports/drift/latest.json
reports/drift/drift_report_<timestamp>.json
reports/drift/simulations/<scenario>_latest.json
reports/drift/simulations/<scenario>_<timestamp>.json
reports/drift/simulations/<scenario>_<timestamp>.csv
reports/drift/simulations/<scenario>_<timestamp>_dashboard.html
reports/drift/simulations/plots/*.png
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

Simulation reports also contain:

| Section | Meaning |
| --- | --- |
| `windows` | Per-window drift scores and flags |
| `latest_window` | Final simulated window |
| `prediction_error_mae` | MAE for the current synthetic window |
| `prediction_error_p95` | 95th percentile absolute prediction error |
| `artifacts` | Paths to CSV and PNG plots |
| `plotly_dashboard` | Path to standalone Plotly HTML dashboard |

## Prometheus Metrics

After `/drift/run` or `/drift/simulate`, the API updates these gauges:

| Metric | Meaning |
| --- | --- |
| `data_drift_score` | Latest data drift score |
| `data_drift_detected` | `1` when data drift is detected |
| `target_drift_score` | Latest target drift score |
| `target_drift_detected` | `1` when target drift is detected |
| `concept_drift_score` | Latest concept drift score |
| `concept_drift_detected` | `1` when concept drift is detected |
| `drifted_features_count` | Number of drifted features |
| `drift_simulation_window` | Latest simulation window number |
| `drift_simulation_scenario{scenario=...}` | Active simulation scenario flag |
| `model_prediction_error_mae` | Latest synthetic-window MAE |
| `model_prediction_error_p95` | Latest synthetic-window p95 absolute error |
| `model_actual_rul_mean` | Latest synthetic-window actual RUL mean |
| `model_predicted_rul_mean` | Latest synthetic-window predicted RUL mean |

Prometheus reads them from:

```text
http://localhost:8080/metrics
```

Useful Grafana queries for the demo:

```promql
data_drift_score
target_drift_score
concept_drift_score
data_drift_detected
target_drift_detected
concept_drift_detected
drifted_features_count
model_prediction_error_mae
model_prediction_error_p95
drift_simulation_window
drift_simulation_scenario
```

For visual warnings, use Stat panels for `*_drift_detected` gauges with a
threshold where `0` is OK and `1` is alerting.

## Runtime Paths

| Variable | Default |
| --- | --- |
| `DRIFT_DATA_PATH` | `data/raw` |
| `REPORTS_DIR` | `reports` |
| `PREDICTION_DB_PATH` | `state/predictions.db` |

Docker Compose mounts local `data/raw` into the API container as read-only data
for drift calculation.
