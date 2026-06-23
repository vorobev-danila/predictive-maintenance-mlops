# Monitoring and Drift

[Back to README](../../README.md)

## Contents

- [Drift workflow](#drift-workflow)
- [API endpoints](#api-endpoints)
- [Drift simulations](#drift-simulations)
- [Plotly debug dashboard](#plotly-debug-dashboard)
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

Run prediction-window drift calculation:

```bash
curl -X POST http://localhost:8080/drift/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "FD001", "scenario": "all", "intensity": 1.0}'
```

Read latest report:

```bash
curl http://localhost:8080/drift/latest
```

Read saved drift reports:

```bash
curl http://localhost:8080/drift/reports?limit=10
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

There are two complementary demo flows:

| Flow | Endpoint/tool | Purpose |
| --- | --- | --- |
| Runtime stream | `/predict` + `/drift/run` | Feed simulated rows into the API and monitor the latest prediction window in Prometheus/Grafana |
| Offline simulation report | `/drift/simulate` or `python -m monitoring.drift_simulation` | Generate multi-window JSON/CSV/PNG/Plotly reports for debugging distributions |

The offline simulation endpoint creates controlled drift scenarios from the
CMAPSS current data. The baseline is the unmodified `test_FD001` dataset, and
each window is a synthetic modification of that same dataset. This isolates the
demonstration from natural train/test distribution differences.

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

Run a single offline scenario:

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

Stream simulated prediction rows through the public API:

```bash
uv run python -m data.drift_simulation_stream --scenario all --windows 7 --delay 1
```

## Plotly Debug Dashboard

Each offline simulation also exports a standalone Plotly HTML dashboard. It is
useful when Grafana looks flat and you need to inspect the raw simulation
results without Prometheus scraping delay.

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
| `prediction_summary` | Latest prediction-window summary, when using `/drift/run` |

Simulation reports also contain:

| Section | Meaning |
| --- | --- |
| `windows` | Per-window drift scores and flags |
| `latest_window` | Final simulated window |
| `prediction_error_mae` | MAE for the current synthetic window |
| `prediction_error_p95` | 95th percentile absolute prediction error |
| `plots` | Paths to PNG plots |
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
| `feature_drift_score{feature=...}` | Latest drift score by feature |
| `feature_drift_detected{feature=...}` | Latest drift flag by feature |
| `feature_reference_mean{feature=...}` | Reference mean by feature |
| `feature_current_mean{feature=...}` | Current mean by feature |
| `prediction_window_predicted_rul_mean` | Mean predicted RUL in latest `/drift/run` window |
| `prediction_window_actual_rul_mean` | Mean actual RUL in latest labeled `/drift/run` window |
| `prediction_window_absolute_error_mae` | MAE in latest labeled `/drift/run` window |
| `prediction_window_absolute_error_p95` | p95 absolute error in latest labeled `/drift/run` window |
| `drift_simulation_window` | Latest `/drift/simulate` window number |
| `drift_simulation_scenario{scenario=...}` | Active offline simulation scenario flag |
| `model_prediction_error_mae` | Latest offline synthetic-window MAE |
| `model_prediction_error_p95` | Latest offline synthetic-window p95 absolute error |
| `model_actual_rul_mean` | Latest offline synthetic-window actual RUL mean |
| `model_predicted_rul_mean` | Latest offline synthetic-window predicted RUL mean |

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
feature_drift_score
model_prediction_error_mae
prediction_window_absolute_error_mae
drift_simulation_window
drift_simulation_scenario
```

For visual warnings, use Stat panels for `*_drift_detected` gauges with a
threshold where `0` is OK and `1` is alerting.

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
