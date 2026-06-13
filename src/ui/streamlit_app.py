import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from data.data_loader import RAW_FEATURES

API_URL = os.getenv("API_URL", "http://127.0.0.1:8080").rstrip("/")
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", API_URL).rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
PROMETHEUS_PUBLIC_URL = os.getenv("PROMETHEUS_PUBLIC_URL", PROMETHEUS_URL).rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000").rstrip("/")
MLFLOW_API_URL = os.getenv("MLFLOW_API_URL", MLFLOW_URL).rstrip("/")
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9001").rstrip("/")


def api_get(path, params=None):
    return get_json(f"{API_URL}{path}", params=params)


def get_json(url, params=None):
    if params:
        url = f"{url}?{urlencode(params)}"
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post(path, payload=None):
    return post_json(f"{API_URL}{path}", payload)


def post_json(url, payload=None):
    body = json.dumps(payload or {}).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def safe_call(callback, fallback=None):
    try:
        return callback()
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        st.error(str(error))
        return fallback


def render_drift_alerts(report):
    if not report:
        st.warning("Drift report is not available")
        return

    flags = {
        "data": report.get("data_drift", {}).get("drift_detected", False),
        "target": report.get("target_drift", {}).get("drift_detected", False),
        "concept": report.get("concept_drift", {}).get("drift_detected", False),
    }
    for name, detected in flags.items():
        if detected:
            st.error(f"{name} drift: YES")
        else:
            st.success(f"{name} drift: NO")


def render_prometheus_alerts():
    response = safe_call(
        lambda: get_json(f"{PROMETHEUS_URL}/api/v1/alerts"),
        fallback=None,
    )
    if not response:
        st.warning("Prometheus alerts are not available")
        return

    alerts = response.get("data", {}).get("alerts", [])
    if not alerts:
        st.success("Prometheus alerts: none")
        return

    rows = []
    for alert in alerts:
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        state = alert.get("state")
        rows.append(
            {
                "state": state,
                "alert": labels.get("alertname"),
                "severity": labels.get("severity"),
                "drift_type": labels.get("drift_type"),
                "summary": annotations.get("summary"),
            }
        )
        if state == "firing":
            st.error(f"Prometheus alert firing: {labels.get('alertname')}")
        else:
            st.info(f"Prometheus alert pending: {labels.get('alertname')}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_inference_page():
    st.header("Inference")
    if st.button("Random"):
        sample = safe_call(lambda: api_get("/samples/random", {"dataset_id": "FD001"}))
        if sample:
            payload = sample["payload"]
            for feature in RAW_FEATURES:
                st.session_state[f"input_{feature}"] = float(payload[feature])
            st.session_state["include_actual_rul"] = True
            st.session_state["input_actual_rul"] = float(payload["actual_rul"])
            st.success(f"Loaded random sample from {sample['source']}")

    with st.form("inference_form"):
        cols = st.columns(4)
        for index, feature in enumerate(RAW_FEATURES):
            state_key = f"input_{feature}"
            if state_key not in st.session_state:
                st.session_state[state_key] = 100.0 if feature == "setting3" else 0.0
            cols[index % 4].number_input(
                feature,
                value=float(st.session_state[state_key]),
                step=1.0,
                key=state_key,
            )
        include_actual = st.checkbox("actual_rul", key="include_actual_rul")
        if "input_actual_rul" not in st.session_state:
            st.session_state["input_actual_rul"] = 0.0
        st.number_input("actual_rul value", step=1.0, key="input_actual_rul")
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            feature: float(st.session_state[f"input_{feature}"])
            for feature in RAW_FEATURES
        }
        if include_actual:
            payload["actual_rul"] = float(st.session_state["input_actual_rul"])
        result = safe_call(lambda: api_post("/predict", payload))
        if result:
            st.metric("Predicted RUL", round(float(result["rul"]), 3))
            st.json(result)


def render_predictions_page():
    st.header("Recent Predictions")
    predictions = safe_call(
        lambda: api_get("/predictions/recent", {"limit": 50}),
        fallback=[],
    )
    if not predictions:
        st.info("No predictions yet")
        return

    rows = []
    for item in predictions:
        rows.append(
            {
                "id": item["id"],
                "created_at": item["created_at"],
                "predicted_rul": item["predicted_rul"],
                "actual_rul": item.get("actual_rul"),
                "anomaly_flag": item.get("anomaly_flag"),
                "model_version": item.get("model_version"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_drift_page():
    st.header("Drift")
    report = safe_call(lambda: api_get("/drift/latest"))
    render_prometheus_alerts()
    if not report:
        return

    cols = st.columns(4)
    cols[0].metric("Data Score", round(report["data_drift"]["score"], 3))
    cols[1].metric("Target Score", round(report["target_drift"]["score"], 3))
    cols[2].metric("Concept Score", round(report["concept_drift"]["score"], 3))
    cols[3].metric(
        "Drifted Features",
        report["data_drift"].get("drifted_features_count", 0),
    )

    st.link_button("Open Grafana", GRAFANA_URL)
    components.iframe(GRAFANA_URL, height=760, scrolling=True)


def render_reports_page():
    st.header("Drift Reports")
    reports = safe_call(lambda: api_get("/drift/reports", {"limit": 20}), fallback=[])
    if not reports:
        st.info("No drift reports yet")
        return

    summary = [
        {
            "file": item["file"],
            "created_at": item.get("created_at"),
            "reference": item.get("reference_dataset"),
            "current": item.get("current_dataset"),
            "data": item.get("data_drift"),
            "target": item.get("target_drift"),
            "concept": item.get("concept_drift"),
        }
        for item in reports
    ]
    st.dataframe(pd.DataFrame(summary), use_container_width=True)
    selected = st.selectbox("Report", [item["file"] for item in reports])
    report = next(item["report"] for item in reports if item["file"] == selected)
    st.json(report)


def render_experiments_page():
    st.header("Experiments")
    st.link_button("Open MLflow", f"{MLFLOW_URL}/#/experiments")
    experiments = safe_call(
        lambda: post_json(
            f"{MLFLOW_API_URL}/api/2.0/mlflow/experiments/search",
            {"max_results": 100},
        ),
        fallback=None,
    )
    if not experiments or not experiments.get("experiments"):
        st.info("No MLflow experiments found")
        return

    experiment_rows = []
    for item in experiments["experiments"]:
        experiment_id = item.get("experiment_id")
        experiment_rows.append(
            {
                "experiment_id": experiment_id,
                "name": item.get("name"),
                "lifecycle_stage": item.get("lifecycle_stage"),
                "url": f"{MLFLOW_URL}/#/experiments/{experiment_id}/runs",
            }
        )
    st.dataframe(
        pd.DataFrame(experiment_rows),
        use_container_width=True,
        column_config={"url": st.column_config.LinkColumn("url")},
    )

    selected_experiment = st.selectbox(
        "Experiment",
        experiment_rows,
        format_func=lambda item: f"{item['name']} ({item['experiment_id']})",
    )
    st.link_button("Open Selected Experiment", selected_experiment["url"])

    runs = safe_call(
        lambda: post_json(
            f"{MLFLOW_API_URL}/api/2.0/mlflow/runs/search",
            {
                "experiment_ids": [selected_experiment["experiment_id"]],
                "max_results": 20,
                "order_by": ["attributes.start_time DESC"],
            },
        ),
        fallback=None,
    )
    if not runs or not runs.get("runs"):
        st.info("No MLflow runs found for this experiment")
        return

    run_rows = []
    for run in runs["runs"]:
        info = run.get("info", {})
        metrics = {
            metric.get("key"): metric.get("value")
            for metric in run.get("data", {}).get("metrics", [])
        }
        run_id = info.get("run_id")
        run_rows.append(
            {
                "run_name": info.get("run_name"),
                "status": info.get("status"),
                "official_test_mae": metrics.get("official_test_mae"),
                "official_test_rmse": metrics.get("official_test_rmse"),
                "official_test_r2": metrics.get("official_test_r2"),
                "url": (
                    f"{MLFLOW_URL}/#/experiments/"
                    f"{selected_experiment['experiment_id']}/runs/{run_id}"
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(run_rows),
        use_container_width=True,
        column_config={"url": st.column_config.LinkColumn("url")},
    )


def render_retraining_control():
    if st.button("Run Retraining"):
        result = safe_call(lambda: api_post("/retrain"))
        if result:
            st.json(result)


def main():
    st.set_page_config(page_title="Predictive Maintenance MLOps", layout="wide")
    st.title("Predictive Maintenance MLOps")
    st.sidebar.link_button("FastAPI", f"{API_PUBLIC_URL}/docs")
    st.sidebar.link_button("Grafana", GRAFANA_URL)
    st.sidebar.link_button("Prometheus", PROMETHEUS_PUBLIC_URL)
    st.sidebar.link_button("MLflow", MLFLOW_URL)
    st.sidebar.link_button("MinIO", MINIO_URL)
    st.sidebar.divider()
    render_retraining_control()
    page = st.sidebar.radio(
        "Page",
        ["Inference", "Recent Predictions", "Drift", "Drift Reports", "Experiments"],
    )

    if page == "Inference":
        render_inference_page()
    elif page == "Recent Predictions":
        render_predictions_page()
    elif page == "Drift":
        render_drift_page()
    elif page == "Drift Reports":
        render_reports_page()
    else:
        render_experiments_page()


if __name__ == "__main__":
    main()
