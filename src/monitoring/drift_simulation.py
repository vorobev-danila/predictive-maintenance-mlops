import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

from data.data_loader import RAW_FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUPPORTED_SCENARIOS = {"data_drift", "target_drift", "concept_drift", "all"}
SCENARIO_DRIFT_TYPES = {
    "data_drift": {"data_drift"},
    "target_drift": {"target_drift"},
    "concept_drift": {"concept_drift"},
    "all": {"data_drift", "target_drift", "concept_drift"},
}
MAX_SIMULATION_INTENSITY = 1.0
DATA_SHIFT_MULTIPLIER = 1.0


def validate_scenario(scenario):
    if scenario not in SUPPORTED_SCENARIOS:
        supported = ", ".join(sorted(SUPPORTED_SCENARIOS))
        raise ValueError(
            f"Unsupported drift simulation scenario: {scenario}. Use: {supported}"
        )


def get_active_drift_types(scenario, intensity):
    validate_scenario(scenario)
    bounded_intensity = max(0.0, min(float(intensity), MAX_SIMULATION_INTENSITY))
    if bounded_intensity <= 0.0:
        return set()
    return SCENARIO_DRIFT_TYPES[scenario]


def apply_simulation_scenario(
    current_df,
    scenario,
    intensity,
    random_state=42,
    model=None,
    feature_names=None,
):
    validate_scenario(scenario)
    bounded_intensity = max(0.0, min(float(intensity), MAX_SIMULATION_INTENSITY))
    active_drift_types = get_active_drift_types(scenario, bounded_intensity)
    simulated = current_df.copy()

    if "data_drift" in active_drift_types:
        simulated["sensor2"] = (
            simulated["sensor2"] + 2.0 * DATA_SHIFT_MULTIPLIER * bounded_intensity
        )
        simulated["sensor4"] = (
            simulated["sensor4"] + 20.0 * DATA_SHIFT_MULTIPLIER * bounded_intensity
        )
        simulated["sensor11"] = (
            simulated["sensor11"] + 1.0 * DATA_SHIFT_MULTIPLIER * bounded_intensity
        )
        simulated["sensor15"] = (
            simulated["sensor15"] + 0.12 * DATA_SHIFT_MULTIPLIER * bounded_intensity
        )

    if "target_drift" in active_drift_types:
        multiplier = max(0.05, 1.0 - 0.9 * bounded_intensity)
        simulated["RUL"] = (simulated["RUL"] * multiplier).clip(lower=0)

    if "concept_drift" in active_drift_types:
        shuffled = (
            simulated["RUL"]
            .sample(
                frac=1.0,
                random_state=random_state,
            )
            .reset_index(drop=True)
        )
        simulated["RUL"] = (
            (1.0 - bounded_intensity) * simulated["RUL"].reset_index(drop=True)
            + bounded_intensity * shuffled
        ).to_numpy()

    return simulated


def run_drift_simulation(
    scenario="data_drift",
    data_path="data/raw",
    reports_dir="reports/drift",
    dataset_id="FD001",
    model=None,
    feature_names=None,
    windows=6,
    sleep_seconds=0.0,
    on_window=None,
):
    validate_scenario(scenario)
    if model is None:
        raise ValueError("Model is required for drift simulation")

    from monitoring.drift import load_cmapss_dataset

    feature_names = feature_names or RAW_FEATURES
    _, current_base_df = load_cmapss_dataset(data_path, dataset_id)
    simulation_reference_df = current_base_df.copy()
    reference_predictions = _predict(model, simulation_reference_df, feature_names)
    reference_mae = _mean_absolute_error(
        reference_predictions, simulation_reference_df["RUL"]
    )

    window_results = []
    final_current_df = None
    final_predictions = None

    for window in range(1, windows + 1):
        intensity = MAX_SIMULATION_INTENSITY * window / windows
        current_df = apply_simulation_scenario(
            current_base_df,
            scenario=scenario,
            intensity=intensity,
            random_state=42 + window,
            model=model,
            feature_names=feature_names,
        )
        current_predictions = _predict(model, current_df, feature_names)
        window_result = calculate_simulation_window(
            reference_df=simulation_reference_df,
            current_df=current_df,
            reference_mae=reference_mae,
            current_predictions=current_predictions,
            feature_names=feature_names,
            scenario=scenario,
            window=window,
            intensity=intensity,
        )
        window_results.append(window_result)
        final_current_df = current_df
        final_predictions = current_predictions

        if on_window is not None:
            on_window(window_result)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "dataset_id": dataset_id,
        "data_path": str(data_path),
        "reference_dataset": f"test_{dataset_id}_baseline",
        "current_dataset": f"test_{dataset_id}_simulated",
        "windows": window_results,
        "latest_window": window_results[-1],
    }
    report = save_simulation_report(
        report=report,
        reports_dir=reports_dir,
        reference_df=simulation_reference_df,
        current_df=final_current_df,
        current_predictions=final_predictions,
        feature_names=feature_names,
    )
    return report


def calculate_simulation_window(
    reference_df,
    current_df,
    reference_mae,
    current_predictions,
    feature_names,
    scenario,
    window,
    intensity,
):
    from monitoring.drift import calculate_data_drift, calculate_target_drift

    data_drift = calculate_data_drift(reference_df, current_df, columns=feature_names)
    target_drift = calculate_target_drift(reference_df, current_df)
    current_mae = _mean_absolute_error(current_predictions, current_df["RUL"])
    current_errors = _absolute_errors(current_predictions, current_df["RUL"])
    concept_score = (
        0.0
        if reference_mae == 0 and current_mae == 0
        else (
            10.0
            if reference_mae == 0
            else max(0.0, (current_mae - reference_mae) / reference_mae)
        )
    )
    concept_drift = {
        "drift_detected": bool(concept_score > 0.3),
        "score": float(concept_score),
        "threshold": 0.3,
        "reference_mae": float(reference_mae),
        "current_mae": float(current_mae),
        "status": "calculated",
    }

    window_result = {
        "scenario": scenario,
        "window": int(window),
        "intensity": float(intensity),
        "data_drift": _compact_data_drift(data_drift),
        "target_drift": {
            "drift_detected": target_drift["drift_detected"],
            "score": float(target_drift["score"]),
            "threshold": float(target_drift["threshold"]),
            "status": target_drift.get("status", "calculated"),
        },
        "concept_drift": concept_drift,
        "prediction_error_mae": float(current_mae),
        "prediction_error_p95": float(current_errors.quantile(0.95)),
        "actual_rul_mean": float(current_df["RUL"].mean()),
        "predicted_rul_mean": float(pd.Series(current_predictions).mean()),
    }
    return focus_result_on_scenario(window_result, scenario, intensity)


def focus_result_on_scenario(report, scenario, intensity=None):
    if intensity is None:
        intensity = report.get("intensity", MAX_SIMULATION_INTENSITY)
    active_drift_types = get_active_drift_types(scenario, intensity)
    report["active_drift_types"] = sorted(active_drift_types)

    if "data_drift" not in active_drift_types:
        report["data_drift"] = {
            **report["data_drift"],
            "drift_detected": False,
            "score": 0.0,
            "drifted_features_count": 0,
            "drifted_features": [],
            "status": "not_applicable_for_scenario",
        }

    if "target_drift" not in active_drift_types:
        report["target_drift"] = {
            **report["target_drift"],
            "drift_detected": False,
            "score": 0.0,
            "status": "not_applicable_for_scenario",
        }

    if "concept_drift" not in active_drift_types:
        report["concept_drift"] = {
            **report["concept_drift"],
            "drift_detected": False,
            "score": 0.0,
            "status": "not_applicable_for_scenario",
        }

    return report


def save_simulation_report(
    report,
    reports_dir,
    reference_df,
    current_df,
    current_predictions,
    feature_names,
):
    simulations_dir = Path(reports_dir) / "simulations"
    plots_dir = simulations_dir / "plots"
    simulations_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    scenario = report["scenario"]
    timestamp = report["created_at"].replace(":", "-").replace("+", "Z")
    report_path = simulations_dir / f"{scenario}_{timestamp}.json"
    latest_path = simulations_dir / f"{scenario}_latest.json"
    csv_path = simulations_dir / f"{scenario}_{timestamp}.csv"

    plot_paths = create_simulation_plots(
        plots_dir=plots_dir,
        scenario=scenario,
        timestamp=timestamp,
        reference_df=reference_df,
        current_df=current_df,
        current_predictions=current_predictions,
        feature_names=feature_names,
    )
    report["plots"] = {key: str(path) for key, path in plot_paths.items()}
    dashboard_path = create_plotly_dashboard(
        simulations_dir=simulations_dir,
        scenario=scenario,
        timestamp=timestamp,
        report=report,
        reference_df=reference_df,
        current_df=current_df,
        current_predictions=current_predictions,
        feature_names=feature_names,
    )
    report["plotly_dashboard"] = str(dashboard_path)

    pd.DataFrame(report["windows"]).to_csv(csv_path, index=False)
    report["windows_csv"] = str(csv_path)

    _write_json(report_path, report)
    _write_json(latest_path, report)
    return report


def create_simulation_plots(
    plots_dir,
    scenario,
    timestamp,
    reference_df,
    current_df,
    current_predictions,
    feature_names,
):
    top_feature = _choose_plot_feature(reference_df, current_df, feature_names)
    paths = {
        "feature_distribution": plots_dir
        / f"{scenario}_{timestamp}_{top_feature}_distribution.png",
        "target_distribution": plots_dir
        / f"{scenario}_{timestamp}_rul_distribution.png",
        "prediction_error": plots_dir / f"{scenario}_{timestamp}_prediction_error.png",
    }

    _histogram_plot(
        reference_df[top_feature],
        current_df[top_feature],
        title=f"{top_feature} distribution: reference vs simulated",
        xlabel=top_feature,
        path=paths["feature_distribution"],
    )
    _histogram_plot(
        reference_df["RUL"],
        current_df["RUL"],
        title="RUL distribution: reference vs simulated",
        xlabel="RUL",
        path=paths["target_distribution"],
    )
    errors = _absolute_errors(current_predictions, current_df["RUL"])
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30, color="#d95f02", alpha=0.8)
    plt.title("Prediction absolute error distribution")
    plt.xlabel("absolute error")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(paths["prediction_error"])
    plt.close()
    return paths


def create_plotly_dashboard(
    simulations_dir,
    scenario,
    timestamp,
    report,
    reference_df,
    current_df,
    current_predictions,
    feature_names,
):
    dashboard_path = simulations_dir / f"{scenario}_{timestamp}_dashboard.html"
    top_feature = _choose_plot_feature(reference_df, current_df, feature_names)
    errors = _absolute_errors(current_predictions, current_df["RUL"])
    dashboard_data = {
        "scenario": scenario,
        "created_at": report["created_at"],
        "windows": report["windows"],
        "top_feature": top_feature,
        "reference_feature": _sample_values(reference_df[top_feature]),
        "current_feature": _sample_values(current_df[top_feature]),
        "reference_rul": _sample_values(reference_df["RUL"]),
        "current_rul": _sample_values(current_df["RUL"]),
        "prediction_errors": _sample_values(errors),
        "drifted_features": report["latest_window"]["data_drift"]["drifted_features"],
    }

    html = _build_plotly_dashboard_html(dashboard_data)
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def _compact_data_drift(data_drift):
    return {
        "drift_detected": data_drift["drift_detected"],
        "score": float(data_drift["score"]),
        "threshold": float(data_drift["threshold"]),
        "drifted_features_count": int(data_drift["drifted_features_count"]),
        "drifted_features": data_drift["drifted_features"],
        "features": data_drift.get("features", {}),
        "status": data_drift.get("status", "calculated"),
    }


def _choose_plot_feature(reference_df, current_df, feature_names):
    scores = {}
    for feature in feature_names:
        reference_std = reference_df[feature].std()
        if reference_std == 0:
            scores[feature] = 0.0
        else:
            scores[feature] = (
                abs(current_df[feature].mean() - reference_df[feature].mean())
                / reference_std
            )
    return max(scores, key=scores.get)


def _sample_values(series, limit=5000):
    values = pd.Series(series).dropna()
    if len(values) > limit:
        values = values.sample(n=limit, random_state=42)
    return [float(value) for value in values]


def _build_plotly_dashboard_html(data):
    payload = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drift Simulation Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #1f2937;
      background: #f8fafc;
    }}
    header {{
      padding: 24px 32px;
      background: #111827;
      color: #f9fafb;
    }}
    main {{
      padding: 24px 32px 40px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .panel {{
      min-height: 360px;
      padding: 14px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #ffffff;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      padding: 14px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #ffffff;
    }}
    .metric strong {{
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
      color: #6b7280;
    }}
    .metric span {{
      font-size: 24px;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Drift Simulation Dashboard</h1>
    <p id="subtitle"></p>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section class="grid">
      <div class="panel" id="scores"></div>
      <div class="panel" id="flags"></div>
      <div class="panel" id="errors"></div>
      <div class="panel" id="rul-means"></div>
      <div class="panel" id="feature-distribution"></div>
      <div class="panel" id="rul-distribution"></div>
      <div class="panel" id="error-distribution"></div>
      <div class="panel" id="drifted-features"></div>
    </section>
  </main>
  <script>
    const data = {payload};
    const windows = data.windows.map((item) => item.window);
    const latest = data.windows[data.windows.length - 1];
    const layoutBase = {{
      margin: {{ t: 44, r: 18, b: 44, l: 54 }},
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{ family: "Arial, sans-serif" }},
    }};

    document.getElementById("subtitle").textContent =
      `${{data.scenario}} - ${{data.created_at}} - feature: ${{data.top_feature}}`;

    const summary = [
      ["Data drift", latest.data_drift.drift_detected ? "YES" : "NO"],
      ["Target drift", latest.target_drift.drift_detected ? "YES" : "NO"],
      ["Concept drift", latest.concept_drift.drift_detected ? "YES" : "NO"],
      ["Drifted features", latest.data_drift.drifted_features_count],
      ["MAE", latest.prediction_error_mae.toFixed(2)],
      ["Error p95", latest.prediction_error_p95.toFixed(2)],
    ];
    document.getElementById("summary").innerHTML = summary
      .map(([label, value]) => `<div class="metric"><strong>${{label}}</strong><span>${{value}}</span></div>`)
      .join("");

    Plotly.newPlot("scores", [
      {{
        x: windows,
        y: data.windows.map((item) => item.data_drift.score),
        name: "data drift",
        mode: "lines+markers",
      }},
      {{
        x: windows,
        y: data.windows.map((item) => item.target_drift.score),
        name: "target drift",
        mode: "lines+markers",
      }},
      {{
        x: windows,
        y: data.windows.map((item) => item.concept_drift.score),
        name: "concept drift",
        mode: "lines+markers",
      }},
    ], {{ ...layoutBase, title: "Drift scores by simulation window", xaxis: {{ title: "window" }} }});

    Plotly.newPlot("flags", [
      {{
        x: windows,
        y: data.windows.map((item) => Number(item.data_drift.drift_detected)),
        name: "data drift flag",
        type: "bar",
      }},
      {{
        x: windows,
        y: data.windows.map((item) => Number(item.target_drift.drift_detected)),
        name: "target drift flag",
        type: "bar",
      }},
      {{
        x: windows,
        y: data.windows.map((item) => Number(item.concept_drift.drift_detected)),
        name: "concept drift flag",
        type: "bar",
      }},
    ], {{ ...layoutBase, title: "Drift flags", barmode: "group", yaxis: {{ range: [0, 1.2] }} }});

    Plotly.newPlot("errors", [
      {{
        x: windows,
        y: data.windows.map((item) => item.prediction_error_mae),
        name: "MAE",
        mode: "lines+markers",
      }},
      {{
        x: windows,
        y: data.windows.map((item) => item.prediction_error_p95),
        name: "p95 error",
        mode: "lines+markers",
      }},
    ], {{ ...layoutBase, title: "Prediction error by window", xaxis: {{ title: "window" }} }});

    Plotly.newPlot("rul-means", [
      {{
        x: windows,
        y: data.windows.map((item) => item.actual_rul_mean),
        name: "actual RUL mean",
        mode: "lines+markers",
      }},
      {{
        x: windows,
        y: data.windows.map((item) => item.predicted_rul_mean),
        name: "predicted RUL mean",
        mode: "lines+markers",
      }},
    ], {{ ...layoutBase, title: "Actual vs predicted RUL mean" }});

    Plotly.newPlot("feature-distribution", [
      {{
        x: data.reference_feature,
        name: "reference",
        type: "histogram",
        opacity: 0.65,
      }},
      {{
        x: data.current_feature,
        name: "simulated",
        type: "histogram",
        opacity: 0.65,
      }},
    ], {{ ...layoutBase, title: `${{data.top_feature}} distribution`, barmode: "overlay" }});

    Plotly.newPlot("rul-distribution", [
      {{ x: data.reference_rul, name: "reference", type: "histogram", opacity: 0.65 }},
      {{ x: data.current_rul, name: "simulated", type: "histogram", opacity: 0.65 }},
    ], {{ ...layoutBase, title: "RUL distribution", barmode: "overlay" }});

    Plotly.newPlot("error-distribution", [
      {{
        x: data.prediction_errors,
        name: "absolute error",
        type: "histogram",
        marker: {{ color: "#dc2626" }},
      }},
    ], {{ ...layoutBase, title: "Prediction absolute error distribution" }});

    Plotly.newPlot("drifted-features", [
      {{
        x: data.drifted_features,
        y: data.drifted_features.map(() => 1),
        type: "bar",
        marker: {{ color: "#7c3aed" }},
      }},
    ], {{
      ...layoutBase,
      title: "Features marked as drifted in the final window",
      xaxis: {{ tickangle: -45 }},
      yaxis: {{ visible: false }},
    }});
  </script>
</body>
</html>
"""


def _histogram_plot(reference, current, title, xlabel, path):
    plt.figure(figsize=(8, 4))
    plt.hist(reference, bins=30, alpha=0.55, label="reference", color="#1b9e77")
    plt.hist(current, bins=30, alpha=0.55, label="simulated", color="#7570b3")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _predict(model, df, feature_names):
    return model.predict(df[feature_names])


def _absolute_errors(predictions, actual):
    return (
        pd.Series(predictions).reset_index(drop=True)
        - pd.Series(actual).reset_index(drop=True)
    ).abs()


def _mean_absolute_error(predictions, actual):
    return float(_absolute_errors(predictions, actual).mean())


def _write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run local drift simulation report.")
    parser.add_argument(
        "--scenario", choices=sorted(SUPPORTED_SCENARIOS), default="all"
    )
    parser.add_argument("--data-path", default="data/raw")
    parser.add_argument("--reports-dir", default="reports/drift")
    parser.add_argument("--model-path", default="models/pipeline.pkl")
    parser.add_argument("--features-path", default="models/features.json")
    parser.add_argument("--windows", type=int, default=6)
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    with Path(args.features_path).open("r", encoding="utf-8") as file:
        feature_names = json.load(file)
    report = run_drift_simulation(
        scenario=args.scenario,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
        model=model,
        feature_names=feature_names,
        windows=args.windows,
    )
    print(json.dumps(report["latest_window"], indent=2))


if __name__ == "__main__":
    main()
