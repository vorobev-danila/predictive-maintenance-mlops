from pathlib import Path

import pandas as pd

from data.data_loader import RAW_FEATURES
from monitoring.drift_simulation import run_drift_simulation


class RuleBasedRulModel:
    def predict(self, features):
        return 10.0 - pd.Series(features["cycle"]).reset_index(drop=True)


def write_rows(path: Path, rows):
    content = "\n".join(" ".join(str(value) for value in row) for row in rows)
    path.write_text(f"{content}\n", encoding="utf-8")


def make_row(unit, cycle, base_value):
    settings = [0.0, 0.0, 100.0]
    sensors = [base_value + sensor_idx * 0.1 for sensor_idx in range(1, 22)]
    return [unit, cycle, *settings, *sensors]


def write_simulation_dataset(data_path):
    data_path.mkdir(parents=True, exist_ok=True)
    train_rows = []
    test_rows = []
    for unit in range(1, 4):
        for cycle in range(1, 11):
            train_rows.append(make_row(unit=unit, cycle=cycle, base_value=1.0))
            test_rows.append(make_row(unit=unit, cycle=cycle, base_value=1.0))

    write_rows(data_path / "train_FD001.txt", train_rows)
    write_rows(data_path / "test_FD001.txt", test_rows)
    write_rows(data_path / "RUL_FD001.txt", [[0], [0], [0]])


def run_scenario(tmp_path, scenario):
    data_path = tmp_path / "data" / "raw"
    reports_dir = tmp_path / "reports" / "drift"
    write_simulation_dataset(data_path)
    return run_drift_simulation(
        scenario=scenario,
        data_path=data_path,
        reports_dir=reports_dir,
        model=RuleBasedRulModel(),
        feature_names=RAW_FEATURES,
        windows=4,
    )


def test_data_drift_simulation_sets_data_flag(tmp_path):
    report = run_scenario(tmp_path, "data_drift")

    latest = report["latest_window"]

    assert latest["data_drift"]["drift_detected"] is True
    assert latest["target_drift"]["drift_detected"] is False
    assert latest["concept_drift"]["drift_detected"] is False
    assert latest["data_drift"]["drifted_features_count"] > 0
    assert Path(report["plots"]["feature_distribution"]).exists()
    dashboard = Path(report["plotly_dashboard"])
    assert dashboard.exists()
    assert "Plotly.newPlot" in dashboard.read_text(encoding="utf-8")


def test_target_drift_simulation_sets_target_flag(tmp_path):
    report = run_scenario(tmp_path, "target_drift")

    latest = report["latest_window"]

    assert latest["target_drift"]["drift_detected"] is True
    assert latest["data_drift"]["drift_detected"] is False
    assert latest["concept_drift"]["drift_detected"] is False
    assert latest["target_drift"]["score"] > 0.3
    assert Path(report["plots"]["target_distribution"]).exists()


def test_concept_drift_simulation_sets_concept_flag(tmp_path):
    report = run_scenario(tmp_path, "concept_drift")

    latest = report["latest_window"]

    assert latest["concept_drift"]["drift_detected"] is True
    assert latest["data_drift"]["drift_detected"] is False
    assert latest["target_drift"]["drift_detected"] is False
    assert (
        latest["concept_drift"]["current_mae"]
        > latest["concept_drift"]["reference_mae"]
    )
    assert Path(report["plots"]["prediction_error"]).exists()


def test_all_drift_simulation_sets_all_flags(tmp_path):
    report = run_scenario(tmp_path, "all")

    latest = report["latest_window"]

    assert latest["data_drift"]["drift_detected"] is True
    assert latest["target_drift"]["drift_detected"] is True
    assert latest["concept_drift"]["drift_detected"] is True
