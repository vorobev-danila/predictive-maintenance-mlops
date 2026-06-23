import pandas as pd

from monitoring.drift import (
    MAX_DISPLAY_DRIFT_SCORE,
    add_test_rul,
    add_train_rul,
    build_clean_reference_window,
    calculate_calibrated_thresholds,
    calculate_concept_drift,
    calculate_data_drift,
    calculate_target_drift,
    run_simulated_test_drift_report,
    save_drift_report,
    load_latest_drift_report,
)
from monitoring.drift_simulation import get_active_drift_types


class MeanFeatureModel:
    def predict(self, features):
        return features.mean(axis=1).to_numpy()


def test_add_train_and_test_rul_columns():
    train = pd.DataFrame({"unit": [1, 1, 2], "cycle": [1, 2, 1]})
    test = pd.DataFrame({"unit": [1, 1, 2], "cycle": [1, 2, 1]})
    rul = pd.DataFrame({"final_rul": [10, 20]})

    train_with_rul = add_train_rul(train)
    test_with_rul = add_test_rul(test, rul)

    assert train_with_rul["RUL"].tolist() == [1, 0, 0]
    assert test_with_rul["RUL"].tolist() == [11.0, 10.0, 20.0]


def test_calculate_data_and_target_drift_detects_shift():
    reference = pd.DataFrame(
        {
            "sensor1": [10.0, 11.0, 12.0, 13.0],
            "sensor2": [1.0, 1.0, 1.0, 1.0],
            "RUL": [80.0, 70.0, 60.0, 50.0],
        }
    )
    current = pd.DataFrame(
        {
            "sensor1": [20.0, 21.0, 22.0, 23.0],
            "sensor2": [1.0, 1.0, 1.0, 1.0],
            "RUL": [30.0, 25.0, 20.0, 15.0],
        }
    )

    data_drift = calculate_data_drift(
        reference,
        current,
        columns=["sensor1", "sensor2"],
        threshold=0.3,
    )
    target_drift = calculate_target_drift(reference, current, threshold=0.3)

    assert data_drift["drift_detected"] is True
    assert data_drift["drifted_features"] == ["sensor1"]
    assert data_drift["features"]["sensor2"]["status"] == "skipped"
    assert target_drift["drift_detected"] is True


def test_constant_feature_shift_uses_bounded_drift_score():
    reference = pd.DataFrame({"sensor1": [1.0, 1.0, 1.0, 1.0]})
    current = pd.DataFrame({"sensor1": [2.0, 2.0, 2.0, 2.0]})

    data_drift = calculate_data_drift(
        reference,
        current,
        columns=["sensor1"],
        threshold=0.3,
    )

    assert data_drift["drift_detected"] is True
    assert data_drift["score"] == MAX_DISPLAY_DRIFT_SCORE
    assert data_drift["features"]["sensor1"]["score"] == MAX_DISPLAY_DRIFT_SCORE


def test_constant_feature_ignores_float_noise():
    reference = pd.DataFrame({"sensor10": [1.3, 1.3, 1.3, 1.3]})
    current = pd.DataFrame({"sensor10": [1.2999999999999996, 1.2999999999999996]})

    data_drift = calculate_data_drift(
        reference,
        current,
        columns=["sensor10"],
        threshold=0.3,
    )

    assert data_drift["drift_detected"] is False
    assert data_drift["score"] == 0.0
    assert data_drift["features"]["sensor10"]["status"] == "skipped"


def test_calculate_concept_drift_detects_error_growth():
    result = calculate_concept_drift(
        reference_predictions=[10.0, 20.0, 30.0],
        reference_actual=[11.0, 19.0, 31.0],
        current_predictions=[10.0, 20.0, 30.0],
        current_actual=[20.0, 35.0, 45.0],
        threshold=0.25,
    )

    assert result["drift_detected"] is True
    assert result["current_mae"] > result["reference_mae"]


def test_all_simulation_scenario_runs_all_drifts_together():
    expected_drift_types = {
        "data_drift",
        "target_drift",
        "concept_drift",
    }

    assert get_active_drift_types("all", 0.0) == set()
    assert get_active_drift_types("all", 1 / 6) == expected_drift_types
    assert get_active_drift_types("all", 3 / 6) == expected_drift_types
    assert get_active_drift_types("all", 5 / 6) == expected_drift_types


def test_save_and_load_latest_drift_report(tmp_path):
    report = {
        "created_at": "2026-06-01T12:00:00+00:00",
        "data_drift": {},
        "target_drift": {},
        "concept_drift": {},
    }

    save_drift_report(report, tmp_path)

    assert load_latest_drift_report(tmp_path) == report


def test_calibrated_thresholds_are_derived_from_reference_windows():
    reference = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "cycle": [1.0, 2.0, 1.0, 2.0],
            "sensor1": [10.0, 11.0, 12.0, 13.0],
            "RUL": [1.0, 0.0, 1.0, 0.0],
        }
    )

    thresholds = calculate_calibrated_thresholds(
        reference_df=reference,
        baseline_current_df=reference,
        model=MeanFeatureModel(),
        feature_names=["cycle", "sensor1"],
        window_size=2,
    )

    assert thresholds["data_drift"] >= 0.0
    assert thresholds["target_drift"] >= 0.0
    assert thresholds["concept_drift"] >= 0.0
    assert thresholds["threshold_source"].startswith(
        "calibrated_data_p95_target_p85_concept_p95"
    )


def test_clean_reference_window_matches_prediction_unit_cycles():
    train_reference = pd.DataFrame(
        {
            "unit": [1.0, 1.0, 2.0],
            "cycle": [1.0, 2.0, 1.0],
            "sensor1": [10.0, 20.0, 30.0],
            "RUL": [1.0, 0.0, 5.0],
        }
    )
    window = pd.DataFrame(
        {
            "unit": [1.0, 1.0],
            "cycle": [2.0, 1.0],
            "sensor1": [99.0, 88.0],
            "RUL": [0.0, 1.0],
            "predicted_rul": [3.0, 4.0],
        }
    )

    reference_window = build_clean_reference_window(
        train_reference,
        window,
        feature_names=["cycle", "sensor1"],
    )

    assert reference_window.attrs["source"] == "clean_train_matching_prediction_window"
    assert reference_window["sensor1"].tolist() == [20.0, 10.0]
    assert reference_window["RUL"].tolist() == [0.0, 1.0]


def test_simulated_report_uses_clean_train_reference_and_simulated_train_current(
    tmp_path,
):
    data_path = tmp_path / "data"
    data_path.mkdir()
    rows = [
        "1 1 0 0 100 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21",
        "1 2 0 0 100 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22",
        "2 1 0 0 100 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23",
        "2 2 0 0 100 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24",
    ]
    (data_path / "train_FD001.txt").write_text("\n".join(rows) + "\n")
    (data_path / "test_FD001.txt").write_text("\n".join(rows) + "\n")
    (data_path / "RUL_FD001.txt").write_text("5\n7\n")

    feature_names = ["cycle", *[f"sensor{i}" for i in range(1, 22)]]
    report = run_simulated_test_drift_report(
        data_path=data_path,
        reports_dir=tmp_path / "reports",
        model=MeanFeatureModel(),
        feature_names=feature_names,
        scenario="target_drift",
        intensity=1.0,
    )

    assert report["reference_dataset"] == "train_FD001_baseline"
    assert report["current_dataset"] == "simulated_FD001"
    assert report["target_drift"]["status"] == "calculated"
    assert report["data_drift"]["status"] == "not_applicable_for_scenario"
    assert report["prediction_summary"]["window_rows"] == 4
