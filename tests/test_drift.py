import pandas as pd

from monitoring.drift import (
    add_test_rul,
    add_train_rul,
    calculate_concept_drift,
    calculate_data_drift,
    calculate_target_drift,
    save_drift_report,
    load_latest_drift_report,
)


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


def test_save_and_load_latest_drift_report(tmp_path):
    report = {
        "created_at": "2026-06-01T12:00:00+00:00",
        "data_drift": {},
        "target_drift": {},
        "concept_drift": {},
    }

    save_drift_report(report, tmp_path)

    assert load_latest_drift_report(tmp_path) == report
