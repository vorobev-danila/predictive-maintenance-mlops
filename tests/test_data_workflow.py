from pathlib import Path

import pandas as pd
import pytest

from data.data_loader import load_and_prepare_data
from features.feature_engineering import prepare_data, select_all_sensors


def write_cmapss_file(path: Path, rows):
    path.write_text("\n".join(" ".join(str(value) for value in row) for row in rows))


def make_cmapss_row(unit, cycle):
    settings = [0.0, 0.0, 100.0]
    sensors = [float(cycle + sensor_idx + unit) for sensor_idx in range(1, 22)]
    return [unit, cycle, *settings, *sensors]


def test_load_and_prepare_data_adds_rul_column(tmp_path):
    train_rows = [
        make_cmapss_row(unit=1, cycle=1),
        make_cmapss_row(unit=1, cycle=2),
        make_cmapss_row(unit=2, cycle=1),
        make_cmapss_row(unit=2, cycle=2),
        make_cmapss_row(unit=2, cycle=3),
    ]
    test_rows = [
        make_cmapss_row(unit=1, cycle=1),
        make_cmapss_row(unit=2, cycle=1),
    ]

    write_cmapss_file(tmp_path / "train_FD001.txt", train_rows)
    write_cmapss_file(tmp_path / "test_FD001.txt", test_rows)
    write_cmapss_file(tmp_path / "RUL_FD001.txt", [[10], [20]])

    train_with_rul, test, rul = load_and_prepare_data(data_path=tmp_path)

    assert "RUL" in train_with_rul.columns
    assert train_with_rul.loc[train_with_rul["unit"] == 1, "RUL"].tolist() == [1, 0]
    assert train_with_rul.loc[train_with_rul["unit"] == 2, "RUL"].tolist() == [2, 1, 0]
    assert test.shape[0] == 2
    assert rul["RUL"].tolist() == [10, 20]


def make_feature_frame(n_units=10, cycles_per_unit=4):
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            rows.append(
                {
                    "unit": unit,
                    "cycle": cycle,
                    "setting1": 0.1 * unit,
                    "setting2": 0.01 * cycle,
                    "setting3": 100.0,
                    "sensor1": cycle,
                    "sensor2": unit + cycle,
                    "sensor3": 1.0,
                    "RUL": cycles_per_unit - cycle,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_select_all_sensors_skips_constant_sensor():
    frame = make_feature_frame()

    sensors = select_all_sensors(frame)

    assert "sensor1" in sensors
    assert "sensor2" in sensors
    assert "sensor3" not in sensors


def test_prepare_data_splits_by_units_and_scales_features():
    frame = make_feature_frame()

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, base_features = (
        prepare_data(
            frame,
            ["sensor1", "sensor2"],
        )
    )

    assert X_train.shape[1] == 5
    assert X_val.shape[1] == 5
    assert X_test.shape[1] == 5
    assert len(y_train) == X_train.shape[0]
    assert len(y_val) == X_val.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert base_features == ["sensor1", "sensor2", "setting1", "setting2", "setting3"]
    assert hasattr(scaler, "mean_")
