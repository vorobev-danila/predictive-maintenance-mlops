from pathlib import Path

import pandas as pd

from data.data_loader import (
    RAW_FEATURES,
    get_official_test_last_rows,
    iter_official_test_rows,
    iter_train_rows,
    load_and_prepare_data,
    load_official_test_with_rul,
)
from data.drift_simulation_stream import (
    build_simulated_window,
    calculate_window_intensity,
)


def write_cmapss_file(path: Path, rows):
    path.write_text("\n".join(" ".join(str(value) for value in row) for row in rows))


def make_cmapss_row(unit, cycle):
    settings = [0.0, 0.0, 100.0]
    sensors = [float(cycle + sensor_idx + unit) for sensor_idx in range(1, 22)]
    return [unit, cycle, *settings, *sensors]


def test_raw_features_contain_cycle_settings_and_all_sensors():
    assert len(RAW_FEATURES) == 25
    assert RAW_FEATURES[:4] == ["cycle", "setting1", "setting2", "setting3"]
    assert RAW_FEATURES[-1] == "sensor21"


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
        make_cmapss_row(unit=1, cycle=2),
        make_cmapss_row(unit=2, cycle=1),
    ]

    write_cmapss_file(tmp_path / "train_FD001.txt", train_rows)
    write_cmapss_file(tmp_path / "test_FD001.txt", test_rows)
    write_cmapss_file(tmp_path / "RUL_FD001.txt", [[10], [20]])

    train_with_rul, test, rul = load_and_prepare_data(data_path=tmp_path)
    last_rows, y_true = get_official_test_last_rows(test, rul)

    assert "RUL" in train_with_rul.columns
    assert train_with_rul.loc[train_with_rul["unit"] == 1, "RUL"].tolist() == [1, 0]
    assert train_with_rul.loc[train_with_rul["unit"] == 2, "RUL"].tolist() == [2, 1, 0]
    assert last_rows["cycle"].tolist() == [2, 1]
    assert y_true.tolist() == [10, 20]


def test_load_official_test_with_rul_adds_cycle_level_targets(tmp_path):
    train_rows = [
        make_cmapss_row(unit=1, cycle=1),
        make_cmapss_row(unit=2, cycle=1),
    ]
    test_rows = [
        make_cmapss_row(unit=1, cycle=1),
        make_cmapss_row(unit=1, cycle=2),
        make_cmapss_row(unit=1, cycle=3),
        make_cmapss_row(unit=2, cycle=1),
        make_cmapss_row(unit=2, cycle=2),
    ]

    write_cmapss_file(tmp_path / "train_FD001.txt", train_rows)
    write_cmapss_file(tmp_path / "test_FD001.txt", test_rows)
    write_cmapss_file(tmp_path / "RUL_FD001.txt", [[10], [20]])

    test_with_rul = load_official_test_with_rul(data_path=tmp_path)
    unit_1_rows = test_with_rul[test_with_rul["unit"] == 1]
    payloads = list(iter_official_test_rows(data_path=tmp_path, unit_id=1))

    assert unit_1_rows["RUL"].tolist() == [12.0, 11.0, 10.0]
    assert len(payloads) == 3
    assert set(RAW_FEATURES).issubset(payloads[0])
    assert payloads[0]["cycle"] == 1.0
    assert payloads[0]["actual_rul"] == 12.0


def test_iter_train_rows_uses_train_rul_targets(tmp_path):
    train_rows = [
        make_cmapss_row(unit=1, cycle=1),
        make_cmapss_row(unit=1, cycle=2),
        make_cmapss_row(unit=1, cycle=3),
        make_cmapss_row(unit=2, cycle=1),
    ]
    test_rows = [make_cmapss_row(unit=1, cycle=1)]

    write_cmapss_file(tmp_path / "train_FD001.txt", train_rows)
    write_cmapss_file(tmp_path / "test_FD001.txt", test_rows)
    write_cmapss_file(tmp_path / "RUL_FD001.txt", [[10]])

    payloads = list(iter_train_rows(data_path=tmp_path, unit_id=1))

    assert [payload["cycle"] for payload in payloads] == [1.0, 2.0, 3.0]
    assert [payload["actual_rul"] for payload in payloads] == [2.0, 1.0, 0.0]
    assert set(RAW_FEATURES).issubset(payloads[0])


def test_drift_simulation_intensity_reaches_one_on_last_window():
    intensities = [
        round(calculate_window_intensity(window_index, total_windows=7), 2)
        for window_index in range(1, 8)
    ]

    assert intensities == [0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0]
    assert calculate_window_intensity(window_index=1, total_windows=1) == 0.0


def test_concept_simulation_is_applied_inside_selected_window():
    rows = []
    for cycle in range(1, 7):
        row = make_cmapss_row(unit=1, cycle=cycle)
        rows.append(row)
    clean_train = pd.DataFrame(
        rows,
        columns=[
            "unit",
            "cycle",
            "setting1",
            "setting2",
            "setting3",
            *[f"sensor{i}" for i in range(1, 22)],
        ],
    )
    clean_train["RUL"] = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    clean_window = clean_train.iloc[0:4].reset_index(drop=True)

    simulated = build_simulated_window(
        clean_train,
        scenario="concept_drift",
        intensity=1.0,
        window_index=1,
        window_size=4,
        random_state=43,
    ).reset_index(drop=True)

    assert simulated[RAW_FEATURES].equals(clean_window[RAW_FEATURES])
    assert simulated["RUL"].mean() == clean_window["RUL"].mean()
    assert sorted(simulated["RUL"].tolist()) == sorted(clean_window["RUL"].tolist())
    assert simulated["RUL"].tolist() != clean_window["RUL"].tolist()


def test_drift_simulation_start_row_offsets_first_window():
    rows = [make_cmapss_row(unit=1, cycle=cycle) for cycle in range(1, 7)]
    clean_train = pd.DataFrame(
        rows,
        columns=[
            "unit",
            "cycle",
            "setting1",
            "setting2",
            "setting3",
            *[f"sensor{i}" for i in range(1, 22)],
        ],
    )
    clean_train["RUL"] = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]

    simulated = build_simulated_window(
        clean_train,
        scenario="data_drift",
        intensity=0.0,
        window_index=1,
        window_size=2,
        random_state=43,
        start_row=2,
    ).reset_index(drop=True)

    assert simulated["cycle"].tolist() == [3, 4]
