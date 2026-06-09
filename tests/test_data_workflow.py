from pathlib import Path

from data.data_loader import (
    RAW_FEATURES,
    get_official_test_last_rows,
    load_and_prepare_data,
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
