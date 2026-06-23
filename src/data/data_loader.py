from pathlib import Path

import pandas as pd

CMAPSS_COLUMNS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [
    f"sensor{i}" for i in range(1, 22)
]

RAW_FEATURES = [
    "cycle",
    "setting1",
    "setting2",
    "setting3",
    *[f"sensor{i}" for i in range(1, 22)],
]


def read_cmapss_file(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def add_train_rul(df):
    df = df.copy()
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]
    return df


def load_cmapss_dataset(data_path="data/raw", dataset_id="FD001"):
    data_dir = Path(data_path)
    train = read_cmapss_file(data_dir / f"train_{dataset_id}.txt")
    test = read_cmapss_file(data_dir / f"test_{dataset_id}.txt")
    official_rul = pd.read_csv(
        data_dir / f"RUL_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )
    return add_train_rul(train), test, official_rul


def load_cmapss_fd001(data_path="data/raw"):
    return load_cmapss_dataset(data_path=data_path, dataset_id="FD001")


def get_official_test_last_rows(test_df, official_rul):
    last_rows = (
        test_df.sort_values(["unit", "cycle"])
        .groupby("unit", as_index=False)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )
    y_true = official_rul["RUL"].reset_index(drop=True)
    if len(last_rows) != len(y_true):
        raise ValueError("Number of test engines does not match official RUL targets.")
    return last_rows, y_true


def load_official_test_with_rul(data_path="data/raw", dataset_id="FD001"):
    _, test, official_rul = load_cmapss_dataset(
        data_path=data_path, dataset_id=dataset_id
    )
    max_cycles = test.groupby("unit")["cycle"].transform("max")
    final_rul_by_unit = official_rul["RUL"].reset_index(drop=True)
    unit_to_final_rul = {
        unit: float(final_rul_by_unit.iloc[index])
        for index, unit in enumerate(sorted(test["unit"].unique()))
    }

    test_with_rul = test.copy()
    test_with_rul["RUL"] = test_with_rul.apply(
        lambda row: unit_to_final_rul[row["unit"]]
        + float(max_cycles.loc[row.name] - row["cycle"]),
        axis=1,
    )
    return test_with_rul


def iter_official_test_rows(data_path="data/raw", dataset_id="FD001", unit_id=None):
    test_with_rul = load_official_test_with_rul(
        data_path=data_path,
        dataset_id=dataset_id,
    )
    if unit_id is not None:
        test_with_rul = test_with_rul[test_with_rul["unit"] == unit_id]

    ordered_rows = test_with_rul.sort_values(["unit", "cycle"])
    for _, row in ordered_rows.iterrows():
        payload = {feature: float(row[feature]) for feature in RAW_FEATURES}
        payload["actual_rul"] = float(row["RUL"])
        yield payload


def iter_train_rows(data_path="data/raw", dataset_id="FD001", unit_id=None):
    train_with_rul, _, _ = load_cmapss_dataset(
        data_path=data_path,
        dataset_id=dataset_id,
    )
    if unit_id is not None:
        train_with_rul = train_with_rul[train_with_rul["unit"] == unit_id]

    ordered_rows = train_with_rul.sort_values(["unit", "cycle"])
    for _, row in ordered_rows.iterrows():
        payload = {feature: float(row[feature]) for feature in RAW_FEATURES}
        payload["actual_rul"] = float(row["RUL"])
        yield payload


def load_and_prepare_data(data_path="data/raw"):
    train_with_rul, test, rul = load_cmapss_fd001(data_path=data_path)
    print(
        f"Loaded FD001 train: {train_with_rul.shape[0]} rows, "
        f"{train_with_rul['unit'].nunique()} engines"
    )
    print(f"Loaded FD001 test: {test.shape[0]} rows, {test['unit'].nunique()} engines")
    return train_with_rul, test, rul
