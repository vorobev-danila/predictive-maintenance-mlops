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


def load_cmapss_fd001(data_path="data/raw"):
    data_dir = Path(data_path)
    train = read_cmapss_file(data_dir / "train_FD001.txt")
    test = read_cmapss_file(data_dir / "test_FD001.txt")
    official_rul = pd.read_csv(
        data_dir / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )
    return add_train_rul(train), test, official_rul


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
        raise ValueError("Number of test engines does not match RUL_FD001 targets.")
    return last_rows, y_true


def load_and_prepare_data(data_path="data/raw"):
    train_with_rul, test, rul = load_cmapss_fd001(data_path=data_path)
    print(
        f"Loaded FD001 train: {train_with_rul.shape[0]} rows, "
        f"{train_with_rul['unit'].nunique()} engines"
    )
    print(f"Loaded FD001 test: {test.shape[0]} rows, {test['unit'].nunique()} engines")
    return train_with_rul, test, rul
