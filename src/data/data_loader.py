import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(data_path="src/data/datasets/CMAPSSData"):
    # Загрузка и подготовка данных CMAPSS
    print("Загрузка данных")
    print(f"Путь: {data_path}")

    columns = ["unit", "cycle", "setting1", "setting2", "setting3"] + [
        f"sensor{i}" for i in range(1, 22)
    ]

    train = pd.read_csv(
        os.path.join(data_path, "train_FD001.txt"),
        sep=r"\s+",
        header=None,
        names=columns,
    )
    test = pd.read_csv(
        os.path.join(data_path, "test_FD001.txt"),
        sep=r"\s+",
        header=None,
        names=columns,
    )
    rul = pd.read_csv(
        os.path.join(data_path, "RUL_FD001.txt"),
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    train = train.dropna(axis=1, how="all")
    test = test.dropna(axis=1, how="all")

    # Добавление целевой переменной RUL
    def add_rul_column(df):
        df = df.copy()
        max_cycles = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = max_cycles - df["cycle"]
        return df

    train_with_rul = add_rul_column(train)

    print(
        f"Обучающая выборка: {train_with_rul.shape[0]} строк, {train_with_rul.shape[1]} колонок"
    )
    print(f"Количество двигателей: {train_with_rul['unit'].nunique()}")

    return train_with_rul, test, rul
