import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def select_all_sensors(train_with_rul):
    # Отбор всех сенсоров, исключая постоянные (корреляция nan)
    sensor_cols = [col for col in train_with_rul.columns if col.startswith("sensor")]

    correlations = {}
    for sensor in sensor_cols:
        corr = train_with_rul[sensor].corr(train_with_rul["RUL"])
        if not pd.isna(corr):
            correlations[sensor] = corr

    corr_list = [(sensor, corr) for sensor, corr in correlations.items()]
    corr_list.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Топ-10 сенсоров по корреляции с RUL:")
    for i, (sensor, corr) in enumerate(corr_list[:10]):
        print(f"  {i+1}. {sensor}: {corr:.4f}")

    all_sensors = [sensor for sensor, corr in corr_list]
    print(f"Отобрано сенсоров: {len(all_sensors)}")

    return all_sensors


def prepare_data(train_with_rul, top_sensors):
    # Подготовка данных и разделение по двигателям
    base_features = top_sensors + ["setting1", "setting2", "setting3"]
    print(f"Всего признаков: {len(base_features)}")

    # Разделение двигателей на train, val, test
    unique_units = train_with_rul["unit"].unique()
    train_units, temp_units = train_test_split(
        unique_units, test_size=0.3, random_state=42
    )
    val_units, test_units = train_test_split(temp_units, test_size=0.5, random_state=42)

    train_data = train_with_rul[train_with_rul["unit"].isin(train_units)]
    val_data = train_with_rul[train_with_rul["unit"].isin(val_units)]
    test_data = train_with_rul[train_with_rul["unit"].isin(test_units)]

    X_train = train_data[base_features].dropna()
    y_train = train_data.loc[X_train.index, "RUL"]
    X_val = val_data[base_features].dropna()
    y_val = val_data.loc[X_val.index, "RUL"]
    X_test = test_data[base_features].dropna()
    y_test = test_data.loc[X_test.index, "RUL"]

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Обучающих двигателей: {len(train_units)}")
    print(f"Валидационных двигателей: {len(val_units)}")
    print(f"Тестовых двигателей: {len(test_units)}")
    print(f"Размер обучающей выборки: {X_train_scaled.shape[0]} строк")
    print(f"Размер валидационной выборки: {X_val_scaled.shape[0]} строк")
    print(f"Размер тестовой выборки: {X_test_scaled.shape[0]} строк")

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
        base_features,
    )
