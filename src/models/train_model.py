from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def train_random_forest(X_train, y_train, X_val, y_val):
    # Обучение случайного леса
    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)

    print("Результаты обучения")
    print(f"Обучающая MAE: {train_mae:.2f}")
    print(f"Валидационная MAE: {val_mae:.2f}")
    print(f"Обучающая RMSE: {train_rmse:.2f}")
    print(f"Валидационная RMSE: {val_rmse:.2f}")
    print(f"Обучающая R2: {train_r2:.3f}")
    print(f"Валидационная R2: {val_r2:.3f}")

    return model, train_mae, val_mae, train_rmse, val_rmse, train_r2, val_r2
