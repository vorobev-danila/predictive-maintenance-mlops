import os
import json
import joblib


def save_model(model, scaler, base_features, metrics, models_path="models"):
    # Сохранение модели, нормализатора, признаков и метрик в одну папку
    os.makedirs(models_path, exist_ok=True)

    joblib.dump(model, os.path.join(models_path, "random_forest_model.pkl"))
    joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))

    with open(os.path.join(models_path, "features.json"), "w") as f:
        json.dump(base_features, f)

    with open(os.path.join(models_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Модель и артефакты сохранены в {models_path}")
