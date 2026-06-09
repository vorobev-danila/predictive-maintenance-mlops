import json
import os

import joblib


def save_model(model, base_features, metrics, models_path="models"):
    os.makedirs(models_path, exist_ok=True)

    joblib.dump(model, os.path.join(models_path, "model.pkl"))
    joblib.dump(model, os.path.join(models_path, "pipeline.pkl"))

    # Backward-compatible filename for deployments that have not updated env/config yet.
    joblib.dump(model, os.path.join(models_path, "random_forest_model.pkl"))

    with open(os.path.join(models_path, "features.json"), "w", encoding="utf-8") as f:
        json.dump(base_features, f, indent=2)

    with open(os.path.join(models_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model artifacts saved to {models_path}")
