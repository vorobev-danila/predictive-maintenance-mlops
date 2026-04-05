import os
import json
import joblib

def save_model(model, scaler, base_features, metrics, models_path='models', metrics_path='reports/metrics'):
    # Сохранение модели, нормализатора, признаков и метрик
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    
    joblib.dump(model, os.path.join(models_path, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(models_path, 'scaler.pkl'))
    
    with open(os.path.join(models_path, 'features.json'), 'w') as f:
        json.dump(base_features, f)
    
    with open(os.path.join(metrics_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Модель сохранена в {models_path}")
    print(f"Метрики сохранены в {metrics_path}")