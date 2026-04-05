# Основной файл для запуска всего пайплайна

import sys
import os

from data.data_loader import load_and_prepare_data
from data.analysis import print_basic_info, print_statistics, analyze_engine_lifetime, plot_engine_lifetime, plot_sensors_dynamics, plot_rul_distribution
from features.feature_engineering import select_all_sensors, prepare_data
from models.train_model import train_random_forest
from evaluation.evaluate import evaluate_on_test
from models.save_model import save_model

def main():
    print("Запуск пайплайна обучения модели")
    
    # Загрузка данных
    train_with_rul, test_original, rul_original = load_and_prepare_data()
    
    # Анализ данных
    print_basic_info(train_with_rul, test_original)
    print_statistics(train_with_rul)
    engine_lifetime = analyze_engine_lifetime(train_with_rul)
    plot_engine_lifetime(engine_lifetime)
    plot_sensors_dynamics(train_with_rul)
    plot_rul_distribution(train_with_rul)
    
    # Отбор признаков
    top_sensors = select_all_sensors(train_with_rul)
    
    # Подготовка данных
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, base_features = prepare_data(train_with_rul, top_sensors)
    
    # Обучение модели
    model, train_mae, val_mae, train_rmse, val_rmse, train_r2, val_r2 = train_random_forest(X_train, y_train, X_val, y_val)
    
    # Оценка на тесте
    test_mae, test_rmse, test_r2 = evaluate_on_test(model, X_test, y_test)
    
    # Сохранение метрик
    metrics = {
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'features': base_features,
        'n_estimators': 30,
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 10
    }
    
    # Сохранение модели и артефактов
    save_model(model, scaler, base_features, metrics)
    
    # Итоговые выводы

    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"Модель: RandomForestRegressor")
    print(f"Количество признаков: {len(base_features)}")
    print(f"Параметры: n_estimators=30, max_depth=5, min_samples_split=20, min_samples_leaf=10")
    print(f"\nКачество на валидации:")
    print(f"  MAE: {val_mae:.2f} циклов")
    print(f"  R²: {val_r2:.3f}")
    print(f"\nКачество на тесте:")
    print(f"  MAE: {test_mae:.2f} циклов")
    print(f"  R²: {test_r2:.3f}")
    print(f"\nСредняя длительность жизни двигателя: {engine_lifetime.mean():.1f} циклов")
    print(f"Относительная ошибка на тесте: {(test_mae / engine_lifetime.mean()) * 100:.1f}%")
    
    if test_r2 > 0.5:
        print("\nМодель показывает хорошее качество. R² > 0.5")
    else:
        print("\nКачество модели ниже ожидаемого. Требуется доработка")
    
    print("\nПайплайн выполнен успешно")

if __name__ == "__main__":
    main()