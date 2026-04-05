import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def print_basic_info(train, test):
    # Вывод основной информации о данных
    print("Общая информация")
    print(f"Обучающая выборка: {train.shape[0]} строк, {train.shape[1]} колонок")
    print(f"Тестовая выборка: {test.shape[0]} строк, {test.shape[1]} колонок")
    print(f"Количество двигателей в обучении: {train['unit'].nunique()}")
    print(f"Диапазон циклов: {train['cycle'].min()} - {train['cycle'].max()}")

def print_statistics(train):
    # Вывод статистики и проверка пропусков
    print("Статистика по данным")
    print(train.describe())
    print(f"Пропуски: {train.isnull().sum().sum()}")

def analyze_engine_lifetime(train):
    # Анализ длительности жизни двигателей
    engine_lifetime = train.groupby('unit')['cycle'].max()
    print("Длительность жизни двигателей")
    print(f"Минимум: {engine_lifetime.min()} циклов")
    print(f"Максимум: {engine_lifetime.max()} циклов")
    print(f"Среднее: {engine_lifetime.mean():.1f} циклов")
    print(f"Медиана: {engine_lifetime.median()} циклов")
    return engine_lifetime

def plot_engine_lifetime(engine_lifetime):
    # Визуализация распределения длительности жизни
    plt.figure(figsize=(10, 4))
    plt.hist(engine_lifetime, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Количество циклов')
    plt.ylabel('Количество двигателей')
    plt.title('Распределение длительности жизни двигателей')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_sensors_dynamics(train):
    # Визуализация динамики датчиков для первого двигателя
    engine_1 = train[train['unit'] == 1]
    sensors_plot = ['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8', 'sensor11']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, sensor in enumerate(sensors_plot):
        axes[i].plot(engine_1['cycle'], engine_1[sensor], linewidth=1.5)
        axes[i].set_xlabel('Цикл')
        axes[i].set_ylabel(sensor)
        axes[i].set_title(f'{sensor} (двигатель 1)')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_rul_distribution(train_with_rul):
    # Визуализация распределения RUL
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(train_with_rul['RUL'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('RUL (циклы)')
    plt.ylabel('Частота')
    plt.title('Распределение RUL')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for unit in range(1, 6):
        unit_data = train_with_rul[train_with_rul['unit'] == unit]
        plt.plot(unit_data['cycle'], unit_data['RUL'], label=f'Engine {unit}')
    plt.xlabel('Цикл')
    plt.ylabel('RUL')
    plt.title('Деградация двигателей (первые 5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()