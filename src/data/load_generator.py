import pandas as pd
import requests
import time

print("Загрузка датасетов...")
test_df = pd.read_csv("data/raw/test_FD001.txt", sep="\s+", header=None)
rul_df = pd.read_csv("data/raw/RUL_FD001.txt", sep="\s+", header=None)

max_cycles = test_df.groupby(0)[1].max()

print("Запуск имитации (Двигатель 1)...")
engine_1_data = test_df[test_df[0] == 1]

for index, row in engine_1_data.iterrows():
    engine_id = int(row[0])
    cycle = int(row[1])
    
    final_rul = rul_df.iloc[engine_id - 1, 0]
    max_cycle = max_cycles[engine_id]
    current_actual_rul = final_rul + (max_cycle - cycle)   # реальный RUL на текущем цикле
    
    payload = {f"sensor{i+1}": float(row[i+5]) for i in range(21)}  # создаем словарь с 21 ключом
    payload.update({
        "setting1": float(row[2]), 
        "setting2": float(row[3]), 
        "setting3": float(row[4]),
        "actual_rul": float(current_actual_rul)
    })

    # Отправка запроса
    try:
        response = requests.post("http://localhost:8080/predict", json=payload) # отправляем POST запрос
        predicted = response.json()['rul']   # извлекает из ответа предсказанный RUL
        print(f"Цикл {cycle} | Предсказано: {predicted:.1f} | Реально: {current_actual_rul} | Ошибка: {abs(predicted - current_actual_rul):.1f}")
    except Exception as e:
        print(f"Ошибка: {e}")

    time.sleep(15)

# Сброс метрик после завершения всех циклов
print("Имитация завершена. Отправка сигнала сброса метрик...")
reset_payload = {f"sensor{i+1}": 0.0 for i in range(21)}
reset_payload.update({
    "setting1": 0.0,
    "setting2": 0.0,
    "setting3": 0.0,
    "actual_rul": 0.0
})

print("Имитация завершена. Сброс метрик...")
try:
    requests.post("http://localhost:8080/reset_metrics", timeout=2)
    print("Метрики сброшены до 0")
except Exception as e:
    print(f"Ошибка сброса: {e}")