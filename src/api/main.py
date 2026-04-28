# FastAPI сервис для предсказания остаточного ресурса двигателей
import sys
import os
import joblib
import uvicorn
import pandas as pd
import json
import subprocess
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from prometheus_client import Gauge, REGISTRY

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные для загруженных артефактов
model = None
scaler = None
feature_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_names
    
    model_path = "models/random_forest_model.pkl"
    scaler_path = "models/scaler.pkl"
    features_path = "models/features.json"
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, "r") as f:
            feature_names = json.load(f)
        print("Модель и артефакты успешно загружены")
    except Exception as e:
        print(f"Ошибка загрузки артефактов: {e}")
    
    yield
    print("Завершение работы приложения")

# Создаём экземпляр приложения FastAPI
app = FastAPI(
    title="Predictive Maintenance API",
    description="API для прогнозирования остаточного ресурса (RUL) авиационных двигателей",
    version="1.0.0",
    lifespan=lifespan
)

# Инициализация Instrumentator для сбора метрик
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)

# Добавляем стандартные метрики
instrumentator.add().instrument(app).expose(app, endpoint="/metrics")

try:
    predicted_rul_gauge = Gauge("model_predicted_rul", "Predicted RUL value")
except ValueError:
    # Если Uvicorn перезагрузил код и метрика уже существует, просто берем её из памяти
    predicted_rul_gauge = REGISTRY._names_to_collectors["model_predicted_rul"]

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# Определяем модель данных для входного запроса
class SensorData(BaseModel):
    sensor1: float = Field(..., description="Полная температура на входе в вентилятор")
    sensor2: float = Field(..., description="Полная температура на выходе компрессора низкого давления")
    sensor3: float = Field(..., description="Полная температура на выходе компрессора высокого давления")
    sensor4: float = Field(..., description="Полная температура на выходе турбины низкого давления")
    sensor5: float = Field(..., description="Давление на входе в вентилятор")
    sensor6: float = Field(..., description="Полное давление в перепускном канале")
    sensor7: float = Field(..., description="Полное давление на выходе компрессора высокого давления")
    sensor8: float = Field(..., description="Физическая скорость вращения вентилятора")
    sensor9: float = Field(..., description="Физическая скорость вращения компрессора")
    sensor10: float = Field(..., description="Отношение давлений в двигателе")
    sensor11: float = Field(..., description="Статическое давление на выходе компрессора высокого давления")
    sensor12: float = Field(..., description="Отношение расхода топлива к статическому давлению")
    sensor13: float = Field(..., description="Приведенная скорость вращения вентилятора")
    sensor14: float = Field(..., description="Приведенная скорость вращения компрессора")
    sensor15: float = Field(..., description="Коэффициент двухконтурности")
    sensor16: float = Field(..., description="Отношение топливо-воздух в камере сгорания")
    sensor17: float = Field(..., description="Энтальпия отбора воздуха")
    sensor18: float = Field(..., description="Заданная скорость вращения вентилятора")
    sensor19: float = Field(..., description="Заданная приведенная скорость вращения вентилятора")
    sensor20: float = Field(..., description="Охлаждающий отбор воздуха из турбины высокого давления")
    sensor21: float = Field(..., description="Охлаждающий отбор воздуха из турбины низкого давления")
    setting1: float = Field(..., description="Настройка режима 1")
    setting2: float = Field(..., description="Настройка режима 2")
    setting3: float = Field(..., description="Настройка режима 3")

class PredictionResponse(BaseModel):
    rul: float = Field(..., description="Предсказанный остаточный ресурс в циклах")
    status: str = Field(..., description="Статус выполнения запроса")

# Эндпоинт для проверки работоспособности
@app.get("/health")
async def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ok"}

# Эндпоинт для получения метрик модели
@app.get("/model_metrics")
async def get_model_metricss():
    metrics_path = "models/metrics.json"
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для запуска переобучения
@app.post("/retrain")
async def retrain():
    try:
        logger.info("Запуск переобучения модели")
        
        # Путь к python из виртуального окружения
        venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".venv", "Scripts", "python.exe")
        
        # Если venv не найден, пробуем просто python
        if not os.path.exists(venv_python):
            venv_python = "python"
        
        result = subprocess.run(
            [venv_python, "src/pipeline.py"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        
        if result.returncode == 0:
            logger.info("Переобучение успешно завершено")
            # После успешного переобучения перезагружаем модель
            await reload_model()
            return {"status": "success", "message": "Retraining completed successfully", "output": result.stdout}
        else:
            logger.error(f"Ошибка переобучения: {result.stderr}")
            return {"status": "error", "message": "Retraining failed", "error": result.stderr}
    except subprocess.TimeoutExpired:
        logger.error("Переобучение превысило время ожидания")
        raise HTTPException(status_code=504, detail="Retraining timeout")
    except Exception as e:
        logger.error(f"Ошибка при переобучении: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def reload_model():
    global model, scaler, feature_names
    
    model_path = "models/random_forest_model.pkl"
    scaler_path = "models/scaler.pkl"
    features_path = "models/features.json"
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, "r") as f:
            feature_names = json.load(f)
        logger.info("Модель успешно перезагружена после переобучения")
    except Exception as e:
        logger.error(f"Ошибка перезагрузки модели: {e}")
        
# Основной эндпоинт для предсказания
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    predicted_rul_gauge.set(float(prediction)) 

    return PredictionResponse(rul=float(prediction), status="success")

if __name__ == "__main__":
    print("Сервер запущен. Откройте в браузере: http://localhost:8080/docs")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8080, reload=True)