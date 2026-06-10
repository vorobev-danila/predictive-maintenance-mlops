# FastAPI сервис для предсказания остаточного ресурса двигателей
import sys
import os
import joblib
import uvicorn
import pandas as pd
import json
import subprocess
import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, REGISTRY
from monitoring.drift import load_latest_drift_report, run_cmapss_drift_report
from monitoring.drift_simulation import SUPPORTED_SCENARIOS, run_drift_simulation
from storage.prediction_repository import PredictionRepository

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные для загруженных артефактов
model = None
scaler = None
feature_names = None
prediction_repository = None


def get_prediction_db_path():
    return os.getenv("PREDICTION_DB_PATH", "state/predictions.db")


def get_reports_dir():
    return os.getenv("REPORTS_DIR", "reports")


def get_drift_reports_dir():
    return os.path.join(get_reports_dir(), "drift")


def get_drift_data_path():
    return os.getenv("DRIFT_DATA_PATH", "data/raw")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_names, prediction_repository

    model_path = "models/pipeline.pkl"
    fallback_model_path = "models/model.pkl"
    features_path = "models/features.json"
    prediction_repository = PredictionRepository(get_prediction_db_path())
    prediction_repository.initialize()

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = joblib.load(fallback_model_path)
        scaler = None
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
    lifespan=lifespan,
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


def get_or_create_gauge(name, description, labelnames=None):
    try:
        return Gauge(name, description, labelnames or [])
    except ValueError:
        return REGISTRY._names_to_collectors[name]


predicted_rul_gauge = get_or_create_gauge("model_predicted_rul", "Predicted RUL value")
actual_rul_gauge = get_or_create_gauge("model_actual_rul", "Actual Ground Truth RUL")
data_drift_score_gauge = get_or_create_gauge(
    "data_drift_score", "Latest data drift score"
)
data_drift_detected_gauge = get_or_create_gauge(
    "data_drift_detected", "Latest data drift flag"
)
target_drift_score_gauge = get_or_create_gauge(
    "target_drift_score", "Latest target drift score"
)
target_drift_detected_gauge = get_or_create_gauge(
    "target_drift_detected", "Latest target drift flag"
)
concept_drift_score_gauge = get_or_create_gauge(
    "concept_drift_score", "Latest concept drift score"
)
concept_drift_detected_gauge = get_or_create_gauge(
    "concept_drift_detected", "Latest concept drift flag"
)
drifted_features_count_gauge = get_or_create_gauge(
    "drifted_features_count", "Number of features with detected drift"
)
drift_simulation_window_gauge = get_or_create_gauge(
    "drift_simulation_window", "Latest drift simulation window"
)
drift_simulation_scenario_gauge = get_or_create_gauge(
    "drift_simulation_scenario",
    "Active drift simulation scenario",
    ["scenario"],
)
model_prediction_error_mae_gauge = get_or_create_gauge(
    "model_prediction_error_mae", "Latest simulated prediction MAE"
)
model_prediction_error_p95_gauge = get_or_create_gauge(
    "model_prediction_error_p95", "Latest simulated prediction error p95"
)
model_actual_rul_mean_gauge = get_or_create_gauge(
    "model_actual_rul_mean", "Latest simulated actual RUL mean"
)
model_predicted_rul_mean_gauge = get_or_create_gauge(
    "model_predicted_rul_mean", "Latest simulated predicted RUL mean"
)


# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request, call_next):
    import time

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )
    return response


# Определяем модель данных для входного запроса
class SensorData(BaseModel):
    cycle: float = Field(..., description="Engine cycle")
    sensor1: float = Field(..., description="Полная температура на входе в вентилятор")
    sensor2: float = Field(
        ..., description="Полная температура на выходе компрессора низкого давления"
    )
    sensor3: float = Field(
        ..., description="Полная температура на выходе компрессора высокого давления"
    )
    sensor4: float = Field(
        ..., description="Полная температура на выходе турбины низкого давления"
    )
    sensor5: float = Field(..., description="Давление на входе в вентилятор")
    sensor6: float = Field(..., description="Полное давление в перепускном канале")
    sensor7: float = Field(
        ..., description="Полное давление на выходе компрессора высокого давления"
    )
    sensor8: float = Field(..., description="Физическая скорость вращения вентилятора")
    sensor9: float = Field(..., description="Физическая скорость вращения компрессора")
    sensor10: float = Field(..., description="Отношение давлений в двигателе")
    sensor11: float = Field(
        ..., description="Статическое давление на выходе компрессора высокого давления"
    )
    sensor12: float = Field(
        ..., description="Отношение расхода топлива к статическому давлению"
    )
    sensor13: float = Field(
        ..., description="Приведенная скорость вращения вентилятора"
    )
    sensor14: float = Field(
        ..., description="Приведенная скорость вращения компрессора"
    )
    sensor15: float = Field(..., description="Коэффициент двухконтурности")
    sensor16: float = Field(
        ..., description="Отношение топливо-воздух в камере сгорания"
    )
    sensor17: float = Field(..., description="Энтальпия отбора воздуха")
    sensor18: float = Field(..., description="Заданная скорость вращения вентилятора")
    sensor19: float = Field(
        ..., description="Заданная приведенная скорость вращения вентилятора"
    )
    sensor20: float = Field(
        ..., description="Охлаждающий отбор воздуха из турбины высокого давления"
    )
    sensor21: float = Field(
        ..., description="Охлаждающий отбор воздуха из турбины низкого давления"
    )
    setting1: float = Field(..., description="Настройка режима 1")
    setting2: float = Field(..., description="Настройка режима 2")
    setting3: float = Field(..., description="Настройка режима 3")
    actual_rul: float = Field(
        default=None, description="Реальный RUL (опционально для тестов)"
    )


class PredictionResponse(BaseModel):
    prediction_id: int = Field(..., description="ID сохраненного предсказания")
    rul: float = Field(..., description="Предсказанный остаточный ресурс в циклах")
    status: str = Field(..., description="Статус выполнения запроса")


class PredictionHistoryItem(BaseModel):
    id: int
    created_at: str
    input: dict
    predicted_rul: float
    actual_rul: float | None = None
    anomaly_flag: bool
    model_version: str | None = None


class DriftRunRequest(BaseModel):
    dataset_id: str = Field(default="FD001", description="CMAPSS dataset suffix")


class DriftRunResponse(BaseModel):
    status: str
    report: dict


class DriftSimulationRequest(BaseModel):
    scenario: str = Field(default="all", description="Drift simulation scenario")
    dataset_id: str = Field(default="FD001", description="CMAPSS dataset suffix")
    windows: int = Field(default=6, ge=1, le=30)
    sleep_seconds: float = Field(default=0.0, ge=0.0, le=10.0)


class DriftSimulationResponse(BaseModel):
    status: str
    report: dict


# Эндпоинт для проверки работоспособности
@app.get("/health")
async def health_check():
    if model is None:
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
        venv_python = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            ".venv",
            "Scripts",
            "python.exe",
        )

        # Если venv не найден, пробуем просто python
        if not os.path.exists(venv_python):
            venv_python = "python"

        result = subprocess.run(
            [venv_python, "src/pipeline.py"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
        )

        if result.returncode == 0:
            logger.info("Переобучение успешно завершено")
            # После успешного переобучения перезагружаем модель
            await reload_model()
            return {
                "status": "success",
                "message": "Retraining completed successfully",
                "output": result.stdout,
            }
        else:
            logger.error(f"Ошибка переобучения: {result.stderr}")
            return {
                "status": "error",
                "message": "Retraining failed",
                "error": result.stderr,
            }
    except subprocess.TimeoutExpired:
        logger.error("Переобучение превысило время ожидания")
        raise HTTPException(status_code=504, detail="Retraining timeout")
    except Exception as e:
        logger.error(f"Ошибка при переобучении: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def reload_model():
    global model, scaler, feature_names

    model_path = "models/pipeline.pkl"
    fallback_model_path = "models/model.pkl"
    features_path = "models/features.json"

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = joblib.load(fallback_model_path)
        scaler = None
        with open(features_path, "r") as f:
            feature_names = json.load(f)
        logger.info("Модель успешно перезагружена после переобучения")
    except Exception as e:
        logger.error(f"Ошибка перезагрузки модели: {e}")


# Основной эндпоинт для предсказания
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorData):
    if model is None or prediction_repository is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]
    prediction = model.predict(input_df)[0]
    predicted_rul_gauge.set(float(prediction))

    # проверка на реальный RUL
    if data.actual_rul is not None:
        actual_rul_gauge.set(float(data.actual_rul))

    saved_prediction = prediction_repository.create_prediction(
        input_payload=input_dict,
        predicted_rul=float(prediction),
        actual_rul=data.actual_rul,
        anomaly_flag=False,
        model_version=os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME", "predictive-maintenance-gradient-boosting"
        ),
    )

    return PredictionResponse(
        prediction_id=saved_prediction["id"], rul=float(prediction), status="success"
    )


@app.get("/predictions/recent", response_model=list[PredictionHistoryItem])
async def get_recent_predictions(limit: int = Query(default=20, ge=1, le=100)):
    if prediction_repository is None:
        raise HTTPException(status_code=503, detail="Prediction storage is not ready")

    return prediction_repository.list_recent(limit=limit)


@app.post("/drift/run", response_model=DriftRunResponse)
async def run_drift(request: DriftRunRequest):
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        report = run_cmapss_drift_report(
            data_path=get_drift_data_path(),
            reports_dir=get_drift_reports_dir(),
            dataset_id=request.dataset_id,
            model=model,
            scaler=scaler,
            feature_names=feature_names,
        )
        update_drift_metrics(report)
        return DriftRunResponse(status="success", report=report)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        logger.error(f"Ошибка расчета drift: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/drift/latest")
async def get_latest_drift():
    report = load_latest_drift_report(get_drift_reports_dir())
    if report is None:
        raise HTTPException(status_code=404, detail="Drift report not found")
    return report


@app.post("/drift/simulate", response_model=DriftSimulationResponse)
async def simulate_drift(request: DriftSimulationRequest):
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    if request.scenario not in SUPPORTED_SCENARIOS:
        raise HTTPException(status_code=400, detail="Unsupported simulation scenario")

    try:
        report = run_drift_simulation(
            scenario=request.scenario,
            data_path=get_drift_data_path(),
            reports_dir=get_drift_reports_dir(),
            dataset_id=request.dataset_id,
            model=model,
            feature_names=feature_names,
            windows=request.windows,
            sleep_seconds=request.sleep_seconds,
            on_window=update_drift_simulation_metrics,
        )
        return DriftSimulationResponse(status="success", report=report)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        logger.error(f"Ошибка drift simulation: {error}")
        raise HTTPException(status_code=500, detail=str(error))


def update_drift_metrics(report):
    data_drift = report["data_drift"]
    target_drift = report["target_drift"]
    concept_drift = report["concept_drift"]

    data_drift_score_gauge.set(float(data_drift["score"]))
    data_drift_detected_gauge.set(float(data_drift["drift_detected"]))
    drifted_features_count_gauge.set(float(data_drift["drifted_features_count"]))
    target_drift_score_gauge.set(float(target_drift["score"]))
    target_drift_detected_gauge.set(float(target_drift["drift_detected"]))
    concept_drift_score_gauge.set(float(concept_drift["score"]))
    concept_drift_detected_gauge.set(float(concept_drift["drift_detected"]))


def update_drift_simulation_metrics(window_result):
    update_drift_metrics(window_result)
    scenario = window_result["scenario"]

    for scenario_name in SUPPORTED_SCENARIOS:
        drift_simulation_scenario_gauge.labels(scenario=scenario_name).set(
            float(scenario_name == scenario)
        )

    drift_simulation_window_gauge.set(float(window_result["window"]))
    model_prediction_error_mae_gauge.set(float(window_result["prediction_error_mae"]))
    model_prediction_error_p95_gauge.set(float(window_result["prediction_error_p95"]))
    model_actual_rul_mean_gauge.set(float(window_result["actual_rul_mean"]))
    model_predicted_rul_mean_gauge.set(float(window_result["predicted_rul_mean"]))


@app.post("/reset_metrics")
async def reset_metrics():
    predicted_rul_gauge.set(0.0)
    actual_rul_gauge.set(0.0)
    data_drift_score_gauge.set(0.0)
    data_drift_detected_gauge.set(0.0)
    target_drift_score_gauge.set(0.0)
    target_drift_detected_gauge.set(0.0)
    concept_drift_score_gauge.set(0.0)
    concept_drift_detected_gauge.set(0.0)
    drifted_features_count_gauge.set(0.0)
    drift_simulation_window_gauge.set(0.0)
    for scenario_name in SUPPORTED_SCENARIOS:
        drift_simulation_scenario_gauge.labels(scenario=scenario_name).set(0.0)
    model_prediction_error_mae_gauge.set(0.0)
    model_prediction_error_p95_gauge.set(0.0)
    model_actual_rul_mean_gauge.set(0.0)
    model_predicted_rul_mean_gauge.set(0.0)
    return {"status": "metrics reset to 0"}


if __name__ == "__main__":
    print("Сервер запущен. Откройте в браузере: http://localhost:8080/docs")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8080, reload=True)
