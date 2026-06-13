# FastAPI сервис для предсказания остаточного ресурса двигателей
import sys
import os
import joblib
import uvicorn
import pandas as pd
import json
import subprocess
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, REGISTRY
from data.data_loader import RAW_FEATURES
from monitoring.drift import (
    get_colleague_thresholds,
    load_train_dataset,
    load_latest_drift_report,
    run_prediction_window_drift_report,
    save_drift_report,
)
from monitoring.drift_simulation import MAX_SIMULATION_INTENSITY
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


def get_drift_window_size():
    return int(os.getenv("DRIFT_WINDOW_SIZE", "10"))


def get_drift_min_window_size():
    return int(os.getenv("DRIFT_MIN_WINDOW_SIZE", "5"))


def get_reference_mae():
    metrics_path = "models/metrics.json"
    try:
        with open(metrics_path, "r") as file:
            metrics = json.load(file)
        return metrics.get("official_test_mae")
    except FileNotFoundError:
        return None


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


def get_or_create_gauge(name, description):
    try:
        return Gauge(name, description)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


def get_or_create_labeled_gauge(name, description, labelnames):
    try:
        return Gauge(name, description, labelnames)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


def get_or_create_counter(name, description):
    try:
        return Counter(name, description)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


predicted_rul_gauge = get_or_create_gauge("model_predicted_rul", "Predicted RUL value")
actual_rul_gauge = get_or_create_gauge("model_actual_rul", "Actual Ground Truth RUL")
prediction_error_gauge = get_or_create_gauge(
    "model_prediction_error",
    "Latest signed prediction error: predicted RUL - actual RUL",
)
absolute_error_gauge = get_or_create_gauge(
    "model_absolute_error", "Latest absolute prediction error"
)
prediction_count_counter = get_or_create_counter(
    "model_predictions_total", "Total number of prediction requests"
)
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
prediction_window_predicted_rul_mean_gauge = get_or_create_gauge(
    "prediction_window_predicted_rul_mean",
    "Mean predicted RUL in the latest prediction drift window",
)
prediction_window_actual_rul_mean_gauge = get_or_create_gauge(
    "prediction_window_actual_rul_mean",
    "Mean actual RUL in the latest labeled prediction drift window",
)
prediction_window_absolute_error_mae_gauge = get_or_create_gauge(
    "prediction_window_absolute_error_mae",
    "Mean absolute error in the latest labeled prediction drift window",
)
prediction_window_absolute_error_p95_gauge = get_or_create_gauge(
    "prediction_window_absolute_error_p95",
    "95th percentile absolute error in the latest labeled prediction drift window",
)
feature_drift_score_gauge = get_or_create_labeled_gauge(
    "feature_drift_score",
    "Latest drift score by feature",
    ["feature"],
)
feature_drift_detected_gauge = get_or_create_labeled_gauge(
    "feature_drift_detected",
    "Latest drift flag by feature",
    ["feature"],
)
feature_reference_mean_gauge = get_or_create_labeled_gauge(
    "feature_reference_mean",
    "Reference dataset mean by feature in the latest drift report",
    ["feature"],
)
feature_current_mean_gauge = get_or_create_labeled_gauge(
    "feature_current_mean",
    "Current dataset mean by feature in the latest drift report",
    ["feature"],
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
    unit: float | None = Field(default=None, description="Engine unit identifier")
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
    scenario: str = Field(default="all", description="Drift simulation scenario")
    intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=MAX_SIMULATION_INTENSITY,
        description=f"Simulation intensity from 0.0 to {MAX_SIMULATION_INTENSITY}",
    )


class DriftRunResponse(BaseModel):
    status: str
    report: dict


# Эндпоинт для проверки работоспособности
class RandomSampleResponse(BaseModel):
    dataset_id: str
    source: str
    payload: dict


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
def predict(data: SensorData):
    if model is None or prediction_repository is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    input_dict = data.model_dump()
    model_input = {feature: input_dict[feature] for feature in feature_names}
    input_df = pd.DataFrame([model_input])
    input_df = input_df[feature_names]
    prediction = model.predict(input_df)[0]
    predicted_rul_gauge.set(float(prediction))
    prediction_count_counter.inc()

    # проверка на реальный RUL
    if data.actual_rul is not None:
        actual_rul_gauge.set(float(data.actual_rul))
        prediction_error = float(prediction) - float(data.actual_rul)
        prediction_error_gauge.set(prediction_error)
        absolute_error_gauge.set(abs(prediction_error))

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
def get_recent_predictions(limit: int = Query(default=20, ge=1, le=100)):
    if prediction_repository is None:
        raise HTTPException(status_code=503, detail="Prediction storage is not ready")

    return prediction_repository.list_recent(limit=limit)


@app.get("/samples/random", response_model=RandomSampleResponse)
def get_random_train_sample(dataset_id: str = "FD001"):
    try:
        train_df = load_train_dataset(get_drift_data_path(), dataset_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    sample = train_df.sample(n=1).iloc[0]
    payload = {feature: float(sample[feature]) for feature in RAW_FEATURES}
    payload["unit"] = float(sample["unit"])
    payload["actual_rul"] = float(sample["RUL"])
    return {
        "dataset_id": dataset_id,
        "source": f"train_{dataset_id}",
        "payload": payload,
    }


@app.post("/drift/run", response_model=DriftRunResponse)
def run_drift(request: DriftRunRequest):
    if model is None or feature_names is None or prediction_repository is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        window_size = get_drift_window_size()
        thresholds = get_colleague_thresholds()
        recent_predictions = prediction_repository.list_recent(limit=window_size)
        report = run_prediction_window_drift_report(
            predictions=recent_predictions,
            data_path=get_drift_data_path(),
            reports_dir=get_drift_reports_dir(),
            dataset_id=request.dataset_id,
            feature_names=feature_names,
            reference_mae=get_reference_mae(),
            model=model,
            window_size=window_size,
            min_window_size=get_drift_min_window_size(),
            data_drift_threshold=thresholds["data_drift"],
            target_drift_threshold=thresholds["target_drift"],
            concept_drift_threshold=thresholds["concept_drift"],
            threshold_source=thresholds["threshold_source"],
        )
        report["scenario"] = request.scenario
        report["intensity"] = request.intensity
        report["current_dataset"] = (
            f"simulated_{request.dataset_id}_window_last_"
            f"{report['prediction_summary']['window_rows']}"
        )
        save_drift_report(report, get_drift_reports_dir())
        update_drift_metrics(report)
        return DriftRunResponse(status="success", report=report)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        logger.error(f"Ошибка расчета drift: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/drift/latest")
def get_latest_drift():
    report = load_latest_drift_report(get_drift_reports_dir())
    if report is None:
        raise HTTPException(status_code=404, detail="Drift report not found")
    return report


@app.get("/drift/reports")
def get_drift_reports(limit: int = Query(default=10, ge=1, le=50)):
    reports_dir = get_drift_reports_dir()
    report_paths = sorted(
        Path(reports_dir).glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    reports = []
    for report_path in report_paths:
        try:
            with open(report_path, "r") as file:
                report = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        reports.append(
            {
                "file": report_path.name,
                "created_at": report.get("created_at"),
                "reference_dataset": report.get("reference_dataset"),
                "current_dataset": report.get("current_dataset"),
                "data_drift": report.get("data_drift", {}).get("drift_detected"),
                "target_drift": report.get("target_drift", {}).get("drift_detected"),
                "concept_drift": report.get("concept_drift", {}).get("drift_detected"),
                "report": report,
            }
        )
    return reports


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
    update_prediction_window_summary_metrics(report.get("prediction_summary", {}))
    update_feature_drift_metrics(data_drift)


def update_prediction_window_summary_metrics(summary):
    _set_optional_gauge(
        prediction_window_predicted_rul_mean_gauge,
        summary.get("predicted_rul_mean"),
    )
    _set_optional_gauge(
        prediction_window_actual_rul_mean_gauge,
        summary.get("actual_rul_mean"),
    )
    _set_optional_gauge(
        prediction_window_absolute_error_mae_gauge,
        summary.get("absolute_error_mae"),
    )
    _set_optional_gauge(
        prediction_window_absolute_error_p95_gauge,
        summary.get("absolute_error_p95"),
    )


def _set_optional_gauge(gauge, value):
    if value is not None:
        gauge.set(float(value))


def update_feature_drift_metrics(data_drift):
    for feature_name, feature_result in data_drift.get("features", {}).items():
        feature_drift_score_gauge.labels(feature=feature_name).set(
            float(feature_result["score"])
        )
        feature_drift_detected_gauge.labels(feature=feature_name).set(
            float(feature_result["drifted"])
        )
        feature_reference_mean_gauge.labels(feature=feature_name).set(
            float(feature_result["reference_mean"])
        )
        feature_current_mean_gauge.labels(feature=feature_name).set(
            float(feature_result["current_mean"])
        )


@app.post("/reset_metrics")
def reset_metrics():
    predicted_rul_gauge.set(0.0)
    actual_rul_gauge.set(0.0)
    prediction_error_gauge.set(0.0)
    absolute_error_gauge.set(0.0)
    data_drift_score_gauge.set(0.0)
    data_drift_detected_gauge.set(0.0)
    target_drift_score_gauge.set(0.0)
    target_drift_detected_gauge.set(0.0)
    concept_drift_score_gauge.set(0.0)
    concept_drift_detected_gauge.set(0.0)
    drifted_features_count_gauge.set(0.0)
    prediction_window_predicted_rul_mean_gauge.set(0.0)
    prediction_window_actual_rul_mean_gauge.set(0.0)
    prediction_window_absolute_error_mae_gauge.set(0.0)
    prediction_window_absolute_error_p95_gauge.set(0.0)
    return {"status": "metrics reset to 0"}


if __name__ == "__main__":
    print("Сервер запущен. Откройте в браузере: http://localhost:8080/docs")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8080, reload=True)
