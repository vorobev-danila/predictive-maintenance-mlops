# Predictive Maintenance MLOps

MLOps-проект для прогнозирования остаточного ресурса авиационных двигателей (RUL, Remaining Useful Life) на данных NASA CMAPSS. В проекте есть обучение модели, трекинг экспериментов в MLflow, версионирование данных через DVC, FastAPI-сервис для инференса, Docker/docker-compose, Kubernetes-манифесты и CI/CD в GitHub Actions.

## Что внутри

- Датасет: NASA CMAPSS, файлы FD001-FD004 в `data/raw`.
- Модель: `RandomForestRegressor` из scikit-learn.
- API: FastAPI с OpenAPI-документацией.
- Трекинг: MLflow, артефакты хранятся в MinIO.
- Данные: DVC с S3-compatible remote в MinIO.
- Мониторинг: Prometheus + Grafana.
- Контейнеризация: `Dockerfile` и `docker-compose.yml`.
- Оркестрация: Kubernetes manifests в `k8s/`.
- CI/CD: GitHub Actions workflow `.github/workflows/ci-cd.yml`.
- Шаблонизация: Cookiecutter template в `cookiecutter-template/`.

## Структура проекта

```text
.
|-- data/                     # DVC-managed raw data
|-- models/                   # обученная модель, scaler, metrics, features
|-- src/
|   |-- api/main.py           # FastAPI приложение
|   |-- data/                 # загрузка и анализ данных
|   |-- features/             # feature engineering
|   |-- models/               # обучение и сохранение модели
|   |-- evaluation/           # оценка качества
|   `-- pipeline.py           # полный training pipeline + MLflow
|-- tests/                    # pytest тесты API
|-- k8s/                      # Kubernetes Deployment и Service
|-- cookiecutter-template/    # шаблон проекта
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
`-- uv.lock
```

## Быстрый старт

Требования:

- Python 3.12+
- `uv`
- Docker Desktop
- Git
- DVC, если нужно работать с версионированием данных

Установить зависимости:

```bash
uv sync --all-extras --frozen
```

Запустить инфраструктуру:

```bash
docker compose up -d minio mlflow prometheus grafana
```

Обучить модель и залогировать experiment в MLflow:

```bash
uv run python src/pipeline.py
```

Запустить API локально:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Открыть OpenAPI:

```text
http://localhost:8080/docs
```

## Docker Compose

Запустить весь стек:

```bash
docker compose up --build
```

Основные адреса:

- FastAPI: `http://localhost:8080/docs`
- MLflow: `http://localhost:5000`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Доступы MinIO:

```text
login: minio
password: minio123
```

Доступ Grafana:

```text
login: admin
password: admin
```

## API

Основные эндпоинты:

- `GET /health` - проверка загрузки модели.
- `POST /predict` - предсказание RUL по sensor/settings payload.
- `GET /model_metrics` - метрики последней сохраненной модели.
- `POST /retrain` - запуск переобучения модели.
- `GET /metrics` - Prometheus metrics.
- `POST /reset_metrics` - сброс пользовательских Prometheus gauge.

Пример проверки OpenAPI:

```bash
curl http://localhost:8080/openapi.json
```

## MLflow и MinIO

Training pipeline пишет в MLflow:

- параметры модели;
- метрики `train_mae`, `val_mae`, `test_mae`, `train_rmse`, `val_rmse`, `test_rmse`, `train_r2`, `val_r2`, `test_r2`;
- артефакты `metrics.json`, `features.json`, `scaler.pkl`;
- зарегистрированную sklearn-модель `predictive-maintenance-random-forest`.

Метрики и параметры хранятся в backend store MLflow. Артефакты сохраняются в MinIO bucket `mlflow`.

## DVC

Данные `data/raw` добавлены в DVC:

```bash
dvc status
dvc pull
dvc push
```

DVC remote настроен на MinIO bucket `dvc`.

## Тесты и линтеры

Запустить тесты:

```bash
uv run pytest
```

Запустить линтер:

```bash
uv run flake8 src tests
```

Проверить форматирование:

```bash
uv run black --check src tests
```

Автоматически отформатировать код:

```bash
uv run black src tests
```

## Docker build

Собрать image вручную:

```bash
docker build -t predictive-maintenance-mlops:latest .
```

Запустить контейнер API:

```bash
docker run --rm -p 8080:8080 predictive-maintenance-mlops:latest
```

## Kubernetes / minikube

Собрать image и загрузить его в minikube:

```bash
minikube start
docker build -t predictive-maintenance-mlops:latest .
minikube image load predictive-maintenance-mlops:latest
kubectl apply -f k8s/
kubectl rollout status deployment/predictive-maintenance-api
```

Получить URL сервиса:

```bash
minikube service predictive-maintenance-api --url
```

## CI/CD

Workflow: `.github/workflows/ci-cd.yml`.

Запускается на:

- `push` в `main`;
- `push` в `features`;
- `pull_request` в `main`;
- ручной запуск через `workflow_dispatch`.

Pipeline делает:

1. `uv sync --all-extras --frozen`
2. `uv run flake8 src tests`
3. `uv run black --check src tests`
4. `uv run pytest`
5. `docker build`
6. deploy в временный kind Kubernetes cluster для PR в `main` и push в `main`

## Cookiecutter

Сгенерировать новый проект из шаблона:

```bash
uvx cookiecutter cookiecutter-template
```

Шаблон содержит заготовки для FastAPI, MLflow pipeline, Docker, Kubernetes, tests и GitHub Actions.

## Полезные команды

Проверить состояние Git:

```bash
git status
```

Запушить ветку features:

```bash
git push origin features
```

Запустить полный локальный quality check:

```bash
uv run flake8 src tests
uv run black --check src tests
uv run pytest
docker build -t predictive-maintenance-mlops:ci-check .
```
