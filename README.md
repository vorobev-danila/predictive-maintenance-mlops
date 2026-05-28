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
|-- k8s/                      # Kubernetes manifests для API, MLflow, MinIO, Prometheus, Grafana
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

Подготовить локальные переменные окружения:

```bash
cp .env.example .env
```

Файл `.env` используется Docker Compose автоматически. В репозиторий он не коммитится; в Git хранится только безопасный пример `.env.example`.

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
- `GET /predictions/recent` - последние сохраненные предсказания из SQLite.
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

Для локального доступа к MinIO DVC берет credentials из переменных окружения:

```bash
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
```

В PowerShell:

```powershell
$env:AWS_ACCESS_KEY_ID="minio"
$env:AWS_SECRET_ACCESS_KEY="minio123"
```

Сами ключи не должны храниться в `.dvc/config`; в Kubernetes они должны задаваться через `Secret`.

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

Основной целевой запуск проекта - Kubernetes/Minikube. `docker-compose.yml` остается удобным debug-режимом для локальной разработки.

Собрать image, загрузить его в minikube и применить все manifests:

```bash
minikube start
docker build -t predictive-maintenance-mlops:latest .
minikube image load predictive-maintenance-mlops:latest
kubectl apply -f k8s/
kubectl rollout status deployment/predictive-maintenance-api
kubectl rollout status deployment/minio
kubectl rollout status deployment/mlflow
kubectl rollout status deployment/prometheus
kubectl rollout status deployment/grafana
```

Получить URL API:

```bash
minikube service predictive-maintenance-api --url
```

Открыть остальные сервисы:

```bash
minikube service mlflow --url
minikube service minio --url
minikube service prometheus --url
minikube service grafana --url
```

Если `minikube service` не открывает URL автоматически, можно использовать port-forward:

```bash
kubectl port-forward service/predictive-maintenance-api 8080:8080
kubectl port-forward service/mlflow 5000:5000
kubectl port-forward service/minio 9000:9000
kubectl port-forward service/minio 9001:9001
kubectl port-forward service/prometheus 9090:9090
kubectl port-forward service/grafana 3000:3000
```

Kubernetes manifests включают:

- API Deployment и NodePort Service;
- MinIO Deployment, Service и Job для создания buckets `mlflow` и `dvc`;
- MLflow Deployment и Service;
- Prometheus Deployment, Service и ConfigMap;
- Grafana Deployment и Service;
- PVC для API state, MinIO, MLflow, Prometheus и Grafana;
- ConfigMap и Secret для runtime-настроек.

Для будущей истории предсказаний и drift metadata выбран SQLite. Файл БД будет храниться в PVC API по пути:

```text
/app/state/predictions.db
```

Это проще, чем отдельный PostgreSQL-сервис, и достаточно для учебного Minikube-стенда. Если позже потребуется production-like режим с несколькими replica API, SQLite можно заменить на PostgreSQL через repository-слой без переписывания бизнес-логики.

API уже записывает каждое успешное предсказание в SQLite и отдает последние записи:

```bash
curl "http://localhost:8080/predictions/recent?limit=20"
```

Эта история будет использоваться следующими этапами: web UI, флаги аномалий и расчет drift по окнам предсказаний.

## CI/CD

Workflow: `.github/workflows/ci-cd.yml`.

Запускается на:

- `push` в `main`;
- `push` в `features`;
- `pull_request` в `main`;
- ручной запуск через `workflow_dispatch`.

Pipeline делает:

1. `uv sync --all-extras --frozen`
2. `docker compose config --quiet`
3. `kubeconform -strict -summary k8s/`
4. `uv run flake8 src tests`
5. `uv run black --check src tests`
6. `uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=60`
7. `docker build`
8. deploy в временный kind Kubernetes cluster для PR в `main` и push в `main`
9. smoke test `/health` после Kubernetes deploy

## Git flow

В проекте используется простой feature branch flow:

1. `main` - стабильная ветка.
2. Разработка ведется в feature-ветках, например `features` или `feature/<task-name>`.
3. Из feature-ветки открывается Pull Request в `main`.
4. GitHub Actions должен пройти lint, format check, tests, Docker build и deploy/smoke test.
5. После успешного PR изменения вливаются в `main`.

Коммиты оформляются в стиле Conventional Commits:

```text
feat(api): add prediction history endpoint
fix(ci): validate kubernetes manifests
test: add drift calculator tests
docs: update minikube launch guide
chore(dvc): move credentials to environment
```

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
