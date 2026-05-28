# Predictive Maintenance MLOps

[![CI/CD](https://github.com/vorobev-danila/predictive-maintenance-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/vorobev-danila/predictive-maintenance-mlops/actions/workflows/ci-cd.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](docs/guides/api.md)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](docs/guides/mlflow-dvc.md)

MLOps-проект для прогнозирования остаточного ресурса авиационных двигателей
(RUL, Remaining Useful Life) на данных NASA CMAPSS.

Проект объединяет полный учебный MLOps-контур: обучение модели, MLflow tracking,
DVC-версионирование данных, FastAPI-инференс, историю предсказаний в SQLite,
Docker, Kubernetes/Minikube, Prometheus/Grafana и CI/CD в GitHub Actions.

## Quick Links

| Раздел | Ссылка |
| --- | --- |
| Быстрый старт | [docs/guides/quickstart.md](docs/guides/quickstart.md) |
| API и OpenAPI | [docs/guides/api.md](docs/guides/api.md) |
| MLflow, MinIO и DVC | [docs/guides/mlflow-dvc.md](docs/guides/mlflow-dvc.md) |
| Docker и Docker Compose | [docs/guides/docker.md](docs/guides/docker.md) |
| Kubernetes / Minikube | [docs/guides/kubernetes.md](docs/guides/kubernetes.md) |
| CI/CD | [docs/workflows/ci-cd.md](docs/workflows/ci-cd.md) |
| Git flow | [docs/workflows/git-flow.md](docs/workflows/git-flow.md) |
| Cookiecutter template | [docs/templates/cookiecutter.md](docs/templates/cookiecutter.md) |
| Полезные команды | [docs/reference/commands.md](docs/reference/commands.md) |

## Что внутри

- **Датасет:** NASA CMAPSS, файлы FD001-FD004 в `data/raw`.
- **Базовая модель:** `RandomForestRegressor` из scikit-learn.
- **API:** FastAPI с OpenAPI-документацией на `/docs`.
- **Prediction history:** SQLite-backed repository и endpoint `/predictions/recent`.
- **Трекинг:** MLflow, backend store и артефакты через MinIO.
- **Данные:** DVC с S3-compatible remote в MinIO.
- **Мониторинг:** Prometheus metrics и Grafana.
- **Контейнеризация:** `Dockerfile` и `docker-compose.yml`.
- **Оркестрация:** Kubernetes manifests в `k8s/`.
- **CI/CD:** GitHub Actions workflow `.github/workflows/ci-cd.yml`.
- **Шаблонизация:** Cookiecutter template в `cookiecutter-template/`.

## Documentation Map

### Project

- [Overview](docs/project/overview.md) - назначение проекта и состав MLOps-контура.
- [Repository structure](docs/project/structure.md) - дерево директорий и ответственность модулей.
- [License status](docs/project/license.md) - текущий статус лицензирования.

### Guides

- [Quickstart](docs/guides/quickstart.md) - локальный запуск с `uv`, Docker Compose и API.
- [API guide](docs/guides/api.md) - endpoints, OpenAPI, история предсказаний.
- [MLflow and DVC](docs/guides/mlflow-dvc.md) - tracking, registry, MinIO buckets, DVC remote.
- [Docker guide](docs/guides/docker.md) - compose, адреса сервисов, ручная сборка image.
- [Kubernetes guide](docs/guides/kubernetes.md) - запуск в Minikube и состав manifests.

### Workflows

- [CI/CD](docs/workflows/ci-cd.md) - проверки, сборка и deploy в kind.
- [Git flow](docs/workflows/git-flow.md) - feature branch flow и Conventional Commits.

### Templates and Reference

- [Cookiecutter](docs/templates/cookiecutter.md) - генерация нового проекта из шаблона.
- [Commands](docs/reference/commands.md) - короткая шпаргалка команд.
- [Contributing](CONTRIBUTING.md) - как предлагать изменения.

## Start Here

```bash
cp .env.example .env
uv sync --all-extras --frozen
docker compose up -d minio mlflow prometheus grafana
uv run python src/pipeline.py
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

OpenAPI будет доступен по адресу:

```text
http://localhost:8080/docs
```

Полный сценарий запуска описан в [Quickstart](docs/guides/quickstart.md).

## Repository Structure

```text
.
|-- docs/                    # документационный портал
|-- data/                    # DVC-managed raw data
|-- models/                  # модель, scaler, metrics, features
|-- src/                     # API, pipeline, data, features, models
|-- tests/                   # pytest тесты
|-- k8s/                     # Kubernetes manifests
|-- cookiecutter-template/   # шаблон нового MLOps-проекта
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
`-- uv.lock
```

Подробности: [docs/project/structure.md](docs/project/structure.md).

## Contributing

Проект использует простой feature branch flow и Conventional Commits.

Перед pull request запустите локальные проверки:

```bash
uv run flake8 src tests
uv run black --check src tests
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=60
```

Подробные правила: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Лицензия проекта пока не зафиксирована отдельным `LICENSE` файлом.
До добавления лицензии используйте код как учебный проект и не распространяйте его
как open-source пакет без согласования условий.

Подробнее: [docs/project/license.md](docs/project/license.md).
