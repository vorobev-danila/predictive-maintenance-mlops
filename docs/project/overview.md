# Project Overview

[← Back to README](../../README.md)

## Contents

- [Purpose](#purpose)
- [MLOps scope](#mlops-scope)
- [Current components](#current-components)

## Purpose

Predictive Maintenance MLOps прогнозирует остаточный ресурс авиационного
двигателя в циклах эксплуатации. Задача построена вокруг датасета NASA CMAPSS
и production-модели `GradientBoostingRegressor`.

Проект показывает не только обучение модели, но и полный контур вокруг нее:
tracking, serving, storage, CI/CD, контейнеризацию и запуск в Kubernetes.

## MLOps Scope

| Area | Implementation |
| --- | --- |
| Dataset | NASA CMAPSS, `data/raw` |
| Model | scikit-learn `GradientBoostingRegressor` |
| Training | `src/pipeline.py` |
| Experiment tracking | MLflow |
| Artifact storage | MinIO bucket `mlflow` |
| Data versioning | DVC remote in MinIO bucket `dvc` |
| Serving | FastAPI, OpenAPI `/docs` |
| Prediction history | SQLite database `state/predictions.db` |
| Monitoring | Prometheus + Grafana |
| Packaging | Docker image |
| Local debug | Docker Compose |
| Orchestration | Kubernetes / Minikube |
| CI/CD | GitHub Actions + Argo CD |
| Template | Cookiecutter |

## Current Components

- `src/pipeline.py` loads data, builds features, trains the model, evaluates it
  and logs metadata to MLflow.
- `src/api/main.py` exposes inference, metrics, retraining and prediction history.
- `src/storage/prediction_repository.py` isolates SQLite persistence from API logic.
- `k8s/` contains manifests for API, Streamlit UI, MinIO, MLflow, Prometheus and Grafana.
- `argocd/application.yaml` configures GitOps synchronization for Kubernetes manifests.
- `.github/workflows/ci-cd.yml` validates code, tests, Docker build and Kubernetes deploy.

Next planned areas are production image promotion, anomaly flag tuning and richer
Grafana dashboards.
