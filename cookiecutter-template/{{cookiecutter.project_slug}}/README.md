# {{ cookiecutter.project_name }}

MLOps template project for predictive maintenance models.

## Stack

- FastAPI service with OpenAPI docs
- scikit-learn training pipeline
- MLflow tracking and model registry
- DVC-ready data versioning
- Docker and docker-compose
- Kubernetes manifests for minikube
- GitHub Actions CI/CD skeleton

## Quick Start

```bash
uv sync
python -m {{ cookiecutter.package_name }}.pipeline
uvicorn {{ cookiecutter.package_name }}.api.main:app --host 0.0.0.0 --port {{ cookiecutter.fastapi_port }}
```

Open API docs at `http://localhost:{{ cookiecutter.fastapi_port }}/docs`.

## Docker

```bash
docker compose up --build
```

## Kubernetes

```bash
minikube start
kubectl apply -f k8s/
```
