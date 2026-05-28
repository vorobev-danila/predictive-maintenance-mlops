# Command Reference

[← Back to README](../../README.md)

## Local Development

```bash
cp .env.example .env
uv sync --all-extras --frozen
uv run python src/pipeline.py
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

## Quality Checks

```bash
uv run flake8 src tests
uv run black --check src tests
uv run black src tests
uv run pytest
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=60
```

## Docker

```bash
docker compose up -d minio mlflow prometheus grafana
docker compose up --build
docker compose config --quiet
docker build -t predictive-maintenance-mlops:latest .
docker run --rm -p 8080:8080 predictive-maintenance-mlops:latest
```

## DVC

```bash
dvc status
dvc pull
dvc push
```

## Kubernetes

```bash
minikube start
minikube image load predictive-maintenance-mlops:latest
kubectl apply -f k8s/
kubectl rollout status deployment/predictive-maintenance-api
minikube service predictive-maintenance-api --url
```

## Git

```bash
git status
git push origin features
```
