# Docker Guide

[← Back to README](../../README.md)

## Contents

- [Docker Compose](#docker-compose)
- [Service URLs](#service-urls)
- [Manual image build](#manual-image-build)

## Docker Compose

Start the full local stack:

```bash
cp .env.example .env
docker compose up --build
```

Start only infrastructure services:

```bash
cp .env.example .env
docker compose up -d minio mlflow prometheus grafana
```

Validate compose configuration:

```bash
docker compose config --quiet
```

## Service URLs

| Service | URL |
| --- | --- |
| FastAPI | `http://localhost:8080/docs` |
| MLflow | `http://localhost:5000` |
| MinIO API | `http://localhost:9000` |
| MinIO Console | `http://localhost:9001` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

Credentials are read from the local `.env` file. Use strong local values and do
not commit `.env`.

MinIO:

```text
login: MINIO_ROOT_USER from .env
password: MINIO_ROOT_PASSWORD from .env
```

Grafana:

```text
login: admin
password: GRAFANA_ADMIN_PASSWORD from .env
```

## Manual Image Build

Build API image:

```bash
docker build -t predictive-maintenance-mlops:latest .
```

Run API container:

```bash
docker run --rm -p 8080:8080 predictive-maintenance-mlops:latest
```
