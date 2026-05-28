# Quickstart

[← Back to README](../../README.md)

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Run infrastructure](#run-infrastructure)
- [Train model](#train-model)
- [Run API](#run-api)
- [Validate](#validate)

## Requirements

- Python 3.12+
- `uv`
- Docker Desktop
- Git
- DVC, if you need to work with versioned data

## Setup

Prepare local environment variables:

```bash
cp .env.example .env
```

Install dependencies:

```bash
uv sync --all-extras --frozen
```

The `.env` file is consumed by Docker Compose automatically. It is ignored by
Git; only `.env.example` is stored in the repository.

## Run Infrastructure

Start services required for local development:

```bash
docker compose up -d minio mlflow prometheus grafana
```

## Train Model

Run the training pipeline and log an experiment to MLflow:

```bash
uv run python src/pipeline.py
```

The pipeline saves local artifacts to `models/` and logs metrics, parameters and
model artifacts to MLflow.

## Run API

Start FastAPI locally:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open OpenAPI:

```text
http://localhost:8080/docs
```

## Validate

Check API health:

```bash
curl http://localhost:8080/health
```

Check OpenAPI schema:

```bash
curl http://localhost:8080/openapi.json
```

Check recent prediction history after at least one `/predict` call:

```bash
curl "http://localhost:8080/predictions/recent?limit=20"
```
