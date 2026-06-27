# Predictive Maintenance MLOps

[![CI/CD](https://github.com/vorobev-danila/predictive-maintenance-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/vorobev-danila/predictive-maintenance-mlops/actions/workflows/ci-cd.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](docs/guides/api.md)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](docs/guides/mlflow-dvc.md)

An educational end-to-end MLOps project for Remaining Useful Life prediction on
NASA CMAPSS turbofan engine data.

The repository covers model training, MLflow tracking and model registration,
DVC data versioning, FastAPI serving, SQLite prediction history, Docker,
Kubernetes/Minikube, Prometheus/Grafana monitoring, GitHub Actions CI/CD and
Argo CD GitOps delivery.

## Quick Links

| Area | Link |
| --- | --- |
| Quickstart | [docs/guides/quickstart.md](docs/guides/quickstart.md) |
| API and OpenAPI | [docs/guides/api.md](docs/guides/api.md) |
| MLflow, MinIO and DVC | [docs/guides/mlflow-dvc.md](docs/guides/mlflow-dvc.md) |
| Docker and Docker Compose | [docs/guides/docker.md](docs/guides/docker.md) |
| Kubernetes / Minikube | [docs/guides/kubernetes.md](docs/guides/kubernetes.md) |
| Drift monitoring | [docs/guides/monitoring.md](docs/guides/monitoring.md) |
| CI/CD | [docs/workflows/ci-cd.md](docs/workflows/ci-cd.md) |
| Argo CD | [docs/workflows/argocd.md](docs/workflows/argocd.md) |
| Git flow | [docs/workflows/git-flow.md](docs/workflows/git-flow.md) |
| Cookiecutter template | [docs/templates/cookiecutter.md](docs/templates/cookiecutter.md) |
| Command reference | [docs/reference/commands.md](docs/reference/commands.md) |

## What Is Included

- **Dataset:** NASA CMAPSS FD001-FD004 files managed through DVC.
- **Baseline model:** scikit-learn `GradientBoostingRegressor`.
- **Training pipeline:** `src/pipeline.py`.
- **API:** FastAPI service with OpenAPI docs at `/docs`.
- **Prediction history:** SQLite repository and `/predictions/recent` endpoint.
- **Experiment tracking:** MLflow with MinIO-backed artifacts.
- **Data versioning:** DVC S3-compatible remote in MinIO.
- **Monitoring:** Prometheus metrics and Grafana dashboards.
- **Drift:** data drift, target drift, concept drift, JSON reports and Plotly simulation reports.
- **Packaging:** Docker image and Docker Compose for local debugging.
- **Orchestration:** Kubernetes manifests in `k8s/`.
- **CI/CD:** GitHub Actions checks, GHCR image publishing and kind smoke deploy.
- **GitOps CD:** Argo CD application in `argocd/`.
- **Project template:** Cookiecutter template in `cookiecutter-template/`.

## Documentation Map

### Project

- [Overview](docs/project/overview.md) - project purpose and MLOps scope.
- [Repository structure](docs/project/structure.md) - directory layout and module responsibilities.
- [License status](docs/project/license.md) - current license status.

### Guides

- [Quickstart](docs/guides/quickstart.md) - local run with `uv`, Docker Compose and API.
- [API guide](docs/guides/api.md) - endpoints, OpenAPI and prediction history.
- [MLflow and DVC](docs/guides/mlflow-dvc.md) - tracking, registry, MinIO buckets and DVC remote.
- [Docker guide](docs/guides/docker.md) - compose services, ports and manual image build.
- [Kubernetes guide](docs/guides/kubernetes.md) - Minikube runbook and manifest overview.
- [Monitoring and drift](docs/guides/monitoring.md) - drift calculation, reports and Prometheus metrics.

### Workflows

- [CI/CD](docs/workflows/ci-cd.md) - checks, build, GHCR push and kind deploy.
- [Argo CD](docs/workflows/argocd.md) - GitOps delivery in Kubernetes / Minikube.
- [Git flow](docs/workflows/git-flow.md) - feature branch flow and Conventional Commits.

### Templates and Reference

- [Cookiecutter](docs/templates/cookiecutter.md) - generate a new project from the template.
- [Commands](docs/reference/commands.md) - useful command snippets.
- [Contributing](CONTRIBUTING.md) - contribution workflow.

## Start Here

```bash
cp .env.example .env
uv sync --all-extras --frozen
docker compose up -d minio mlflow prometheus grafana
uv run python src/pipeline.py
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

OpenAPI is available at:

```text
http://localhost:8080/docs
```

See the full scenario in [Quickstart](docs/guides/quickstart.md).

## Repository Structure

```text
.
|-- argocd/                 # Argo CD Application manifest
|-- docs/                   # documentation hub
|-- data/                   # DVC-managed raw data
|-- models/                 # model, metrics and feature artifacts
|-- src/                    # API, pipeline, data, monitoring and UI
|-- tests/                  # pytest tests
|-- k8s/                    # Kubernetes manifests
|-- grafana/                # Grafana dashboards for Docker Compose
|-- cookiecutter-template/  # reusable MLOps project template
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
`-- uv.lock
```

More details: [docs/project/structure.md](docs/project/structure.md).

## Contributing

The project uses a simple feature branch flow and Conventional Commits.

Before opening a pull request, run:

```bash
uv run flake8 src tests
uv run black --check src tests
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=60
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

The project does not currently include a dedicated `LICENSE` file. Treat it as
an educational repository until explicit license terms are added.

More details: [docs/project/license.md](docs/project/license.md).
