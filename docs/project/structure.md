# Repository Structure

[← Back to README](../../README.md)

## Contents

- [Tree](#tree)
- [Main modules](#main-modules)
- [Generated state](#generated-state)

## Tree

```text
.
|-- .github/workflows/       # GitHub Actions CI/CD
|-- configs/                 # model and pipeline configuration
|-- cookiecutter-template/   # reusable project template
|-- data/                    # DVC-managed data
|-- docs/                    # documentation hub
|-- docker/                  # supporting Docker assets
|-- k8s/                     # Kubernetes manifests
|-- models/                  # trained model artifacts
|-- notebooks/               # exploratory notebooks
|-- src/
|   |-- api/                 # FastAPI application
|   |-- data/                # data loading and analysis
|   |-- evaluation/          # model evaluation
|   |-- features/            # feature engineering
|   |-- models/              # model training and persistence helpers
|   |-- storage/             # repository layer for local state
|   `-- pipeline.py          # training pipeline
|-- tests/                   # pytest suite
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
`-- uv.lock
```

## Main Modules

| Path | Responsibility |
| --- | --- |
| `src/api/main.py` | FastAPI app, OpenAPI, inference, Prometheus metrics, retraining |
| `src/pipeline.py` | End-to-end model training and MLflow logging |
| `src/data/data_loader.py` | Loading NASA CMAPSS files |
| `src/features/feature_engineering.py` | Feature preparation |
| `src/models/train_model.py` | Model training |
| `src/evaluation/evaluate.py` | Quality metrics |
| `src/storage/prediction_repository.py` | SQLite-backed prediction history |
| `tests/` | API, pipeline helpers and repository tests |
| `k8s/` | Minikube/Kubernetes runtime manifests |

## Generated State

The following local files are generated during runtime and should not be
committed:

- `.env`
- `.coverage`
- `.pytest_cache/`
- `state/predictions.db`
- downloaded raw data under `data/raw/`
- generated reports under `reports/`
