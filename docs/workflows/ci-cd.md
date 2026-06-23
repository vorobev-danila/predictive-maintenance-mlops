# CI/CD Workflow

[← Back to README](../../README.md)

## Contents

- [Workflow file](#workflow-file)
- [Triggers](#triggers)
- [Pipeline steps](#pipeline-steps)
- [Local parity](#local-parity)

## Workflow File

CI/CD is defined in:

```text
.github/workflows/ci-cd.yml
```

GitOps delivery is defined separately in:

```text
argocd/application.yaml
```

## Triggers

The workflow runs on:

- `push` to `main`;
- `push` to `features`;
- `pull_request` to `main`;
- manual `workflow_dispatch`.

## Pipeline Steps

The `lint-test-build` job runs:

1. checkout repository;
2. set up Python 3.12;
3. install `uv`;
4. `uv sync --all-extras --frozen`;
5. `docker compose config --quiet`;
6. `kubeconform -strict -summary k8s/`;
7. `uv run flake8 src tests`;
8. `uv run black --check src tests`;
9. `uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=60`;
10. `docker build`.

The `deploy` job runs for pull requests and `main`:

1. build Docker image;
2. create temporary kind cluster;
3. load image into kind;
4. `kubectl apply -f k8s/`;
5. wait for API rollout;
6. smoke test `/health`.

For a persistent Kubernetes / Minikube environment, Argo CD watches the `k8s/`
directory and keeps the cluster synchronized with Git. See
[Argo CD workflow](argocd.md).

## Local Parity

Before pushing, run:

```bash
uv run flake8 src tests
uv run black --check src tests
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=60
docker compose config --quiet
docker build -t predictive-maintenance-mlops:ci-check .
```
