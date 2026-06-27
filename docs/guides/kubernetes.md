# Kubernetes / Minikube Guide

[Back to README](../../README.md)

## Contents

- [Role in the project](#role-in-the-project)
- [Run in Minikube](#run-in-minikube)
- [Open services](#open-services)
- [Manifests](#manifests)
- [Argo CD delivery](#argo-cd-delivery)
- [SQLite state](#sqlite-state)

## Role in the Project

Kubernetes is the target runtime for the project. Docker Compose remains a
local debug mode, while Kubernetes shows how the service behaves closer to a
real deployment environment: deployments, services, secrets, config maps,
persistent volumes and rollout checks.

## Run in Minikube

By default, Kubernetes manifests use the GHCR image:

```text
ghcr.io/vorobev-danila/predictive-maintenance-mlops:latest
```

After `main` is updated, GitHub Actions publishes this image automatically.
For a local-only Minikube demo before GHCR has the image, build and load the
same tag manually:

```bash
minikube start
docker build -t ghcr.io/vorobev-danila/predictive-maintenance-mlops:latest .
minikube image load ghcr.io/vorobev-danila/predictive-maintenance-mlops:latest
```

Create runtime secrets from local environment variables before applying
manifests. Do not commit real secret values:

```bash
kubectl create secret generic predictive-maintenance-secrets \
  --from-literal=MINIO_ROOT_USER="$MINIO_ROOT_USER" \
  --from-literal=MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --from-literal=GRAFANA_ADMIN_PASSWORD="$GRAFANA_ADMIN_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -
```

An example manifest with placeholder values is available at
[`docs/reference/kubernetes-secret.example.yaml`](../reference/kubernetes-secret.example.yaml).

Then apply manifests:

```bash
kubectl apply -f k8s/
kubectl rollout status deployment/predictive-maintenance-api
kubectl rollout status deployment/minio
kubectl rollout status deployment/mlflow
kubectl rollout status deployment/prometheus
kubectl rollout status deployment/grafana
```

## Open Services

Get API URL:

```bash
minikube service predictive-maintenance-api --url
```

Open supporting services:

```bash
minikube service mlflow --url
minikube service minio --url
minikube service prometheus --url
minikube service grafana --url
```

If `minikube service` is not convenient, use port-forward:

```bash
kubectl port-forward service/predictive-maintenance-api 8080:8080
kubectl port-forward service/mlflow 5000:5000
kubectl port-forward service/minio 9000:9000
kubectl port-forward service/minio 9001:9001
kubectl port-forward service/prometheus 9090:9090
kubectl port-forward service/grafana 3000:3000
```

## Manifests

Kubernetes manifests include:

- API Deployment and NodePort Service;
- Streamlit UI Deployment and Service;
- MinIO Deployment, Service and bucket creation Job;
- MLflow Deployment and Service;
- Prometheus Deployment, Service and ConfigMap;
- Grafana Deployment, Service and provisioning ConfigMap;
- PVC for API state, MinIO, MLflow, Prometheus and Grafana;
- ConfigMap for runtime settings;
- Kubernetes Secret created outside the committed `k8s/` directory.

Validate manifests:

```bash
docker run --rm -v "${PWD}:/work" ghcr.io/yannh/kubeconform:latest -strict -summary /work/k8s
```

## Argo CD Delivery

Argo CD GitOps delivery is configured separately from raw Kubernetes manifests:

```text
argocd/application.yaml
```

See [Argo CD workflow](../workflows/argocd.md) for installation, sync and
validation commands.

## SQLite State

Prediction history and drift reports use SQLite and JSON files in the API state
PVC.

Kubernetes paths:

```text
/app/state/predictions.db
/app/state/reports
```

This is enough for a single-replica educational Minikube setup. If the API later
needs multiple replicas or production-like concurrency, replace SQLite with
PostgreSQL behind the same repository layer.
