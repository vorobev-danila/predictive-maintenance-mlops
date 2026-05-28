# Kubernetes / Minikube Guide

[← Back to README](../../README.md)

## Contents

- [Role in the project](#role-in-the-project)
- [Run in Minikube](#run-in-minikube)
- [Open services](#open-services)
- [Manifests](#manifests)
- [SQLite state](#sqlite-state)

## Role in the Project

Kubernetes is the target runtime for the project. Docker Compose remains a
local debug mode, while Kubernetes shows how the service behaves closer to a
real deployment environment: deployments, services, secrets, config maps,
persistent volumes and rollout checks.

## Run in Minikube

Build the image, load it into Minikube and apply manifests:

```bash
minikube start
docker build -t predictive-maintenance-mlops:latest .
minikube image load predictive-maintenance-mlops:latest
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
- MinIO Deployment, Service and bucket creation Job;
- MLflow Deployment and Service;
- Prometheus Deployment, Service and ConfigMap;
- Grafana Deployment and Service;
- PVC for API state, MinIO, MLflow, Prometheus and Grafana;
- ConfigMap and Secret for runtime settings.

Validate manifests:

```bash
docker run --rm -v "${PWD}:/work" ghcr.io/yannh/kubeconform:latest -strict -summary /work/k8s
```

## SQLite State

Prediction history and future drift metadata use SQLite in the API state PVC.

Kubernetes path:

```text
/app/state/predictions.db
```

This is enough for a single-replica educational Minikube setup. If the API later
needs multiple replicas or production-like concurrency, replace SQLite with
PostgreSQL behind the same repository layer.
