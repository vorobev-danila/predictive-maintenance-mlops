# Argo CD

[Back to README](../../README.md)

## Contents

- [Purpose](#purpose)
- [Application manifest](#application-manifest)
- [Run in Minikube](#run-in-minikube)
- [Sync and validate](#sync-and-validate)
- [Image flow](#image-flow)

## Purpose

Argo CD is the GitOps CD layer for the Kubernetes manifests in `k8s/`.
GitHub Actions still performs linting, tests, Docker build and a kind smoke
deploy, while Argo CD continuously reconciles the target Minikube/Kubernetes
cluster from the repository.

## Application Manifest

The Argo CD application is defined in:

```text
argocd/application.yaml
```

It watches:

```text
repoURL: https://github.com/vorobev-danila/predictive-maintenance-mlops.git
targetRevision: main
path: k8s
```

For forks or local demos, update `repoURL` and `targetRevision` before applying
the manifest.

## Run in Minikube

Start Minikube and install Argo CD:

```bash
minikube start
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl rollout status deployment/argocd-server -n argocd --timeout=180s
```

The default Kubernetes manifests use the GHCR image:

```text
ghcr.io/vorobev-danila/predictive-maintenance-mlops:latest
```

After a merge to `main`, GitHub Actions builds and pushes this image
automatically. For a local-only demo before the image is available in GHCR,
build and load the same tag into Minikube:

```bash
docker build -t ghcr.io/vorobev-danila/predictive-maintenance-mlops:latest .
minikube image load ghcr.io/vorobev-danila/predictive-maintenance-mlops:latest
```

Create the Argo CD application:

```bash
kubectl apply -f argocd/application.yaml
```

## Sync and Validate

Open the Argo CD UI:

```bash
kubectl port-forward svc/argocd-server -n argocd 8081:443
```

Then open:

```text
https://localhost:8081
```

Initial admin password:

```bash
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d
```

Check application state:

```bash
kubectl get application predictive-maintenance-mlops -n argocd
kubectl get pods
```

Open project services:

```bash
kubectl port-forward service/predictive-maintenance-api 8080:8080
kubectl port-forward service/predictive-maintenance-ui 8501:8501
kubectl port-forward service/grafana 3000:3000
kubectl port-forward service/prometheus 9090:9090
```

## Image Flow

Argo CD applies Kubernetes manifests but does not build Docker images. The image
flow is:

1. GitHub Actions validates the project.
2. On `push` to `main`, GitHub Actions pushes `latest` and the commit SHA tag to
   GHCR.
3. Argo CD syncs `k8s/` manifests that reference the GHCR image.

If the GHCR package is private, make it public for the educational demo or add a
Kubernetes `imagePullSecret`.
