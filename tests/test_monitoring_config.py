from pathlib import Path

import yaml


def test_prometheus_drift_alert_rules_are_configured():
    prometheus_config = Path("prometheus.yml").read_text(encoding="utf-8")
    alert_rules = yaml.safe_load(Path("prometheus_alert_rules.yml").read_text())

    assert "/etc/prometheus/alert_rules.yml" in prometheus_config
    alert_names = {
        rule["alert"] for group in alert_rules["groups"] for rule in group["rules"]
    }
    assert {
        "DataDriftDetected",
        "TargetDriftDetected",
        "ConceptDriftDetected",
        "AnyDriftDetected",
    }.issubset(alert_names)


def test_kubernetes_prometheus_drift_alert_rules_are_configured():
    prometheus_documents = list(
        yaml.safe_load_all(Path("k8s/prometheus.yaml").read_text())
    )
    config_map = next(
        document for document in prometheus_documents if document["kind"] == "ConfigMap"
    )
    deployment = next(
        document
        for document in prometheus_documents
        if document["kind"] == "Deployment"
    )
    alert_rules = yaml.safe_load(config_map["data"]["alert_rules.yml"])
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    mount_paths = {
        mount["mountPath"]
        for mount in container["volumeMounts"]
        if mount["name"] == "prometheus-config"
    }
    alert_names = {
        rule["alert"] for group in alert_rules["groups"] for rule in group["rules"]
    }

    assert "rule_files:" in config_map["data"]["prometheus.yml"]
    assert "/etc/prometheus/alert_rules.yml" in mount_paths
    assert {
        "DataDriftDetected",
        "TargetDriftDetected",
        "ConceptDriftDetected",
        "AnyDriftDetected",
    }.issubset(alert_names)
    assert deployment["spec"]["strategy"]["type"] == "Recreate"


def test_argocd_application_targets_k8s_manifests():
    application = yaml.safe_load(Path("argocd/application.yaml").read_text())

    assert application["kind"] == "Application"
    assert application["metadata"]["namespace"] == "argocd"
    assert application["spec"]["source"]["path"] == "k8s"
    assert application["spec"]["destination"]["namespace"] == "default"
    assert application["spec"]["syncPolicy"]["automated"]["selfHeal"] is True


def test_kubernetes_grafana_provisioning_is_mounted():
    provisioning = yaml.safe_load(Path("k8s/grafana-provisioning.yaml").read_text())
    grafana_documents = list(yaml.safe_load_all(Path("k8s/grafana.yaml").read_text()))
    deployment = next(
        document for document in grafana_documents if document["kind"] == "Deployment"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    mount_paths = {
        mount["mountPath"]
        for mount in container["volumeMounts"]
        if mount["name"] == "grafana-provisioning"
    }

    assert provisioning["kind"] == "ConfigMap"
    assert {"prometheus.yml", "dashboard.yml", "mlops-monitoring.json"}.issubset(
        provisioning["data"]
    )
    assert {
        "/etc/grafana/provisioning/datasources/prometheus.yml",
        "/etc/grafana/provisioning/dashboards/dashboard.yml",
        "/var/lib/grafana/dashboards/mlops-monitoring.json",
    }.issubset(mount_paths)
    assert deployment["spec"]["strategy"]["type"] == "Recreate"


def test_kubernetes_grafana_password_comes_from_secret():
    grafana_documents = list(yaml.safe_load_all(Path("k8s/grafana.yaml").read_text()))
    deployment = next(
        document for document in grafana_documents if document["kind"] == "Deployment"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    admin_password_env = next(
        item
        for item in container["env"]
        if item["name"] == "GF_SECURITY_ADMIN_PASSWORD"
    )

    assert admin_password_env["valueFrom"]["secretKeyRef"] == {
        "name": "predictive-maintenance-secrets",
        "key": "GRAFANA_ADMIN_PASSWORD",
    }


def test_kubernetes_retrain_enables_mlflow_logging():
    config_map = yaml.safe_load(Path("k8s/configmap.yaml").read_text())

    assert config_map["data"]["ENABLE_MLFLOW_LOGGING"] == "true"
    assert config_map["data"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert config_map["data"]["MLFLOW_S3_ENDPOINT_URL"] == "http://minio:9000"


def test_kubernetes_application_images_use_ghcr():
    deployment = yaml.safe_load(Path("k8s/deployment.yaml").read_text())
    ui_documents = list(yaml.safe_load_all(Path("k8s/ui.yaml").read_text()))
    ui_deployment = next(
        document for document in ui_documents if document["kind"] == "Deployment"
    )

    api_image = deployment["spec"]["template"]["spec"]["containers"][0]["image"]
    ui_image = ui_deployment["spec"]["template"]["spec"]["containers"][0]["image"]

    assert api_image == "ghcr.io/vorobev-danila/predictive-maintenance-mlops:latest"
    assert ui_image == "ghcr.io/vorobev-danila/predictive-maintenance-mlops:latest"


def test_runtime_configs_do_not_commit_known_demo_passwords():
    checked_paths = [
        Path(".env.example"),
        Path("docker-compose.yml"),
        *Path("k8s").glob("*.yaml"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked_paths)

    assert "minio123" not in combined
    assert "GRAFANA_ADMIN_PASSWORD=admin" not in combined
    assert 'value: "admin"' not in combined
    assert not Path("k8s/secret.yaml").exists()
