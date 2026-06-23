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


def test_argocd_application_targets_k8s_manifests():
    application = yaml.safe_load(Path("argocd/application.yaml").read_text())

    assert application["kind"] == "Application"
    assert application["metadata"]["namespace"] == "argocd"
    assert application["spec"]["source"]["path"] == "k8s"
    assert application["spec"]["destination"]["namespace"] == "default"
    assert application["spec"]["syncPolicy"]["automated"]["selfHeal"] is True
