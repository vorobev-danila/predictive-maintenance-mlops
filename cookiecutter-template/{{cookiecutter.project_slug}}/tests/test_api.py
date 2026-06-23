from fastapi.testclient import TestClient

from {{ cookiecutter.package_name }}.api.main import app


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
