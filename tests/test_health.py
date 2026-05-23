from fastapi.testclient import TestClient

from app.main import app


def test_healthz() -> None:
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_system() -> None:
    client = TestClient(app)
    response = client.get("/v1/system")
    assert response.status_code == 200
    assert response.json()["app_name"]
