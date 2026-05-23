from fastapi.testclient import TestClient

from app.main import app


def test_admin_home_loads() -> None:
    client = TestClient(app)
    response = client.get("/admin")

    assert response.status_code == 200
    assert "Phytena Lab" in response.text


def test_admin_comparison_runs_all_pipelines() -> None:
    client = TestClient(app)
    response = client.post(
        "/admin/comparisons",
        data={
            "question": "Пшеница желтеет, нижние листья сохнут, что делать?",
            "crop": "пшеница",
            "region": "Узбекистан",
            "language": "ru",
        },
    )

    assert response.status_code == 200
    assert "A_PURE_LLM" in response.text
    assert "B_HYBRID_RAG" in response.text
    assert "C_HYBRID_RAG_RERANK" in response.text
