from fastapi.testclient import TestClient

from app.main import app


def test_query_stub_pipeline() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/query",
        json={
            "question": "Пшеница желтеет, нижние листья сохнут, что делать?",
            "pipeline": "B_HYBRID_RAG",
            "context": {"crop": "пшеница", "language": "ru"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline"] == "B_HYBRID_RAG"
    assert payload["answer"]["citations"]
    assert payload["trace"]["retrieval"]["enabled"] is True


def test_compare_returns_all_default_pipelines() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/query/compare",
        json={"question": "Pomidor barglari sargayib qolyapti, nima qilish kerak?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [result["pipeline"] for result in payload["results"]] == [
        "A_PURE_LLM",
        "B_HYBRID_RAG",
        "C_HYBRID_RAG_RERANK",
    ]
    assert payload["results"][0]["trace"]["query"]["language_detected"] == "uz_latn"
