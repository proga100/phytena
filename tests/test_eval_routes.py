from fastapi.testclient import TestClient

from app.main import app


def test_eval_run_returns_pipeline_summary() -> None:
    client = TestClient(app)
    response = client.post("/v1/evals/run", json={"limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["cases"]) == 1
    assert set(payload["summary"].keys()) == {
        "A_PURE_LLM",
        "B_HYBRID_RAG",
        "C_HYBRID_RAG_RERANK",
    }
    assert "citation_rate" in payload["summary"]["B_HYBRID_RAG"]
