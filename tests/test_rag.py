from fastapi.testclient import TestClient

from app.main import app
from app.rag.normalize import normalize_query_to_russian
from app.rag.rrf import RankedCandidate, reciprocal_rank_fusion


def test_normalize_uzbek_latin_query_to_russian_terms() -> None:
    normalized = normalize_query_to_russian(
        "Pomidor barglari sargayib qolyapti, nima qilish kerak?"
    )

    assert normalized.language_detected == "uz_latn"
    assert "томат" in normalized.normalized_ru
    assert normalized.entities["crop"] == "томат"


def test_rrf_merges_ranked_lists() -> None:
    merged = reciprocal_rank_fusion(
        {
            "lexical": [RankedCandidate("a", 1), RankedCandidate("b", 2)],
            "vector": [RankedCandidate("b", 1), RankedCandidate("a", 2)],
        }
    )

    assert {candidate.chunk_id for candidate in merged} == {"a", "b"}
    assert merged[0].rrf_score == merged[1].rrf_score


def test_retrieve_endpoint_returns_stub_candidates() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/retrieve",
        json={"query": "Пшеница желтеет, нижние листья сохнут", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["language_detected"] == "ru"
    assert payload["candidates"]
    assert payload["candidates"][0]["rrf_score"] > 0
