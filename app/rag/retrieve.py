from app.rag.normalize import NormalizedQuery, normalize_query_to_russian
from app.rag.rrf import RankedCandidate, reciprocal_rank_fusion
from app.schemas import RetrievalCandidate, RetrieveResponse


def retrieve_stub(query: str, language: str = "auto", top_k: int = 10) -> RetrieveResponse:
    normalized = normalize_query_to_russian(query, language)
    lexical = _lexical_stub_candidates(normalized)
    vector = _vector_stub_candidates(normalized)
    merged = reciprocal_rank_fusion({"lexical": lexical, "vector": vector})[:top_k]

    candidates = [
        RetrievalCandidate(
            chunk_id=candidate.chunk_id,
            title="Stub Russian-first agronomy KB",
            text_preview=_preview_for_chunk(candidate.chunk_id),
            rrf_score=round(candidate.rrf_score, 6),
            lexical_rank=candidate.ranks.get("lexical"),
            vector_rank=candidate.ranks.get("vector"),
            reranker_score=None,
            included_in_prompt=True,
        )
        for candidate in merged
    ]
    return RetrieveResponse(
        query=query,
        normalized_ru=normalized.normalized_ru,
        language_detected=normalized.language_detected,
        entities=normalized.entities,
        candidates=candidates,
    )


def _lexical_stub_candidates(normalized: NormalizedQuery) -> list[RankedCandidate]:
    text = normalized.normalized_ru.lower()
    candidates: list[RankedCandidate] = []
    if "пшениц" in text or normalized.entities.get("crop") == "пшеница":
        candidates.append(RankedCandidate("stub-wheat-nitrogen-001", 1, 0.92))
        candidates.append(RankedCandidate("stub-wheat-septoria-002", 2, 0.73))
    if "томат" in text or normalized.entities.get("crop") == "томат":
        candidates.append(RankedCandidate("stub-tomato-yellowing-001", 1, 0.88))
    if not candidates:
        candidates.append(RankedCandidate("stub-general-diagnosis-001", 1, 0.5))
    return candidates


def _vector_stub_candidates(normalized: NormalizedQuery) -> list[RankedCandidate]:
    symptoms = normalized.entities.get("symptoms", [])
    if isinstance(symptoms, list) and "пожелтение листьев" in symptoms:
        return [
            RankedCandidate("stub-wheat-nitrogen-001", 1, 0.84),
            RankedCandidate("stub-tomato-yellowing-001", 2, 0.77),
            RankedCandidate("stub-general-diagnosis-001", 3, 0.61),
        ]
    return [RankedCandidate("stub-general-diagnosis-001", 1, 0.56)]


def _preview_for_chunk(chunk_id: str) -> str:
    previews = {
        "stub-wheat-nitrogen-001": "Дефицит азота часто начинается с пожелтения нижних листьев.",
        "stub-wheat-septoria-002": "Септориоз может вызывать пятна и усыхание листьев пшеницы.",
        "stub-tomato-yellowing-001": "Пожелтение листьев томата может быть связано с питанием, поливом или болезнями.",
        "stub-general-diagnosis-001": "Для точной диагностики нужны культура, фото симптома и история ухода.",
    }
    return previews.get(chunk_id, "Stub retrieval candidate.")
