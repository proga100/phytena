import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.embeddings import EmbeddingsClient
from app.config import get_settings
from app.rag.normalize import NormalizedQuery, normalize_query_to_russian
from app.rag.rrf import RankedCandidate, reciprocal_rank_fusion
from app.schemas import RetrievalCandidate, RetrieveResponse


async def fetch_treatments_for_diseases(
    db: AsyncSession, rag_disease_ids: list[str], per_disease: int = 8
) -> dict[str, list[dict]]:
    """Fetch treatment options for retrieved diseases (the second step of the flow:
    symptom -> disease -> treatments). Keyed by rag_disease_id.
    """
    if not rag_disease_ids:
        return {}
    ids = [uuid.UUID(i) for i in rag_disease_ids]
    rows = await db.execute(
        text("""
            SELECT rag_disease_id, drug_name, drug_description, dose_min, dose_max, type
            FROM rag_treatments
            WHERE rag_disease_id = ANY(:ids) AND drug_name IS NOT NULL
        """),
        {"ids": ids},
    )
    out: dict[str, list[dict]] = {}
    for did, drug, drug_desc, lo, hi, ttype in rows:
        bucket = out.setdefault(str(did), [])
        if len(bucket) >= per_disease:
            continue
        bucket.append(
            {"drug": drug, "description": drug_desc, "dose_min": lo, "dose_max": hi, "type": ttype}
        )
    return out

async def retrieve(
    query: str, 
    db: AsyncSession, 
    language: str = "auto", 
    top_k: int = 5,
) -> RetrieveResponse:
    settings = get_settings()
    normalized = normalize_query_to_russian(query, language)
    
    # 1. Vector Search. The Growz docs are Uzbek, so embed the original query text
    # (not the Russian normalization) to keep the query in the same language space.
    embed_client = EmbeddingsClient(api_key=settings.gemini_api_key, model=settings.embeddings_model)
    query_vector = await embed_client.get_embedding(
        query,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=settings.embedding_dimension
    )

    # Retrieval targets the Growz Uzbek disease docs (rag_diseases). The data is
    # Uzbek, so FTS uses the 'simple' config (no Russian stemmer) and the query is
    # matched against the original wording, not the Russian normalization.
    vector_stmt = text("""
        SELECT id, 1 - (embedding <=> :query_vector) as score
        FROM rag_diseases
        ORDER BY embedding <=> :query_vector
        LIMIT :limit
    """)

    lexical_stmt = text("""
        SELECT id, ts_rank_cd(fts, plainto_tsquery('simple', :query_text)) as score
        FROM rag_diseases
        WHERE fts @@ plainto_tsquery('simple', :query_text)
        ORDER BY score DESC
        LIMIT :limit
    """)

    # Execute searches. Note: query_vector should be passed as a list of floats.
    try:
        vector_results = await db.execute(vector_stmt, {"query_vector": query_vector, "limit": top_k * 2})
        vector_candidates = [
            RankedCandidate(str(row[0]), i + 1, float(row[1])) 
            for i, row in enumerate(vector_results)
        ]
    except Exception:
        # Fallback to string representation if list fails
        vector_results = await db.execute(vector_stmt, {"query_vector": str(query_vector), "limit": top_k * 2})
        vector_candidates = [
            RankedCandidate(str(row[0]), i + 1, float(row[1])) 
            for i, row in enumerate(vector_results)
        ]

    lexical_results = await db.execute(lexical_stmt, {"query_text": query, "limit": top_k * 2})
    lexical_candidates = [
        RankedCandidate(str(row[0]), i + 1, float(row[1])) 
        for i, row in enumerate(lexical_results)
    ]

    # 2. RRF Fusion
    merged = reciprocal_rank_fusion({"lexical": lexical_candidates, "vector": vector_candidates})[:top_k]

    # 3. Fetch full text for candidates in one go
    if not merged:
        return RetrieveResponse(
            query=query,
            normalized_ru=normalized.normalized_ru,
            language_detected=normalized.language_detected,
            entities=normalized.entities,
            candidates=[],
        )

    chunk_ids = [m.chunk_id for m in merged]
    chunk_stmt = text(
        "SELECT id, search_text, disease_name, crop_name FROM rag_diseases WHERE id = ANY(:ids)"
    )
    # Convert string IDs back to UUIDs
    import uuid
    uuid_ids = [uuid.UUID(cid) for cid in chunk_ids]
    
    chunk_rows = await db.execute(chunk_stmt, {"ids": uuid_ids})
    chunk_map = {str(row[0]): row for row in chunk_rows}

    final_candidates = []
    for m in merged:
        chunk_data = chunk_map.get(m.chunk_id)
        if chunk_data:
            final_candidates.append(
                RetrievalCandidate(
                    chunk_id=m.chunk_id,
                    title=f"{chunk_data[3] or 'Общее'}: {chunk_data[2] or 'Раздел'}",
                    text_preview=chunk_data[1],
                    rrf_score=round(m.rrf_score, 6),
                    lexical_rank=m.ranks.get("lexical"),
                    vector_rank=m.ranks.get("vector"),
                    reranker_score=None,
                    included_in_prompt=True,
                )
            )

    return RetrieveResponse(
        query=query,
        normalized_ru=normalized.normalized_ru,
        language_detected=normalized.language_detected,
        entities=normalized.entities,
        candidates=final_candidates,
    )


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
