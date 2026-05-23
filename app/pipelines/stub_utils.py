from time import perf_counter
from typing import Any
from uuid import uuid4

from app.fusion.confidence import ConfidenceInputs, fuse_confidence
from app.fusion.decisions import decide_response
from app.safety.validator import validate_answer_safety
from app.schemas import AssistantAnswer, Diagnosis, PipelineId, PipelineMetrics, PipelineResponse, QueryRequest


def detect_language(question: str, fallback: str = "auto") -> str:
    if fallback != "auto":
        return fallback
    latin_uz_markers = (" nima ", " barg", " sarg", " qilyap", " qolyap", " kerak")
    lowered = f" {question.lower()} "
    if any(marker in lowered for marker in latin_uz_markers):
        return "uz_latn"
    if any("а" <= char <= "я" or "А" <= char <= "Я" for char in question):
        return "ru"
    return "unknown"


def normalize_to_russian(question: str, language: str) -> str:
    if language == "uz_latn" and "pomidor" in question.lower():
        return "У помидора желтеют листья. Возможные причины и безопасные действия?"
    return question


def build_stub_answer(
    *,
    pipeline: PipelineId,
    request: QueryRequest,
    confidence: str,
    citations: bool,
    reranked: bool = False,
) -> PipelineResponse:
    started = perf_counter()
    language = detect_language(request.question, request.context.language)
    normalized = normalize_to_russian(request.question, language)
    crop = request.context.crop or "не указано"
    confidence_result = fuse_confidence(
        ConfidenceInputs(
            image_confidence=0.45 if confidence == "medium" else 0.3,
            quality_status="warn",
            rag_alignment="partial" if citations else "none",
            context_match="match" if request.context.crop else "missing",
        )
    )
    decision = decide_response(confidence_result.final_confidence, quality_status="warn")

    citation_list = []
    if citations:
        citation_list = [
            {
                "chunk_id": "stub-wheat-nitrogen-001",
                "title": "Russian-first agronomy KB stub",
                "section": "Symptom differentiation",
                "url": None,
            }
        ]

    answer = AssistantAnswer(
        diagnoses=[
            Diagnosis(
                name="Недостаточно данных для точного диагноза",
                category="unknown",
                confidence=0.45 if confidence == "medium" else 0.3,
                evidence=["Prototype stub response; real vision/RAG is not wired yet."],
            )
        ],
        confidence=confidence,
        answer=(
            "Это демонстрационный ответ прототипа. Система пока показывает архитектурный поток: "
            "вопрос нормализуется, затем выбранный pipeline готовит осторожный ответ. "
            "Для реальной рекомендации нужны подключенные Vision API, RAG-база и валидация источников."
        ),
        actions=[
            "Проверить качество фото и культуру.",
            "Сравнить ответы Pipeline A/B/C в админке.",
            "Не выдавать химические рекомендации без подтвержденного источника.",
        ],
        warnings=["Не применять пестициды по этому демонстрационному ответу."],
        citations=citation_list,
        needs_clarification=True,
        clarification_question="Пришлите крупное фото пораженного листа при дневном свете.",
        escalate_to_agronomist=False,
    )
    safety = validate_answer_safety(answer)

    trace: dict[str, Any] = {
        "query": {
            "original": request.question,
            "language_detected": language,
            "normalized_ru": normalized,
            "entities": {"crop": crop},
        },
        "pipeline": {"id": pipeline.value, "mode": "stub"},
        "retrieval": {
            "enabled": pipeline != PipelineId.PURE_LLM,
            "reranker_enabled": reranked,
            "hits": citation_list,
        },
        "fusion": {
            "image_confidence": confidence_result.image_confidence,
            "quality_multiplier": confidence_result.quality_multiplier,
            "rag_alignment_multiplier": confidence_result.rag_alignment_multiplier,
            "context_multiplier": confidence_result.context_multiplier,
            "final_confidence": confidence_result.final_confidence,
            "decision": decision.type,
            "allowed_recommendation_level": decision.allowed_recommendation_level,
            "reason_codes": decision.reason_codes,
        },
        "safety": {
            "json_valid": safety.json_valid,
            "has_citations": safety.has_citations,
            "unsafe_pesticide_advice": safety.unsafe_pesticide_advice,
            "unsupported_chemical_claim": safety.unsupported_chemical_claim,
            "needs_agronomist": safety.needs_agronomist,
            "flags": safety.flags,
            "blocked_claims": safety.blocked_claims,
            "requires_caveat": True,
            "policy": "no unsupported pesticide dosage",
        },
    }

    elapsed_ms = int((perf_counter() - started) * 1000)
    return PipelineResponse(
        run_id=uuid4(),
        pipeline=pipeline,
        answer=answer,
        metrics=PipelineMetrics(latency_ms=elapsed_ms),
        trace=trace if request.return_trace else {},
    )
