from time import perf_counter
from uuid import uuid4


from app.clients.gemini import GeminiClient, GeminiClientError
from app.config import get_settings
from app.llm.prompts import PIPELINE_B_PROMPT_VERSION, build_pipeline_b_prompt
from app.logging import logger
from app.pipelines.base import Pipeline
from app.pipelines.image_guard import build_image_rejection_response
from app.pipelines.stub_utils import build_stub_answer
from app.rag.retrieve import retrieve
from app.safety.validator import validate_answer_safety
from app.schemas import PipelineId, PipelineMetrics, PipelineResponse, QueryRequest
from app.vision.input import prepare_image_input


def _build_context(candidates: list, treatments_by_disease: dict[str, list[dict]]) -> list[str]:
    """Compose one KB doc per retrieved disease: its symptom text plus the available
    treatment options, so the LLM can both diagnose and recommend treatments.
    """
    docs: list[str] = []
    for c in candidates:
        parts = [c.text_preview]
        treatments = treatments_by_disease.get(c.chunk_id, [])
        if treatments:
            lines = ["Davolash usullari (treatments):"]
            for t in treatments:
                dose = ""
                if t.get("dose_min") is not None or t.get("dose_max") is not None:
                    dose = f" (doza: {t.get('dose_min')}-{t.get('dose_max')})"
                desc = f" — {t['description']}" if t.get("description") else ""
                lines.append(f"- {t['drug']}{dose}{desc}")
            parts.append("\n".join(lines))
        docs.append("\n".join(parts))
    return docs


class HybridRagPipeline(Pipeline):
    pipeline_id = PipelineId.HYBRID_RAG

    async def run(self, request: QueryRequest) -> PipelineResponse:
        logger.info(f"Running HybridRagPipeline for question: {request.question[:50]}...")
        settings = get_settings()
        started = perf_counter()
        image = prepare_image_input(request.image, request.image_mime_type)
        if image.provided and not image.send_to_llm:
            return build_image_rejection_response(
                pipeline=self.pipeline_id,
                image=image,
                started=started,
                return_trace=request.return_trace,
            )

        if settings.llm_provider.lower() not in ("gemini", "google") or not settings.gemini_api_key:
            logger.warning("LLM provider not configured or API key missing.")
            response = build_stub_answer(
                pipeline=self.pipeline_id,
                request=request,
                confidence="low",
                citations=True,
            )
            response.trace["image"] = image.trace()
            return response

        # 1. Retrieval against the Growz Uzbek RAG DB (rag_diseases), then fetch each
        # retrieved disease's treatments (symptom -> disease -> treatments).
        from app.db import GrowzRagSessionLocal
        from app.rag.retrieve import fetch_treatments_for_diseases

        logger.info("Performing retrieval...")
        async with GrowzRagSessionLocal() as db:
            retrieval_response = await retrieve(request.question, db, language=request.context.language)
            disease_ids = [c.chunk_id for c in retrieval_response.candidates]
            treatments_by_disease = await fetch_treatments_for_diseases(db, disease_ids)

        context_chunks = _build_context(retrieval_response.candidates, treatments_by_disease)
        logger.info(f"Retrieved {len(context_chunks)} chunks.")
        
        if not context_chunks:
            logger.info("No relevant chunks found in KB.")
            return PipelineResponse(
                run_id=uuid4(),
                pipeline=self.pipeline_id,
                answer={
                    "answer": "Я не нашел точной информации в базе знаний по вашему вопросу.",
                    "confidence": "low",
                    "diagnoses": [],
                    "actions": ["Попробуйте другой вопрос."],
                    "warnings": [],
                    "citations": [],
                },
                metrics=PipelineMetrics(latency_ms=int((perf_counter() - started) * 1000)),
                trace={"retrieval_status": "no_results"}
            )

        # 2. Grounded Generation
        logger.info("Generating grounded response...")
        prompt = build_pipeline_b_prompt(request, context_chunks)
        client = GeminiClient(api_key=settings.gemini_api_key, model=settings.llm_model)
        
        try:
            completion = await client.generate_structured_answer(
                prompt,
                image_b64=request.image if image.send_to_llm else None,
                image_mime_type=image.mime_type or "image/jpeg",
            )
            logger.info(f"RAG generation successful. Tokens: {completion.input_tokens}/{completion.output_tokens}")
        except GeminiClientError as exc:
            logger.error(f"Gemini API Error in HybridRagPipeline: {str(exc)}")
            raise exc

        elapsed_ms = int((perf_counter() - started) * 1000)

        # Safety gate: flag chemical/dosage advice and escalate when unsupported.
        safety = validate_answer_safety(completion.answer)

        return PipelineResponse(
            run_id=uuid4(),
            pipeline=self.pipeline_id,
            answer=completion.answer,
            metrics=PipelineMetrics(
                latency_ms=elapsed_ms,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            ),
            trace={
                "image": image.trace(),
                "retrieval": retrieval_response.model_dump(),
                "prompt": prompt,
                "model": settings.llm_model,
                "version": PIPELINE_B_PROMPT_VERSION,
                "safety": {
                    "has_citations": safety.has_citations,
                    "unsafe_pesticide_advice": safety.unsafe_pesticide_advice,
                    "unsupported_chemical_claim": safety.unsupported_chemical_claim,
                    "needs_agronomist": safety.needs_agronomist,
                    "flags": safety.flags,
                },
            } if request.return_trace else {},
        )
