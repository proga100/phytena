from time import perf_counter
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.gemini import GeminiClient, GeminiClientError
from app.config import get_settings
from app.db import get_session
from app.llm.prompts import PIPELINE_B_PROMPT_VERSION, build_pipeline_b_prompt
from app.logging import logger
from app.pipelines.base import Pipeline
from app.pipelines.stub_utils import build_stub_answer
from app.rag.retrieve import retrieve
from app.schemas import PipelineId, PipelineMetrics, PipelineResponse, QueryRequest


class HybridRagPipeline(Pipeline):
    pipeline_id = PipelineId.HYBRID_RAG

    async def run(self, request: QueryRequest) -> PipelineResponse:
        logger.info(f"Running HybridRagPipeline for question: {request.question[:50]}...")
        settings = get_settings()
        if settings.llm_provider.lower() not in ("gemini", "google") or not settings.gemini_api_key:
            logger.warning("LLM provider not configured or API key missing.")
            return build_stub_answer(
                pipeline=self.pipeline_id,
                request=request,
                confidence="low",
                citations=False,
            )

        started = perf_counter()
        # 1. Retrieval
        from app.db import AsyncSessionLocal
        
        logger.info("Performing retrieval...")
        async with AsyncSessionLocal() as db:
            retrieval_response = await retrieve(request.question, db, language=request.context.language)
        
        context_chunks = [c.text_preview for c in retrieval_response.candidates]
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
            completion = await client.generate_structured_answer(prompt, image_b64=request.image)
            logger.info(f"RAG generation successful. Tokens: {completion.input_tokens}/{completion.output_tokens}")
        except GeminiClientError as exc:
            logger.error(f"Gemini API Error in HybridRagPipeline: {str(exc)}")
            raise exc

        elapsed_ms = int((perf_counter() - started) * 1000)
        
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
                "retrieval": retrieval_response.model_dump(),
                "prompt": prompt,
                "model": settings.llm_model,
                "version": PIPELINE_B_PROMPT_VERSION,
            } if request.return_trace else {},
        )
