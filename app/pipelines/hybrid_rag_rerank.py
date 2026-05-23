from time import perf_counter
from uuid import uuid4
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker

from app.clients.gemini import GeminiClient, GeminiClientError
from app.config import get_settings
from app.llm.prompts import PIPELINE_B_PROMPT_VERSION, build_pipeline_b_prompt
from app.logging import logger
from app.pipelines.base import Pipeline
from app.pipelines.stub_utils import build_stub_answer
from app.rag.retrieve import retrieve
from app.schemas import PipelineId, PipelineMetrics, PipelineResponse, QueryRequest


class HybridRagRerankPipeline(Pipeline):
    pipeline_id = PipelineId.HYBRID_RAG_RERANK

    async def run(self, request: QueryRequest) -> PipelineResponse:
        logger.info(f"Running HybridRagRerankPipeline for question: {request.question[:50]}...")
        started = perf_counter()
        settings = get_settings()

        if settings.llm_provider.lower() not in ("gemini", "google") or not settings.gemini_api_key:
            logger.warning("LLM provider not configured or API key missing.")
            return build_stub_answer(
                pipeline=self.pipeline_id,
                request=request,
                confidence="low",
                citations=False,
            )

        # 1. Retrieval (Fetch more candidates for reranking)
        from app.db import AsyncSessionLocal
        
        logger.info("Performing deep retrieval (top 15)...")
        async with AsyncSessionLocal() as db:
            # Get top 15 for reranking
            retrieval_response = await retrieve(request.question, db, language=request.context.language, top_k=15)
        
        if not retrieval_response.candidates:
            logger.info("No relevant chunks found in KB.")
            return PipelineResponse(
                run_id=uuid4(),
                pipeline=self.pipeline_id,
                answer={
                    "answer": "Я не нашел точной информации в базе знаний по вашему вопросу. Пожалуйста, уточните симптомы или культуру.",
                    "confidence": "low",
                    "diagnoses": [],
                    "actions": ["Попробуйте переформулировать вопрос.", "Загрузите более четкое фото."],
                    "warnings": [],
                    "citations": [],
                    "needs_clarification": True,
                },
                metrics=PipelineMetrics(latency_ms=int((perf_counter() - started) * 1000)),
                trace={"retrieval_status": "no_results"}
            )

        # 2. Grounded Generation with In-Context Reranking
        logger.info(f"Generating grounded response with {len(retrieval_response.candidates)} candidates...")
        client = GeminiClient(api_key=settings.gemini_api_key, model=settings.llm_model)
        
        try:
            # HIGHER LEVEL STRATEGY for Option C:
            # We provide ALL 15 candidates to Gemini and ask it to be EXTREMELY 
            # critical and pick only the most relevant ones for the final answer.
            
            final_prompt = build_pipeline_b_prompt(request, [c.text_preview for c in retrieval_response.candidates])
            final_prompt = final_prompt.replace(
                "Answer the farmer's question", 
                "Carefully evaluate ALL provided documents (up to 15 chunks). Filter out irrelevant noise, identify the most relevant facts, and answer the question"
            )
            
            completion = await client.generate_structured_answer(final_prompt, image_b64=request.image)
            logger.info(f"Rerank generation successful. Tokens: {completion.input_tokens}/{completion.output_tokens}")
            
        except GeminiClientError as exc:
            logger.error(f"Gemini API Error in HybridRagRerankPipeline: {str(exc)}")
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
                "mode": "hybrid_rerank",
                "reranker": "gemini-2.5-in-context",
                "model": settings.llm_model,
            } if request.return_trace else {},
        )

    def _build_rerank_prompt(self, question: str, documents: list[str]) -> str:
        # This is a helper for future discrete reranking steps
        return f"Rank these docs for question: {question}\n\n" + "\n".join(documents)
