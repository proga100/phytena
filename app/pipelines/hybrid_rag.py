from time import perf_counter
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.gemini import GeminiClient, GeminiClientError
from app.config import get_settings
from app.db import get_session
from app.llm.prompts import PIPELINE_B_PROMPT_VERSION, build_pipeline_b_prompt
from app.pipelines.base import Pipeline
from app.pipelines.stub_utils import build_stub_answer
from app.rag.retrieve import retrieve
from app.schemas import PipelineId, PipelineMetrics, PipelineResponse, QueryRequest


class HybridRagPipeline(Pipeline):
    pipeline_id = PipelineId.HYBRID_RAG

    async def run(self, request: QueryRequest) -> PipelineResponse:
        settings = get_settings()
        if settings.llm_provider.lower() not in ("gemini", "google") or not settings.gemini_api_key:
            return build_stub_answer(
                pipeline=self.pipeline_id,
                request=request,
                confidence="low",
                citations=False,
            )

        started = perf_counter()
        # 1. Retrieval
        # We need a DB session here. Since Pipeline.run is not designed with Depends, 
        # we'll use the async context manager for the session.
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.orm import sessionmaker
        
        engine = create_async_engine(settings.database_url)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as db:
            retrieval_response = await retrieve(request.question, db, language=request.context.language)
        
        await engine.dispose()

        context_chunks = [c.text_preview for c in retrieval_response.candidates]
        
        if not context_chunks:
            # Fallback to stub or refusal if no context found
            return build_stub_answer(
                pipeline=self.pipeline_id,
                request=request,
                confidence="low",
                citations=False,
            )

        # 2. Grounded Generation
        prompt = build_pipeline_b_prompt(request, context_chunks)
        client = GeminiClient(api_key=settings.gemini_api_key, model=settings.llm_model)
        
        try:
            # Note: For RAG we usually don't send the image to the generation step 
            # unless the model needs to see it to confirm symptoms in the docs.
            # We'll send it if present for better reasoning.
            completion = await client.generate_structured_answer(prompt, image_b64=request.image)
        except GeminiClientError as exc:
            return build_stub_answer(
                pipeline=self.pipeline_id,
                request=request,
                confidence="low",
                citations=True,
            )

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
