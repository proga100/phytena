from time import perf_counter
from uuid import uuid4

from app.clients.gemini import GeminiClient, GeminiClientError
from app.config import get_settings
from app.llm.prompts import PIPELINE_A_PROMPT_VERSION, build_pipeline_a_prompt
from app.pipelines.base import Pipeline
from app.pipelines.stub_utils import build_stub_answer
from app.safety.validator import validate_answer_safety
from app.schemas import PipelineId, PipelineMetrics, PipelineResponse, QueryRequest


class PureLlmPipeline(Pipeline):
    pipeline_id = PipelineId.PURE_LLM

    async def run(self, request: QueryRequest) -> PipelineResponse:
        settings = get_settings()
        if settings.llm_provider.lower() != "gemini" or not settings.gemini_api_key:
            return build_stub_answer(
                pipeline=self.pipeline_id,
                request=request,
                confidence="low",
                citations=False,
            )

        started = perf_counter()
        prompt = build_pipeline_a_prompt(request)
        client = GeminiClient(api_key=settings.gemini_api_key, model=settings.llm_model)
        try:
            completion = await client.generate_structured_answer(prompt)
        except GeminiClientError as exc:
            fallback = build_stub_answer(
                pipeline=self.pipeline_id,
                request=request,
                confidence="low",
                citations=False,
            )
            fallback.trace["llm"] = {
                "provider": "gemini",
                "model": settings.llm_model,
                "prompt_version": PIPELINE_A_PROMPT_VERSION,
                "status": "error_fallback_to_stub",
                "error": str(exc),
            }
            return fallback

        safety = validate_answer_safety(completion.answer)
        latency_ms = int((perf_counter() - started) * 1000)
        trace = {
            "pipeline": {"id": self.pipeline_id.value, "mode": "real_llm"},
            "llm": {
                "provider": "gemini",
                "model": settings.llm_model,
                "prompt_version": PIPELINE_A_PROMPT_VERSION,
                "status": "success",
                "raw_output": completion.raw_text,
            },
            "safety": {
                "json_valid": safety.json_valid,
                "has_citations": safety.has_citations,
                "unsafe_pesticide_advice": safety.unsafe_pesticide_advice,
                "unsupported_chemical_claim": safety.unsupported_chemical_claim,
                "needs_agronomist": safety.needs_agronomist,
                "flags": safety.flags,
                "blocked_claims": safety.blocked_claims,
            },
        }
        return PipelineResponse(
            run_id=uuid4(),
            pipeline=self.pipeline_id,
            answer=completion.answer,
            metrics=PipelineMetrics(
                latency_ms=latency_ms,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            ),
            trace=trace if request.return_trace else {},
        )

    async def run_stub(self, request: QueryRequest) -> PipelineResponse:
        return build_stub_answer(
            pipeline=self.pipeline_id,
            request=request,
            confidence="low",
            citations=False,
        )
