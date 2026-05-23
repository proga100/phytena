from app.pipelines.base import Pipeline
from app.pipelines.hybrid_rag import HybridRagPipeline
from app.pipelines.hybrid_rag_rerank import HybridRagRerankPipeline
from app.pipelines.pure_llm import PureLlmPipeline
from app.schemas import CompareRequest, CompareResponse, PipelineId, PipelineResponse, QueryRequest


class PipelineRunner:
    def __init__(self, pipelines: list[Pipeline] | None = None) -> None:
        pipeline_list = pipelines or [PureLlmPipeline(), HybridRagPipeline(), HybridRagRerankPipeline()]
        self._pipelines = {pipeline.pipeline_id: pipeline for pipeline in pipeline_list}

    async def run_one(self, request: QueryRequest) -> PipelineResponse:
        return await self._pipelines[request.pipeline].run(request)

    async def compare(self, request: CompareRequest) -> CompareResponse:
        results = []
        for pipeline_id in request.pipelines:
            query_request = QueryRequest(
                question=request.question,
                context=request.context,
                pipeline=pipeline_id,
                return_trace=request.return_trace,
            )
            results.append(await self.run_one(query_request))
        return CompareResponse(question=request.question, context=request.context, results=results)

    def available_pipeline_ids(self) -> list[PipelineId]:
        return list(self._pipelines.keys())
