from app.pipelines.base import Pipeline
from app.pipelines.stub_utils import build_stub_answer
from app.schemas import PipelineId, PipelineResponse, QueryRequest


class HybridRagPipeline(Pipeline):
    pipeline_id = PipelineId.HYBRID_RAG

    async def run(self, request: QueryRequest) -> PipelineResponse:
        return build_stub_answer(
            pipeline=self.pipeline_id,
            request=request,
            confidence="medium",
            citations=True,
        )
