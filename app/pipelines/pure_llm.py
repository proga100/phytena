from app.pipelines.base import Pipeline
from app.pipelines.stub_utils import build_stub_answer
from app.schemas import PipelineId, PipelineResponse, QueryRequest


class PureLlmPipeline(Pipeline):
    pipeline_id = PipelineId.PURE_LLM

    async def run(self, request: QueryRequest) -> PipelineResponse:
        return build_stub_answer(
            pipeline=self.pipeline_id,
            request=request,
            confidence="low",
            citations=False,
        )
