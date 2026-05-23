from abc import ABC, abstractmethod

from app.schemas import PipelineId, PipelineResponse, QueryRequest


class Pipeline(ABC):
    pipeline_id: PipelineId

    @abstractmethod
    async def run(self, request: QueryRequest) -> PipelineResponse:
        """Run one architecture variant for a query."""
