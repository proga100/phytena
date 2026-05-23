from fastapi import APIRouter, Depends

from app.dependencies import get_pipeline_runner
from app.pipelines import PipelineRunner
from app.schemas import CompareRequest, CompareResponse, PipelineResponse, QueryRequest

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query", response_model=PipelineResponse)
async def query(
    request: QueryRequest,
    runner: PipelineRunner = Depends(get_pipeline_runner),
) -> PipelineResponse:
    return await runner.run_one(request)


@router.post("/query/compare", response_model=CompareResponse)
async def compare(
    request: CompareRequest,
    runner: PipelineRunner = Depends(get_pipeline_runner),
) -> CompareResponse:
    return await runner.compare(request)
