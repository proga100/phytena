from fastapi import APIRouter, Depends

from app.dependencies import get_pipeline_runner
from app.eval.runner import DEFAULT_GOLDEN_CASES, run_stub_eval
from app.pipelines import PipelineRunner
from app.schemas import EvalRunRequest, EvalRunResponse

router = APIRouter(prefix="/v1/evals", tags=["eval"])


@router.post("/run", response_model=EvalRunResponse)
async def run_eval(
    request: EvalRunRequest,
    runner: PipelineRunner = Depends(get_pipeline_runner),
) -> EvalRunResponse:
    cases = DEFAULT_GOLDEN_CASES[: request.limit] if request.limit else DEFAULT_GOLDEN_CASES
    result = await run_stub_eval(runner, cases=cases)
    return EvalRunResponse(
        eval_run_id=result.eval_run_id,
        cases=[
            {
                "case_id": case.case_id,
                "question": case.question,
                "pipeline_results": [
                    {
                        "pipeline": pipeline_result.pipeline,
                        "score": pipeline_result.score,
                        "latency_ms": pipeline_result.latency_ms,
                        "has_citations": pipeline_result.has_citations,
                        "safety_flags": pipeline_result.safety_flags,
                    }
                    for pipeline_result in case.pipeline_results
                ],
            }
            for case in result.cases
        ],
        summary=result.summary,
    )
