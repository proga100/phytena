from dataclasses import dataclass, field
from uuid import UUID, uuid4

from app.pipelines import PipelineRunner
from app.schemas import CompareRequest, PipelineId, QueryContext


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    context: QueryContext = field(default_factory=QueryContext)
    expected_terms: list[str] = field(default_factory=list)
    safety_critical: bool = False


@dataclass(frozen=True)
class EvalPipelineResult:
    pipeline: PipelineId
    score: float
    latency_ms: int
    has_citations: bool
    safety_flags: list[str]


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    question: str
    pipeline_results: list[EvalPipelineResult]


@dataclass(frozen=True)
class EvalRunResult:
    eval_run_id: UUID
    cases: list[EvalCaseResult]
    summary: dict[str, dict[str, float]]


DEFAULT_GOLDEN_CASES = [
    GoldenCase(
        id="wheat-yellowing-001",
        question="Пшеница желтеет, нижние листья сохнут, что делать?",
        context=QueryContext(crop="пшеница", region="Узбекистан", language="ru"),
        expected_terms=["пшеница", "лист"],
    ),
    GoldenCase(
        id="tomato-uz-latin-001",
        question="Pomidor barglari sargayib qolyapti, nima qilish kerak?",
        context=QueryContext(crop="томат", region="Узбекистан", language="auto"),
        expected_terms=["томат", "листь"],
    ),
]


async def run_stub_eval(
    runner: PipelineRunner,
    cases: list[GoldenCase] | None = None,
) -> EvalRunResult:
    selected_cases = cases or DEFAULT_GOLDEN_CASES
    case_results: list[EvalCaseResult] = []

    for golden_case in selected_cases:
        comparison = await runner.compare(
            CompareRequest(question=golden_case.question, context=golden_case.context)
        )
        pipeline_results = [
            EvalPipelineResult(
                pipeline=result.pipeline,
                score=_score_stub_answer(result.answer.answer, golden_case.expected_terms),
                latency_ms=result.metrics.latency_ms,
                has_citations=bool(result.answer.citations),
                safety_flags=result.trace.get("safety", {}).get("flags", []),
            )
            for result in comparison.results
        ]
        case_results.append(
            EvalCaseResult(
                case_id=golden_case.id,
                question=golden_case.question,
                pipeline_results=pipeline_results,
            )
        )

    return EvalRunResult(
        eval_run_id=uuid4(),
        cases=case_results,
        summary=_summarize(case_results),
    )


def _score_stub_answer(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 0.0
    lowered = answer.lower()
    matches = sum(1 for term in expected_terms if term.lower() in lowered)
    return round(matches / len(expected_terms), 3)


def _summarize(case_results: list[EvalCaseResult]) -> dict[str, dict[str, float]]:
    accumulator: dict[str, list[EvalPipelineResult]] = {}
    for case_result in case_results:
        for pipeline_result in case_result.pipeline_results:
            accumulator.setdefault(pipeline_result.pipeline.value, []).append(pipeline_result)

    summary = {}
    for pipeline, results in accumulator.items():
        summary[pipeline] = {
            "avg_stub_score": round(sum(result.score for result in results) / len(results), 3),
            "avg_latency_ms": round(sum(result.latency_ms for result in results) / len(results), 3),
            "citation_rate": round(
                sum(1 for result in results if result.has_citations) / len(results), 3
            ),
        }
    return summary
