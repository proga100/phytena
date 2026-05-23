from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PipelineId(StrEnum):
    PURE_LLM = "A_PURE_LLM"
    HYBRID_RAG = "B_HYBRID_RAG"
    HYBRID_RAG_RERANK = "C_HYBRID_RAG_RERANK"


class Diagnosis(BaseModel):
    name: str
    category: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: str | None = None
    title: str | None = None
    section: str | None = None
    url: str | None = None


class AssistantAnswer(BaseModel):
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    confidence: str = "low"
    answer: str
    actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    escalate_to_agronomist: bool = False


class QueryContext(BaseModel):
    crop: str | None = None
    region: str | None = None
    growth_stage: str | None = None
    language: str = "auto"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    pipeline: PipelineId = PipelineId.HYBRID_RAG
    context: QueryContext = Field(default_factory=QueryContext)
    return_trace: bool = True


class PipelineMetrics(BaseModel):
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class PipelineResponse(BaseModel):
    run_id: UUID
    pipeline: PipelineId
    answer: AssistantAnswer
    metrics: PipelineMetrics
    trace: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    question: str = Field(min_length=1)
    context: QueryContext = Field(default_factory=QueryContext)
    pipelines: list[PipelineId] = Field(
        default_factory=lambda: [
            PipelineId.PURE_LLM,
            PipelineId.HYBRID_RAG,
            PipelineId.HYBRID_RAG_RERANK,
        ]
    )
    return_trace: bool = True


class CompareResponse(BaseModel):
    question: str
    context: QueryContext
    results: list[PipelineResponse]


class ImageQualityResponse(BaseModel):
    width: int
    height: int
    normalized_width: int
    normalized_height: int
    blur_score: float
    exposure_score: float
    quality_score: float
    status: str
    issues: list[str] = Field(default_factory=list)
    recommended_user_action: str | None = None
