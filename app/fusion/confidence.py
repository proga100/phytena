from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInputs:
    image_confidence: float
    quality_status: str
    rag_alignment: str
    context_match: str


@dataclass(frozen=True)
class ConfidenceResult:
    image_confidence: float
    quality_multiplier: float
    rag_alignment_multiplier: float
    context_multiplier: float
    final_confidence: float


QUALITY_MULTIPLIERS = {
    "pass": 1.0,
    "warn": 0.75,
    "fail": 0.2,
}

RAG_ALIGNMENT_MULTIPLIERS = {
    "strong": 1.1,
    "partial": 1.0,
    "none": 0.7,
    "conflict": 0.6,
}

CONTEXT_MULTIPLIERS = {
    "match": 1.05,
    "missing": 0.9,
    "conflict": 0.65,
}


def fuse_confidence(inputs: ConfidenceInputs) -> ConfidenceResult:
    quality_multiplier = QUALITY_MULTIPLIERS.get(inputs.quality_status, 0.75)
    rag_multiplier = RAG_ALIGNMENT_MULTIPLIERS.get(inputs.rag_alignment, 0.7)
    context_multiplier = CONTEXT_MULTIPLIERS.get(inputs.context_match, 0.9)

    final = (
        inputs.image_confidence
        * quality_multiplier
        * rag_multiplier
        * context_multiplier
    )
    final = round(max(0.0, min(0.95, final)), 3)
    return ConfidenceResult(
        image_confidence=inputs.image_confidence,
        quality_multiplier=quality_multiplier,
        rag_alignment_multiplier=rag_multiplier,
        context_multiplier=context_multiplier,
        final_confidence=final,
    )
