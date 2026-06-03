import re
from dataclasses import dataclass, field

from app.schemas import AssistantAnswer


CHEMICAL_TERMS = (
    # Russian
    "пестицид",
    "фунгицид",
    "инсектицид",
    "гербицид",
    "карбамид",
    "селитра",
    "доза",
    "л/га",
    "кг/га",
    "ml/ha",
    "l/ha",
    "kg/ha",
    # Uzbek (Latin)
    "pestitsid",
    "fungitsid",
    "insektitsid",
    "gerbitsid",
    "preparat",
    "doza",
    "dozasida",
)

DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:л/га|кг/га|г/л|ml/ha|l/ha|kg/ha|l/ga|kg/ga|g/l|doza)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SafetyValidation:
    json_valid: bool = True
    has_citations: bool = False
    unsafe_pesticide_advice: bool = False
    unsupported_chemical_claim: bool = False
    needs_agronomist: bool = False
    flags: list[str] = field(default_factory=list)
    blocked_claims: list[str] = field(default_factory=list)


def validate_answer_safety(answer: AssistantAnswer) -> SafetyValidation:
    text = " ".join(
        [
            answer.answer,
            " ".join(answer.actions),
            " ".join(answer.warnings),
        ]
    ).lower()
    has_chemical_terms = any(term in text for term in CHEMICAL_TERMS)
    has_dosage = bool(DOSAGE_PATTERN.search(text))
    has_citations = bool(answer.citations)

    flags: list[str] = []
    blocked_claims: list[str] = []
    unsupported_chemical_claim = has_chemical_terms and not has_citations
    unsafe_pesticide_advice = has_dosage and not has_citations

    if unsupported_chemical_claim:
        flags.append("UNSUPPORTED_CHEMICAL_CLAIM")
        blocked_claims.append("Chemical advice requires citation-backed context.")
    if unsafe_pesticide_advice:
        flags.append("UNSUPPORTED_DOSAGE")
        blocked_claims.append("Dosage advice requires verified local source and label constraints.")
    if answer.escalate_to_agronomist:
        flags.append("ESCALATE_TO_AGRONOMIST")

    return SafetyValidation(
        has_citations=has_citations,
        unsafe_pesticide_advice=unsafe_pesticide_advice,
        unsupported_chemical_claim=unsupported_chemical_claim,
        needs_agronomist=answer.escalate_to_agronomist or unsafe_pesticide_advice,
        flags=flags,
        blocked_claims=blocked_claims,
    )
