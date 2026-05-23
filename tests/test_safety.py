from app.safety.validator import validate_answer_safety
from app.schemas import AssistantAnswer, Citation


def test_safety_validator_flags_uncited_dosage() -> None:
    answer = AssistantAnswer(
        answer="Опрыскайте фунгицидом 1 л/га.",
        actions=[],
        warnings=[],
    )

    result = validate_answer_safety(answer)

    assert result.unsafe_pesticide_advice is True
    assert "UNSUPPORTED_DOSAGE" in result.flags
    assert result.needs_agronomist is True


def test_safety_validator_allows_cited_chemical_mention() -> None:
    answer = AssistantAnswer(
        answer="Фунгицидные меры требуют проверки местной регистрации.",
        citations=[Citation(chunk_id="chunk-1", title="Verified source")],
    )

    result = validate_answer_safety(answer)

    assert result.has_citations is True
    assert result.unsupported_chemical_claim is False
    assert result.unsafe_pesticide_advice is False
