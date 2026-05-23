from app.fusion.confidence import ConfidenceInputs, fuse_confidence
from app.fusion.decisions import decide_response


def test_fuse_confidence_clamps_to_maximum() -> None:
    result = fuse_confidence(
        ConfidenceInputs(
            image_confidence=0.99,
            quality_status="pass",
            rag_alignment="strong",
            context_match="match",
        )
    )

    assert result.final_confidence == 0.95


def test_decide_response_medium_confidence() -> None:
    decision = decide_response(0.61, quality_status="warn")

    assert decision.type == "medium_confidence_clarify"
    assert decision.allowed_recommendation_level == "cultural_controls"


def test_decide_response_bad_image_overrides_confidence() -> None:
    decision = decide_response(0.9, quality_status="fail")

    assert decision.type == "bad_image_request_new_photo"
    assert decision.reason_codes == ["LOW_IMAGE_QUALITY"]
