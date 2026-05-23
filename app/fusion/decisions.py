from dataclasses import dataclass, field


@dataclass(frozen=True)
class Decision:
    type: str
    reason_codes: list[str] = field(default_factory=list)
    allowed_recommendation_level: str = "general"
    requires_disclaimer: bool = True


def decide_response(final_confidence: float, *, quality_status: str = "pass") -> Decision:
    if quality_status == "fail":
        return Decision(
            type="bad_image_request_new_photo",
            reason_codes=["LOW_IMAGE_QUALITY"],
            allowed_recommendation_level="general",
        )
    if final_confidence >= 0.75:
        return Decision(
            type="high_confidence_answer",
            reason_codes=["HIGH_CONFIDENCE"],
            allowed_recommendation_level="diagnostic_steps",
        )
    if final_confidence >= 0.50:
        return Decision(
            type="medium_confidence_clarify",
            reason_codes=["MEDIUM_CONFIDENCE"],
            allowed_recommendation_level="cultural_controls",
        )
    if final_confidence >= 0.30:
        return Decision(
            type="low_confidence_escalate",
            reason_codes=["LOW_CONFIDENCE"],
            allowed_recommendation_level="diagnostic_steps",
        )
    return Decision(
        type="refuse_diagnosis",
        reason_codes=["VERY_LOW_CONFIDENCE"],
        allowed_recommendation_level="general",
    )
