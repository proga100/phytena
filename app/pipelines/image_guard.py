from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from app.schemas import AssistantAnswer, Diagnosis, PipelineId, PipelineMetrics, PipelineResponse
from app.vision.input import PreparedImage


def build_image_rejection_response(
    *,
    pipeline: PipelineId,
    image: PreparedImage,
    started: float,
    return_trace: bool,
) -> PipelineResponse:
    quality = image.quality
    issues = quality.issues if quality else []
    action = (
        quality.recommended_user_action
        if quality and quality.recommended_user_action
        else "Загрузите JPEG, PNG или WebP до 10 МБ, снятый крупно и при хорошем освещении."
    )
    return PipelineResponse(
        run_id=uuid4(),
        pipeline=pipeline,
        answer=AssistantAnswer(
            diagnoses=[
                Diagnosis(
                    name="Недостаточно качественное фото",
                    category="image_quality",
                    confidence=0.0,
                    evidence=issues or [image.error or "image_rejected"],
                )
            ],
            confidence="low",
            answer=(
                "Я не отправил фото в модель, потому что изображение не прошло предварительную "
                "проверку качества или формата. Для диагностики нужно более четкое фото пораженной "
                "части растения."
            ),
            actions=[action],
            warnings=["Не применяйте препараты по этому результату. Сначала загрузите подходящее фото."],
            citations=[],
            needs_clarification=True,
            clarification_question="Пришлите крупное фото пораженного листа или стебля при дневном свете.",
            escalate_to_agronomist=False,
        ),
        metrics=PipelineMetrics(latency_ms=int((perf_counter() - started) * 1000)),
        trace={"image": image.trace(), "pipeline": {"id": pipeline.value, "mode": "image_rejected"}}
        if return_trace
        else {},
    )
