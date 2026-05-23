import base64
from io import BytesIO

import pytest
from PIL import Image

from app.schemas import QueryRequest
from app.pipelines.pure_llm import PureLlmPipeline
from app.vision.input import prepare_image_input


def make_image_b64(size: tuple[int, int] = (800, 800)) -> str:
    image = Image.new("RGB", size, (80, 140, 90))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_prepare_image_input_rejects_bad_base64() -> None:
    result = prepare_image_input("not-base64", "image/jpeg")

    assert result.provided is True
    assert result.valid is False
    assert result.send_to_llm is False
    assert result.trace()["rejection_reason"] == "invalid_base64_image"


def test_prepare_image_input_records_quality_trace() -> None:
    result = prepare_image_input(make_image_b64(), "image/jpeg")

    trace = result.trace()
    assert trace["provided"] is True
    assert trace["valid"] is True
    assert trace["mime_type"] == "image/jpeg"
    assert trace["size_bytes"] > 0
    assert trace["quality"]["width"] == 800
    assert "sent_to_gemini" in trace


@pytest.mark.asyncio
async def test_pipeline_rejects_invalid_image_before_llm() -> None:
    response = await PureLlmPipeline().run(
        QueryRequest(
            question="Что с растением?",
            pipeline="A_PURE_LLM",
            image="not-base64",
            image_mime_type="image/jpeg",
        )
    )

    assert response.trace["pipeline"]["mode"] == "image_rejected"
    assert response.trace["image"]["sent_to_gemini"] is False
    assert response.trace["image"]["rejection_reason"] == "invalid_base64_image"
