from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.vision.quality import assess_image_quality


def make_image(size: tuple[int, int] = (800, 600), color: tuple[int, int, int] = (80, 140, 90)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_assess_image_quality_returns_metrics() -> None:
    result = assess_image_quality(make_image())

    assert result.width == 800
    assert result.height == 600
    assert 0 <= result.exposure_score <= 1
    assert result.status in {"pass", "warn", "fail"}


def test_quality_check_endpoint_accepts_jpeg() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/image/quality-check",
        files={"image": ("leaf.jpg", make_image(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["width"] == 800
    assert payload["height"] == 600
    assert "blur_score" in payload


def test_quality_check_endpoint_rejects_non_image() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/image/quality-check",
        files={"image": ("note.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
