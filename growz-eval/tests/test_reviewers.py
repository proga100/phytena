from io import BytesIO

from PIL import Image

from growz_eval.verification import reviewers


def test_small_image_is_sent_unchanged(tmp_path):
    image_path = tmp_path / "leaf.png"
    Image.new("RGB", (32, 24), "green").save(image_path)
    original = image_path.read_bytes()

    image_bytes, mime_type = reviewers._gemini_image_part(image_path)

    assert image_bytes == original
    assert mime_type == "image/png"


def test_large_image_is_resized_and_converted_to_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(reviewers, "MAX_GEMINI_IMAGE_SIDE", 50)
    image_path = tmp_path / "leaf.png"
    Image.new("RGBA", (120, 80), (0, 128, 0, 160)).save(image_path)

    image_bytes, mime_type = reviewers._gemini_image_part(image_path)

    with Image.open(BytesIO(image_bytes)) as normalized:
        assert mime_type == "image/jpeg"
        assert normalized.format == "JPEG"
        assert normalized.mode == "RGB"
        assert max(normalized.size) == 50
