from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass

from app.vision.quality import ImageQualityResult, assess_image_quality

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class PreparedImage:
    provided: bool
    valid: bool
    send_to_llm: bool
    mime_type: str | None = None
    size_bytes: int = 0
    quality: ImageQualityResult | None = None
    error: str | None = None

    def trace(self) -> dict:
        return {
            "provided": self.provided,
            "valid": self.valid,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "quality": asdict(self.quality) if self.quality else None,
            "sent_to_gemini": self.send_to_llm,
            "rejection_reason": self.error,
        }


def prepare_image_input(image_b64: str | None, mime_type: str | None) -> PreparedImage:
    if not image_b64:
        return PreparedImage(provided=False, valid=True, send_to_llm=False)

    resolved_mime_type = mime_type or DEFAULT_IMAGE_MIME_TYPE
    if resolved_mime_type not in ALLOWED_IMAGE_TYPES:
        return PreparedImage(
            provided=True,
            valid=False,
            send_to_llm=False,
            mime_type=resolved_mime_type,
            error="unsupported_image_mime_type",
        )

    try:
        content = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        return PreparedImage(
            provided=True,
            valid=False,
            send_to_llm=False,
            mime_type=resolved_mime_type,
            error="invalid_base64_image",
        )

    size_bytes = len(content)
    if size_bytes > MAX_IMAGE_BYTES:
        return PreparedImage(
            provided=True,
            valid=False,
            send_to_llm=False,
            mime_type=resolved_mime_type,
            size_bytes=size_bytes,
            error="image_too_large",
        )

    try:
        quality = assess_image_quality(content)
    except Exception:
        return PreparedImage(
            provided=True,
            valid=False,
            send_to_llm=False,
            mime_type=resolved_mime_type,
            size_bytes=size_bytes,
            error="unreadable_image",
        )

    return PreparedImage(
        provided=True,
        valid=True,
        send_to_llm=quality.status != "fail",
        mime_type=resolved_mime_type,
        size_bytes=size_bytes,
        quality=quality,
        error=None if quality.status != "fail" else "image_quality_failed",
    )
