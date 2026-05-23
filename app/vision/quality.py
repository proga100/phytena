from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageQualityResult:
    width: int
    height: int
    normalized_width: int
    normalized_height: int
    blur_score: float
    exposure_score: float
    quality_score: float
    status: str
    issues: list[str]
    recommended_user_action: str | None


def load_image(content: bytes) -> Image.Image:
    image = Image.open(BytesIO(content))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_long_edge(image: Image.Image, long_edge: int = 1280) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= long_edge:
        return image.copy()
    scale = long_edge / float(longest)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def estimate_blur_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


def estimate_exposure_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    too_dark = float(np.mean(gray < 0.08))
    too_bright = float(np.mean(gray > 0.92))
    penalty = max(too_dark, too_bright)
    return max(0.0, min(1.0, 1.0 - penalty))


def assess_image_quality(content: bytes, long_edge: int = 1280) -> ImageQualityResult:
    image = load_image(content)
    width, height = image.size
    normalized = resize_long_edge(image, long_edge=long_edge)
    normalized_width, normalized_height = normalized.size

    blur_score = estimate_blur_score(normalized)
    exposure_score = estimate_exposure_score(normalized)
    resolution_ok = min(width, height) >= 512

    issues: list[str] = []
    if not resolution_ok:
        issues.append("low_resolution")
    if blur_score < 60:
        issues.append("blurry")
    elif blur_score < 120:
        issues.append("borderline_blur")
    if exposure_score < 0.75:
        issues.append("poor_exposure")

    if "low_resolution" in issues or "blurry" in issues or exposure_score < 0.5:
        status = "fail"
    elif issues:
        status = "warn"
    else:
        status = "pass"

    blur_component = min(1.0, blur_score / 160.0)
    resolution_component = 1.0 if resolution_ok else 0.4
    quality_score = round(
        max(0.0, min(1.0, 0.45 * blur_component + 0.35 * exposure_score + 0.2 * resolution_component)),
        3,
    )

    action = None
    if status != "pass":
        action = "Retake the photo closer to the affected plant part, in daylight, without shadow or motion blur."

    return ImageQualityResult(
        width=width,
        height=height,
        normalized_width=normalized_width,
        normalized_height=normalized_height,
        blur_score=round(blur_score, 3),
        exposure_score=round(exposure_score, 3),
        quality_score=quality_score,
        status=status,
        issues=issues,
        recommended_user_action=action,
    )
