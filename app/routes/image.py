from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import ImageQualityResponse
from app.vision.input import ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES
from app.vision.quality import assess_image_quality

router = APIRouter(prefix="/v1/image", tags=["image"])


@router.post("/quality-check", response_model=ImageQualityResponse)
async def quality_check(image: UploadFile = File(...)) -> ImageQualityResponse:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are supported.")

    content = await image.read()
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")

    try:
        result = assess_image_quality(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read image.") from exc

    return ImageQualityResponse(**result.__dict__)
