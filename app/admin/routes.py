import base64
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.dependencies import get_pipeline_runner
from app.pipelines import PipelineRunner
from app.schemas import CompareRequest, QueryContext

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="app/admin/templates")


@router.get("", response_class=HTMLResponse)
async def admin_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@router.post("/comparisons", response_class=HTMLResponse)
async def create_comparison(
    request: Request,
    question: str = Form(...),
    crop: str = Form(""),
    region: str = Form(""),
    language: str = Form("auto"),
    image: UploadFile | None = File(None),
    runner: PipelineRunner = Depends(get_pipeline_runner),
) -> HTMLResponse:
    image_b64 = None
    if image and image.filename:
        content = await image.read()
        image_b64 = base64.b64encode(content).decode("utf-8")

    comparison = await runner.compare(
        CompareRequest(
            question=question,
            context=QueryContext(
                crop=crop or None,
                region=region or None,
                language=language or "auto",
            ),
            image=image_b64,
        )
    )
    return templates.TemplateResponse(request, "comparison.html", {"comparison": comparison})
