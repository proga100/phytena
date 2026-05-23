import time
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.routes import router as admin_router
from app.config import Settings, get_settings
from app.db import get_session
from app.logging import logger
from app.routes.eval import router as eval_router
from app.routes.image import router as image_router
from app.routes.query import router as query_router
from app.routes.rag import router as rag_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Method: {request.method} Path: {request.url.path} "
            f"Status: {response.status_code} Duration: {process_time:.2f}ms"
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal Server Error",
                "message": str(exc),
                "type": exc.__class__.__name__,
            },
        )

    app.include_router(admin_router)
    app.include_router(eval_router)
    app.include_router(image_router)
    app.include_router(query_router)
    app.include_router(rag_router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(
        session: AsyncSession = Depends(get_session),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, str | int]:
        await session.execute(text("SELECT 1"))
        return {
            "status": "ready",
            "app_env": settings.app_env,
            "embedding_dimension": settings.embedding_dimension,
        }

    @app.get("/v1/system")
    async def system(settings: Settings = Depends(get_settings)) -> dict[str, str | int]:
        return {
            "app_name": settings.app_name,
            "app_env": settings.app_env,
            "llm_provider": settings.llm_provider,
            "vision_provider": settings.vision_provider,
            "embeddings_provider": settings.embeddings_provider,
            "reranker_provider": settings.reranker_provider,
            "embedding_dimension": settings.embedding_dimension,
        }

    return app


app = create_app()
