from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.routes import router as admin_router
from app.config import Settings, get_settings
from app.db import get_session
from app.routes.query import router as query_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(admin_router)
    app.include_router(query_router)

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
