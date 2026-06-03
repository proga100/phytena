from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Dedicated engine/session for the Growz Uzbek RAG DB (rag_diseases / rag_crops).
growz_rag_engine = create_async_engine(settings.growz_rag_database_url, pool_pre_ping=True)
GrowzRagSessionLocal = async_sessionmaker(growz_rag_engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def check_database() -> bool:
    async with engine.begin() as connection:
        await connection.execute(text("SELECT 1"))
    return True
