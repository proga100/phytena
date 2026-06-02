from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class GrowzBase(DeclarativeBase):
    pass


settings = get_settings()
growz_engine = create_async_engine(settings.growz_database_url, pool_pre_ping=True)
GrowzSessionLocal = async_sessionmaker(growz_engine, expire_on_commit=False)
