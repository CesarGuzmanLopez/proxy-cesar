"""Async SQLModel engine and session factory.

python.md §7: async-first with AsyncSession.
Uses SQLModel (python.md §6.2) which combines SQLAlchemy + Pydantic.
"""

from collections.abc import AsyncGenerator

from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config.settings import settings

engine = create_async_engine(settings.database_url, echo=False, pool_size=5, max_overflow=10)
session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency: yields an async session, closes on completion."""
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
