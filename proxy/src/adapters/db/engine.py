"""Database engine and session factory.

IMPORTANT: The engine is created in main.py during lifespan startup.
This module only provides the session factory type and a helper.
"""

from collections.abc import AsyncGenerator

from sqlmodel.ext.asyncio.session import AsyncSession


async def get_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency: yields an async session, closes on completion.

    Requires db_session_factory to be set on app.state first.
    This is a placeholder — the actual factory is created in main.py.
    """
    raise RuntimeError(
        "get_session() is not used directly. Use request.app.state.db_session_factory() instead."
    )
