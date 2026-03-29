"""
ARKEN — Database Session v3.0
Full no-op fallback when PostgreSQL is unavailable.
App starts and runs without a database.
"""

import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

_db_available = False


# ── No-op session (used when DB unavailable) ──────────────────────────────────

class _NoOpSession:
    """Silent no-op for all DB operations."""

    def add(self, obj): pass
    def add_all(self, objs): pass

    async def flush(self): pass
    async def commit(self): pass
    async def rollback(self): pass
    async def close(self): pass

    async def execute(self, *args, **kwargs):
        return _NoOpResult()

    async def get(self, model, pk):
        return None

    def __aenter__(self): return self
    async def __aexit__(self, *args): pass


class _NoOpResult:
    def scalars(self): return self
    def all(self): return []
    def first(self): return None
    def one_or_none(self): return None
    def scalar_one_or_none(self): return None
    def fetchall(self): return []


# ── Real async setup ──────────────────────────────────────────────────────────

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
    from config import settings

    engine = create_async_engine(
        settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        echo=settings.DEBUG,
        pool_pre_ping=True,
    )
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    class Base(DeclarativeBase):
        pass

    class TimestampMixin:
        created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
        updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    class UUIDMixin:
        id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    _sqlalchemy_available = True

except ImportError:
    _sqlalchemy_available = False
    class Base: pass
    class TimestampMixin: pass
    class UUIDMixin: pass
    AsyncSession = _NoOpSession


async def get_db() -> AsyncGenerator:
    if not _db_available or not _sqlalchemy_available:
        yield _NoOpSession()
        return
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    global _db_available
    if not _sqlalchemy_available:
        logger.warning("[DB] SQLAlchemy not available — running without database")
        return
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            from db.models import Base as ModelBase
            await conn.run_sync(ModelBase.metadata.create_all)
        _db_available = True
        logger.info("[DB] Database initialised successfully")
    except Exception as e:
        logger.warning(f"[DB] Database unavailable ({e}) — running without DB")
        _db_available = False
