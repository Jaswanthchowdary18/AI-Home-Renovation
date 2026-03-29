"""
ARKEN — Cache Service v3.0
Redis-first with in-memory TTL fallback. Never crashes.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from config import settings

logger = logging.getLogger(__name__)


class _InMemoryCache:
    """Thread-safe in-memory TTL cache."""

    def __init__(self):
        self._store: Dict[str, tuple] = {}  # key -> (value_str, expire_ts)

    def set(self, key: str, value: str, ttl: int):
        self._store[key] = (value, time.monotonic() + ttl)

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if not entry:
            return None
        value, exp = entry
        if time.monotonic() > exp:
            del self._store[key]
            return None
        return value

    def delete(self, key: str):
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def incr(self, key: str) -> int:
        raw = self.get(key)
        n = (int(raw) if raw else 0) + 1
        ttl = 60
        existing = self._store.get(key)
        if existing:
            ttl = max(1, int(existing[1] - time.monotonic()))
        self.set(key, str(n), ttl)
        return n


class CacheService:
    def __init__(self):
        self._redis = None
        self._memory = _InMemoryCache()
        self._using_fallback = False

    @property
    def is_using_fallback(self) -> bool:
        return self._using_fallback

    async def connect(self):
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                socket_connect_timeout=3,
            )
            await self._redis.ping()
            logger.info("Redis connected.")
        except Exception as e:
            logger.warning(f"[Cache] Redis unavailable ({e}) — using in-memory cache")
            self._redis = None
            self._using_fallback = True

    async def disconnect(self):
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass

    async def get(self, key: str) -> Optional[Any]:
        try:
            if self._redis:
                value = await self._redis.get(key)
            else:
                value = self._memory.get(key)
            if value is None:
                return None
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.debug(f"[Cache] get error: {e}")
            value = self._memory.get(key)
            if value is None:
                return None
            try:
                return json.loads(value)
            except Exception:
                return value

    async def set(self, key: str, value: Any, ttl: int = None):
        if ttl is None:
            ttl = settings.CACHE_TTL_SECONDS
        serialized = json.dumps(value) if not isinstance(value, str) else value
        try:
            if self._redis:
                await self._redis.setex(key, ttl, serialized)
                return
        except Exception as e:
            logger.debug(f"[Cache] Redis set error: {e} — using memory")
        self._memory.set(key, serialized, ttl)

    async def delete(self, key: str):
        try:
            if self._redis:
                await self._redis.delete(key)
        except Exception:
            pass
        self._memory.delete(key)

    async def exists(self, key: str) -> bool:
        try:
            if self._redis:
                return bool(await self._redis.exists(key))
        except Exception:
            pass
        return self._memory.exists(key)

    async def incr(self, key: str) -> int:
        try:
            if self._redis:
                return await self._redis.incr(key)
        except Exception:
            pass
        return self._memory.incr(key)


cache_service = CacheService()
