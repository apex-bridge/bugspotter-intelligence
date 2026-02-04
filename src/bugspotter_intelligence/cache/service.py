"""Cache service using existing Redis infrastructure"""

import json
import logging
from typing import Any
from uuid import UUID

from bugspotter_intelligence.rate_limiting.redis_client import get_redis, is_redis_available
from .keys import CacheKeyBuilder

logger = logging.getLogger(__name__)


class CacheService:
    """
    Caching service built on the existing Redis client.

    Graceful degradation: all methods are no-ops when Redis is unavailable.
    Uses tenant version counters for O(1) cache invalidation.
    """

    def __init__(self):
        self._keys = CacheKeyBuilder()

    @property
    def _redis(self):
        return get_redis()

    @property
    def available(self) -> bool:
        return is_redis_available()

    async def get(self, key: str) -> Any | None:
        """
        Get cached value.

        Returns None on cache miss or if Redis is unavailable.
        """
        if not self.available:
            return None

        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.debug(f"Cache get failed for {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """
        Set cached value with TTL.

        Returns False if Redis is unavailable or the operation fails.
        """
        if not self.available:
            return False

        try:
            raw = json.dumps(value)
            await self._redis.set(key, raw, ex=ttl_seconds)
            return True
        except Exception as e:
            logger.debug(f"Cache set failed for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a cached key."""
        if not self.available:
            return False

        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.debug(f"Cache delete failed for {key}: {e}")
            return False

    async def invalidate_tenant(self, tenant_id: UUID) -> None:
        """
        Increment tenant version counter.

        O(1) operation. Old cache entries with the previous version
        will never be accessed and expire via TTL naturally.
        """
        if not self.available:
            return

        try:
            key = CacheKeyBuilder.tenant_version_key(tenant_id)
            await self._redis.incr(key)
        except Exception as e:
            logger.debug(f"Cache invalidation failed for tenant {tenant_id}: {e}")

    async def get_tenant_version(self, tenant_id: UUID) -> int:
        """
        Get current tenant version.

        Returns 0 if no version set or Redis is unavailable.
        """
        if not self.available:
            return 0

        try:
            key = CacheKeyBuilder.tenant_version_key(tenant_id)
            version = await self._redis.get(key)
            return int(version) if version is not None else 0
        except Exception as e:
            logger.debug(f"Failed to get tenant version for {tenant_id}: {e}")
            return 0


    async def get_stats(self) -> dict:
        """
        Get cache statistics from Redis.

        Returns hit/miss counts and hit rate from Redis INFO stats.
        Returns zeroes if Redis is unavailable.
        """
        if not self.available:
            return {
                "available": False,
                "keyspace_hits": 0,
                "keyspace_misses": 0,
                "hit_rate": 0.0,
            }

        try:
            info = await self._redis.info("stats")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            hit_rate = round(hits / total, 4) if total > 0 else 0.0

            return {
                "available": True,
                "keyspace_hits": hits,
                "keyspace_misses": misses,
                "hit_rate": hit_rate,
            }
        except Exception as e:
            logger.debug(f"Failed to get cache stats: {e}")
            return {
                "available": True,
                "keyspace_hits": 0,
                "keyspace_misses": 0,
                "hit_rate": 0.0,
            }


_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """Get singleton CacheService instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
