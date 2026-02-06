"""Cache service using existing Redis infrastructure"""

import json
import logging
import time
from typing import Any
from uuid import UUID

from bugspotter_intelligence.rate_limiting.redis_client import (
    get_redis,
    is_redis_available,
)

from .keys import CacheKeyBuilder

logger = logging.getLogger(__name__)


class CacheService:
    """
    Caching service built on the existing Redis client.

    Graceful degradation: all methods are no-ops when Redis is unavailable.
    Uses tenant invalidation timestamps for O(1) cache invalidation.

    Consistency Model:
    ------------------
    This cache implements **eventual consistency** with a post-commit invalidation
    pattern. When data is modified (e.g., new bug inserted, status updated):

    1. Database transaction commits first
    2. Cache invalidation occurs after commit

    This creates a small window where concurrent read requests may receive cached
    results that don't reflect the latest changes. This is acceptable for this
    use case because:

    - Bug tracking doesn't require strict consistency (not financial data)
    - Stale results (missing recent bugs) are tolerable for brief periods
    - Cache TTLs ensure eventual consistency (typical: 5-15 minutes)
    - The alternative (pre-invalidate + re-populate) adds complexity and latency

    Alternative Pattern (if stricter consistency needed):
    - Invalidate BEFORE database commit
    - Re-populate cache AFTER commit
    - Trade-off: Higher latency, risk of cache misses if commit fails
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
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt cache data for {key}: {e}")
            return None
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
        except (TypeError, ValueError) as e:
            logger.warning(f"Cannot serialize value for {key}: {e}")
            return False
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
        Update tenant invalidation timestamp.

        O(1) operation. Old cache entries with the previous token
        will never be accessed and expire via TTL naturally.

        Sets a 30-day TTL on the timestamp key to allow cleanup of
        inactive tenants while maintaining invalidation for active ones.
        """
        if not self.available:
            return

        try:
            key = CacheKeyBuilder.tenant_version_key(tenant_id)
            timestamp_ms = time.time_ns() // 1_000_000
            # Set 30-day TTL to prevent unbounded memory growth from inactive tenants
            await self._redis.set(key, timestamp_ms, ex=30 * 24 * 60 * 60)
        except Exception as e:
            logger.debug(f"Cache invalidation failed for tenant {tenant_id}: {e}")

    async def get_tenant_version(self, tenant_id: UUID) -> int:
        """
        Get current tenant invalidation token.

        Returns 0 if no token set or Redis is unavailable.
        """
        if not self.available:
            return 0

        try:
            key = CacheKeyBuilder.tenant_version_key(tenant_id)
            token = await self._redis.get(key)
            return int(token) if token is not None else 0
        except Exception as e:
            logger.debug(f"Failed to get tenant token for {tenant_id}: {e}")
            return 0

    async def get_stats(self) -> dict:
        """
        Get cache statistics from Redis.

        Returns global (system-wide) hit/miss counts and hit rate from Redis INFO stats.
        These metrics are aggregated across all tenants and all cache operations.
        Returns zeroes if Redis is unavailable.

        Note: Not tenant-specific. Use for overall system monitoring.
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
                "available": False,
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
