"""Redis client management for rate limiting"""

import logging

import redis.asyncio as redis

from bugspotter_intelligence.config import Settings

logger = logging.getLogger(__name__)

_redis_client: redis.Redis | None = None


async def init_redis(settings: Settings) -> None:
    """
    Initialize Redis connection.

    Args:
        settings: Application settings with Redis configuration

    Raises:
        redis.ConnectionError: If unable to connect to Redis
    """
    global _redis_client

    if not settings.rate_limit_enabled:
        logger.info("Rate limiting disabled, skipping Redis initialization")
        return

    try:
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        # Test connection
        await _redis_client.ping()
        logger.info(f"Redis connected at {settings.redis_host}:{settings.redis_port}")
    except redis.ConnectionError as e:
        logger.warning(f"Failed to connect to Redis: {e}. Rate limiting will be disabled.")
        _redis_client = None


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client

    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


def get_redis() -> redis.Redis | None:
    """
    Get Redis client.

    Returns:
        Redis client if connected, None if not available
    """
    return _redis_client


def is_redis_available() -> bool:
    """Check if Redis is available."""
    return _redis_client is not None
