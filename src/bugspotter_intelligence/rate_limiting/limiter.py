"""Sliding window rate limiter implementation"""

import time
from uuid import UUID

import redis.asyncio as redis


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter using Redis sorted sets.

    Uses a sliding window counter algorithm to provide accurate
    rate limiting without the burst issues of fixed windows.
    """

    def __init__(self, redis_client: redis.Redis, window_seconds: int = 60):
        """
        Initialize the rate limiter.

        Args:
            redis_client: Redis async client
            window_seconds: Size of the sliding window in seconds
        """
        self.redis = redis_client
        self.window_seconds = window_seconds

    async def is_allowed(
        self,
        key_id: UUID,
        limit: int,
    ) -> tuple[bool, int, int]:
        """
        Check if a request is allowed under the rate limit.

        Args:
            key_id: Unique identifier for the rate limit key (API key ID)
            limit: Maximum requests allowed in the window

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds)
            - allowed: True if request should be allowed
            - remaining: Number of requests remaining in window
            - retry_after: Seconds to wait before retrying (0 if allowed)
        """
        key = f"rate_limit:{key_id}"
        now = time.time()
        window_start = now - self.window_seconds

        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current requests in window
        pipe.zcard(key)
        # Add current request (will be committed only if allowed)
        pipe.zadd(key, {str(now): now})
        # Set expiry on the key
        pipe.expire(key, self.window_seconds)

        results = await pipe.execute()
        current_count = results[1]

        if current_count >= limit:
            # Over limit - calculate retry_after
            # Get the oldest entry to determine when window will shift
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = int(oldest_time + self.window_seconds - now) + 1
            else:
                retry_after = self.window_seconds

            # Remove the request we just added (over limit)
            await self.redis.zrem(key, str(now))

            return False, 0, max(1, retry_after)

        # Under limit
        remaining = limit - current_count - 1
        return True, max(0, remaining), 0

    async def get_usage(self, key_id: UUID) -> int:
        """
        Get current usage for a rate limit key.

        Args:
            key_id: Unique identifier for the rate limit key

        Returns:
            Number of requests in the current window
        """
        key = f"rate_limit:{key_id}"
        now = time.time()
        window_start = now - self.window_seconds

        # Remove expired and count
        await self.redis.zremrangebyscore(key, 0, window_start)
        return await self.redis.zcard(key)

    async def reset(self, key_id: UUID) -> None:
        """
        Reset rate limit for a key.

        Args:
            key_id: Unique identifier for the rate limit key
        """
        key = f"rate_limit:{key_id}"
        await self.redis.delete(key)
