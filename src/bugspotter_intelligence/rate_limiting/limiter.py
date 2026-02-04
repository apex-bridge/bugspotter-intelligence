"""Sliding window rate limiter implementation"""

import time
from uuid import UUID, uuid4

import redis.asyncio as redis

# Lua script for atomic rate limiting.
# Runs entirely within Redis — no TOCTOU race between check and update.
# Uses a caller-provided unique member to avoid sorted set collisions.
_LUA_RATE_LIMIT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_start = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
local window_seconds = tonumber(ARGV[5])

-- Remove expired entries
redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

-- Count current requests in window
local count = redis.call('ZCARD', key)

if count >= limit then
    -- Over limit: find oldest entry to calculate retry_after
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = window_seconds
    if #oldest > 0 then
        local oldest_time = tonumber(oldest[2])
        retry_after = math.ceil(oldest_time + window_seconds - now)
        if retry_after < 1 then
            retry_after = 1
        end
    end
    return {0, 0, retry_after}
end

-- Under limit: add and set expiry
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, window_seconds)

local remaining = limit - count - 1
if remaining < 0 then
    remaining = 0
end

return {1, remaining, 0}
"""


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter using Redis sorted sets.

    Uses a Lua script for atomic check-and-update, with unique
    member IDs to prevent sorted set collisions under concurrency.
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
        self._script = self.redis.register_script(_LUA_RATE_LIMIT)

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
        member = f"{now}:{uuid4()}"

        result = await self._script(
            keys=[key],
            args=[now, window_start, limit, member, self.window_seconds],
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        retry_after = int(result[2])

        return allowed, remaining, retry_after

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
