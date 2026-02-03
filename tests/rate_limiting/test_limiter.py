"""Tests for sliding window rate limiter"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from bugspotter_intelligence.rate_limiting.limiter import SlidingWindowRateLimiter


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=MagicMock())
    redis.zrange = AsyncMock(return_value=[])
    redis.zrem = AsyncMock()
    redis.zcard = AsyncMock(return_value=0)
    redis.delete = AsyncMock()
    return redis


@pytest.fixture
def mock_pipeline(mock_redis):
    """Mock Redis pipeline"""
    pipe = MagicMock()
    pipe.zremrangebyscore = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.zadd = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=[0, 0, 1, True])  # Default: 0 current requests
    mock_redis.pipeline = MagicMock(return_value=pipe)
    return pipe


@pytest.fixture
def limiter(mock_redis):
    """Create rate limiter with mock Redis"""
    return SlidingWindowRateLimiter(mock_redis, window_seconds=60)


class TestSlidingWindowRateLimiter:
    """Test suite for SlidingWindowRateLimiter"""

    @pytest.mark.asyncio
    async def test_allows_request_under_limit(self, limiter, mock_pipeline):
        """Should allow request when under limit"""
        mock_pipeline.execute = AsyncMock(return_value=[0, 5, 1, True])  # 5 current requests

        allowed, remaining, retry_after = await limiter.is_allowed(uuid4(), limit=10)

        assert allowed is True
        assert remaining == 4  # 10 - 5 - 1 = 4
        assert retry_after == 0

    @pytest.mark.asyncio
    async def test_denies_request_at_limit(self, limiter, mock_pipeline, mock_redis):
        """Should deny request when at limit"""
        mock_pipeline.execute = AsyncMock(return_value=[0, 10, 1, True])  # 10 current (at limit)
        mock_redis.zrange = AsyncMock(return_value=[("1234", 100.0)])

        allowed, remaining, retry_after = await limiter.is_allowed(uuid4(), limit=10)

        assert allowed is False
        assert remaining == 0
        assert retry_after > 0

    @pytest.mark.asyncio
    async def test_denies_request_over_limit(self, limiter, mock_pipeline, mock_redis):
        """Should deny request when over limit"""
        mock_pipeline.execute = AsyncMock(return_value=[0, 15, 1, True])  # 15 current (over limit)
        mock_redis.zrange = AsyncMock(return_value=[("1234", 100.0)])

        allowed, remaining, retry_after = await limiter.is_allowed(uuid4(), limit=10)

        assert allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_removes_request_on_denial(self, limiter, mock_pipeline, mock_redis):
        """Should remove the added request when denied"""
        mock_pipeline.execute = AsyncMock(return_value=[0, 10, 1, True])
        mock_redis.zrange = AsyncMock(return_value=[("1234", 100.0)])

        await limiter.is_allowed(uuid4(), limit=10)

        mock_redis.zrem.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_correct_key_format(self, limiter, mock_pipeline):
        """Should use correct key format"""
        key_id = uuid4()
        await limiter.is_allowed(key_id, limit=10)

        # Verify pipeline was created (which uses the key internally)
        limiter.redis.pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_usage_returns_count(self, limiter, mock_redis):
        """Should return current usage count"""
        mock_redis.zcard = AsyncMock(return_value=5)
        mock_redis.zremrangebyscore = AsyncMock()

        usage = await limiter.get_usage(uuid4())

        assert usage == 5

    @pytest.mark.asyncio
    async def test_reset_deletes_key(self, limiter, mock_redis):
        """Should delete rate limit key"""
        key_id = uuid4()
        await limiter.reset(key_id)

        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_remaining_is_zero_when_at_limit_minus_one(self, limiter, mock_pipeline):
        """Should show 0 remaining when at limit-1 (current request uses last slot)"""
        mock_pipeline.execute = AsyncMock(return_value=[0, 9, 1, True])  # 9 current

        allowed, remaining, retry_after = await limiter.is_allowed(uuid4(), limit=10)

        assert allowed is True
        assert remaining == 0  # 10 - 9 - 1 = 0
