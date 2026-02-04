"""Tests for sliding window rate limiter"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from bugspotter_intelligence.rate_limiting.limiter import SlidingWindowRateLimiter


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = MagicMock()
    redis.zremrangebyscore = AsyncMock()
    redis.zcard = AsyncMock(return_value=0)
    redis.delete = AsyncMock()

    # register_script returns a callable script object
    mock_script = AsyncMock()
    redis.register_script = MagicMock(return_value=mock_script)

    return redis


@pytest.fixture
def limiter(mock_redis):
    """Create rate limiter with mock Redis"""
    return SlidingWindowRateLimiter(mock_redis, window_seconds=60)


@pytest.fixture
def mock_script(limiter):
    """Access the mock Lua script from the limiter"""
    return limiter._script


class TestSlidingWindowRateLimiter:
    """Test suite for SlidingWindowRateLimiter"""

    @pytest.mark.asyncio
    async def test_registers_lua_script(self, mock_redis):
        """Should register the Lua script on init"""
        SlidingWindowRateLimiter(mock_redis, window_seconds=60)
        mock_redis.register_script.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_request_under_limit(self, limiter, mock_script):
        """Should allow request when under limit"""
        mock_script.return_value = [1, 4, 0]  # allowed, 4 remaining, no retry

        allowed, remaining, retry_after = await limiter.is_allowed(uuid4(), limit=10)

        assert allowed is True
        assert remaining == 4
        assert retry_after == 0

    @pytest.mark.asyncio
    async def test_denies_request_at_limit(self, limiter, mock_script):
        """Should deny request when at limit"""
        mock_script.return_value = [0, 0, 15]  # denied, 0 remaining, retry in 15s

        allowed, remaining, retry_after = await limiter.is_allowed(uuid4(), limit=10)

        assert allowed is False
        assert remaining == 0
        assert retry_after == 15

    @pytest.mark.asyncio
    async def test_denies_request_over_limit(self, limiter, mock_script):
        """Should deny request when over limit"""
        mock_script.return_value = [0, 0, 30]

        allowed, remaining, retry_after = await limiter.is_allowed(uuid4(), limit=10)

        assert allowed is False
        assert remaining == 0
        assert retry_after > 0

    @pytest.mark.asyncio
    async def test_script_called_with_correct_key(self, limiter, mock_script):
        """Should call Lua script with correct Redis key"""
        mock_script.return_value = [1, 9, 0]
        key_id = uuid4()

        await limiter.is_allowed(key_id, limit=10)

        call_kwargs = mock_script.call_args
        assert call_kwargs.kwargs["keys"] == [f"rate_limit:{key_id}"]

    @pytest.mark.asyncio
    async def test_script_called_with_correct_args(self, limiter, mock_script):
        """Should pass limit and window to Lua script"""
        mock_script.return_value = [1, 9, 0]

        await limiter.is_allowed(uuid4(), limit=10)

        call_kwargs = mock_script.call_args
        args = call_kwargs.kwargs["args"]
        # args: [now, window_start, limit, member, window_seconds]
        assert len(args) == 5
        assert args[2] == 10  # limit
        assert args[4] == 60  # window_seconds

    @pytest.mark.asyncio
    async def test_member_is_unique_per_call(self, limiter, mock_script):
        """Should use unique member for each request"""
        mock_script.return_value = [1, 9, 0]

        await limiter.is_allowed(uuid4(), limit=10)
        member1 = mock_script.call_args.kwargs["args"][3]

        await limiter.is_allowed(uuid4(), limit=10)
        member2 = mock_script.call_args.kwargs["args"][3]

        assert member1 != member2

    @pytest.mark.asyncio
    async def test_no_extra_redis_calls_when_allowed(self, limiter, mock_script, mock_redis):
        """Should not make separate Redis calls when allowed (all in Lua)"""
        mock_script.return_value = [1, 5, 0]

        await limiter.is_allowed(uuid4(), limit=10)

        # Only the script call, no separate zrem/zrange
        mock_redis.zrem.assert_not_called() if hasattr(mock_redis, 'zrem') else None
        mock_redis.zrange.assert_not_called() if hasattr(mock_redis, 'zrange') else None

    @pytest.mark.asyncio
    async def test_no_extra_redis_calls_when_denied(self, limiter, mock_script, mock_redis):
        """Should not make separate Redis calls when denied (all in Lua)"""
        mock_script.return_value = [0, 0, 10]

        await limiter.is_allowed(uuid4(), limit=10)

        # Rejection is handled entirely in Lua — no separate zrem needed
        mock_redis.zrem.assert_not_called() if hasattr(mock_redis, 'zrem') else None
        mock_redis.zrange.assert_not_called() if hasattr(mock_redis, 'zrange') else None

    @pytest.mark.asyncio
    async def test_remaining_is_zero_when_at_limit_minus_one(self, limiter, mock_script):
        """Should show 0 remaining when last slot is used"""
        mock_script.return_value = [1, 0, 0]  # allowed, 0 remaining

        allowed, remaining, retry_after = await limiter.is_allowed(uuid4(), limit=10)

        assert allowed is True
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_get_usage_returns_count(self, limiter, mock_redis):
        """Should return current usage count"""
        mock_redis.zcard = AsyncMock(return_value=5)

        usage = await limiter.get_usage(uuid4())

        assert usage == 5

    @pytest.mark.asyncio
    async def test_get_usage_cleans_expired(self, limiter, mock_redis):
        """Should remove expired entries before counting"""
        mock_redis.zcard = AsyncMock(return_value=3)

        await limiter.get_usage(uuid4())

        mock_redis.zremrangebyscore.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_deletes_key(self, limiter, mock_redis):
        """Should delete rate limit key"""
        key_id = uuid4()
        await limiter.reset(key_id)

        mock_redis.delete.assert_called_once_with(f"rate_limit:{key_id}")

    @pytest.mark.asyncio
    async def test_custom_window_seconds(self, mock_redis):
        """Should use custom window size"""
        mock_redis.register_script = MagicMock(return_value=AsyncMock(return_value=[1, 29, 0]))
        limiter = SlidingWindowRateLimiter(mock_redis, window_seconds=30)

        await limiter.is_allowed(uuid4(), limit=30)

        args = limiter._script.call_args.kwargs["args"]
        assert args[4] == 30  # window_seconds
