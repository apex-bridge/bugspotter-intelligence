"""Tests for cache service"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from bugspotter_intelligence.cache.service import CacheService, get_cache_service


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    redis.incr = AsyncMock(return_value=1)
    return redis


@pytest.fixture
def cache_service(mock_redis):
    """Create cache service with mock Redis"""
    service = CacheService()
    with (
        patch(
            "bugspotter_intelligence.cache.service.get_redis",
            return_value=mock_redis,
        ),
        patch(
            "bugspotter_intelligence.cache.service.is_redis_available",
            return_value=True,
        ),
    ):
        yield service


class TestCacheServiceGet:
    """Tests for CacheService.get"""

    @pytest.mark.asyncio
    async def test_returns_none_when_redis_unavailable(self):
        """Should return None when Redis is not available"""
        service = CacheService()

        with patch(
            "bugspotter_intelligence.cache.service.is_redis_available",
            return_value=False,
        ):
            result = await service.get("some:key")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_cache_miss(self, cache_service, mock_redis):
        """Should return None when key doesn't exist"""
        mock_redis.get = AsyncMock(return_value=None)

        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
        ):
            result = await cache_service.get("missing:key")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_deserialized_value_on_hit(self, cache_service, mock_redis):
        """Should deserialize JSON and return cached value"""
        mock_redis.get = AsyncMock(return_value='{"foo": "bar", "count": 42}')

        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
        ):
            result = await cache_service.get("hit:key")

        assert result == {"foo": "bar", "count": 42}

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, cache_service, mock_redis):
        """Should return None and not raise on Redis errors"""
        mock_redis.get = AsyncMock(side_effect=Exception("connection lost"))

        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
        ):
            result = await cache_service.get("error:key")

        assert result is None


class TestCacheServiceSet:
    """Tests for CacheService.set"""

    @pytest.mark.asyncio
    async def test_returns_false_when_redis_unavailable(self):
        """Should return False when Redis is not available"""
        service = CacheService()

        with patch(
            "bugspotter_intelligence.cache.service.is_redis_available",
            return_value=False,
        ):
            result = await service.set("key", {"data": 1}, ttl_seconds=300)

        assert result is False

    @pytest.mark.asyncio
    async def test_serializes_and_stores_value(self, cache_service, mock_redis):
        """Should serialize to JSON and set with TTL"""
        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
        ):
            result = await cache_service.set("test:key", {"data": 1}, ttl_seconds=300)

        assert result is True
        mock_redis.set.assert_called_once_with("test:key", '{"data": 1}', ex=300)

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self, cache_service, mock_redis):
        """Should return False on Redis errors"""
        mock_redis.set = AsyncMock(side_effect=Exception("write failed"))

        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
        ):
            result = await cache_service.set("key", {"data": 1}, ttl_seconds=300)

        assert result is False


class TestCacheServiceDelete:
    """Tests for CacheService.delete"""

    @pytest.mark.asyncio
    async def test_deletes_key(self, cache_service, mock_redis):
        """Should delete the key from Redis"""
        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
        ):
            result = await cache_service.delete("del:key")

        assert result is True
        mock_redis.delete.assert_called_once_with("del:key")

    @pytest.mark.asyncio
    async def test_returns_false_when_redis_unavailable(self):
        """Should return False when Redis is not available"""
        service = CacheService()

        with patch(
            "bugspotter_intelligence.cache.service.is_redis_available",
            return_value=False,
        ):
            result = await service.delete("key")

        assert result is False


class TestCacheServiceInvalidateTenant:
    """Tests for CacheService.invalidate_tenant"""

    @pytest.mark.asyncio
    async def test_sets_timestamp_token(self, cache_service, mock_redis):
        """Should set the tenant invalidation timestamp"""
        tid = uuid4()
        time_ns = 1_700_000_000_123_000_000
        expected_ms = 1_700_000_000_123

        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
            patch(
                "bugspotter_intelligence.cache.service.time.time_ns",
                return_value=time_ns,
            ),
        ):
            await cache_service.invalidate_tenant(tid)

        mock_redis.set.assert_called_once_with(f"tenant:ts:{tid}", expected_ms)

    @pytest.mark.asyncio
    async def test_no_op_when_redis_unavailable(self):
        """Should do nothing when Redis is not available"""
        service = CacheService()

        with patch(
            "bugspotter_intelligence.cache.service.is_redis_available",
            return_value=False,
        ):
            await service.invalidate_tenant(uuid4())


class TestCacheServiceGetTenantVersion:
    """Tests for CacheService.get_tenant_version"""

    @pytest.mark.asyncio
    async def test_returns_token_from_redis(self, cache_service, mock_redis):
        """Should return the invalidation token value"""
        tid = uuid4()
        mock_redis.get = AsyncMock(return_value="5")

        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
        ):
            result = await cache_service.get_tenant_version(tid)

        assert result == 5

    @pytest.mark.asyncio
    async def test_returns_zero_for_new_tenant(self, cache_service, mock_redis):
        """Should return 0 when no version exists"""
        mock_redis.get = AsyncMock(return_value=None)

        with (
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
        ):
            result = await cache_service.get_tenant_version(uuid4())

        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_redis_unavailable(self):
        """Should return 0 when Redis is not available"""
        service = CacheService()

        with patch(
            "bugspotter_intelligence.cache.service.is_redis_available",
            return_value=False,
        ):
            result = await service.get_tenant_version(uuid4())

        assert result == 0


class TestCacheServiceGetStats:
    """Tests for CacheService.get_stats"""

    @pytest.mark.asyncio
    async def test_returns_unavailable_when_redis_not_available(self):
        """Should return available: False when Redis is not available"""
        service = CacheService()

        with patch(
            "bugspotter_intelligence.cache.service.is_redis_available",
            return_value=False,
        ):
            result = await service.get_stats()

        assert result == {
            "available": False,
            "keyspace_hits": 0,
            "keyspace_misses": 0,
            "hit_rate": 0.0,
        }

    @pytest.mark.asyncio
    async def test_returns_stats_when_redis_available(self):
        """Should return stats from Redis INFO command"""
        service = CacheService()
        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(
            return_value={
                "keyspace_hits": 150,
                "keyspace_misses": 50,
            }
        )

        with (
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
        ):
            result = await service.get_stats()

        assert result["available"] is True
        assert result["keyspace_hits"] == 150
        assert result["keyspace_misses"] == 50
        assert result["hit_rate"] == 0.75  # 150 / 200

    @pytest.mark.asyncio
    async def test_returns_unavailable_on_redis_error(self):
        """Should return available: False when Redis.info() raises exception"""
        service = CacheService()
        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(side_effect=Exception("Redis connection error"))

        with (
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
        ):
            result = await service.get_stats()

        # Bug fix: should return available: False, not True
        assert result["available"] is False
        assert result["keyspace_hits"] == 0
        assert result["keyspace_misses"] == 0
        assert result["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_calculates_hit_rate_correctly(self):
        """Should calculate hit rate as hits / (hits + misses)"""
        service = CacheService()
        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(
            return_value={
                "keyspace_hits": 75,
                "keyspace_misses": 25,
            }
        )

        with (
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
        ):
            result = await service.get_stats()

        assert result["hit_rate"] == 0.75  # 75 / 100

    @pytest.mark.asyncio
    async def test_returns_zero_hit_rate_when_no_activity(self):
        """Should return 0.0 hit rate when no hits or misses"""
        service = CacheService()
        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(
            return_value={
                "keyspace_hits": 0,
                "keyspace_misses": 0,
            }
        )

        with (
            patch(
                "bugspotter_intelligence.cache.service.is_redis_available",
                return_value=True,
            ),
            patch(
                "bugspotter_intelligence.cache.service.get_redis",
                return_value=mock_redis,
            ),
        ):
            result = await service.get_stats()

        assert result["hit_rate"] == 0.0


class TestGetCacheServiceSingleton:
    """Tests for get_cache_service singleton"""

    def test_returns_cache_service_instance(self):
        """Should return a CacheService instance"""
        with patch("bugspotter_intelligence.cache.service._cache_service", None):
            result = get_cache_service()
        assert isinstance(result, CacheService)

    def test_returns_singleton(self):
        """Should return same instance on subsequent calls"""
        with patch("bugspotter_intelligence.cache.service._cache_service", None):
            result1 = get_cache_service()
            result2 = get_cache_service()
        assert result1 is result2
