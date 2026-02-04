"""Tests for rate limiting dependencies"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException, Request

from bugspotter_intelligence.auth.models import TenantContext
from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.rate_limiting.dependencies import (
    RateLimitResult,
    check_rate_limit,
    check_rate_limit_admin,
    get_rate_limiter,
)
from bugspotter_intelligence.rate_limiting.limiter import SlidingWindowRateLimiter


@pytest.fixture
def mock_tenant():
    """Create mock tenant context"""
    return TenantContext(
        tenant_id=uuid4(),
        api_key_id=uuid4(),
        is_admin=False,
        rate_limit_per_minute=60,
    )


@pytest.fixture
def mock_admin_tenant():
    """Create mock admin tenant context"""
    return TenantContext(
        tenant_id=uuid4(),
        api_key_id=uuid4(),
        is_admin=True,
        rate_limit_per_minute=100,
    )


@pytest.fixture
def mock_settings():
    """Create mock settings"""
    settings = MagicMock(spec=Settings)
    settings.rate_limit_enabled = True
    settings.rate_limit_window_seconds = 60
    return settings


@pytest.fixture
def mock_limiter():
    """Create mock rate limiter"""
    limiter = MagicMock()
    limiter.is_allowed = AsyncMock(return_value=(True, 59, 0))
    return limiter


@pytest.fixture
def mock_request():
    """Create mock FastAPI request"""
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    return request


class TestCheckRateLimit:
    """Tests for check_rate_limit dependency"""

    @pytest.mark.asyncio
    async def test_allows_request_under_limit(
        self, mock_request, mock_tenant, mock_settings, mock_limiter
    ):
        """Should allow request when under rate limit"""
        mock_limiter.is_allowed = AsyncMock(return_value=(True, 59, 0))

        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.is_redis_available",
            return_value=True,
        ):
            result = await check_rate_limit(
                request=mock_request,
                tenant=mock_tenant,
                settings=mock_settings,
                limiter=mock_limiter,
            )

            assert result == mock_tenant
            mock_limiter.is_allowed.assert_called_once_with(
                mock_tenant.api_key_id, mock_tenant.rate_limit_per_minute
            )

    @pytest.mark.asyncio
    async def test_stores_rate_limit_info_in_request_state(
        self, mock_request, mock_tenant, mock_settings, mock_limiter
    ):
        """Should store rate limit info in request state for middleware"""
        mock_limiter.is_allowed = AsyncMock(return_value=(True, 45, 0))

        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.is_redis_available",
            return_value=True,
        ):
            await check_rate_limit(
                request=mock_request,
                tenant=mock_tenant,
                settings=mock_settings,
                limiter=mock_limiter,
            )

            assert hasattr(mock_request.state, "rate_limit")
            rate_limit = mock_request.state.rate_limit
            assert rate_limit.limit == mock_tenant.rate_limit_per_minute
            assert rate_limit.remaining == 45

    @pytest.mark.asyncio
    async def test_raises_429_when_over_limit(
        self, mock_request, mock_tenant, mock_settings, mock_limiter
    ):
        """Should raise 429 when rate limit exceeded"""
        mock_limiter.is_allowed = AsyncMock(return_value=(False, 0, 30))

        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.is_redis_available",
            return_value=True,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit(
                    request=mock_request,
                    tenant=mock_tenant,
                    settings=mock_settings,
                    limiter=mock_limiter,
                )

            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in exc_info.value.detail
            assert exc_info.value.headers["Retry-After"] == "30"

    @pytest.mark.asyncio
    async def test_skips_when_rate_limiting_disabled(
        self, mock_request, mock_tenant, mock_settings, mock_limiter
    ):
        """Should skip rate limiting when disabled in settings"""
        mock_settings.rate_limit_enabled = False

        result = await check_rate_limit(
            request=mock_request,
            tenant=mock_tenant,
            settings=mock_settings,
            limiter=mock_limiter,
        )

        assert result == mock_tenant
        mock_limiter.is_allowed.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_redis_unavailable(
        self, mock_request, mock_tenant, mock_settings, mock_limiter
    ):
        """Should skip rate limiting when Redis not available"""
        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.is_redis_available",
            return_value=False,
        ):
            result = await check_rate_limit(
                request=mock_request,
                tenant=mock_tenant,
                settings=mock_settings,
                limiter=mock_limiter,
            )

            assert result == mock_tenant
            mock_limiter.is_allowed.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_limiter_none(
        self, mock_request, mock_tenant, mock_settings
    ):
        """Should skip rate limiting when limiter is None"""
        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.is_redis_available",
            return_value=True,
        ):
            result = await check_rate_limit(
                request=mock_request,
                tenant=mock_tenant,
                settings=mock_settings,
                limiter=None,
            )

            assert result == mock_tenant


class TestCheckRateLimitAdmin:
    """Tests for check_rate_limit_admin dependency"""

    @pytest.mark.asyncio
    async def test_allows_admin_request(
        self, mock_request, mock_admin_tenant, mock_settings, mock_limiter
    ):
        """Should allow admin request under rate limit"""
        mock_limiter.is_allowed = AsyncMock(return_value=(True, 99, 0))

        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.is_redis_available",
            return_value=True,
        ):
            result = await check_rate_limit_admin(
                request=mock_request,
                tenant=mock_admin_tenant,
                settings=mock_settings,
                limiter=mock_limiter,
            )

            assert result == mock_admin_tenant

    @pytest.mark.asyncio
    async def test_raises_403_for_non_admin(
        self, mock_request, mock_tenant, mock_settings, mock_limiter
    ):
        """Should raise 403 for non-admin users"""
        mock_limiter.is_allowed = AsyncMock(return_value=(True, 59, 0))

        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.is_redis_available",
            return_value=True,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit_admin(
                    request=mock_request,
                    tenant=mock_tenant,
                    settings=mock_settings,
                    limiter=mock_limiter,
                )

            assert exc_info.value.status_code == 403
            assert "Admin privileges required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_rate_limit_checked_before_admin(
        self, mock_request, mock_admin_tenant, mock_settings, mock_limiter
    ):
        """Should check rate limit before admin check"""
        mock_limiter.is_allowed = AsyncMock(return_value=(False, 0, 30))

        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.is_redis_available",
            return_value=True,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit_admin(
                    request=mock_request,
                    tenant=mock_admin_tenant,
                    settings=mock_settings,
                    limiter=mock_limiter,
                )

            # Should get 429 (rate limit) not 403 (admin)
            assert exc_info.value.status_code == 429


class TestRateLimitResult:
    """Tests for RateLimitResult class"""

    def test_creation(self):
        """Should create rate limit result with values"""
        result = RateLimitResult(limit=60, remaining=45, retry_after=0)

        assert result.limit == 60
        assert result.remaining == 45
        assert result.retry_after == 0

    def test_default_retry_after(self):
        """Should default retry_after to 0"""
        result = RateLimitResult(limit=60, remaining=45)

        assert result.retry_after == 0


class TestGetRateLimiter:
    """Tests for get_rate_limiter dependency"""

    def test_returns_none_when_redis_not_available(self, mock_settings):
        """Should return None when Redis client not available"""
        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.get_redis",
            return_value=None,
        ):
            # Reset singleton
            import bugspotter_intelligence.rate_limiting.dependencies as deps
            deps._limiter = None

            result = get_rate_limiter(mock_settings)

            assert result is None

    def test_returns_limiter_when_redis_available(self, mock_settings):
        """Should return limiter when Redis client available"""
        mock_redis = MagicMock()

        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.get_redis",
            return_value=mock_redis,
        ):
            # Reset singleton
            import bugspotter_intelligence.rate_limiting.dependencies as deps
            deps._limiter = None

            result = get_rate_limiter(mock_settings)

            assert result is not None
            assert isinstance(result, SlidingWindowRateLimiter)

    def test_returns_singleton(self, mock_settings):
        """Should return same instance on subsequent calls"""
        mock_redis = MagicMock()

        with patch(
            "bugspotter_intelligence.rate_limiting.dependencies.get_redis",
            return_value=mock_redis,
        ):
            # Reset singleton
            import bugspotter_intelligence.rate_limiting.dependencies as deps
            deps._limiter = None

            result1 = get_rate_limiter(mock_settings)
            result2 = get_rate_limiter(mock_settings)

            assert result1 is result2
