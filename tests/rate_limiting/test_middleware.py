"""Tests for rate limiting middleware"""

from unittest.mock import MagicMock

import pytest
from fastapi import Request
from starlette.responses import Response

from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.rate_limiting.dependencies import RateLimitResult
from bugspotter_intelligence.rate_limiting.middleware import RateLimitMiddleware


@pytest.fixture
def mock_settings():
    """Create mock settings"""
    settings = MagicMock(spec=Settings)
    settings.rate_limit_enabled = True
    return settings


@pytest.fixture
def mock_app():
    """Create mock ASGI app"""
    return MagicMock()


@pytest.fixture
def middleware(mock_app, mock_settings):
    """Create middleware instance"""
    return RateLimitMiddleware(mock_app, mock_settings)


@pytest.fixture
def mock_request():
    """Create mock request"""
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.path = "/bugs/123"
    request.state = MagicMock()
    return request


@pytest.fixture
def mock_response():
    """Create mock response"""
    response = MagicMock(spec=Response)
    response.headers = {}
    return response


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware"""

    @pytest.mark.asyncio
    async def test_adds_rate_limit_headers(
        self, middleware, mock_request, mock_response
    ):
        """Should add rate limit headers when rate_limit in request state"""
        mock_request.state.rate_limit = RateLimitResult(limit=60, remaining=45)

        async def call_next(request):
            return mock_response

        result = await middleware.dispatch(mock_request, call_next)

        assert result.headers["X-RateLimit-Limit"] == "60"
        assert result.headers["X-RateLimit-Remaining"] == "45"

    @pytest.mark.asyncio
    async def test_no_headers_when_no_rate_limit_info(
        self, middleware, mock_request, mock_response
    ):
        """Should not add headers when no rate_limit in request state"""
        mock_request.state.rate_limit = None

        async def call_next(request):
            return mock_response

        result = await middleware.dispatch(mock_request, call_next)

        assert "X-RateLimit-Limit" not in result.headers
        assert "X-RateLimit-Remaining" not in result.headers

    @pytest.mark.asyncio
    async def test_skips_health_endpoint(
        self, middleware, mock_request, mock_response
    ):
        """Should skip header addition for health endpoint"""
        mock_request.url.path = "/health"
        mock_request.state.rate_limit = RateLimitResult(limit=60, remaining=45)

        async def call_next(request):
            return mock_response

        result = await middleware.dispatch(mock_request, call_next)

        # Headers should NOT be added for health check
        assert "X-RateLimit-Limit" not in result.headers

    @pytest.mark.asyncio
    async def test_skips_when_rate_limiting_disabled(
        self, mock_app, mock_request, mock_response
    ):
        """Should skip header addition when rate limiting disabled"""
        settings = MagicMock(spec=Settings)
        settings.rate_limit_enabled = False
        middleware = RateLimitMiddleware(mock_app, settings)

        mock_request.state.rate_limit = RateLimitResult(limit=60, remaining=45)

        async def call_next(request):
            return mock_response

        result = await middleware.dispatch(mock_request, call_next)

        # Headers should NOT be added when disabled
        assert "X-RateLimit-Limit" not in result.headers

    @pytest.mark.asyncio
    async def test_handles_missing_state_attribute(
        self, middleware, mock_request, mock_response
    ):
        """Should handle case where rate_limit attribute doesn't exist"""
        # Remove rate_limit attribute
        del mock_request.state.rate_limit

        async def call_next(request):
            return mock_response

        # Should not raise
        result = await middleware.dispatch(mock_request, call_next)

        assert "X-RateLimit-Limit" not in result.headers

    @pytest.mark.asyncio
    async def test_passes_request_to_call_next(
        self, middleware, mock_request, mock_response
    ):
        """Should pass request through to next handler"""
        received_request = None

        async def call_next(request):
            nonlocal received_request
            received_request = request
            return mock_response

        await middleware.dispatch(mock_request, call_next)

        assert received_request is mock_request
