"""Rate limiting middleware for FastAPI"""

import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from bugspotter_intelligence.config import Settings

from .limiter import SlidingWindowRateLimiter
from .redis_client import get_redis, is_redis_available

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.

    Applies rate limits based on the API key's configured limit.
    Rate limit info is passed via headers:
    - X-RateLimit-Limit: Maximum requests per window
    - X-RateLimit-Remaining: Requests remaining in window
    - X-RateLimit-Reset: Seconds until window resets (on 429)
    """

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self._limiter: SlidingWindowRateLimiter | None = None

    @property
    def limiter(self) -> SlidingWindowRateLimiter | None:
        """Lazy initialization of rate limiter."""
        if self._limiter is None:
            redis_client = get_redis()
            if redis_client:
                self._limiter = SlidingWindowRateLimiter(
                    redis_client,
                    self.settings.rate_limit_window_seconds,
                )
        return self._limiter

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request through rate limiter."""
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Skip if rate limiting is disabled
        if not self.settings.rate_limit_enabled:
            return await call_next(request)

        # Skip if Redis not available (graceful degradation)
        if not is_redis_available() or self.limiter is None:
            return await call_next(request)

        # Get tenant context from request state (set by auth dependency)
        # This is set after authentication, so we check if it exists
        tenant_ctx = getattr(request.state, "tenant_context", None)

        # If no tenant context, let the request through
        # (it will be rejected by auth middleware if needed)
        if not tenant_ctx:
            response = await call_next(request)

            # Try to get tenant context after auth runs
            tenant_ctx = getattr(request.state, "tenant_context", None)
            if tenant_ctx:
                # Add rate limit headers for informational purposes
                response.headers["X-RateLimit-Limit"] = str(
                    tenant_ctx.rate_limit_per_minute
                )

            return response

        # Apply rate limiting
        allowed, remaining, retry_after = await self.limiter.is_allowed(
            tenant_ctx.api_key_id,
            tenant_ctx.rate_limit_per_minute,
        )

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for API key {tenant_ctx.api_key_id}"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(tenant_ctx.rate_limit_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(tenant_ctx.rate_limit_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
