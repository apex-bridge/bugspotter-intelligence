"""Rate limiting dependencies for FastAPI"""

import logging

from fastapi import Depends, HTTPException, Request, status

from bugspotter_intelligence.auth.dependencies import get_current_tenant
from bugspotter_intelligence.auth.models import TenantContext
from bugspotter_intelligence.config import Settings

from .limiter import SlidingWindowRateLimiter
from .redis_client import get_redis, is_redis_available

logger = logging.getLogger(__name__)

# Global limiter singleton
_limiter: SlidingWindowRateLimiter | None = None


def _get_settings() -> Settings:
    """Get settings - imported here to avoid circular imports"""
    from bugspotter_intelligence.api.deps import get_settings
    return get_settings()


def get_rate_limiter(settings: Settings = Depends(_get_settings)) -> SlidingWindowRateLimiter | None:
    """
    Get rate limiter singleton.

    Returns None if Redis not available (graceful degradation).
    """
    global _limiter
    if _limiter is None:
        redis_client = get_redis()
        if redis_client:
            _limiter = SlidingWindowRateLimiter(
                redis_client,
                settings.rate_limit_window_seconds,
            )
    return _limiter


class RateLimitResult:
    """Result of rate limit check, used to add headers to response."""

    def __init__(
        self,
        limit: int,
        remaining: int,
        retry_after: int = 0,
    ):
        self.limit = limit
        self.remaining = remaining
        self.retry_after = retry_after


async def check_rate_limit(
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
    settings: Settings = Depends(_get_settings),
    limiter: SlidingWindowRateLimiter | None = Depends(get_rate_limiter),
) -> TenantContext:
    """
    Dependency that enforces rate limiting after authentication.

    This runs after get_current_tenant, so tenant context is always available.
    Rate limit info is stored in request.state for middleware to add headers.

    Args:
        request: FastAPI request
        tenant: Authenticated tenant context
        settings: Application settings
        limiter: Rate limiter instance

    Returns:
        TenantContext (pass-through for chaining)

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    # Skip if rate limiting disabled
    if not settings.rate_limit_enabled:
        return tenant

    # Skip if Redis not available (graceful degradation)
    if not is_redis_available() or limiter is None:
        logger.debug("Rate limiting skipped: Redis not available")
        return tenant

    # Check rate limit
    allowed, remaining, retry_after = await limiter.is_allowed(
        tenant.api_key_id,
        tenant.rate_limit_per_minute,
    )

    # Store rate limit info for response headers
    request.state.rate_limit = RateLimitResult(
        limit=tenant.rate_limit_per_minute,
        remaining=remaining,
        retry_after=retry_after,
    )

    if not allowed:
        logger.warning(
            f"Rate limit exceeded for API key {tenant.api_key_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(tenant.rate_limit_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after),
            },
        )

    return tenant


async def check_rate_limit_admin(
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
    settings: Settings = Depends(_get_settings),
    limiter: SlidingWindowRateLimiter | None = Depends(get_rate_limiter),
) -> TenantContext:
    """
    Rate limit check that also requires admin privileges.

    Combines rate limiting with admin check for admin endpoints.
    """
    # First check rate limit
    tenant = await check_rate_limit(request, tenant, settings, limiter)

    # Then check admin
    if not tenant.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return tenant
