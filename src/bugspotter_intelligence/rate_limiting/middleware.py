"""Rate limiting middleware for FastAPI"""

import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from bugspotter_intelligence.config import Settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds rate limit headers to responses.

    Rate limiting enforcement is done via the check_rate_limit dependency.
    This middleware only adds informational headers to successful responses.

    Headers added:
    - X-RateLimit-Limit: Maximum requests per window
    - X-RateLimit-Remaining: Requests remaining in window
    """

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Add rate limit headers to response."""
        # Process request (rate limiting enforced by dependency)
        response = await call_next(request)

        # Skip header addition for health check
        if request.url.path == "/health":
            return response

        # Skip if rate limiting is disabled
        if not self.settings.rate_limit_enabled:
            return response

        # Get rate limit info from request state (set by check_rate_limit dependency)
        rate_limit = getattr(request.state, "rate_limit", None)

        if rate_limit:
            response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_limit.remaining)

        return response
