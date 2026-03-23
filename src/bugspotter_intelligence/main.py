import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bugspotter_intelligence.api.routes import admin, ask, bugs, search
from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.db.database import close_db, get_pool, init_db
from bugspotter_intelligence.db.migrations import create_tables
from bugspotter_intelligence.rate_limiting import close_redis, init_redis
from bugspotter_intelligence.rate_limiting.middleware import RateLimitMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

API_PREFIX = "/api/v1"

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    settings = Settings()

    try:
        # Initialize database
        await init_db(settings)
        logger.info("Database pool initialized")

        # Run migrations
        pool = get_pool()
        async with pool.connection() as conn:
            await create_tables(conn)

        # Initialize Redis for rate limiting
        await init_redis(settings)

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise  # Re-raise to prevent app from starting

    yield  # App runs here

    # Shutdown
    try:
        await close_redis()
        await close_db()
        logger.info("All connections closed")
    except Exception as e:
        logger.warning(f"Error during shutdown: {e}")


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    app.include_router(ask.router, prefix=API_PREFIX)
    app.include_router(bugs.router, prefix=API_PREFIX)
    app.include_router(search.router, prefix=API_PREFIX)
    app.include_router(admin.router, prefix=API_PREFIX)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = Settings()
    app = FastAPI(
        title="BugSpotter Intelligence API",
        description="Store bugs in the knowledge base and filter out the duplicates",
        version="0.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware, settings=settings)

    register_routes(app)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Log validation errors so they're visible in docker logs.

        The `input` field is stripped from the log line to avoid leaking raw
        request values (which may include secrets mis-sent in the wrong field).
        The full errors — including `input` — are still returned to the client
        so callers can diagnose their own mistakes.
        """
        errors = exc.errors()
        sanitized = [
            {key: e.get(key) for key in ("loc", "msg", "type")}
            for e in errors
        ]
        logger.warning(
            "Request validation failed: %s %s — %s",
            request.method,
            request.url.path,
            sanitized,
        )
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(errors)},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Log unhandled exceptions with full traceback, return clean response."""
        logger.exception(
            f"Unhandled exception on {request.method} {request.url.path}"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint (no auth required)."""
        return {"status": "healthy"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "bugspotter_intelligence.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
