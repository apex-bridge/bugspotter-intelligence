"""Authentication dependencies for FastAPI"""

import secrets
from uuid import UUID

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.db.database import get_db_connection

from .models import TenantContext
from .service import APIKeyService

# Bearer token security scheme
security = HTTPBearer(auto_error=False)

# Global singletons
_api_key_service: APIKeyService | None = None


def get_api_key_service() -> APIKeyService:
    """
    Get API key service singleton.

    Returns:
        APIKeyService instance
    """
    global _api_key_service
    if _api_key_service is None:
        _api_key_service = APIKeyService(key_prefix=_get_settings().api_key_prefix)
    return _api_key_service


def _get_settings() -> Settings:
    """Get settings - imported here to avoid circular imports"""
    from bugspotter_intelligence.api.deps import get_settings
    return get_settings()


async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
    conn=Depends(get_db_connection),
    settings: Settings = Depends(_get_settings),
) -> TenantContext:
    """
    Dependency that extracts and validates API key.

    Returns TenantContext for multi-tenant filtering.

    Raises:
        HTTPException: 401 if missing or invalid API key

    Example:
        @router.get("/bugs")
        async def list_bugs(tenant: TenantContext = Depends(get_current_tenant)):
            # tenant.tenant_id is available for filtering
            ...
    """
    # If auth is disabled, return a default development tenant
    if not settings.auth_enabled:
        return TenantContext(
            tenant_id=UUID("00000000-0000-0000-0000-000000000000"),
            api_key_id=UUID("00000000-0000-0000-0000-000000000000"),
            is_admin=True,
            rate_limit_per_minute=10000,
        )

    # Check for credentials
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate the API key
    service = get_api_key_service()
    tenant_ctx = await service.validate_key(conn, credentials.credentials)

    if not tenant_ctx:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return tenant_ctx


async def require_admin(
    tenant: TenantContext = Depends(get_current_tenant),
) -> TenantContext:
    """
    Dependency that requires admin privileges.

    Use this for admin-only endpoints.

    Raises:
        HTTPException: 403 if not an admin

    Example:
        @router.post("/admin/keys")
        async def create_key(tenant: TenantContext = Depends(require_admin)):
            # Only admins can reach this
            ...
    """
    if not tenant.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return tenant


async def require_master_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
    settings: Settings = Depends(_get_settings),
) -> None:
    """
    Dependency that validates a master API key for cross-tenant operations.

    The master key is configured via MASTER_API_KEY env var and allows
    creating API keys for arbitrary tenants (e.g. provisioning per-org keys).

    Raises:
        HTTPException: 401 if missing or invalid, 503 if master key not configured
    """
    master_key_value = (
        settings.master_api_key.get_secret_value() if settings.master_api_key else None
    )
    if not master_key_value:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Master API key not configured on this server",
        )
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing master API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(credentials.credentials, master_key_value):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid master API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_tenant(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
    conn=Depends(get_db_connection),
    settings: Settings = Depends(_get_settings),
) -> TenantContext | None:
    """
    Dependency that optionally extracts tenant context.

    Returns None if no credentials provided (for public endpoints).
    Still validates if credentials are provided.

    Example:
        @router.get("/public")
        async def public_endpoint(tenant: TenantContext | None = Depends(get_optional_tenant)):
            if tenant:
                # Authenticated request
            else:
                # Anonymous request
    """
    if not credentials:
        return None

    if not settings.auth_enabled:
        return TenantContext(
            tenant_id=UUID("00000000-0000-0000-0000-000000000000"),
            api_key_id=UUID("00000000-0000-0000-0000-000000000000"),
            is_admin=True,
            rate_limit_per_minute=10000,
        )

    service = get_api_key_service()
    return await service.validate_key(conn, credentials.credentials)
