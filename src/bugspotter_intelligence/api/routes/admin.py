"""Admin API endpoints for key management"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from psycopg import AsyncConnection

from bugspotter_intelligence.auth import (
    APIKeyService,
    TenantContext,
    get_api_key_service,
)
from bugspotter_intelligence.db.database import get_db_connection
from bugspotter_intelligence.models.requests import CreateAPIKeyRequest
from bugspotter_intelligence.rate_limiting import check_rate_limit_admin
from bugspotter_intelligence.models.responses import (
    APIKeyListResponse,
    APIKeyResponse,
    CreateAPIKeyResponse,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/api-keys", response_model=CreateAPIKeyResponse, status_code=201)
async def create_api_key(
    body: CreateAPIKeyRequest,
    request: Request,
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
    service: APIKeyService = Depends(get_api_key_service),
) -> CreateAPIKeyResponse:
    """
    Create a new API key (admin only).

    The plain key is returned only once in this response.
    Store it securely - it cannot be retrieved again.

    Security:
        Admins can only create keys for their own tenant.
        Cross-tenant key creation is not permitted to prevent privilege escalation.
    """
    # Security: Prevent admins from creating keys for other tenants
    # This prevents privilege escalation attacks
    if body.tenant_id is not None and body.tenant_id != tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create API keys for other tenants. Admins can only create keys for their own tenant.",
        )

    target_tenant = tenant.tenant_id

    api_key, plain_key = await service.create_key(
        conn=conn,
        tenant_id=target_tenant,
        name=body.name,
        rate_limit_per_minute=body.rate_limit_per_minute,
        is_admin=body.is_admin,
    )

    return CreateAPIKeyResponse(
        api_key=APIKeyResponse(
            id=api_key.id,
            tenant_id=api_key.tenant_id,
            key_prefix=api_key.key_prefix,
            name=api_key.name,
            created_at=api_key.created_at,
            last_used_at=api_key.last_used_at,
            is_active=api_key.is_active,
            rate_limit_per_minute=api_key.rate_limit_per_minute,
            is_admin=api_key.is_admin,
        ),
        plain_key=plain_key,
    )


@router.get("/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
    request: Request,
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
    service: APIKeyService = Depends(get_api_key_service),
) -> APIKeyListResponse:
    """
    List all API keys for the tenant (admin only).

    Keys are masked - only the prefix is shown.
    """
    keys = await service.list_keys(conn, tenant.tenant_id)

    return APIKeyListResponse(
        keys=[
            APIKeyResponse(
                id=k.id,
                tenant_id=k.tenant_id,
                key_prefix=k.key_prefix,
                name=k.name,
                created_at=k.created_at,
                last_used_at=k.last_used_at,
                is_active=k.is_active,
                rate_limit_per_minute=k.rate_limit_per_minute,
                is_admin=k.is_admin,
            )
            for k in keys
        ],
        total=len(keys),
    )


@router.get("/api-keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: UUID,
    request: Request,
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
    service: APIKeyService = Depends(get_api_key_service),
) -> APIKeyResponse:
    """
    Get a specific API key by ID (admin only).
    """
    api_key = await service.get_key(conn, key_id, tenant.tenant_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found",
        )

    return APIKeyResponse(
        id=api_key.id,
        tenant_id=api_key.tenant_id,
        key_prefix=api_key.key_prefix,
        name=api_key.name,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        is_active=api_key.is_active,
        rate_limit_per_minute=api_key.rate_limit_per_minute,
        is_admin=api_key.is_admin,
    )


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: UUID,
    request: Request,
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
    service: APIKeyService = Depends(get_api_key_service),
) -> None:
    """
    Revoke an API key (admin only).

    This is a soft delete - the key is marked as revoked but not deleted.
    Revoked keys cannot be used for authentication.
    """
    success = await service.revoke_key(conn, key_id, tenant.tenant_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found",
        )
