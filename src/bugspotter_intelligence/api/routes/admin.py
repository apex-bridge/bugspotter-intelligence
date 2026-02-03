"""Admin API endpoints for key management"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg import AsyncConnection

from bugspotter_intelligence.auth import (
    APIKeyService,
    TenantContext,
    get_api_key_service,
    require_admin,
)
from bugspotter_intelligence.db.database import get_db_connection
from bugspotter_intelligence.models.requests import CreateAPIKeyRequest
from bugspotter_intelligence.models.responses import (
    APIKeyListResponse,
    APIKeyResponse,
    CreateAPIKeyResponse,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/api-keys", response_model=CreateAPIKeyResponse, status_code=201)
async def create_api_key(
    request: CreateAPIKeyRequest,
    tenant: TenantContext = Depends(require_admin),
    conn: AsyncConnection = Depends(get_db_connection),
    service: APIKeyService = Depends(get_api_key_service),
) -> CreateAPIKeyResponse:
    """
    Create a new API key (admin only).

    The plain key is returned only once in this response.
    Store it securely - it cannot be retrieved again.
    """
    # Admin can create keys for any tenant, or default to their own
    target_tenant = request.tenant_id or tenant.tenant_id

    try:
        api_key, plain_key = await service.create_key(
            conn=conn,
            tenant_id=target_tenant,
            name=request.name,
            rate_limit_per_minute=request.rate_limit_per_minute,
            is_admin=request.is_admin,
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}",
        )


@router.get("/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
    tenant: TenantContext = Depends(require_admin),
    conn: AsyncConnection = Depends(get_db_connection),
    service: APIKeyService = Depends(get_api_key_service),
) -> APIKeyListResponse:
    """
    List all API keys for the tenant (admin only).

    Keys are masked - only the prefix is shown.
    """
    try:
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}",
        )


@router.get("/api-keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: UUID,
    tenant: TenantContext = Depends(require_admin),
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
    tenant: TenantContext = Depends(require_admin),
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
