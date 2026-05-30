"""Admin API endpoints for key management and system stats"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from psycopg import AsyncConnection

from bugspotter_intelligence.api.deps import get_cache
from bugspotter_intelligence.auth import (
    APIKeyService,
    TenantContext,
    get_api_key_service,
)
from bugspotter_intelligence.auth.dependencies import require_master_key
from bugspotter_intelligence.cache import CacheService
from bugspotter_intelligence.db.database import get_db_connection
from bugspotter_intelligence.models.requests import CreateAPIKeyRequest, CreateTenantAPIKeyRequest
from bugspotter_intelligence.models.responses import (
    APIKeyListResponse,
    APIKeyResponse,
    CacheStatsResponse,
    CreateAPIKeyResponse,
    ObservabilityAccuracyResponse,
    ObservabilityEvent,
    ObservabilityEventsResponse,
    ObservabilityOpStat,
    ObservabilitySummaryResponse,
)
from bugspotter_intelligence.rate_limiting import check_rate_limit_admin

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/api-keys", response_model=CreateAPIKeyResponse, status_code=201)
async def create_api_key(
    body: CreateAPIKeyRequest,
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
        api_key=APIKeyResponse.model_validate(api_key),
        plain_key=plain_key,
    )


@router.get("/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
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
        keys=[APIKeyResponse.model_validate(k) for k in keys],
        total=len(keys),
    )


@router.get("/api-keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: UUID,
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

    return APIKeyResponse.model_validate(api_key)


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: UUID,
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


@router.post(
    "/tenants/{tenant_id}/api-keys",
    response_model=CreateAPIKeyResponse,
    status_code=201,
    dependencies=[Depends(require_master_key)],
)
async def create_tenant_api_key(
    tenant_id: UUID,
    body: CreateTenantAPIKeyRequest,
    conn: AsyncConnection = Depends(get_db_connection),
    service: APIKeyService = Depends(get_api_key_service),
) -> CreateAPIKeyResponse:
    """
    Create an API key for any tenant (master key required).

    Used by the BugSpotter backend to provision per-org intelligence keys
    with proper tenant isolation. The plain key is returned only once.
    """
    api_key, plain_key = await service.create_key(
        conn=conn,
        tenant_id=tenant_id,
        name=body.name,
        rate_limit_per_minute=body.rate_limit_per_minute,
        is_admin=False,  # master-key provisioned keys are always non-admin
    )

    return CreateAPIKeyResponse(
        api_key=APIKeyResponse.model_validate(api_key),
        plain_key=plain_key,
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    tenant: TenantContext = Depends(check_rate_limit_admin),
    cache: CacheService = Depends(get_cache),
) -> CacheStatsResponse:
    """
    Get cache statistics (admin only).

    Returns global Redis keyspace hit/miss counts and hit rate across all tenants.

    Note: These are system-wide metrics, not tenant-specific. Admin users from any
    tenant can view overall system cache performance, which may reveal aggregate
    usage patterns across all tenants. This is intentional for operational monitoring
    and capacity planning. Tenant-specific data is not exposed, only aggregate metrics.

    If strict tenant isolation of operational metrics is required, consider restricting
    this endpoint to super-admin roles only.
    """
    stats = await cache.get_stats()
    return CacheStatsResponse(**stats)


def _build_time_window(
    where: list[str],
    params: list,
    from_ts: datetime | None,
    to_ts: datetime | None,
) -> None:
    if from_ts is not None:
        where.append("created_at >= %s")
        params.append(from_ts)
    if to_ts is not None:
        where.append("created_at <= %s")
        params.append(to_ts)


@router.get("/observability/summary", response_model=ObservabilitySummaryResponse)
async def observability_summary(
    tenant_id: UUID | None = Query(None, description="Scope to one tenant; omit for all"),
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
) -> ObservabilitySummaryResponse:
    """Aggregated stats over intelligence_event in a time window."""
    where: list[str] = []
    params: list = []
    if tenant_id is not None:
        where.append("tenant_id = %s")
        params.append(tenant_id)
    _build_time_window(where, params, from_ts, to_ts)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT
                COUNT(*) AS calls,
                COALESCE(SUM(cost_micros_usd), 0) AS cost_micros,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)::float AS p50,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::float AS p95,
                COUNT(*) FILTER (WHERE status = 'error') AS errors
            FROM intelligence_event
            {where_sql}
            """,
            params,
        )
        row = await cur.fetchone()
        calls, cost_micros, p50, p95, errors = row

        await cur.execute(
            f"""
            SELECT
                operation,
                COUNT(*) AS calls,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)::float AS p50,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::float AS p95,
                COALESCE(SUM(cost_micros_usd), 0) AS cost_micros
            FROM intelligence_event
            {where_sql}
            GROUP BY operation
            ORDER BY calls DESC
            """,
            params,
        )
        by_op_rows = await cur.fetchall()

    by_operation = [
        ObservabilityOpStat(
            operation=r[0], calls=r[1], p50_ms=r[2], p95_ms=r[3], cost_micros_usd=r[4],
        )
        for r in by_op_rows
    ]
    error_rate = (errors / calls) if calls else 0.0

    return ObservabilitySummaryResponse(
        tenant_id=tenant_id,
        from_ts=from_ts,
        to_ts=to_ts,
        calls=calls,
        cost_micros_usd=cost_micros,
        p50_ms=p50,
        p95_ms=p95,
        error_rate=error_rate,
        by_operation=by_operation,
    )


@router.get("/observability/events", response_model=ObservabilityEventsResponse)
async def observability_events(
    tenant_id: UUID | None = Query(None),
    operation: str | None = Query(None),
    event_status: Literal["ok", "error"] | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
) -> ObservabilityEventsResponse:
    """Recent intelligence_event rows, newest first; for ad-hoc debugging."""
    where: list[str] = []
    params: list = []
    if tenant_id is not None:
        where.append("tenant_id = %s")
        params.append(tenant_id)
    if operation is not None:
        where.append("operation = %s")
        params.append(operation)
    if event_status is not None:
        where.append("status = %s")
        params.append(event_status)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    params_with_pagination = params + [limit, offset]

    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT id, tenant_id, operation, bug_id, provider, model, prompt_version,
                   tokens_in, tokens_out, cost_micros_usd, latency_ms,
                   confidence, status, error_kind, cached, created_at
            FROM intelligence_event
            {where_sql}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            params_with_pagination,
        )
        rows = await cur.fetchall()

    events = [
        ObservabilityEvent(
            id=r[0], tenant_id=r[1], operation=r[2], bug_id=r[3],
            provider=r[4], model=r[5], prompt_version=r[6],
            tokens_in=r[7], tokens_out=r[8], cost_micros_usd=r[9],
            latency_ms=r[10], confidence=r[11], status=r[12],
            error_kind=r[13], cached=r[14], created_at=r[15],
        )
        for r in rows
    ]
    return ObservabilityEventsResponse(events=events, limit=limit, offset=offset)


@router.get("/observability/accuracy", response_model=ObservabilityAccuracyResponse)
async def observability_accuracy(
    tenant_id: UUID | None = Query(None),
    operation: str | None = Query(None),
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
) -> ObservabilityAccuracyResponse:
    """Verdict counts and precision over intelligence_feedback joined with intelligence_event."""
    where: list[str] = []
    params: list = []
    if tenant_id is not None:
        where.append("f.tenant_id = %s")
        params.append(tenant_id)
    if operation is not None:
        where.append("e.operation = %s")
        params.append(operation)
    if from_ts is not None:
        where.append("f.created_at >= %s")
        params.append(from_ts)
    if to_ts is not None:
        where.append("f.created_at <= %s")
        params.append(to_ts)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE f.verdict = 'correct') AS correct,
                COUNT(*) FILTER (WHERE f.verdict = 'incorrect') AS incorrect,
                COUNT(*) FILTER (WHERE f.verdict = 'partial') AS partial
            FROM intelligence_feedback f
            JOIN intelligence_event e ON e.id = f.event_id
            {where_sql}
            """,
            params,
        )
        total, correct, incorrect, partial = await cur.fetchone()

    denom = correct + incorrect
    precision = (correct / denom) if denom > 0 else None

    return ObservabilityAccuracyResponse(
        tenant_id=tenant_id,
        operation=operation,
        feedback_count=total,
        correct=correct,
        incorrect=incorrect,
        partial=partial,
        precision=precision,
    )
