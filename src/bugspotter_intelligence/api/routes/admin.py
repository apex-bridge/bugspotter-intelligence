"""Admin API endpoints for key management and system stats"""

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from bugspotter_intelligence.api.deps import get_cache, get_settings
from bugspotter_intelligence.auth import (
    APIKeyService,
    TenantContext,
    get_api_key_service,
)
from bugspotter_intelligence.auth.dependencies import require_master_key
from bugspotter_intelligence.cache import CacheService
from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.db.database import get_db_connection
from bugspotter_intelligence.models.requests import CreateAPIKeyRequest, CreateTenantAPIKeyRequest
from bugspotter_intelligence.models.responses import (
    APIKeyListResponse,
    APIKeyResponse,
    CacheStatsResponse,
    CreateAPIKeyResponse,
    EmbeddingHealth,
    ObservabilityAccuracyResponse,
    ObservabilityDayStat,
    ObservabilityEvent,
    ObservabilityEventsResponse,
    ObservabilityOpStat,
    ObservabilitySummaryResponse,
    ServiceStatusResponse,
)
from bugspotter_intelligence.rate_limiting import check_rate_limit_admin

try:
    _SERVICE_VERSION = _pkg_version("bugspotter-intelligence")
except PackageNotFoundError:  # pragma: no cover - only when run from a non-installed tree
    _SERVICE_VERSION = "unknown"

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


@router.get(
    "/status",
    response_model=ServiceStatusResponse,
    dependencies=[Depends(require_master_key)],
)
async def get_service_status(
    settings: Settings = Depends(get_settings),
    conn: AsyncConnection = Depends(get_db_connection),
) -> ServiceStatusResponse:
    """
    Operator-only service status (master key required).

    Reports the active generation provider/model, whether cloud API keys are
    configured (booleans only — secret values are never returned), the dedup
    thresholds, and embedding-pipeline health (total rows, NULL count, and the
    minimum stored dimension — `nulls > 0` is the silent-failure signal from a
    dimension mismatch). Reads config + a single count query; no Ollama probe.
    """
    model_by_provider = {
        "ollama": settings.ollama_model,
        "claude": settings.claude_model,
        "openai": settings.openai_model,
    }
    provider = settings.llm_provider.lower()

    async with conn.cursor() as cur:
        # Sample one non-NULL embedding's dimension via a scalar subquery rather
        # than MIN(vector_dims(embedding)) over the whole table — the latter
        # detoasts every vector. The column type is fixed vector(N), so all
        # non-NULL rows share that dimension; a dimension mismatch surfaces as
        # NULL inserts (the `nulls` count), not as a differing stored dim.
        await cur.execute(
            """
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE embedding IS NULL) AS nulls,
                   (SELECT vector_dims(embedding)
                      FROM bug_embeddings
                     WHERE embedding IS NOT NULL
                     LIMIT 1) AS sampled_dim
            FROM bug_embeddings
            """
        )
        row = await cur.fetchone()
        total, nulls, min_dim = row if row else (0, 0, None)

    return ServiceStatusResponse(
        version=_SERVICE_VERSION,
        llm_provider=provider,
        llm_model=model_by_provider.get(provider),
        anthropic_key_configured=bool(settings.anthropic_api_key),
        openai_key_configured=bool(settings.openai_api_key),
        similarity_threshold=settings.similarity_threshold,
        duplicate_threshold=settings.duplicate_threshold,
        embeddings=EmbeddingHealth(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            total=total,
            nulls=nulls,
            min_dim=min_dim,
            healthy=(nulls == 0),
        ),
    )


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
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
) -> ObservabilitySummaryResponse:
    """Aggregated stats over intelligence_event in a time window; scoped to caller's tenant."""
    where: list[str] = ["tenant_id = %s"]
    params: list = [tenant.tenant_id]
    _build_time_window(where, params, from_ts, to_ts)
    where_sql = "WHERE " + " AND ".join(where)

    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT
                COUNT(*) AS calls,
                COALESCE(SUM(cost_micros_usd), 0) AS cost_micros,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms::float) AS p50,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms::float) AS p95,
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
                percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms::float) AS p50,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms::float) AS p95,
                COALESCE(SUM(cost_micros_usd), 0) AS cost_micros
            FROM intelligence_event
            {where_sql}
            GROUP BY operation
            ORDER BY calls DESC
            """,
            params,
        )
        by_op_rows = await cur.fetchall()

        await cur.execute(
            f"""
            SELECT
                date_trunc('day', created_at)::date AS day,
                COUNT(*) AS calls,
                COALESCE(SUM(cost_micros_usd), 0) AS cost_micros,
                COALESCE(SUM(tokens_in), 0) AS tokens_in,
                COALESCE(SUM(tokens_out), 0) AS tokens_out
            FROM intelligence_event
            {where_sql}
            GROUP BY day
            ORDER BY day
            """,
            params,
        )
        by_day_rows = await cur.fetchall()

    by_operation = [
        ObservabilityOpStat(
            operation=r[0], calls=r[1], p50_ms=r[2], p95_ms=r[3], cost_micros_usd=r[4],
        )
        for r in by_op_rows
    ]
    by_day = [
        ObservabilityDayStat(
            day=r[0], calls=r[1], cost_micros_usd=r[2], tokens_in=r[3], tokens_out=r[4],
        )
        for r in by_day_rows
    ]
    error_rate = (errors / calls) if calls else 0.0

    return ObservabilitySummaryResponse(
        tenant_id=tenant.tenant_id,
        from_ts=from_ts,
        to_ts=to_ts,
        calls=calls,
        cost_micros_usd=cost_micros,
        p50_ms=p50,
        p95_ms=p95,
        error_rate=error_rate,
        by_operation=by_operation,
        by_day=by_day,
    )


@router.get("/observability/events", response_model=ObservabilityEventsResponse)
async def observability_events(
    operation: str | None = Query(None),
    event_status: Literal["ok", "error"] | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
) -> ObservabilityEventsResponse:
    """Recent intelligence_event rows, newest first; scoped to caller's tenant."""
    where: list[str] = ["tenant_id = %s"]
    params: list = [tenant.tenant_id]
    if operation is not None:
        where.append("operation = %s")
        params.append(operation)
    if event_status is not None:
        where.append("status = %s")
        params.append(event_status)
    where_sql = "WHERE " + " AND ".join(where)
    params_with_pagination = params + [limit, offset]

    # dict_row maps each column to its name, so adding/removing columns
    # can't silently rotate a positional unpacking — Pydantic validates by
    # field name and a mismatch surfaces as an explicit ValidationError.
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            f"""
            SELECT id, tenant_id, operation, bug_id, provider, model, prompt_version,
                   tokens_in, tokens_out, cost_micros_usd, latency_ms,
                   confidence, rationale, status, error_kind, cached, created_at
            FROM intelligence_event
            {where_sql}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            params_with_pagination,
        )
        rows = await cur.fetchall()

    events = [ObservabilityEvent.model_validate(r) for r in rows]
    return ObservabilityEventsResponse(events=events, limit=limit, offset=offset)


@router.get("/observability/accuracy", response_model=ObservabilityAccuracyResponse)
async def observability_accuracy(
    operation: str | None = Query(None),
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    tenant: TenantContext = Depends(check_rate_limit_admin),
    conn: AsyncConnection = Depends(get_db_connection),
) -> ObservabilityAccuracyResponse:
    """Verdict counts + precision; scoped to caller's tenant via feedback × event JOIN."""
    where: list[str] = ["f.tenant_id = %s"]
    params: list = [tenant.tenant_id]
    if operation is not None:
        where.append("e.operation = %s")
        params.append(operation)
    if from_ts is not None:
        where.append("f.created_at >= %s")
        params.append(from_ts)
    if to_ts is not None:
        where.append("f.created_at <= %s")
        params.append(to_ts)
    where_sql = "WHERE " + " AND ".join(where)

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
        tenant_id=tenant.tenant_id,
        operation=operation,
        feedback_count=total,
        correct=correct,
        incorrect=incorrect,
        partial=partial,
        precision=precision,
    )
