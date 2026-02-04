"""Bug analysis endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection

from bugspotter_intelligence.api.deps import (
    get_bug_command_service,
    get_bug_query_service,
    get_db_connection,
)
from bugspotter_intelligence.auth import TenantContext
from bugspotter_intelligence.models.requests import AnalyzeBugRequest, UpdateResolutionRequest
from bugspotter_intelligence.rate_limiting import check_rate_limit
from bugspotter_intelligence.models.responses import (
    AnalyzeBugResponse,
    BugDetailResponse,
    MitigationResponse,
    ResolutionUpdateResponse,
    SimilarBug,
    SimilarBugsResponse,
)
from bugspotter_intelligence.services import BugCommandService, BugQueryService

router = APIRouter(prefix="/bugs", tags=["Bugs"])


@router.post("/analyze", response_model=AnalyzeBugResponse, status_code=201)
async def analyze_bug(
    body: AnalyzeBugRequest,
    tenant: TenantContext = Depends(check_rate_limit),
    conn: AsyncConnection = Depends(get_db_connection),
    service: BugCommandService = Depends(get_bug_command_service),
) -> AnalyzeBugResponse:
    """
    Analyze a new bug and store its embedding.

    This endpoint:
    1. Extracts relevant information from logs and metadata
    2. Generates vector embedding for similarity search
    3. Stores bug information in the database

    The main BugSpotter app should call this when a new bug is reported.
    """
    result = await service.analyze_and_store_bug(
        conn=conn,
        bug_id=body.bug_id,
        title=body.title,
        description=body.description,
        console_logs=body.console_logs,
        network_logs=body.network_logs,
        metadata=body.metadata,
        tenant_id=tenant.tenant_id,
    )

    return AnalyzeBugResponse(
        bug_id=result["bug_id"],
        embedding_generated=result["embedding_generated"],
    )


@router.get("/{bug_id}", response_model=BugDetailResponse)
async def get_bug(
    bug_id: str,
    tenant: TenantContext = Depends(check_rate_limit),
    conn: AsyncConnection = Depends(get_db_connection),
    service: BugQueryService = Depends(get_bug_query_service),
) -> BugDetailResponse:
    """Get bug details by ID."""
    bug = await service.get_bug(conn, bug_id, tenant_id=tenant.tenant_id)

    if not bug:
        raise HTTPException(status_code=404, detail=f"Bug {bug_id} not found")

    return BugDetailResponse(**bug)


@router.get("/{bug_id}/similar", response_model=SimilarBugsResponse)
async def find_similar_bugs(
    bug_id: str,
    threshold: float | None = None,
    limit: int | None = None,
    tenant: TenantContext = Depends(check_rate_limit),
    conn: AsyncConnection = Depends(get_db_connection),
    service: BugQueryService = Depends(get_bug_query_service),
) -> SimilarBugsResponse:
    """
    Find bugs similar to the given bug.

    Uses vector similarity search to find potentially duplicate or related bugs.
    Returns similarity scores and duplicate detection.
    """
    try:
        result = await service.find_similar_bugs(
            conn=conn,
            bug_id=bug_id,
            similarity_threshold=threshold,
            limit=limit,
            tenant_id=tenant.tenant_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    similar_bugs = [SimilarBug(**bug) for bug in result["similar_bugs"]]

    return SimilarBugsResponse(
        bug_id=result["bug_id"],
        is_duplicate=result["is_duplicate"],
        similar_bugs=similar_bugs,
        threshold_used=result["threshold_used"],
    )


@router.get("/{bug_id}/mitigation", response_model=MitigationResponse)
async def get_mitigation_suggestion(
    bug_id: str,
    use_similar_bugs: bool = True,
    tenant: TenantContext = Depends(check_rate_limit),
    conn: AsyncConnection = Depends(get_db_connection),
    service: BugQueryService = Depends(get_bug_query_service),
) -> MitigationResponse:
    """
    Get AI-powered mitigation suggestion for a bug.

    Optionally uses similar resolved bugs as context to provide
    more relevant suggestions.
    """
    try:
        result = await service.get_mitigation_suggestion(
            conn=conn,
            bug_id=bug_id,
            use_similar_bugs=use_similar_bugs,
            tenant_id=tenant.tenant_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return MitigationResponse(**result)


@router.patch("/{bug_id}/resolution", response_model=ResolutionUpdateResponse)
async def update_resolution(
    bug_id: str,
    body: UpdateResolutionRequest,
    tenant: TenantContext = Depends(check_rate_limit),
    conn: AsyncConnection = Depends(get_db_connection),
    service: BugCommandService = Depends(get_bug_command_service),
) -> ResolutionUpdateResponse:
    """
    Update bug with resolution information.

    Called by the main BugSpotter app when a bug is resolved.
    Generates an AI summary of the resolution for future reference.
    """
    result = await service.update_bug_resolution(
        conn=conn,
        bug_id=bug_id,
        resolution=body.resolution,
        status=body.status,
        tenant_id=tenant.tenant_id,
    )

    if not result.get("updated"):
        raise HTTPException(
            status_code=404,
            detail=f"Bug {bug_id} not found",
        )

    return ResolutionUpdateResponse(**result)
