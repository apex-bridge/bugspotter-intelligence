"""Bug analysis endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection

from bugspotter_intelligence.api.deps import (
    get_bug_command_service,
    get_bug_query_service,
    get_db_connection,
)
from bugspotter_intelligence.auth import TenantContext, get_current_tenant
from bugspotter_intelligence.models.requests import AnalyzeBugRequest, UpdateResolutionRequest
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
    request: AnalyzeBugRequest,
    tenant: TenantContext = Depends(get_current_tenant),
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
    try:
        result = await service.analyze_and_store_bug(
            conn=conn,
            bug_id=request.bug_id,
            title=request.title,
            description=request.description,
            console_logs=request.console_logs,
            network_logs=request.network_logs,
            metadata=request.metadata,
            tenant_id=tenant.tenant_id,
        )

        return AnalyzeBugResponse(
            bug_id=result["bug_id"],
            embedding_generated=result["embedding_generated"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze bug: {str(e)}",
        )


@router.get("/{bug_id}", response_model=BugDetailResponse)
async def get_bug(
    bug_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
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
    tenant: TenantContext = Depends(get_current_tenant),
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

        # Convert to response model
        similar_bugs = [SimilarBug(**bug) for bug in result["similar_bugs"]]

        return SimilarBugsResponse(
            bug_id=result["bug_id"],
            is_duplicate=result["is_duplicate"],
            similar_bugs=similar_bugs,
            threshold_used=result["threshold_used"],
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find similar bugs: {str(e)}",
        )


@router.get("/{bug_id}/mitigation", response_model=MitigationResponse)
async def get_mitigation_suggestion(
    bug_id: str,
    use_similar_bugs: bool = True,
    tenant: TenantContext = Depends(get_current_tenant),
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

        return MitigationResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate mitigation: {str(e)}",
        )


@router.patch("/{bug_id}/resolution", response_model=ResolutionUpdateResponse)
async def update_resolution(
    bug_id: str,
    request: UpdateResolutionRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    conn: AsyncConnection = Depends(get_db_connection),
    service: BugCommandService = Depends(get_bug_command_service),
) -> ResolutionUpdateResponse:
    """
    Update bug with resolution information.

    Called by the main BugSpotter app when a bug is resolved.
    Generates an AI summary of the resolution for future reference.
    """
    try:
        result = await service.update_bug_resolution(
            conn=conn,
            bug_id=bug_id,
            resolution=request.resolution,
            status=request.status,
            tenant_id=tenant.tenant_id,
        )

        if not result.get("updated"):
            raise HTTPException(
                status_code=404,
                detail=f"Bug {bug_id} not found",
            )

        return ResolutionUpdateResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update resolution: {str(e)}",
        )