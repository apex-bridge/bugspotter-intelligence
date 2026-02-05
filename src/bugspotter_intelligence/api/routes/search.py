"""Search endpoint for natural language bug search"""

from fastapi import APIRouter, Depends
from psycopg import AsyncConnection

from bugspotter_intelligence.api.deps import get_db_connection, get_search_service
from bugspotter_intelligence.auth import TenantContext
from bugspotter_intelligence.models.requests import SearchRequest
from bugspotter_intelligence.models.responses import SearchResponse, SearchResult
from bugspotter_intelligence.rate_limiting import check_rate_limit
from bugspotter_intelligence.services import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
async def search_bugs(
    body: SearchRequest,
    tenant: TenantContext = Depends(check_rate_limit),
    conn: AsyncConnection = Depends(get_db_connection),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Search bugs using natural language.

    Modes:
    - **fast**: Vector similarity search (instant)
    - **smart**: LLM-reranked results (higher quality, slightly slower)

    Supports filtering by status and date range, with pagination.
    """
    # Extract all filter parameters from request body
    search_kwargs = body.model_dump(exclude={"query", "mode"})
    search_kwargs["tenant_id"] = tenant.tenant_id

    if body.mode == "smart":
        result = await service.search_smart(conn, body.query, **search_kwargs)
    else:
        result = await service.search_fast(conn, body.query, **search_kwargs)

    search_results = [SearchResult(**r) for r in result["results"]]

    return SearchResponse(
        results=search_results,
        total=result["total"],
        limit=result["limit"],
        offset=result["offset"],
        mode=result["mode"],
        query=result["query"],
        cached=result["cached"],
    )
