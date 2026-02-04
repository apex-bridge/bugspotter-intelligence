"""Search service for natural language bug search"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from psycopg import AsyncConnection

from bugspotter_intelligence.cache.keys import CacheKeyBuilder
from bugspotter_intelligence.db.bug_repository import BugRepository
from bugspotter_intelligence.services.embeddings import EmbeddingProvider
from bugspotter_intelligence.services.reranker import LLMReranker

if TYPE_CHECKING:
    from bugspotter_intelligence.cache.service import CacheService

logger = logging.getLogger(__name__)


class SearchService:
    """
    Handles natural language bug search.

    Embeds the query, then uses vector similarity to find relevant bugs.
    Supports fast (vector-only) and smart (LLM-reranked) modes.
    Optional cache integration for search results.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[LLMReranker] = None,
        smart_candidate_limit: int = 20,
        cache: Optional["CacheService"] = None,
        cache_ttl_fast: int = 300,
        cache_ttl_smart: int = 900,
    ):
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.smart_candidate_limit = smart_candidate_limit
        self.cache = cache
        self.cache_ttl_fast = cache_ttl_fast
        self.cache_ttl_smart = cache_ttl_smart

    async def search_fast(
        self,
        conn: AsyncConnection,
        query: str,
        *,
        tenant_id: UUID,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> dict:
        """
        Fast search: embed query → vector ANN → return ranked results.

        Args:
            conn: Database connection
            query: Natural language search query
            tenant_id: Tenant UUID for isolation
            limit: Page size
            offset: Page offset
            status: Optional status filter
            date_from: Optional created_at lower bound
            date_to: Optional created_at upper bound

        Returns:
            Dict with results, total, pagination info, mode, query, cached flag
        """
        logger.debug(f"Fast search: query={query!r}, tenant={tenant_id}")

        # Check cache
        cached = await self._cache_get(query, tenant_id, "fast", limit, offset, status, date_from, date_to)
        if cached is not None:
            cached["cached"] = True
            return cached

        embedding = self.embedding_provider.embed(query)

        results, total = await BugRepository.search(
            conn,
            embedding,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )

        result = {
            "results": results,
            "total": total,
            "limit": limit,
            "offset": offset,
            "mode": "fast",
            "query": query,
            "cached": False,
        }

        # Populate cache
        await self._cache_set(query, tenant_id, "fast", limit, offset, status, date_from, date_to, result)

        return result

    async def search_smart(
        self,
        conn: AsyncConnection,
        query: str,
        *,
        tenant_id: UUID,
        limit: int = 5,
        offset: int = 0,
        status: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> dict:
        """
        Smart search: retrieve top candidates → LLM rerank → return best matches.

        Fetches a larger candidate set using fast search, then uses the LLM
        reranker to score relevance. Falls back to fast mode if the reranker
        is unavailable or fails.

        Args:
            conn: Database connection
            query: Natural language search query
            tenant_id: Tenant UUID for isolation
            limit: Final number of results after reranking
            offset: Pagination offset (applied after reranking)
            status: Optional status filter
            date_from: Optional created_at lower bound
            date_to: Optional created_at upper bound

        Returns:
            Dict with results, total, pagination info, mode, query, cached flag
        """
        if self.reranker is None:
            logger.info("No reranker available, falling back to fast search")
            result = await self.search_fast(
                conn,
                query,
                tenant_id=tenant_id,
                limit=limit,
                offset=offset,
                status=status,
                date_from=date_from,
                date_to=date_to,
            )
            return result

        logger.debug(f"Smart search: query={query!r}, tenant={tenant_id}")

        # Check cache
        cached = await self._cache_get(query, tenant_id, "smart", limit, offset, status, date_from, date_to)
        if cached is not None:
            cached["cached"] = True
            return cached

        # Fetch larger candidate set (bypass cache for internal fast search)
        embedding = self.embedding_provider.embed(query)
        candidates, total = await BugRepository.search(
            conn,
            embedding,
            tenant_id=tenant_id,
            limit=self.smart_candidate_limit,
            offset=0,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )

        # Rerank candidates
        reranked, llm_used = await self.reranker.rerank(
            query, candidates, return_limit=offset + limit
        )

        # Apply pagination to reranked results
        paginated = reranked[offset:offset + limit]

        result = {
            "results": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "mode": "smart" if llm_used else "fast",
            "query": query,
            "cached": False,
        }

        # Only cache if LLM was actually used (not a fallback)
        if llm_used:
            await self._cache_set(query, tenant_id, "smart", limit, offset, status, date_from, date_to, result)

        return result

    async def _cache_get(
        self,
        query: str,
        tenant_id: UUID,
        mode: str,
        limit: int,
        offset: int,
        status: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
    ) -> Optional[dict]:
        """Check cache for search results."""
        if self.cache is None:
            return None

        version = await self.cache.get_tenant_version(tenant_id)
        filters = self._build_filter_dict(status, date_from, date_to, limit, offset)
        query_hash = CacheKeyBuilder.hash_query(query, filters)
        key = CacheKeyBuilder.search_key(tenant_id, query_hash, mode, version)

        return await self.cache.get(key)

    async def _cache_set(
        self,
        query: str,
        tenant_id: UUID,
        mode: str,
        limit: int,
        offset: int,
        status: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        result: dict,
    ) -> None:
        """Store search results in cache."""
        if self.cache is None:
            return

        version = await self.cache.get_tenant_version(tenant_id)
        filters = self._build_filter_dict(status, date_from, date_to, limit, offset)
        query_hash = CacheKeyBuilder.hash_query(query, filters)
        key = CacheKeyBuilder.search_key(tenant_id, query_hash, mode, version)

        ttl = self.cache_ttl_smart if mode == "smart" else self.cache_ttl_fast

        # Convert datetime objects for JSON serialization
        serializable = dict(result)
        serializable["results"] = [
            {
                **r,
                "created_at": r["created_at"].isoformat() if isinstance(r.get("created_at"), datetime) else r.get("created_at"),
            }
            for r in result["results"]
        ]

        await self.cache.set(key, serializable, ttl_seconds=ttl)

    @staticmethod
    def _build_filter_dict(
        status: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        limit: int,
        offset: int,
    ) -> dict:
        """Build deterministic filter dict for cache key hashing."""
        return {
            "status": status,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "limit": limit,
            "offset": offset,
        }
