"""Tests for SearchService"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from bugspotter_intelligence.services.reranker import LLMReranker
from bugspotter_intelligence.services.search_service import SearchService


@pytest.fixture
def search_service(mock_embedding_provider):
    """Create SearchService with mock embedding provider (no reranker)"""
    return SearchService(mock_embedding_provider)


@pytest.fixture
def mock_reranker():
    """Mock LLMReranker"""
    reranker = MagicMock(spec=LLMReranker)
    reranker.rerank = AsyncMock()
    return reranker


@pytest.fixture
def smart_search_service(mock_embedding_provider, mock_reranker):
    """Create SearchService with mock embedding provider and reranker"""
    return SearchService(
        mock_embedding_provider, reranker=mock_reranker, smart_candidate_limit=20
    )


@pytest.fixture
def sample_search_results():
    """Sample search results from repository"""
    return [
        {
            "bug_id": "bug-001",
            "title": "Login page crash",
            "description": "App crashes on login",
            "status": "open",
            "resolution": None,
            "similarity": 0.92,
            "created_at": datetime(2025, 1, 15, 10, 0, 0),
        },
        {
            "bug_id": "bug-002",
            "title": "Auth timeout error",
            "description": "Session expires too quickly",
            "status": "resolved",
            "resolution": "Increased session TTL",
            "similarity": 0.78,
            "created_at": datetime(2025, 1, 10, 8, 30, 0),
        },
    ]


class TestSearchFast:
    """Tests for SearchService.search_fast"""

    @pytest.mark.asyncio
    async def test_generates_embedding_from_query(
        self, search_service, mock_embedding_provider, mock_db_connection
    ):
        """Should embed the query text using the embedding provider"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            await search_service.search_fast(
                mock_db_connection, "login crash", tenant_id=tid
            )

        mock_embedding_provider.embed.assert_called_once_with("login crash")

    @pytest.mark.asyncio
    async def test_passes_embedding_to_repository(
        self, search_service, mock_embedding_provider, mock_db_connection
    ):
        """Should pass the generated embedding to the repository"""
        tid = uuid4()
        expected_embedding = [0.1] * 384

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await search_service.search_fast(
                mock_db_connection, "login crash", tenant_id=tid
            )

        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args.args[1] == expected_embedding

    @pytest.mark.asyncio
    async def test_returns_results_with_metadata(
        self, search_service, mock_db_connection, sample_search_results
    ):
        """Should return results with pagination and mode metadata"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(sample_search_results, 2),
        ):
            result = await search_service.search_fast(
                mock_db_connection,
                "login crash",
                tenant_id=tid,
                limit=10,
                offset=0,
            )

        assert result["results"] == sample_search_results
        assert result["total"] == 2
        assert result["limit"] == 10
        assert result["offset"] == 0
        assert result["mode"] == "fast"
        assert result["query"] == "login crash"
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_passes_tenant_id_to_repository(
        self, search_service, mock_db_connection
    ):
        """Should pass tenant_id for isolation"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await search_service.search_fast(
                mock_db_connection, "query", tenant_id=tid
            )

        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["tenant_id"] == tid

    @pytest.mark.asyncio
    async def test_passes_status_filter(self, search_service, mock_db_connection):
        """Should pass status filter to repository"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await search_service.search_fast(
                mock_db_connection, "query", tenant_id=tid, status="open"
            )

        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["status"] == "open"

    @pytest.mark.asyncio
    async def test_passes_date_filters(self, search_service, mock_db_connection):
        """Should pass date_from and date_to filters to repository"""
        tid = uuid4()
        date_from = datetime(2025, 1, 1)
        date_to = datetime(2025, 6, 30)

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await search_service.search_fast(
                mock_db_connection,
                "query",
                tenant_id=tid,
                date_from=date_from,
                date_to=date_to,
            )

        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["date_from"] == date_from
        assert call_kwargs["date_to"] == date_to

    @pytest.mark.asyncio
    async def test_passes_pagination_params(self, search_service, mock_db_connection):
        """Should pass limit and offset to repository"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await search_service.search_fast(
                mock_db_connection, "query", tenant_id=tid, limit=25, offset=50
            )

        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["limit"] == 25
        assert call_kwargs["offset"] == 50

    @pytest.mark.asyncio
    async def test_empty_results(self, search_service, mock_db_connection):
        """Should handle empty results gracefully"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            result = await search_service.search_fast(
                mock_db_connection, "nonexistent bug", tenant_id=tid
            )

        assert result["results"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_default_pagination(self, search_service, mock_db_connection):
        """Should use default limit=10 and offset=0"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await search_service.search_fast(
                mock_db_connection, "query", tenant_id=tid
            )

        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 0

    @pytest.mark.asyncio
    async def test_none_filters_by_default(self, search_service, mock_db_connection):
        """Should pass None for optional filters when not provided"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await search_service.search_fast(
                mock_db_connection, "query", tenant_id=tid
            )

        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["status"] is None
        assert call_kwargs["date_from"] is None
        assert call_kwargs["date_to"] is None


class TestSearchSmart:
    """Tests for SearchService.search_smart"""

    @pytest.mark.asyncio
    async def test_falls_back_to_fast_when_no_reranker(
        self, search_service, mock_db_connection
    ):
        """Should fall back to fast search when reranker is None"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            result = await search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid
            )

        # Falls back to fast mode since reranker is None
        assert result["mode"] == "fast"

    @pytest.mark.asyncio
    async def test_fetches_larger_candidate_set(
        self, smart_search_service, mock_reranker, mock_db_connection
    ):
        """Should fetch smart_candidate_limit candidates for reranking"""
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(return_value=([], True))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid, limit=5
            )

        # Should fetch 20 candidates (smart_candidate_limit), not 5
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["limit"] == 20

    @pytest.mark.asyncio
    async def test_passes_candidates_to_reranker(
        self,
        smart_search_service,
        mock_reranker,
        mock_db_connection,
        sample_search_results,
    ):
        """Should pass fast search results to the reranker"""
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(return_value=(sample_search_results, True))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(sample_search_results, 2),
        ):
            await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid, limit=5
            )

        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        assert call_args.args[0] == "query"
        assert call_args.args[1] == sample_search_results

    @pytest.mark.asyncio
    async def test_mode_is_smart_when_llm_used(
        self, smart_search_service, mock_reranker, mock_db_connection
    ):
        """Should report mode=smart when LLM reranking succeeded"""
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(return_value=([], True))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid
            )

        assert result["mode"] == "smart"

    @pytest.mark.asyncio
    async def test_mode_is_fast_when_llm_fallback(
        self, smart_search_service, mock_reranker, mock_db_connection
    ):
        """Should report mode=fast when LLM reranking fell back"""
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(return_value=([], False))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid
            )

        assert result["mode"] == "fast"

    @pytest.mark.asyncio
    async def test_paginates_reranked_results(
        self, smart_search_service, mock_reranker, mock_db_connection
    ):
        """Should apply offset/limit pagination to reranked results"""
        tid = uuid4()
        reranked = [
            {"bug_id": f"bug-{i}", "similarity": 0.9 - i * 0.1}
            for i in range(10)
        ]
        mock_reranker.rerank = AsyncMock(return_value=(reranked, True))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(reranked, 50),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid, limit=3, offset=2
            )

        # Should get items at index 2, 3, 4 from reranked results
        assert len(result["results"]) == 3
        assert result["results"][0]["bug_id"] == "bug-2"
        assert result["results"][1]["bug_id"] == "bug-3"
        assert result["results"][2]["bug_id"] == "bug-4"

    @pytest.mark.asyncio
    async def test_preserves_total_from_fast_search(
        self, smart_search_service, mock_reranker, mock_db_connection
    ):
        """Should preserve the total count from the initial fast search"""
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(return_value=([], True))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 42),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid
            )

        assert result["total"] == 42


class TestSearchCacheIntegration:
    """Tests for cache integration in SearchService"""

    @pytest.fixture
    def cached_search_service(self, mock_embedding_provider, mock_cache_service):
        """Create SearchService with cache enabled"""
        return SearchService(
            mock_embedding_provider,
            cache=mock_cache_service,
            cache_ttl_fast=300,
            cache_ttl_smart=900,
        )

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(
        self, cached_search_service, mock_cache_service, mock_db_connection
    ):
        """Should return cached result with cached=True on cache hit"""
        tid = uuid4()
        cached_data = {
            "results": [{"bug_id": "bug-001", "similarity": 0.9}],
            "total": 1,
            "limit": 10,
            "offset": 0,
            "mode": "fast",
            "query": "test query",
            "cached": False,
        }
        mock_cache_service.get = AsyncMock(return_value=cached_data)

        result = await cached_search_service.search_fast(
            mock_db_connection, "test query", tenant_id=tid
        )

        assert result["cached"] is True
        assert result["results"] == cached_data["results"]

    @pytest.mark.asyncio
    async def test_cache_miss_queries_database(
        self, cached_search_service, mock_cache_service, mock_db_connection
    ):
        """Should query database on cache miss"""
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            result = await cached_search_service.search_fast(
                mock_db_connection, "test query", tenant_id=tid
            )

        mock_search.assert_called_once()
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_cache_miss_populates_cache(
        self, cached_search_service, mock_cache_service, mock_db_connection
    ):
        """Should store result in cache after database query"""
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            await cached_search_service.search_fast(
                mock_db_connection, "test query", tenant_id=tid
            )

        mock_cache_service.set.assert_called_once()
        call_kwargs = mock_cache_service.set.call_args
        assert call_kwargs.kwargs["ttl_seconds"] == 300

    @pytest.mark.asyncio
    async def test_works_without_cache(self, search_service, mock_db_connection):
        """Should work normally when cache is None"""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            result = await search_service.search_fast(
                mock_db_connection, "test query", tenant_id=tid
            )

        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_cache_uses_tenant_version(
        self, cached_search_service, mock_cache_service, mock_db_connection
    ):
        """Should include tenant version in cache key lookup"""
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)
        mock_cache_service.get_tenant_version = AsyncMock(return_value=3)

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            await cached_search_service.search_fast(
                mock_db_connection, "test query", tenant_id=tid
            )

        # Should have called get_tenant_version for both get and set
        assert mock_cache_service.get_tenant_version.call_count == 2
