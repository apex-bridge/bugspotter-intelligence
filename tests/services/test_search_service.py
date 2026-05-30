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
        expected_embedding = [0.1] * 1024

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
            await search_service.search_fast(mock_db_connection, "query", tenant_id=tid)

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
            await search_service.search_fast(mock_db_connection, "query", tenant_id=tid)

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
            await search_service.search_fast(mock_db_connection, "query", tenant_id=tid)

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
        mock_reranker.rerank = AsyncMock(return_value=([], True, None))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid, limit=5
            )

        # Should fetch 20 candidates (smart_candidate_limit) when offset+limit < smart_candidate_limit
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["limit"] == 20

    @pytest.mark.asyncio
    async def test_fetches_enough_candidates_for_large_offset(
        self, smart_search_service, mock_reranker, mock_db_connection
    ):
        """Should fetch enough candidates to cover offset + limit when it exceeds smart_candidate_limit"""
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(return_value=([], True, None))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_search:
            # offset=15, limit=10 -> needs 25 total, but smart_candidate_limit=20
            await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid, limit=10, offset=15
            )

        # Should fetch 25 candidates to ensure pagination works
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["limit"] == 25

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
        mock_reranker.rerank = AsyncMock(return_value=(sample_search_results, True, None))

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
    async def test_forwards_tenant_id_and_surfaces_event_id(
        self,
        smart_search_service,
        mock_reranker,
        mock_db_connection,
        sample_search_results,
    ):
        """Smart search must thread tenant_id into rerank and bubble its event_id."""
        from uuid import uuid4 as _uuid4
        tid = uuid4()
        event_id = _uuid4()
        mock_reranker.rerank = AsyncMock(return_value=(sample_search_results, True, event_id))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(sample_search_results, 2),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "query", tenant_id=tid, limit=5
            )

        assert mock_reranker.rerank.call_args.kwargs["tenant_id"] == tid
        assert result["event_id"] == str(event_id)

    @pytest.mark.asyncio
    async def test_mode_is_smart_when_llm_used(
        self, smart_search_service, mock_reranker, mock_db_connection
    ):
        """Should report mode=smart when LLM reranking succeeded"""
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(return_value=([], True, None))

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
        mock_reranker.rerank = AsyncMock(return_value=([], False, None))

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
            {"bug_id": f"bug-{i}", "similarity": 0.9 - i * 0.1} for i in range(10)
        ]
        mock_reranker.rerank = AsyncMock(return_value=(reranked, True, None))

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
        mock_reranker.rerank = AsyncMock(return_value=([], True, None))

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
    async def test_cache_uses_tenant_token(
        self, cached_search_service, mock_cache_service, mock_db_connection
    ):
        """Should include tenant token in cache key lookup"""
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

        get_key = mock_cache_service.get.call_args.args[0]
        set_key = mock_cache_service.set.call_args.args[0]
        assert f":t3:" in get_key
        assert f":t3:" in set_key
        assert str(tid) in get_key
        assert str(tid) in set_key


# ---------------------------------------------------------------------------
# Edge-case coverage. Grouped by what each class surfaces:
#
#   TestSearchContractGaps        — tests that FAIL on current code, exposing
#                                   docstring/contract violations (real bugs).
#   TestSearchDocumentedBehavior  — tests that pass and pin behavior the code
#                                   does today but no test currently asserts.
#   TestSearchEdgeInputs          — defensive coverage for unusual inputs that
#                                   should not blow up.
# ---------------------------------------------------------------------------


class TestSearchContractGaps:
    """
    Tests that EXPOSE real bugs by asserting behavior the docstrings and
    soft-fail conventions promise but the code does not currently deliver.
    Each test failure is a finding, not flaky CI.
    """

    @pytest.fixture
    def cached_smart_service(
        self, mock_embedding_provider, mock_reranker, mock_cache_service
    ):
        return SearchService(
            mock_embedding_provider,
            reranker=mock_reranker,
            smart_candidate_limit=20,
            cache=mock_cache_service,
            cache_ttl_fast=300,
            cache_ttl_smart=900,
        )

    @pytest.mark.asyncio
    async def test_smart_search_falls_back_when_reranker_raises(
        self, smart_search_service, mock_reranker, mock_db_connection,
        sample_search_results,
    ):
        """
        Docstring (line 132): 'Falls back to fast mode if the reranker is
        unavailable OR FAILS.' Today the code only handles the
        unavailable=None case; a raising reranker propagates the exception
        and breaks the request.
        """
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(side_effect=RuntimeError("rerank kaboom"))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(sample_search_results, 2),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "q", tenant_id=tid, limit=5
            )

        # Promised fallback shape: fast results, mode=fast, no event_id.
        assert result["mode"] == "fast"
        assert result["results"] == sample_search_results

    @pytest.mark.asyncio
    async def test_search_fast_soft_fails_on_cache_get_error(
        self, mock_embedding_provider, mock_cache_service, mock_db_connection,
        sample_search_results,
    ):
        """
        Cache is opt-in and best-effort: a failing cache backend must NOT
        break the search. Today cache.get's exception propagates.
        """
        service = SearchService(mock_embedding_provider, cache=mock_cache_service)
        tid = uuid4()
        mock_cache_service.get = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(sample_search_results, 2),
        ):
            result = await service.search_fast(
                mock_db_connection, "q", tenant_id=tid
            )

        assert result["results"] == sample_search_results
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_search_fast_soft_fails_on_cache_set_error(
        self, mock_embedding_provider, mock_cache_service, mock_db_connection,
        sample_search_results,
    ):
        """
        Mirror of the above for the write path. cache.set throwing today
        means the request returns 5xx even though the DB query succeeded.
        """
        service = SearchService(mock_embedding_provider, cache=mock_cache_service)
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)
        mock_cache_service.set = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(sample_search_results, 2),
        ):
            result = await service.search_fast(
                mock_db_connection, "q", tenant_id=tid
            )

        assert result["results"] == sample_search_results
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_search_fast_soft_fails_on_tenant_version_error(
        self, mock_embedding_provider, mock_cache_service, mock_db_connection,
        sample_search_results,
    ):
        """get_tenant_version is called in BOTH _cache_get and _cache_set.
        Either failure should soft-fail rather than break the request."""
        service = SearchService(mock_embedding_provider, cache=mock_cache_service)
        tid = uuid4()
        mock_cache_service.get_tenant_version = AsyncMock(
            side_effect=ConnectionError("redis down")
        )

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(sample_search_results, 2),
        ):
            result = await service.search_fast(
                mock_db_connection, "q", tenant_id=tid
            )

        assert result["results"] == sample_search_results


class TestSearchDocumentedBehavior:
    """
    Tests that pin behavior the code does today but no existing test
    asserts. These should pass — they exist so future refactors can't
    quietly change semantics that callers (admin UI, public API) depend on.
    """

    @pytest.fixture
    def cached_smart_service(
        self, mock_embedding_provider, mock_reranker, mock_cache_service
    ):
        return SearchService(
            mock_embedding_provider,
            reranker=mock_reranker,
            smart_candidate_limit=20,
            cache=mock_cache_service,
            cache_ttl_fast=300,
            cache_ttl_smart=900,
        )

    @pytest.mark.asyncio
    async def test_cache_hit_sets_event_id_to_none(
        self, mock_embedding_provider, mock_cache_service, mock_db_connection,
    ):
        """Cache hit response must include event_id=None — admin UI relies on
        the field being present (not missing) so it can show 'cached' badges."""
        service = SearchService(mock_embedding_provider, cache=mock_cache_service)
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value={
            "results": [], "total": 0, "limit": 10, "offset": 0,
            "mode": "fast", "query": "q", "cached": False,
        })

        result = await service.search_fast(mock_db_connection, "q", tenant_id=tid)

        assert result["event_id"] is None

    @pytest.mark.asyncio
    async def test_smart_cache_hit_skips_reranker(
        self, cached_smart_service, mock_reranker, mock_cache_service,
        mock_db_connection,
    ):
        """Cache hit on smart path must short-circuit BEFORE the reranker
        call. Otherwise the cache saves nothing — the expensive LLM call
        still happens."""
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value={
            "results": [], "total": 0, "limit": 5, "offset": 0,
            "mode": "smart", "query": "q", "cached": False,
        })

        await cached_smart_service.search_smart(
            mock_db_connection, "q", tenant_id=tid
        )

        mock_reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_smart_does_not_cache_when_llm_fallback(
        self, cached_smart_service, mock_reranker, mock_cache_service,
        mock_db_connection,
    ):
        """If the reranker reported llm_used=False (intentional pass-through),
        the result is the SAME as fast mode — caching under the smart key
        would pollute future smart requests."""
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)
        mock_reranker.rerank = AsyncMock(return_value=([], False, None))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            await cached_smart_service.search_smart(
                mock_db_connection, "q", tenant_id=tid
            )

        mock_cache_service.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_smart_caches_with_smart_ttl_on_llm_success(
        self, cached_smart_service, mock_reranker, mock_cache_service,
        mock_db_connection, sample_search_results,
    ):
        """Smart-mode cache writes must use ttl_smart=900, not ttl_fast=300."""
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)
        mock_reranker.rerank = AsyncMock(
            return_value=(sample_search_results, True, None)
        )

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(sample_search_results, 2),
        ):
            await cached_smart_service.search_smart(
                mock_db_connection, "q", tenant_id=tid, limit=5
            )

        mock_cache_service.set.assert_called_once()
        assert mock_cache_service.set.call_args.kwargs["ttl_seconds"] == 900

    @pytest.mark.asyncio
    async def test_cache_key_changes_with_status_filter(
        self, mock_embedding_provider, mock_cache_service, mock_db_connection,
    ):
        """Status filter must be part of the cache key — otherwise 'open' and
        'resolved' queries collide and return each other's results."""
        service = SearchService(mock_embedding_provider, cache=mock_cache_service)
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            await service.search_fast(
                mock_db_connection, "q", tenant_id=tid, status="open"
            )
            key_open = mock_cache_service.get.call_args.args[0]

            await service.search_fast(
                mock_db_connection, "q", tenant_id=tid, status="resolved"
            )
            key_resolved = mock_cache_service.get.call_args.args[0]

        assert key_open != key_resolved

    @pytest.mark.asyncio
    async def test_cache_key_changes_with_date_filters(
        self, mock_embedding_provider, mock_cache_service, mock_db_connection,
    ):
        """Date range filters must also segment the cache key."""
        service = SearchService(mock_embedding_provider, cache=mock_cache_service)
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            await service.search_fast(
                mock_db_connection, "q", tenant_id=tid,
                date_from=datetime(2025, 1, 1),
            )
            key_jan = mock_cache_service.get.call_args.args[0]

            await service.search_fast(
                mock_db_connection, "q", tenant_id=tid,
                date_from=datetime(2025, 6, 1),
            )
            key_jun = mock_cache_service.get.call_args.args[0]

        assert key_jan != key_jun

    @pytest.mark.asyncio
    async def test_cache_key_changes_with_pagination(
        self, mock_embedding_provider, mock_cache_service, mock_db_connection,
    ):
        """Different pages must cache separately."""
        service = SearchService(mock_embedding_provider, cache=mock_cache_service)
        tid = uuid4()
        mock_cache_service.get = AsyncMock(return_value=None)

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            await service.search_fast(
                mock_db_connection, "q", tenant_id=tid, limit=10, offset=0
            )
            key_p1 = mock_cache_service.get.call_args.args[0]

            await service.search_fast(
                mock_db_connection, "q", tenant_id=tid, limit=10, offset=10
            )
            key_p2 = mock_cache_service.get.call_args.args[0]

        assert key_p1 != key_p2


class TestSearchEdgeInputs:
    """Defensive coverage for inputs that should not crash the service."""

    @pytest.mark.asyncio
    async def test_smart_handles_empty_reranker_output(
        self, smart_search_service, mock_reranker, mock_db_connection,
    ):
        """Reranker returning [] (no candidates passed its filter) must yield
        an empty result list, not a crash."""
        tid = uuid4()
        mock_reranker.rerank = AsyncMock(return_value=([], True, None))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([{"bug_id": "bug-1"}], 1),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "q", tenant_id=tid, limit=5
            )

        assert result["results"] == []
        # total still reflects the DB count, not the reranker's filtering.
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_smart_handles_reranker_returning_fewer_than_requested(
        self, smart_search_service, mock_reranker, mock_db_connection,
    ):
        """If reranker returns fewer items than (offset + limit), slicing
        must not error — Python returns the shorter tail silently."""
        tid = uuid4()
        # 3 items returned, caller asks for offset=2 limit=5 → expect [item3]
        reranked = [{"bug_id": f"bug-{i}"} for i in range(3)]
        mock_reranker.rerank = AsyncMock(return_value=(reranked, True, None))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(reranked, 3),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "q", tenant_id=tid, limit=5, offset=2
            )

        assert len(result["results"]) == 1
        assert result["results"][0]["bug_id"] == "bug-2"

    @pytest.mark.asyncio
    async def test_smart_offset_past_results_returns_empty(
        self, smart_search_service, mock_reranker, mock_db_connection,
    ):
        """Pagination past the end of results must return [] cleanly."""
        tid = uuid4()
        reranked = [{"bug_id": f"bug-{i}"} for i in range(3)]
        mock_reranker.rerank = AsyncMock(return_value=(reranked, True, None))

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=(reranked, 3),
        ):
            result = await smart_search_service.search_smart(
                mock_db_connection, "q", tenant_id=tid, limit=5, offset=100
            )

        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_fast_empty_query_string_does_not_crash(
        self, search_service, mock_embedding_provider, mock_db_connection,
    ):
        """Empty string is a valid argument shape — let the embedding provider
        and repo handle it; service must not pre-validate or crash."""
        tid = uuid4()

        with patch(
            "bugspotter_intelligence.services.search_service.BugRepository.search",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            result = await search_service.search_fast(
                mock_db_connection, "", tenant_id=tid
            )

        assert result["query"] == ""
        assert result["results"] == []
        # Embedding still attempted; the provider decides whether to error.
        mock_embedding_provider.embed.assert_called_once_with("")
