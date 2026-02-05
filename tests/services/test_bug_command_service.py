"""Tests for BugCommandService"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from bugspotter_intelligence.services.bug_command_service import BugCommandService


class TestBugCommandService:
    """Test suite for BugCommandService"""

    @pytest.fixture
    def command_service(self, mock_llm_provider, mock_embedding_provider):
        """Create BugCommandService instance"""
        return BugCommandService(mock_llm_provider, mock_embedding_provider)

    @pytest.mark.asyncio
    async def test_analyze_and_store_bug_basic(
            self,
            command_service,
            mock_db_connection,
            mock_embedding_provider
    ):
        """Should analyze and store a basic bug"""
        # Mock the repository
        with patch.object(command_service.repo, 'insert_bug', new_callable=AsyncMock) as mock_insert:
            result = await command_service.analyze_and_store_bug(
                conn=mock_db_connection,
                bug_id="bug-001",
                title="Login crashes",
                description="App crashes on login"
            )

            # Should return success
            assert result["bug_id"] == "bug-001"
            assert result["embedding_generated"] is True
            assert "embedding_text" in result

            # Should have called embedding service
            mock_embedding_provider.embed.assert_called_once()

            # Should have stored in database
            mock_insert.assert_called_once()
            call_args = mock_insert.call_args
            assert call_args.kwargs["bug_id"] == "bug-001"
            assert call_args.kwargs["title"] == "Login crashes"

    @pytest.mark.asyncio
    async def test_analyze_with_console_logs(
            self,
            command_service,
            mock_db_connection,
            mock_embedding_provider
    ):
        """Should include console logs in embedding text"""
        console_logs = [
            {"level": "error", "message": "TypeError: null reference"}
        ]

        with patch.object(command_service.repo, 'insert_bug', new_callable=AsyncMock):
            await command_service.analyze_and_store_bug(
                conn=mock_db_connection,
                bug_id="bug-002",
                title="Login error",
                console_logs=console_logs
            )

            # Should have called embed with text containing error
            called_text = mock_embedding_provider.embed.call_args[0][0]
            assert "TypeError" in called_text

    @pytest.mark.asyncio
    async def test_analyze_with_network_logs(
            self,
            command_service,
            mock_db_connection,
            mock_embedding_provider
    ):
        """Should include network errors in embedding text"""
        network_logs = [
            {"url": "/api/login", "method": "POST", "status": 500, "duration": 123}
        ]

        with patch.object(command_service.repo, 'insert_bug', new_callable=AsyncMock):
            await command_service.analyze_and_store_bug(
                conn=mock_db_connection,
                bug_id="bug-003",
                title="API error",
                network_logs=network_logs
            )

            # Should have called embed with text containing API error
            called_text = mock_embedding_provider.embed.call_args[0][0]
            assert "/api/login" in called_text
            assert "500" in called_text

    @pytest.mark.asyncio
    async def test_analyze_with_all_metadata(
            self,
            command_service,
            mock_db_connection,
            mock_embedding_provider
    ):
        """Should build comprehensive embedding text from all inputs"""
        with patch.object(command_service.repo, 'insert_bug', new_callable=AsyncMock):
            await command_service.analyze_and_store_bug(
                conn=mock_db_connection,
                bug_id="bug-004",
                title="Complete bug report",
                description="Full description",
                console_logs=[{"level": "error", "message": "Error message"}],
                network_logs=[{"url": "/api/test", "method": "GET", "status": 404, "duration": 50}],
                metadata={"browser": "Chrome", "os": "Windows"}
            )

            called_text = mock_embedding_provider.embed.call_args[0][0]
            assert "Complete bug report" in called_text
            assert "Full description" in called_text
            assert "Error message" in called_text
            assert "404" in called_text
            assert "Chrome" in called_text

    @pytest.mark.asyncio
    async def test_update_bug_resolution(
            self,
            command_service,
            mock_db_connection,
            mock_llm_provider
    ):
        """Should update bug resolution and generate summary"""
        with patch.object(command_service.repo, 'update_resolution', new_callable=AsyncMock) as mock_update:
            result = await command_service.update_bug_resolution(
                conn=mock_db_connection,
                bug_id="bug-005",
                resolution="Added null check in AuthService.java:42",
                status="resolved"
            )

            assert result["bug_id"] == "bug-005"
            assert result["status"] == "resolved"
            assert "resolution_summary" in result

            # Should have called LLM to generate summary
            mock_llm_provider.generate.assert_called_once()

            # Should have updated database
            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_stored_correctly(
            self,
            command_service,
            mock_db_connection,
            mock_embedding_provider
    ):
        """Should store the exact embedding returned by provider"""
        mock_embedding = [0.5] * 384
        mock_embedding_provider.embed.return_value = mock_embedding

        with patch.object(command_service.repo, 'insert_bug', new_callable=AsyncMock) as mock_insert:
            await command_service.analyze_and_store_bug(
                conn=mock_db_connection,
                bug_id="bug-006",
                title="Test bug"
            )

            # Should store the exact embedding
            stored_embedding = mock_insert.call_args.kwargs["embedding"]
            assert stored_embedding == mock_embedding


class TestBugCommandServiceCacheIntegration:
    """Tests for cache invalidation in BugCommandService"""

    @pytest.fixture
    def cached_command_service(
        self, mock_llm_provider, mock_embedding_provider, mock_cache_service
    ):
        """Create BugCommandService with cache"""
        return BugCommandService(
            mock_llm_provider, mock_embedding_provider, cache=mock_cache_service
        )

    @pytest.mark.asyncio
    async def test_invalidates_tenant_cache_on_bug_submission(
        self, cached_command_service, mock_db_connection, mock_cache_service
    ):
        """Should invalidate tenant cache when a new bug is stored"""
        tid = uuid4()

        with patch.object(
            cached_command_service.repo, "insert_bug", new_callable=AsyncMock
        ):
            await cached_command_service.analyze_and_store_bug(
                conn=mock_db_connection,
                bug_id="bug-100",
                title="New bug",
                tenant_id=tid,
            )

        mock_cache_service.invalidate_tenant.assert_called_once_with(tid)

    @pytest.mark.asyncio
    async def test_no_invalidation_without_tenant_id(
        self, cached_command_service, mock_db_connection, mock_cache_service
    ):
        """Should not invalidate when no tenant_id is provided"""
        with patch.object(
            cached_command_service.repo, "insert_bug", new_callable=AsyncMock
        ):
            await cached_command_service.analyze_and_store_bug(
                conn=mock_db_connection,
                bug_id="bug-101",
                title="Legacy bug",
                tenant_id=None,
            )

        mock_cache_service.invalidate_tenant.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_invalidation_without_cache(
        self, mock_llm_provider, mock_embedding_provider, mock_db_connection
    ):
        """Should work normally when cache is None"""
        service = BugCommandService(mock_llm_provider, mock_embedding_provider)

        with patch.object(
            service.repo, "insert_bug", new_callable=AsyncMock
        ):
            result = await service.analyze_and_store_bug(
                conn=mock_db_connection,
                bug_id="bug-102",
                title="No cache bug",
            )

        assert result["bug_id"] == "bug-102"
    @pytest.mark.asyncio
    async def test_invalidates_cache_on_resolution_update(
        self, cached_command_service, mock_db_connection, mock_cache_service
    ):
        """Should invalidate tenant cache when bug resolution is updated"""
        tid = uuid4()

        with patch.object(
            cached_command_service.repo, "update_resolution", new_callable=AsyncMock
        ) as mock_update:
            mock_update.return_value = True

            await cached_command_service.update_bug_resolution(
                conn=mock_db_connection,
                bug_id="bug-200",
                resolution="Fixed by adding null check",
                status="resolved",
                tenant_id=tid,
            )

        # Cache should be invalidated when status changes
        mock_cache_service.invalidate_tenant.assert_called_once_with(tid)

    @pytest.mark.asyncio
    async def test_no_invalidation_on_resolution_without_tenant(
        self, cached_command_service, mock_db_connection, mock_cache_service
    ):
        """Should not invalidate cache when no tenant_id on resolution update"""
        with patch.object(
            cached_command_service.repo, "update_resolution", new_callable=AsyncMock
        ) as mock_update:
            mock_update.return_value = True

            await cached_command_service.update_bug_resolution(
                conn=mock_db_connection,
                bug_id="bug-201",
                resolution="Fixed",
                tenant_id=None,
            )

        mock_cache_service.invalidate_tenant.assert_not_called()
