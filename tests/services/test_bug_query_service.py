"""Tests for BugQueryService"""

from unittest.mock import AsyncMock, patch

import pytest

from bugspotter_intelligence.services.bug_query_service import BugQueryService


class TestBugQueryService:
    """Test suite for BugQueryService"""

    @pytest.fixture
    def query_service(self, mock_settings, mock_llm_provider, mock_embedding_provider):
        """Create BugQueryService instance"""
        return BugQueryService(mock_settings, mock_llm_provider, mock_embedding_provider)

    @pytest.mark.asyncio
    async def test_get_bug_found(self, query_service, mock_db_connection):
        """Should return bug when found"""
        mock_bug = {
            "bug_id": "bug-001",
            "title": "Test bug",
            "description": "Test description"
        }

        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value=mock_bug):
            result = await query_service.get_bug(mock_db_connection, "bug-001")

            assert result == mock_bug

    @pytest.mark.asyncio
    async def test_get_bug_not_found(self, query_service, mock_db_connection):
        """Should return None when bug not found"""
        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value=None):
            result = await query_service.get_bug(mock_db_connection, "nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_find_similar_bugs_uses_default_threshold(
            self,
            query_service,
            mock_db_connection,
            mock_settings
    ):
        """Should use default similarity threshold from settings"""
        # Mock database responses
        cursor = mock_db_connection.cursor.return_value.__aenter__.return_value
        cursor.fetchone.return_value = ([0.1] * 384,)  # Mock embedding

        mock_similar = [
            {"bug_id": "bug-002", "title": "Similar bug", "similarity": 0.85}
        ]

        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value={"bug_id": "bug-001"}):
            with patch.object(query_service.repo, 'find_similar', new_callable=AsyncMock, return_value=mock_similar):
                result = await query_service.find_similar_bugs(
                    conn=mock_db_connection,
                    bug_id="bug-001"
                )

                # Should use default threshold
                assert result["threshold_used"] == mock_settings.similarity_threshold

    @pytest.mark.asyncio
    async def test_find_similar_bugs_with_override_threshold(
            self,
            query_service,
            mock_db_connection
    ):
        """Should use provided threshold when overridden"""
        cursor = mock_db_connection.cursor.return_value.__aenter__.return_value
        cursor.fetchone.return_value = ([0.1] * 384,)

        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value={"bug_id": "bug-001"}):
            with patch.object(query_service.repo, 'find_similar', new_callable=AsyncMock, return_value=[]):
                result = await query_service.find_similar_bugs(
                    conn=mock_db_connection,
                    bug_id="bug-001",
                    similarity_threshold=0.95
                )

                assert result["threshold_used"] == 0.95

    @pytest.mark.asyncio
    async def test_find_similar_detects_duplicate(
            self,
            query_service,
            mock_db_connection
    ):
        """Should mark as duplicate when similarity >= duplicate_threshold"""
        cursor = mock_db_connection.cursor.return_value.__aenter__.return_value
        cursor.fetchone.return_value = ([0.1] * 384,)

        # Very similar bug (>= 0.90)
        mock_similar = [
            {"bug_id": "bug-002", "title": "Almost identical", "similarity": 0.95}
        ]

        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value={"bug_id": "bug-001"}):
            with patch.object(query_service.repo, 'find_similar', new_callable=AsyncMock, return_value=mock_similar):
                result = await query_service.find_similar_bugs(
                    conn=mock_db_connection,
                    bug_id="bug-001"
                )

                assert result["is_duplicate"] is True

    @pytest.mark.asyncio
    async def test_find_similar_not_duplicate(
            self,
            query_service,
            mock_db_connection
    ):
        """Should not mark as duplicate when similarity < duplicate_threshold"""
        cursor = mock_db_connection.cursor.return_value.__aenter__.return_value
        cursor.fetchone.return_value = ([0.1] * 384,)

        # Somewhat similar but not duplicate (< 0.90)
        mock_similar = [
            {"bug_id": "bug-002", "title": "Related bug", "similarity": 0.80}
        ]

        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value={"bug_id": "bug-001"}):
            with patch.object(query_service.repo, 'find_similar', new_callable=AsyncMock, return_value=mock_similar):
                result = await query_service.find_similar_bugs(
                    conn=mock_db_connection,
                    bug_id="bug-001"
                )

                assert result["is_duplicate"] is False

    @pytest.mark.asyncio
    async def test_find_similar_passes_exclude_bug_id(
            self,
            query_service,
            mock_db_connection
    ):
        """Should pass exclude_bug_id to repository to filter out self"""
        cursor = mock_db_connection.cursor.return_value.__aenter__.return_value
        cursor.fetchone.return_value = ([0.1] * 384,)

        # Repo returns only other bugs (already filtered by exclude_bug_id)
        mock_similar = [
            {"bug_id": "bug-002", "title": "Similar", "similarity": 0.85}
        ]

        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value={"bug_id": "bug-001"}):
            with patch.object(query_service.repo, 'find_similar', new_callable=AsyncMock, return_value=mock_similar) as mock_find:
                result = await query_service.find_similar_bugs(
                    conn=mock_db_connection,
                    bug_id="bug-001"
                )

                # Verify exclude_bug_id was passed to repo
                mock_find.assert_called_once()
                call_kwargs = mock_find.call_args.kwargs
                assert call_kwargs.get("exclude_bug_id") == "bug-001"

                # Verify results
                bug_ids = [b["bug_id"] for b in result["similar_bugs"]]
                assert "bug-002" in bug_ids

    @pytest.mark.asyncio
    async def test_get_mitigation_with_similar_bugs(
            self,
            query_service,
            mock_db_connection,
            mock_llm_provider
    ):
        """Should use similar bugs with resolutions as context"""
        mock_bug = {"bug_id": "bug-001", "title": "Login error", "description": "Crashes"}

        # Mock similar bugs with resolutions
        mock_similar = {
            "similar_bugs": [
                {
                    "bug_id": "bug-002",
                    "title": "Similar login issue",
                    "resolution": "Added null check"
                }
            ]
        }

        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value=mock_bug):
            with patch.object(query_service, 'find_similar_bugs', new_callable=AsyncMock, return_value=mock_similar):
                result = await query_service.get_mitigation_suggestion(
                    conn=mock_db_connection,
                    bug_id="bug-001",
                    use_similar_bugs=True
                )

                assert result["based_on_similar_bugs"] is True

                # Should have called LLM with context
                mock_llm_provider.generate.assert_called_once()
                call_kwargs = mock_llm_provider.generate.call_args.kwargs
                assert call_kwargs["context"] is not None
                assert len(call_kwargs["context"]) > 0

    @pytest.mark.asyncio
    async def test_get_mitigation_without_similar_bugs(
            self,
            query_service,
            mock_db_connection,
            mock_llm_provider
    ):
        """Should generate mitigation without context when not using similar bugs"""
        mock_bug = {"bug_id": "bug-001", "title": "Error", "description": "Bug"}

        with patch.object(query_service.repo, 'get_bug', new_callable=AsyncMock, return_value=mock_bug):
            result = await query_service.get_mitigation_suggestion(
                conn=mock_db_connection,
                bug_id="bug-001",
                use_similar_bugs=False
            )

            assert result["based_on_similar_bugs"] is False

            # Should have called LLM without context
            call_kwargs = mock_llm_provider.generate.call_args.kwargs
            assert call_kwargs["context"] is None or len(call_kwargs["context"]) == 0