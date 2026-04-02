"""Tests for BugQueryService"""

import json
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

    # ========================================================================
    # enrich_bug tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_enrich_bug_valid_json(self, query_service, mock_llm_provider):
        """Should parse valid JSON from LLM and return enrichment data"""
        mock_llm_provider.generate.return_value = json.dumps({
            "category": "crash",
            "severity": "high",
            "root_cause": "Null pointer in payment processor",
            "components": ["CheckoutForm", "PaymentService"],
            "tags": ["null-pointer", "crash"],
        })

        result = await query_service.enrich_bug(
            bug_id="bug-001",
            title="Checkout crashes on submit",
            description="TypeError when clicking pay",
        )

        assert result["bug_id"] == "bug-001"
        assert result["category"] == "crash"
        assert result["suggested_severity"] == "high"
        assert result["root_cause_summary"] == "Null pointer in payment processor"
        assert "CheckoutForm" in result["affected_components"]
        assert "null-pointer" in result["tags"]
        assert result["confidence"]["category"] == 0.75

    @pytest.mark.asyncio
    async def test_enrich_bug_markdown_wrapped_json(self, query_service, mock_llm_provider):
        """Should extract JSON from markdown code blocks"""
        mock_llm_provider.generate.return_value = """Here is the analysis:

```json
{
  "category": "api",
  "severity": "medium",
  "root_cause": "API timeout on slow network",
  "components": ["APIClient"],
  "tags": ["timeout"]
}
```"""

        result = await query_service.enrich_bug(
            bug_id="bug-002",
            title="API call fails intermittently",
        )

        assert result["category"] == "api"
        assert result["suggested_severity"] == "medium"
        assert result["confidence"]["category"] == 0.75

    @pytest.mark.asyncio
    async def test_enrich_bug_invalid_json_returns_defaults(self, query_service, mock_llm_provider):
        """Should return defaults when LLM returns unparseable output"""
        mock_llm_provider.generate.return_value = "I cannot analyze this bug report."

        result = await query_service.enrich_bug(
            bug_id="bug-003",
            title="Some bug",
        )

        assert result["category"] == "other"
        assert result["suggested_severity"] == "medium"
        assert result["confidence"]["category"] == 0.2

    @pytest.mark.asyncio
    async def test_enrich_bug_filters_placeholder_values(self, query_service, mock_llm_provider):
        """Should filter out placeholder template values from LLM"""
        mock_llm_provider.generate.return_value = json.dumps({
            "category": "ui",
            "severity": "low",
            "root_cause": "<1-2 sentence summary>",
            "components": ["<affected component names>"],
            "tags": ["<descriptive tags>"],
        })

        result = await query_service.enrich_bug(
            bug_id="bug-004",
            title="Button misaligned",
        )

        assert result["category"] == "ui"
        assert result["affected_components"] == []  # Placeholder filtered
        assert result["tags"] == []  # Placeholder filtered
        assert result["confidence"]["root_cause"] == 0.3  # Low confidence

    @pytest.mark.asyncio
    async def test_enrich_bug_takes_last_json_block(self, query_service, mock_llm_provider):
        """Should use the last JSON block when LLM echoes the template"""
        template_echo = json.dumps({
            "category": "<one of: ui, api>",
            "severity": "<one of: critical, high>",
            "root_cause": "<summary>",
            "components": ["<names>"],
            "tags": ["<tags>"],
        })
        actual_response = json.dumps({
            "category": "performance",
            "severity": "high",
            "root_cause": "Database query N+1 in user list",
            "components": ["UserService"],
            "tags": ["n-plus-one", "performance"],
        })
        mock_llm_provider.generate.return_value = (
            f"Template: {template_echo}\n\nActual: {actual_response}"
        )

        result = await query_service.enrich_bug(
            bug_id="bug-005",
            title="User list loads slowly",
        )

        assert result["category"] == "performance"
        assert "UserService" in result["affected_components"]

    # ========================================================================
    # _parse_enrichment_response unit tests
    # ========================================================================

    def test_parse_normalizes_category(self, query_service):
        """Should normalize unknown categories to 'other'"""
        result = query_service._parse_enrichment_response(
            "bug-1",
            json.dumps({"category": "UNKNOWN_CAT", "severity": "high",
                        "root_cause": "x", "components": [], "tags": []}),
        )
        assert result["category"] == "other"

    def test_parse_normalizes_severity(self, query_service):
        """Should normalize unknown severity to 'medium'"""
        result = query_service._parse_enrichment_response(
            "bug-1",
            json.dumps({"category": "ui", "severity": "EXTREME",
                        "root_cause": "x", "components": [], "tags": []}),
        )
        assert result["suggested_severity"] == "medium"