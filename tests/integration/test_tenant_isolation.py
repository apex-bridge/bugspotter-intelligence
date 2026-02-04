"""Tests for tenant isolation in the data layer"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from bugspotter_intelligence.db.bug_repository import BugRepository


@pytest.fixture
def mock_db_connection():
    """Mock database connection with cursor context manager"""
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn.cursor = MagicMock(return_value=cursor)
    conn.commit = AsyncMock()
    return conn


@pytest.fixture
def tenant_a_id():
    """Tenant A UUID"""
    return uuid4()


@pytest.fixture
def tenant_b_id():
    """Tenant B UUID"""
    return uuid4()


class TestBugRepositoryTenantIsolation:
    """Test suite for tenant isolation in BugRepository"""

    @pytest.mark.asyncio
    async def test_insert_bug_includes_tenant_id(self, mock_db_connection, tenant_a_id):
        """Should include tenant_id when inserting bug"""
        cursor = mock_db_connection.cursor.return_value

        await BugRepository.insert_bug(
            conn=mock_db_connection,
            bug_id="bug-001",
            title="Test Bug",
            description="Test description",
            embedding=[0.1] * 384,
            tenant_id=tenant_a_id,
        )

        # Verify execute was called with tenant_id
        call_args = cursor.__aenter__.return_value.execute.call_args
        assert tenant_a_id in call_args[0][1]  # tenant_id in params
        mock_db_connection.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_similar_filters_by_tenant(self, mock_db_connection, tenant_a_id):
        """Should filter similar bugs by tenant_id"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchall = AsyncMock(return_value=[])

        await BugRepository.find_similar(
            conn=mock_db_connection,
            embedding=[0.1] * 384,
            tenant_id=tenant_a_id,
        )

        # Verify query includes tenant filter
        call_args = cursor.__aenter__.return_value.execute.call_args
        query = call_args[0][0]
        assert "tenant_id" in query

    @pytest.mark.asyncio
    async def test_find_similar_includes_legacy_bugs(self, mock_db_connection, tenant_a_id):
        """Should include bugs with NULL tenant_id (legacy data)"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchall = AsyncMock(return_value=[])

        await BugRepository.find_similar(
            conn=mock_db_connection,
            embedding=[0.1] * 384,
            tenant_id=tenant_a_id,
        )

        # Verify query includes OR tenant_id IS NULL
        call_args = cursor.__aenter__.return_value.execute.call_args
        query = call_args[0][0]
        assert "tenant_id IS NULL" in query

    @pytest.mark.asyncio
    async def test_get_bug_filters_by_tenant(self, mock_db_connection, tenant_a_id):
        """Should filter get_bug by tenant_id"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchone = AsyncMock(return_value=None)

        await BugRepository.get_bug(
            conn=mock_db_connection,
            bug_id="bug-001",
            tenant_id=tenant_a_id,
        )

        # Verify query includes tenant filter
        call_args = cursor.__aenter__.return_value.execute.call_args
        query = call_args[0][0]
        assert "tenant_id" in query

    @pytest.mark.asyncio
    async def test_get_bug_without_tenant_returns_any(self, mock_db_connection):
        """Should return any bug when no tenant_id provided"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchone = AsyncMock(
            return_value=(
                "bug-001",
                "Test Bug",
                "Description",
                "open",
                None,
                None,
                datetime.now(),
                datetime.now(),
                None,  # tenant_id
            )
        )

        result = await BugRepository.get_bug(
            conn=mock_db_connection,
            bug_id="bug-001",
            tenant_id=None,
        )

        assert result is not None
        assert result["bug_id"] == "bug-001"

    @pytest.mark.asyncio
    async def test_update_resolution_filters_by_tenant(
        self, mock_db_connection, tenant_a_id
    ):
        """Should only update bug owned by tenant"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.rowcount = 1

        result = await BugRepository.update_resolution(
            conn=mock_db_connection,
            bug_id="bug-001",
            resolution="Fixed the issue",
            tenant_id=tenant_a_id,
        )

        assert result is True
        # Verify query includes tenant filter
        call_args = cursor.__aenter__.return_value.execute.call_args
        query = call_args[0][0]
        assert "tenant_id" in query

    @pytest.mark.asyncio
    async def test_update_resolution_returns_false_for_wrong_tenant(
        self, mock_db_connection, tenant_b_id
    ):
        """Should return False when bug not owned by tenant"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.rowcount = 0  # No rows updated

        result = await BugRepository.update_resolution(
            conn=mock_db_connection,
            bug_id="bug-001",
            resolution="Fixed the issue",
            tenant_id=tenant_b_id,  # Different tenant
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_find_similar_excludes_specified_bug(self, mock_db_connection):
        """Should exclude specified bug_id from results"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchall = AsyncMock(return_value=[])

        await BugRepository.find_similar(
            conn=mock_db_connection,
            embedding=[0.1] * 384,
            exclude_bug_id="bug-001",
        )

        # Verify query excludes the bug
        call_args = cursor.__aenter__.return_value.execute.call_args
        query = call_args[0][0]
        assert "bug_id !=" in query
