"""Tests for API Key repository"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from bugspotter_intelligence.auth.models import APIKey
from bugspotter_intelligence.auth.repository import APIKeyRepository


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
def sample_api_key_row():
    """Sample row returned from database (as dict for dict_row compatibility)"""
    return {
        "id": uuid4(),
        "tenant_id": uuid4(),
        "key_prefix": "bsi_abc12345",
        "name": "Test Key",
        "created_at": datetime.now(),
        "last_used_at": None,
        "revoked_at": None,
        "rate_limit_per_minute": 60,
        "is_admin": False,
    }


class TestAPIKeyRepository:
    """Test suite for APIKeyRepository"""

    @pytest.mark.asyncio
    async def test_insert_key_creates_api_key(
        self, mock_db_connection, sample_api_key_row
    ):
        """Should insert API key and return APIKey object"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchone = AsyncMock(
            return_value=sample_api_key_row
        )

        result = await APIKeyRepository.insert_key(
            conn=mock_db_connection,
            tenant_id=sample_api_key_row["tenant_id"],
            key_hash="abc123hash",
            key_prefix="bsi_abc12345",
            name="Test Key",
            rate_limit_per_minute=60,
            is_admin=False,
        )

        assert isinstance(result, APIKey)
        assert result.name == "Test Key"
        assert result.key_prefix == "bsi_abc12345"
        assert result.is_active is True
        mock_db_connection.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_hash_returns_api_key(
        self, mock_db_connection, sample_api_key_row
    ):
        """Should return APIKey when found by hash"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchone = AsyncMock(
            return_value=sample_api_key_row
        )

        result = await APIKeyRepository.get_by_hash(
            conn=mock_db_connection, key_hash="abc123hash"
        )

        assert isinstance(result, APIKey)
        assert result.tenant_id == sample_api_key_row["tenant_id"]

    @pytest.mark.asyncio
    async def test_get_by_hash_returns_none_when_not_found(self, mock_db_connection):
        """Should return None when key not found"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchone = AsyncMock(return_value=None)

        result = await APIKeyRepository.get_by_hash(
            conn=mock_db_connection, key_hash="nonexistent"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_tenant_returns_keys(
        self, mock_db_connection, sample_api_key_row
    ):
        """Should return list of APIKeys for tenant"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchall = AsyncMock(
            return_value=[sample_api_key_row, sample_api_key_row]
        )

        tenant_id = uuid4()
        result = await APIKeyRepository.list_by_tenant(
            conn=mock_db_connection, tenant_id=tenant_id
        )

        assert len(result) == 2
        assert all(isinstance(k, APIKey) for k in result)

    @pytest.mark.asyncio
    async def test_list_by_tenant_returns_empty_list(self, mock_db_connection):
        """Should return empty list when tenant has no keys"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchall = AsyncMock(return_value=[])

        result = await APIKeyRepository.list_by_tenant(
            conn=mock_db_connection, tenant_id=uuid4()
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_update_last_used_updates_timestamp(self, mock_db_connection):
        """Should update last_used_at timestamp"""
        key_id = uuid4()

        await APIKeyRepository.update_last_used(
            conn=mock_db_connection, key_id=key_id
        )

        mock_db_connection.commit.assert_called_once()
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_key_returns_true_on_success(self, mock_db_connection):
        """Should return True when key is successfully revoked"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.rowcount = 1

        result = await APIKeyRepository.revoke_key(
            conn=mock_db_connection, key_id=uuid4(), tenant_id=uuid4()
        )

        assert result is True
        mock_db_connection.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_key_returns_false_when_not_found(self, mock_db_connection):
        """Should return False when key not found or not owned"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.rowcount = 0

        result = await APIKeyRepository.revoke_key(
            conn=mock_db_connection, key_id=uuid4(), tenant_id=uuid4()
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_by_id_returns_api_key(
        self, mock_db_connection, sample_api_key_row
    ):
        """Should return APIKey when found by ID and tenant"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchone = AsyncMock(
            return_value=sample_api_key_row
        )

        result = await APIKeyRepository.get_by_id(
            conn=mock_db_connection,
            key_id=sample_api_key_row["id"],
            tenant_id=sample_api_key_row["tenant_id"],
        )

        assert isinstance(result, APIKey)

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_for_wrong_tenant(self, mock_db_connection):
        """Should return None when tenant doesn't own the key"""
        cursor = mock_db_connection.cursor.return_value
        cursor.__aenter__.return_value.fetchone = AsyncMock(return_value=None)

        result = await APIKeyRepository.get_by_id(
            conn=mock_db_connection, key_id=uuid4(), tenant_id=uuid4()
        )

        assert result is None
