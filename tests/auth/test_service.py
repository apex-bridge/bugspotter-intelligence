"""Tests for API Key service"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from bugspotter_intelligence.auth.models import APIKey, TenantContext
from bugspotter_intelligence.auth.service import APIKeyService


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    conn = AsyncMock()
    return conn


@pytest.fixture
def api_key_service():
    """Create APIKeyService instance"""
    return APIKeyService(key_prefix="test_")


@pytest.fixture
def sample_api_key():
    """Sample APIKey object"""
    return APIKey(
        id=uuid4(),
        tenant_id=uuid4(),
        key_prefix="test_abc123",
        name="Test Key",
        created_at=datetime.now(),
        last_used_at=None,
        revoked_at=None,
        rate_limit_per_minute=60,
        is_admin=False,
    )


@pytest.fixture
def revoked_api_key(sample_api_key):
    """Sample revoked APIKey"""
    return APIKey(
        **{**sample_api_key.model_dump(), "revoked_at": datetime.now()}
    )


class TestAPIKeyService:
    """Test suite for APIKeyService"""

    @pytest.mark.asyncio
    async def test_create_key_returns_api_key_and_plain_key(
        self, api_key_service, mock_db_connection, sample_api_key
    ):
        """Should return both APIKey object and plain key on creation"""
        with patch.object(
            api_key_service.repo, "insert_key", new_callable=AsyncMock
        ) as mock_insert:
            mock_insert.return_value = sample_api_key

            api_key, plain_key = await api_key_service.create_key(
                conn=mock_db_connection,
                tenant_id=sample_api_key.tenant_id,
                name="Test Key",
            )

            assert isinstance(api_key, APIKey)
            assert plain_key.startswith("test_")
            assert len(plain_key) > 20
            mock_insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_key_with_custom_rate_limit(
        self, api_key_service, mock_db_connection, sample_api_key
    ):
        """Should pass custom rate limit to repository"""
        with patch.object(
            api_key_service.repo, "insert_key", new_callable=AsyncMock
        ) as mock_insert:
            mock_insert.return_value = sample_api_key

            await api_key_service.create_key(
                conn=mock_db_connection,
                tenant_id=sample_api_key.tenant_id,
                name="Test Key",
                rate_limit_per_minute=100,
                is_admin=True,
            )

            call_kwargs = mock_insert.call_args.kwargs
            assert call_kwargs["rate_limit_per_minute"] == 100
            assert call_kwargs["is_admin"] is True

    @pytest.mark.asyncio
    async def test_validate_key_returns_tenant_context(
        self, api_key_service, mock_db_connection, sample_api_key
    ):
        """Should return TenantContext for valid key"""
        with patch.object(
            api_key_service.repo, "get_by_hash", new_callable=AsyncMock
        ) as mock_get, patch.object(
            api_key_service.repo, "update_last_used", new_callable=AsyncMock
        ):
            mock_get.return_value = sample_api_key

            result = await api_key_service.validate_key(
                conn=mock_db_connection,
                plain_key="test_valid_key",
            )

            assert isinstance(result, TenantContext)
            assert result.tenant_id == sample_api_key.tenant_id
            assert result.api_key_id == sample_api_key.id
            assert result.is_admin == sample_api_key.is_admin

    @pytest.mark.asyncio
    async def test_validate_key_returns_none_for_invalid_key(
        self, api_key_service, mock_db_connection
    ):
        """Should return None for non-existent key"""
        with patch.object(
            api_key_service.repo, "get_by_hash", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            result = await api_key_service.validate_key(
                conn=mock_db_connection,
                plain_key="test_invalid_key",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_returns_none_for_revoked_key(
        self, api_key_service, mock_db_connection, revoked_api_key
    ):
        """Should return None for revoked key"""
        with patch.object(
            api_key_service.repo, "get_by_hash", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = revoked_api_key

            result = await api_key_service.validate_key(
                conn=mock_db_connection,
                plain_key="test_revoked_key",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_updates_last_used(
        self, api_key_service, mock_db_connection, sample_api_key
    ):
        """Should update last_used_at on successful validation"""
        with patch.object(
            api_key_service.repo, "get_by_hash", new_callable=AsyncMock
        ) as mock_get, patch.object(
            api_key_service.repo, "update_last_used", new_callable=AsyncMock
        ) as mock_update:
            mock_get.return_value = sample_api_key

            await api_key_service.validate_key(
                conn=mock_db_connection,
                plain_key="test_valid_key",
            )

            mock_update.assert_called_once_with(mock_db_connection, sample_api_key.id)

    @pytest.mark.asyncio
    async def test_list_keys_returns_keys(
        self, api_key_service, mock_db_connection, sample_api_key
    ):
        """Should return list of keys for tenant"""
        with patch.object(
            api_key_service.repo, "list_by_tenant", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = [sample_api_key, sample_api_key]

            result = await api_key_service.list_keys(
                conn=mock_db_connection,
                tenant_id=sample_api_key.tenant_id,
            )

            assert len(result) == 2
            mock_list.assert_called_once_with(
                mock_db_connection, sample_api_key.tenant_id
            )

    @pytest.mark.asyncio
    async def test_revoke_key_returns_true_on_success(
        self, api_key_service, mock_db_connection
    ):
        """Should return True when key is revoked"""
        with patch.object(
            api_key_service.repo, "revoke_key", new_callable=AsyncMock
        ) as mock_revoke:
            mock_revoke.return_value = True

            key_id = uuid4()
            tenant_id = uuid4()
            result = await api_key_service.revoke_key(
                conn=mock_db_connection,
                key_id=key_id,
                tenant_id=tenant_id,
            )

            assert result is True
            mock_revoke.assert_called_once_with(mock_db_connection, key_id, tenant_id)

    @pytest.mark.asyncio
    async def test_revoke_key_returns_false_when_not_found(
        self, api_key_service, mock_db_connection
    ):
        """Should return False when key not found"""
        with patch.object(
            api_key_service.repo, "revoke_key", new_callable=AsyncMock
        ) as mock_revoke:
            mock_revoke.return_value = False

            result = await api_key_service.revoke_key(
                conn=mock_db_connection,
                key_id=uuid4(),
                tenant_id=uuid4(),
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_get_key_returns_api_key(
        self, api_key_service, mock_db_connection, sample_api_key
    ):
        """Should return APIKey when found"""
        with patch.object(
            api_key_service.repo, "get_by_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = sample_api_key

            result = await api_key_service.get_key(
                conn=mock_db_connection,
                key_id=sample_api_key.id,
                tenant_id=sample_api_key.tenant_id,
            )

            assert result == sample_api_key

    @pytest.mark.asyncio
    async def test_get_key_returns_none_when_not_found(
        self, api_key_service, mock_db_connection
    ):
        """Should return None when key not found"""
        with patch.object(
            api_key_service.repo, "get_by_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            result = await api_key_service.get_key(
                conn=mock_db_connection,
                key_id=uuid4(),
                tenant_id=uuid4(),
            )

            assert result is None
