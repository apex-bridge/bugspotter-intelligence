"""Tests for admin API endpoints"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request

from bugspotter_intelligence.auth.models import APIKey, TenantContext


@pytest.fixture
def admin_tenant_context():
    """Admin tenant context"""
    return TenantContext(
        tenant_id=uuid4(),
        api_key_id=uuid4(),
        is_admin=True,
        rate_limit_per_minute=1000,
    )


@pytest.fixture
def non_admin_tenant_context():
    """Non-admin tenant context"""
    return TenantContext(
        tenant_id=uuid4(),
        api_key_id=uuid4(),
        is_admin=False,
        rate_limit_per_minute=60,
    )


@pytest.fixture
def sample_api_key():
    """Sample APIKey for testing"""
    return APIKey(
        id=uuid4(),
        tenant_id=uuid4(),
        key_prefix="bsi_abc12345",
        name="Test Key",
        created_at=datetime.now(),
        last_used_at=None,
        revoked_at=None,
        rate_limit_per_minute=60,
        is_admin=False,
    )


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    return AsyncMock()


@pytest.fixture
def mock_api_key_service(sample_api_key):
    """Mock API key service"""
    service = MagicMock()
    service.create_key = AsyncMock(return_value=(sample_api_key, "bsi_full_key_here"))
    service.list_keys = AsyncMock(return_value=[sample_api_key])
    service.get_key = AsyncMock(return_value=sample_api_key)
    service.revoke_key = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_request():
    """Mock FastAPI request"""
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    return request


class TestCreateAPIKey:
    """Tests for POST /admin/api-keys endpoint"""

    @pytest.mark.asyncio
    async def test_creates_key_for_admin(
        self, admin_tenant_context, mock_db_connection, mock_api_key_service, mock_request
    ):
        """Should create key when caller is admin"""
        from bugspotter_intelligence.api.routes.admin import create_api_key
        from bugspotter_intelligence.models.requests import CreateAPIKeyRequest

        body = CreateAPIKeyRequest(name="New Key")

        with patch(
            "bugspotter_intelligence.api.routes.admin.get_api_key_service",
            return_value=mock_api_key_service,
        ):
            response = await create_api_key(
                body=body,
                tenant=admin_tenant_context,
                conn=mock_db_connection,
                service=mock_api_key_service,
            )

        assert response.plain_key == "bsi_full_key_here"
        assert response.api_key.name == "Test Key"

        # Verify it uses the admin's own tenant_id
        mock_api_key_service.create_key.assert_called_once()
        call_kwargs = mock_api_key_service.create_key.call_args.kwargs
        assert call_kwargs["tenant_id"] == admin_tenant_context.tenant_id

    @pytest.mark.asyncio
    async def test_rejects_cross_tenant_key_creation(
        self, admin_tenant_context, mock_db_connection, mock_api_key_service, mock_request
    ):
        """Should reject attempt to create key for different tenant (security)"""
        from bugspotter_intelligence.api.routes.admin import create_api_key
        from bugspotter_intelligence.models.requests import CreateAPIKeyRequest

        target_tenant = uuid4()  # Different tenant
        body = CreateAPIKeyRequest(name="New Key", tenant_id=target_tenant)

        with pytest.raises(HTTPException) as exc_info:
            await create_api_key(
                body=body,
                tenant=admin_tenant_context,
                conn=mock_db_connection,
                service=mock_api_key_service,
            )

        assert exc_info.value.status_code == 403
        assert "Cannot create API keys for other tenants" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_allows_explicit_own_tenant_id(
        self, admin_tenant_context, mock_db_connection, mock_api_key_service, mock_request
    ):
        """Should allow specifying own tenant_id explicitly"""
        from bugspotter_intelligence.api.routes.admin import create_api_key
        from bugspotter_intelligence.models.requests import CreateAPIKeyRequest

        # Same tenant as the admin
        body = CreateAPIKeyRequest(name="New Key", tenant_id=admin_tenant_context.tenant_id)

        with patch(
            "bugspotter_intelligence.api.routes.admin.get_api_key_service",
            return_value=mock_api_key_service,
        ):
            response = await create_api_key(
                body=body,
                tenant=admin_tenant_context,
                conn=mock_db_connection,
                service=mock_api_key_service,
            )

        assert response.plain_key == "bsi_full_key_here"
        mock_api_key_service.create_key.assert_called_once()
        call_kwargs = mock_api_key_service.create_key.call_args.kwargs
        assert call_kwargs["tenant_id"] == admin_tenant_context.tenant_id


class TestListAPIKeys:
    """Tests for GET /admin/api-keys endpoint"""

    @pytest.mark.asyncio
    async def test_lists_keys_for_tenant(
        self, admin_tenant_context, mock_db_connection, mock_api_key_service, mock_request
    ):
        """Should list all keys for tenant"""
        from bugspotter_intelligence.api.routes.admin import list_api_keys

        with patch(
            "bugspotter_intelligence.api.routes.admin.get_api_key_service",
            return_value=mock_api_key_service,
        ):
            response = await list_api_keys(
                tenant=admin_tenant_context,
                conn=mock_db_connection,
                service=mock_api_key_service,
            )

        assert response.total == 1
        assert len(response.keys) == 1
        assert response.keys[0].name == "Test Key"


class TestGetAPIKey:
    """Tests for GET /admin/api-keys/{key_id} endpoint"""

    @pytest.mark.asyncio
    async def test_gets_key_by_id(
        self, admin_tenant_context, mock_db_connection, mock_api_key_service, sample_api_key, mock_request
    ):
        """Should return key when found"""
        from bugspotter_intelligence.api.routes.admin import get_api_key

        with patch(
            "bugspotter_intelligence.api.routes.admin.get_api_key_service",
            return_value=mock_api_key_service,
        ):
            response = await get_api_key(
                key_id=sample_api_key.id,
                tenant=admin_tenant_context,
                conn=mock_db_connection,
                service=mock_api_key_service,
            )

        assert response.id == sample_api_key.id
        assert response.name == sample_api_key.name

    @pytest.mark.asyncio
    async def test_returns_404_when_not_found(
        self, admin_tenant_context, mock_db_connection, mock_api_key_service, mock_request
    ):
        """Should return 404 when key not found"""
        from bugspotter_intelligence.api.routes.admin import get_api_key

        mock_api_key_service.get_key = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(
                key_id=uuid4(),
                tenant=admin_tenant_context,
                conn=mock_db_connection,
                service=mock_api_key_service,
            )

        assert exc_info.value.status_code == 404


class TestRevokeAPIKey:
    """Tests for DELETE /admin/api-keys/{key_id} endpoint"""

    @pytest.mark.asyncio
    async def test_revokes_key(
        self, admin_tenant_context, mock_db_connection, mock_api_key_service, mock_request
    ):
        """Should revoke key and return 204"""
        from bugspotter_intelligence.api.routes.admin import revoke_api_key

        result = await revoke_api_key(
            key_id=uuid4(),
            tenant=admin_tenant_context,
            conn=mock_db_connection,
            service=mock_api_key_service,
        )

        assert result is None  # 204 No Content

    @pytest.mark.asyncio
    async def test_returns_404_when_not_found(
        self, admin_tenant_context, mock_db_connection, mock_api_key_service, mock_request
    ):
        """Should return 404 when key not found"""
        from bugspotter_intelligence.api.routes.admin import revoke_api_key

        mock_api_key_service.revoke_key = AsyncMock(return_value=False)

        with pytest.raises(HTTPException) as exc_info:
            await revoke_api_key(
                key_id=uuid4(),
                tenant=admin_tenant_context,
                conn=mock_db_connection,
                service=mock_api_key_service,
            )

        assert exc_info.value.status_code == 404
