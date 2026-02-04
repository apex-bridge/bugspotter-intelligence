"""Tests for authentication dependencies"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from bugspotter_intelligence.auth.dependencies import (
    get_api_key_service,
    get_current_tenant,
    get_optional_tenant,
    require_admin,
)
from bugspotter_intelligence.auth.models import TenantContext
from bugspotter_intelligence.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings with auth enabled"""
    settings = MagicMock(spec=Settings)
    settings.auth_enabled = True
    settings.api_key_prefix = "test_"
    return settings


@pytest.fixture
def mock_settings_auth_disabled():
    """Mock settings with auth disabled"""
    settings = MagicMock(spec=Settings)
    settings.auth_enabled = False
    settings.api_key_prefix = "test_"
    return settings


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    return AsyncMock()


@pytest.fixture
def mock_credentials():
    """Mock HTTP credentials"""
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials="test_api_key_123")


@pytest.fixture
def sample_tenant_context():
    """Sample tenant context"""
    return TenantContext(
        tenant_id=uuid4(),
        api_key_id=uuid4(),
        is_admin=False,
        rate_limit_per_minute=60,
    )


@pytest.fixture
def admin_tenant_context():
    """Admin tenant context"""
    return TenantContext(
        tenant_id=uuid4(),
        api_key_id=uuid4(),
        is_admin=True,
        rate_limit_per_minute=1000,
    )


class TestGetApiKeyService:
    """Tests for get_api_key_service"""

    def test_returns_api_key_service(self, mock_settings):
        """Should return APIKeyService instance"""
        # Reset singleton
        import bugspotter_intelligence.auth.dependencies as deps
        deps._api_key_service = None

        with patch("bugspotter_intelligence.auth.dependencies.Settings", return_value=mock_settings):
            service = get_api_key_service(mock_settings)

        from bugspotter_intelligence.auth.service import APIKeyService
        assert isinstance(service, APIKeyService)

    def test_returns_singleton(self, mock_settings):
        """Should return same instance on subsequent calls"""
        import bugspotter_intelligence.auth.dependencies as deps
        deps._api_key_service = None

        service1 = get_api_key_service(mock_settings)
        service2 = get_api_key_service(mock_settings)

        assert service1 is service2


class TestGetCurrentTenant:
    """Tests for get_current_tenant dependency"""

    @pytest.mark.asyncio
    async def test_returns_dev_tenant_when_auth_disabled(
        self, mock_settings_auth_disabled, mock_db_connection
    ):
        """Should return development tenant when auth is disabled"""
        result = await get_current_tenant(
            credentials=None,
            conn=mock_db_connection,
            settings=mock_settings_auth_disabled,
        )

        assert isinstance(result, TenantContext)
        assert result.tenant_id == UUID("00000000-0000-0000-0000-000000000000")
        assert result.is_admin is True

    @pytest.mark.asyncio
    async def test_raises_401_when_no_credentials(
        self, mock_settings, mock_db_connection
    ):
        """Should raise 401 when no credentials provided"""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_tenant(
                credentials=None,
                conn=mock_db_connection,
                settings=mock_settings,
            )

        assert exc_info.value.status_code == 401
        assert "Missing API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_raises_401_when_invalid_key(
        self, mock_settings, mock_db_connection, mock_credentials
    ):
        """Should raise 401 when API key is invalid"""
        import bugspotter_intelligence.auth.dependencies as deps
        deps._api_key_service = None

        with patch.object(
            get_api_key_service(mock_settings),
            "validate_key",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(
                    credentials=mock_credentials,
                    conn=mock_db_connection,
                    settings=mock_settings,
                )

            assert exc_info.value.status_code == 401
            assert "Invalid or revoked" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_returns_tenant_context_for_valid_key(
        self, mock_settings, mock_db_connection, mock_credentials, sample_tenant_context
    ):
        """Should return TenantContext for valid API key"""
        import bugspotter_intelligence.auth.dependencies as deps
        deps._api_key_service = None

        mock_service = MagicMock()
        mock_service.validate_key = AsyncMock(return_value=sample_tenant_context)

        with patch(
            "bugspotter_intelligence.auth.dependencies.get_api_key_service",
            return_value=mock_service,
        ):
            result = await get_current_tenant(
                credentials=mock_credentials,
                conn=mock_db_connection,
                settings=mock_settings,
            )

        assert result == sample_tenant_context


class TestRequireAdmin:
    """Tests for require_admin dependency"""

    @pytest.mark.asyncio
    async def test_returns_tenant_for_admin(self, admin_tenant_context):
        """Should return tenant context for admin users"""
        result = await require_admin(tenant=admin_tenant_context)
        assert result == admin_tenant_context

    @pytest.mark.asyncio
    async def test_raises_403_for_non_admin(self, sample_tenant_context):
        """Should raise 403 for non-admin users"""
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(tenant=sample_tenant_context)

        assert exc_info.value.status_code == 403
        assert "Admin privileges required" in exc_info.value.detail


class TestGetOptionalTenant:
    """Tests for get_optional_tenant dependency"""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_credentials(
        self, mock_settings, mock_db_connection
    ):
        """Should return None when no credentials provided"""
        result = await get_optional_tenant(
            credentials=None,
            conn=mock_db_connection,
            settings=mock_settings,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_tenant_when_valid_credentials(
        self, mock_settings, mock_db_connection, mock_credentials, sample_tenant_context
    ):
        """Should return TenantContext when valid credentials provided"""
        mock_service = MagicMock()
        mock_service.validate_key = AsyncMock(return_value=sample_tenant_context)

        with patch(
            "bugspotter_intelligence.auth.dependencies.get_api_key_service",
            return_value=mock_service,
        ):
            result = await get_optional_tenant(
                credentials=mock_credentials,
                conn=mock_db_connection,
                settings=mock_settings,
            )

        assert result == sample_tenant_context

    @pytest.mark.asyncio
    async def test_returns_none_when_invalid_credentials(
        self, mock_settings, mock_db_connection, mock_credentials
    ):
        """Should return None when credentials are invalid"""
        mock_service = MagicMock()
        mock_service.validate_key = AsyncMock(return_value=None)

        with patch(
            "bugspotter_intelligence.auth.dependencies.get_api_key_service",
            return_value=mock_service,
        ):
            result = await get_optional_tenant(
                credentials=mock_credentials,
                conn=mock_db_connection,
                settings=mock_settings,
            )

        assert result is None
