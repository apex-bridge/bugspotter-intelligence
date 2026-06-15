"""Tests for admin API endpoints"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

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


class TestCreateTenantAPIKey:
    """Tests for POST /admin/tenants/{tenant_id}/api-keys (master key endpoint)"""

    @pytest.mark.asyncio
    async def test_creates_key_for_target_tenant(
        self, mock_db_connection, mock_api_key_service
    ):
        """Master key holder can create a key for any tenant"""
        from bugspotter_intelligence.api.routes.admin import create_tenant_api_key
        from bugspotter_intelligence.models.requests import CreateTenantAPIKeyRequest

        target_tenant_id = uuid4()
        body = CreateTenantAPIKeyRequest(name="org-key", rate_limit_per_minute=120)

        response = await create_tenant_api_key(
            tenant_id=target_tenant_id,
            body=body,
            conn=mock_db_connection,
            service=mock_api_key_service,
        )

        assert response.plain_key == "bsi_full_key_here"
        call_kwargs = mock_api_key_service.create_key.call_args.kwargs
        assert call_kwargs["tenant_id"] == target_tenant_id
        assert call_kwargs["name"] == "org-key"
        assert call_kwargs["rate_limit_per_minute"] == 120
        # Keys created via master key must always be non-admin
        assert call_kwargs["is_admin"] is False

    @pytest.mark.asyncio
    async def test_uses_path_tenant_id_not_body(
        self, mock_db_connection, mock_api_key_service
    ):
        """Tenant ID always comes from path; body has no tenant_id field"""
        from bugspotter_intelligence.api.routes.admin import create_tenant_api_key
        from bugspotter_intelligence.models.requests import CreateTenantAPIKeyRequest

        target_tenant_id = uuid4()
        body = CreateTenantAPIKeyRequest(name="key")

        await create_tenant_api_key(
            tenant_id=target_tenant_id,
            body=body,
            conn=mock_db_connection,
            service=mock_api_key_service,
        )

        call_kwargs = mock_api_key_service.create_key.call_args.kwargs
        assert call_kwargs["tenant_id"] == target_tenant_id

    def test_http_accepts_flat_body(self, mock_api_key_service):
        """Regression: FastAPI must accept a flat JSON body, not {\"body\": {...}}.

        Direct function calls bypass FastAPI's routing layer and cannot catch
        body-embedding bugs (e.g. an unannotated BaseModel parameter in a
        dependency being counted as an extra body parameter, forcing embedded-body
        mode and causing 422s in production). This test exercises the full HTTP path.
        """
        from fastapi import FastAPI
        from bugspotter_intelligence.api.routes.admin import router
        from bugspotter_intelligence.auth.dependencies import _get_settings, get_api_key_service
        from bugspotter_intelligence.db.database import get_db_connection

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")

        master_key = "test-master-key"
        mock_settings = MagicMock()
        mock_settings.master_api_key = SecretStr(master_key)

        # Use dependency_overrides so FastAPI's DI resolves these correctly
        app.dependency_overrides[get_db_connection] = lambda: AsyncMock()
        app.dependency_overrides[get_api_key_service] = lambda: mock_api_key_service
        app.dependency_overrides[_get_settings] = lambda: mock_settings

        with TestClient(app) as client:
            target_tenant_id = uuid4()
            response = client.post(
                f"/api/v1/admin/tenants/{target_tenant_id}/api-keys",
                json={"name": "org-key", "rate_limit_per_minute": 120},
                headers={"Authorization": f"Bearer {master_key}"},
            )

        assert response.status_code == 201, (
            f"Expected 201 with flat body, got {response.status_code}: {response.text}"
        )
        assert "plain_key" in response.json()


class TestRequireMasterKey:
    """Tests for require_master_key dependency"""

    def _make_settings(self, key: str | None):
        from bugspotter_intelligence.config import Settings
        settings = MagicMock(spec=Settings)
        # Use SecretStr for any non-None value, including empty string
        settings.master_api_key = SecretStr(key) if key is not None else None
        return settings

    def _make_credentials(self, token: str):
        from fastapi.security import HTTPAuthorizationCredentials
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    @pytest.mark.asyncio
    async def test_allows_valid_master_key(self):
        """Should pass through with correct master key"""
        from bugspotter_intelligence.auth.dependencies import require_master_key

        settings = self._make_settings("correct-master-key")
        credentials = self._make_credentials("correct-master-key")

        # Should not raise
        result = await require_master_key(credentials=credentials, settings=settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_wrong_master_key(self):
        """Should return 401 for wrong key"""
        from bugspotter_intelligence.auth.dependencies import require_master_key

        settings = self._make_settings("correct-master-key")
        credentials = self._make_credentials("wrong-key")

        with pytest.raises(HTTPException) as exc_info:
            await require_master_key(credentials=credentials, settings=settings)

        assert exc_info.value.status_code == 401
        assert "Invalid master API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_rejects_missing_credentials(self):
        """Should return 401 with distinct message when no Bearer token provided"""
        from bugspotter_intelligence.auth.dependencies import require_master_key

        settings = self._make_settings("correct-master-key")

        with pytest.raises(HTTPException) as exc_info:
            await require_master_key(credentials=None, settings=settings)

        assert exc_info.value.status_code == 401
        assert "Missing master API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_returns_503_when_master_key_not_configured(self):
        """Should return 503 when MASTER_API_KEY is not set on server"""
        from bugspotter_intelligence.auth.dependencies import require_master_key

        settings = self._make_settings(None)
        credentials = self._make_credentials("any-key")

        with pytest.raises(HTTPException) as exc_info:
            await require_master_key(credentials=credentials, settings=settings)

        assert exc_info.value.status_code == 503
        assert "not configured" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_returns_503_when_master_key_is_empty_string(self):
        """Should return 503 when MASTER_API_KEY env var is set to empty string"""
        from bugspotter_intelligence.auth.dependencies import require_master_key

        settings = self._make_settings("")  # SecretStr(""), not None
        credentials = self._make_credentials("any-key")

        with pytest.raises(HTTPException) as exc_info:
            await require_master_key(credentials=credentials, settings=settings)

        assert exc_info.value.status_code == 503
        assert "not configured" in exc_info.value.detail


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


class TestObservabilityIsTenantReadable:
    """The /observability/* endpoints must be readable by a NON-admin tenant key.

    Org keys are provisioned via the master key and are non-admin by design
    (master-key-provisioned keys set is_admin=False). The backend forwards the
    org's own per-tenant key to these endpoints, so gating them on admin made
    the cost/usage dashboard 403 for every org. The reads are tenant-scoped
    (WHERE tenant_id = caller's id), so a regular key is the correct credential.

    These go through the full HTTP/DI path on purpose — direct handler calls
    bypass the Depends() that carries the auth requirement, so only a TestClient
    run can catch the admin-gating regression.
    """

    def _build_app(self, tenant_ctx, conn):
        from fastapi import FastAPI

        from bugspotter_intelligence.api.routes.admin import router
        from bugspotter_intelligence.auth.dependencies import (
            _get_settings,
            get_current_tenant,
        )
        from bugspotter_intelligence.db.database import get_db_connection
        from bugspotter_intelligence.rate_limiting.dependencies import get_rate_limiter

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")

        settings = MagicMock()
        settings.rate_limit_enabled = False  # skip the limiter branch

        app.dependency_overrides[get_current_tenant] = lambda: tenant_ctx
        app.dependency_overrides[_get_settings] = lambda: settings
        app.dependency_overrides[get_db_connection] = lambda: conn
        app.dependency_overrides[get_rate_limiter] = lambda: None
        return app

    @staticmethod
    def _summary_conn():
        """Mock connection whose cursor returns an empty-but-valid summary."""
        cur = AsyncMock()
        cur.execute = AsyncMock()
        cur.fetchone = AsyncMock(return_value=(0, 0, None, None, 0))
        cur.fetchall = AsyncMock(return_value=[])
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cur)
        cm.__aexit__ = AsyncMock(return_value=False)
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=cm)
        return conn

    def test_summary_readable_by_non_admin_and_scoped_to_caller(
        self, non_admin_tenant_context
    ):
        app = self._build_app(non_admin_tenant_context, self._summary_conn())

        with TestClient(app) as client:
            resp = client.get(
                "/api/v1/admin/observability/summary",
                headers={"Authorization": "Bearer any"},
            )

        assert resp.status_code == 200, resp.text
        # Response is scoped to the caller's own tenant — no cross-tenant leak.
        assert resp.json()["tenant_id"] == str(non_admin_tenant_context.tenant_id)

    def test_admin_only_route_still_rejects_non_admin(self, non_admin_tenant_context):
        """Guard: the fix must NOT loosen the genuinely admin-only routes."""
        app = self._build_app(non_admin_tenant_context, self._summary_conn())

        with TestClient(app) as client:
            resp = client.get(
                "/api/v1/admin/api-keys",
                headers={"Authorization": "Bearer any"},
            )

        assert resp.status_code == 403
        assert "Admin privileges required" in resp.text
