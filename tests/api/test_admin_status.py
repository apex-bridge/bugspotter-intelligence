"""Unit tests for GET /admin/status (no DB / app — the route is a plain async fn)."""

import pytest

from bugspotter_intelligence.api.routes.admin import get_service_status
from bugspotter_intelligence.config import Settings


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *args, **kwargs):
        return None

    async def fetchone(self):
        return self._row


class _FakeConn:
    """Stands in for psycopg AsyncConnection; cursor() returns the embedding row."""

    def __init__(self, row):
        self._row = row

    def cursor(self):
        return _FakeCursor(self._row)


def _settings(**overrides):
    # _env_file=None + explicit keys isolates the test from any ambient .env /
    # OPENAI_API_KEY in the dev environment (init kwargs override env vars).
    base = dict(_env_file=None, anthropic_api_key=None, openai_api_key=None)
    base.update(overrides)
    return Settings(**base)


@pytest.mark.asyncio
async def test_status_reports_active_provider_model_and_key_presence():
    settings = _settings(
        llm_provider="claude",
        anthropic_api_key="sk-anthropic",
        claude_model="claude-sonnet-4-6",
    )
    resp = await get_service_status(_=None, settings=settings, conn=_FakeConn((10, 0, 1024)))

    assert resp.llm_provider == "claude"
    assert resp.llm_model == "claude-sonnet-4-6"
    assert resp.anthropic_key_configured is True
    assert resp.openai_key_configured is False
    assert resp.embeddings.total == 10
    assert resp.embeddings.nulls == 0
    assert resp.embeddings.min_dim == 1024
    assert resp.embeddings.healthy is True


@pytest.mark.asyncio
async def test_status_flags_null_embeddings_unhealthy():
    settings = _settings(llm_provider="ollama")
    resp = await get_service_status(_=None, settings=settings, conn=_FakeConn((100, 3, 1024)))

    assert resp.llm_provider == "ollama"
    assert resp.llm_model == settings.ollama_model
    assert resp.embeddings.nulls == 3
    assert resp.embeddings.healthy is False


@pytest.mark.asyncio
async def test_status_empty_table_is_healthy_with_null_min_dim():
    settings = _settings(llm_provider="ollama")
    resp = await get_service_status(_=None, settings=settings, conn=_FakeConn((0, 0, None)))

    assert resp.embeddings.total == 0
    assert resp.embeddings.min_dim is None
    assert resp.embeddings.healthy is True


@pytest.mark.asyncio
async def test_status_never_leaks_key_values():
    settings = _settings(llm_provider="openai", openai_api_key="sk-super-secret")
    resp = await get_service_status(_=None, settings=settings, conn=_FakeConn((1, 0, 1024)))

    assert resp.openai_key_configured is True
    assert "sk-super-secret" not in resp.model_dump_json()
