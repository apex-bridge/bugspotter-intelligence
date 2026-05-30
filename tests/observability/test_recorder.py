"""Unit tests for observability.recorder.

Persistence is mocked via patching `get_pool` — we don't need a real DB; the
focus is on the wrapper's behavior: capturing latency, merging meta, swallowing
persistence failures, propagating LLM exceptions.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from bugspotter_intelligence.llm import Usage
from bugspotter_intelligence.observability import CallContext, record_generate


class _FakeProvider:
    """Minimal LLMProvider stand-in with controllable usage + side effects."""
    def __init__(self, text="answer", usage=None, exc=None, settings=None):
        self._text = text
        self._usage = usage or Usage()
        self._exc = exc
        self.settings = settings

    async def generate_with_usage(self, prompt, context=None, temperature=0.7, max_tokens=1000):
        if self._exc is not None:
            raise self._exc
        return self._text, self._usage


def _make_settings(provider="ollama", model="gemma3:12b"):
    s = MagicMock()
    s.llm_provider = provider
    setattr(s, f"{provider}_model", model)
    return s


def _capture_persist_args() -> tuple[MagicMock, list[dict]]:
    """Patch get_pool() with a mock cursor and capture calls. Returns (pool_mock, captured)."""
    captured: list[dict] = []
    inserted_id = uuid4()

    cursor = MagicMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    async def fake_execute(sql, params):
        captured.append({"sql": sql, "params": params})

    cursor.execute = AsyncMock(side_effect=fake_execute)
    cursor.fetchone = AsyncMock(return_value=(inserted_id,))

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    conn.commit = AsyncMock()

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)

    return pool, captured, inserted_id


@pytest.mark.asyncio
async def test_persists_basic_fields_and_returns_event_id():
    pool, captured, inserted_id = _capture_persist_args()
    provider = _FakeProvider(
        text="hi",
        usage=Usage(input=42, output=7, extra={"eval_duration": 12345}),
        settings=_make_settings(),
    )
    ctx = CallContext(
        tenant_id=uuid4(),
        operation="ask",
        prompt_version="ask.v1",
        meta={"context_size": 0},
    )

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        text, event_id = await record_generate(provider, "Q?", ctx=ctx)

    assert text == "hi"
    assert event_id == inserted_id
    assert len(captured) == 1
    params = captured[0]["params"]
    # Positional order matches the INSERT clause in recorder.py
    (tenant_id, operation, bug_id, prov, model, prompt_version,
     tokens_in, tokens_out, cost, latency_ms,
     confidence, rationale, status, error_kind, cached, meta) = params

    assert operation == "ask"
    assert prov == "ollama"
    assert model == "gemma3:12b"
    assert tokens_in == 42
    assert tokens_out == 7
    assert cost is None              # ollama is not priced
    assert latency_ms >= 0
    assert status == "ok"
    assert error_kind is None
    assert cached is False
    # meta is wrapped in Jsonb; extract via .obj
    raw_meta = getattr(meta, "obj", meta)
    assert raw_meta["context_size"] == 0
    assert raw_meta["eval_duration"] == 12345


@pytest.mark.asyncio
async def test_records_error_and_reraises_when_provider_raises():
    pool, captured, _ = _capture_persist_args()
    provider = _FakeProvider(exc=RuntimeError("boom"), settings=_make_settings())
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        with pytest.raises(RuntimeError, match="boom"):
            await record_generate(provider, "Q?", ctx=ctx)

    assert len(captured) == 1, "must persist an error event before re-raising"
    params = captured[0]["params"]
    assert params[12] == "error"             # status
    assert params[13] == "RuntimeError"      # error_kind
    raw_meta = getattr(params[15], "obj", params[15])
    assert raw_meta["error_message"] == "boom"


@pytest.mark.asyncio
async def test_persistence_failure_does_not_break_caller():
    """If the DB insert raises, the LLM result must still be returned (event_id=None)."""
    pool = MagicMock()
    pool.connection = MagicMock(side_effect=RuntimeError("db down"))
    provider = _FakeProvider(text="ok", usage=Usage(), settings=_make_settings())
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        text, event_id = await record_generate(provider, "Q?", ctx=ctx)

    assert text == "ok"
    assert event_id is None


@pytest.mark.asyncio
async def test_rationale_truncated_to_4096_chars():
    pool, captured, _ = _capture_persist_args()
    provider = _FakeProvider(text="x", settings=_make_settings())
    ctx = CallContext(
        tenant_id=uuid4(),
        operation="ask",
        prompt_version="ask.v1",
        rationale="z" * 10_000,
    )

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        await record_generate(provider, "Q?", ctx=ctx)

    rationale_param = captured[0]["params"][11]
    assert len(rationale_param) == 4096


@pytest.mark.asyncio
async def test_unknown_model_when_settings_missing_attr():
    pool, captured, _ = _capture_persist_args()
    s = MagicMock(spec=["llm_provider"])
    s.llm_provider = "weirdprov"
    provider = _FakeProvider(text="x", settings=s)
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        await record_generate(provider, "Q?", ctx=ctx)

    params = captured[0]["params"]
    assert params[3] == "weirdprov"     # provider
    assert params[4] == "unknown"       # model — fall-through sentinel
