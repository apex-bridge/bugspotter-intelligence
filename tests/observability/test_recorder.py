"""Unit tests for observability.recorder.

Persistence is mocked via patching `get_pool` — we don't need a real DB; the
focus is on the wrapper's behavior: capturing latency, merging meta, swallowing
persistence failures, propagating LLM exceptions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from bugspotter_intelligence.llm import Usage
from bugspotter_intelligence.observability import CallContext, record_generate


class _FakeBase:
    def __init__(self, text="answer", usage=None, exc=None, delay=None, settings=None):
        self._text = text
        self._usage = usage or Usage()
        self._exc = exc
        self._delay = delay
        self.settings = settings

    async def generate_with_usage(self, prompt, context=None, temperature=0.7, max_tokens=1000):
        if self._delay is not None:
            await asyncio.sleep(self._delay)
        if self._exc is not None:
            raise self._exc
        return self._text, self._usage


class OllamaProvider(_FakeBase):
    """Stand-in named so the class-name → 'ollama' resolver path is exercised."""


class NoSuffixProvider(_FakeBase):
    """Locally named the natural way so it DOES match the *Provider convention."""


class WeirdName(_FakeBase):
    """Class name doesn't end in 'Provider' so resolver must fall back to settings."""


def _make_settings(provider="ollama", model="gemma3:12b"):
    s = MagicMock()
    s.llm_provider = provider
    setattr(s, f"{provider}_model", model)
    return s


def _capture_persist_args() -> tuple[MagicMock, list[dict], UUID]:
    """Patch get_pool() with a mock cursor and capture calls. Returns (pool, captured, inserted_id)."""
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
    provider = OllamaProvider(
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
    provider = OllamaProvider(exc=RuntimeError("boom"), settings=_make_settings())
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
    provider = OllamaProvider(text="ok", usage=Usage(), settings=_make_settings())
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        text, event_id = await record_generate(provider, "Q?", ctx=ctx)

    assert text == "ok"
    assert event_id is None


@pytest.mark.asyncio
async def test_rationale_truncated_to_4096_chars():
    pool, captured, _ = _capture_persist_args()
    provider = OllamaProvider(text="x", settings=_make_settings())
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
async def test_provider_name_from_class_overrides_settings():
    """When the provider class follows *Provider, class name wins over settings.llm_provider."""
    pool, captured, _ = _capture_persist_args()
    s = MagicMock(spec=["llm_provider", "weirdprov_model"])
    s.llm_provider = "weirdprov"
    s.weirdprov_model = "should-not-be-picked"
    provider = OllamaProvider(text="x", settings=s)
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        await record_generate(provider, "Q?", ctx=ctx)

    params = captured[0]["params"]
    assert params[3] == "ollama"   # class-name wins
    assert params[4] == "unknown"  # ollama_model not on settings → fallback


@pytest.mark.asyncio
async def test_provider_name_falls_back_to_settings_when_class_doesnt_match():
    pool, captured, _ = _capture_persist_args()
    provider = WeirdName(text="x", settings=_make_settings(provider="anthropic", model="claude-sonnet-4-6"))
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        await record_generate(provider, "Q?", ctx=ctx)

    params = captured[0]["params"]
    assert params[3] == "anthropic"
    assert params[4] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_timeout_param_records_error_event():
    """Passing timeout= records a TimeoutError event and re-raises."""
    pool, captured, _ = _capture_persist_args()
    provider = OllamaProvider(text="x", delay=0.5, settings=_make_settings())
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await record_generate(provider, "Q?", ctx=ctx, timeout=0.05)

    assert len(captured) == 1
    params = captured[0]["params"]
    assert params[12] == "error"   # status
    # asyncio.TimeoutError is TimeoutError on 3.11+ but the kind label is captured
    # off the exception's class — accept either spelling.
    assert params[13] in ("TimeoutError", "asyncio.TimeoutError")


@pytest.mark.asyncio
async def test_event_id_attached_to_exception_on_provider_error():
    """Callers can recover the persisted event_id off the re-raised exception."""
    pool, captured, inserted_id = _capture_persist_args()
    provider = OllamaProvider(exc=RuntimeError("kaboom"), settings=_make_settings())
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
        with pytest.raises(RuntimeError) as ei:
            await record_generate(provider, "Q?", ctx=ctx)

    assert getattr(ei.value, "event_id", None) == inserted_id


@pytest.mark.asyncio
async def test_cancellation_is_recorded_and_re_raised():
    """asyncio.CancelledError from outer cancellation is logged as an error event."""
    pool, captured, _ = _capture_persist_args()
    # Provider sleeps long enough that the outer cancel hits while awaiting.
    provider = OllamaProvider(delay=1.0, settings=_make_settings())
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    async def runner():
        with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
            await record_generate(provider, "Q?", ctx=ctx)

    task = asyncio.create_task(runner())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(captured) == 1
    params = captured[0]["params"]
    assert params[12] == "error"
    assert params[13] == "CancelledError"


@pytest.mark.asyncio
async def test_cancellation_during_persist_propagates_over_pending_exc():
    """If outer cancel hits while persisting AND the provider already raised,
    CancelledError must still propagate — not be swallowed in favor of the
    provider error. The event loop / framework needs the cancel to land.
    """
    captured: list[dict] = []
    cancel_inside = asyncio.Event()
    cancel_acked = asyncio.Event()

    cursor = MagicMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    async def slow_execute(sql, params):
        # Signal we're inside persist, then await an event the test will never
        # set — emulating a real DB call that the outer cancel interrupts.
        captured.append({"sql": sql, "params": params})
        cancel_inside.set()
        try:
            await asyncio.sleep(5)
        finally:
            cancel_acked.set()

    cursor.execute = AsyncMock(side_effect=slow_execute)
    cursor.fetchone = AsyncMock(return_value=(uuid4(),))

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    conn.commit = AsyncMock()

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)

    provider = OllamaProvider(exc=RuntimeError("provider boom"), settings=_make_settings())
    ctx = CallContext(tenant_id=uuid4(), operation="ask", prompt_version="ask.v1")

    async def runner():
        with patch("bugspotter_intelligence.observability.recorder.get_pool", return_value=pool):
            await record_generate(provider, "Q?", ctx=ctx)

    task = asyncio.create_task(runner())
    # Wait until persist is in-flight, then cancel.
    await cancel_inside.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Sanity: the persist was actually entered (one INSERT captured).
    assert len(captured) == 1
