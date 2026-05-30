"""Wraps LLM calls to persist an intelligence_event per call.

Call sites use `record_generate(provider, prompt, ctx=CallContext(...))`
instead of `provider.generate(prompt)`. Returns `(text, event_id)`.

Persistence failures never propagate — the wrapped LLM call's outcome
is what the caller cares about. The original exception (if any) is
re-raised after the event is persisted so caller fallback paths keep
working.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from psycopg.types.json import Jsonb

from bugspotter_intelligence.db.database import get_pool
from bugspotter_intelligence.llm import LLMProvider, Usage

from .pricing import price_micros

logger = logging.getLogger(__name__)

_MAX_RATIONALE_CHARS = 4096
_MAX_ERROR_MESSAGE_CHARS = 1000


@dataclass
class CallContext:
    """Caller-supplied fields for one persisted intelligence_event row."""
    tenant_id: UUID
    operation: str
    prompt_version: str
    bug_id: Optional[str] = None
    cached: bool = False
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


async def record_generate(
    provider: LLMProvider,
    prompt: str,
    *,
    ctx: CallContext,
    context: Optional[list[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> tuple[str, Optional[UUID]]:
    """Invoke provider.generate_with_usage(...), persist an intelligence_event, return (text, event_id)."""
    started = time.perf_counter()
    status = "ok"
    err_kind: Optional[str] = None
    err_message: Optional[str] = None
    text = ""
    usage = Usage()
    pending_exc: Optional[BaseException] = None
    try:
        text, usage = await provider.generate_with_usage(
            prompt=prompt,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        status = "error"
        err_kind = type(exc).__name__
        err_message = str(exc)[:_MAX_ERROR_MESSAGE_CHARS]
        pending_exc = exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    provider_name = _resolve_provider_name(provider)
    model = _resolve_model(provider)
    cost = price_micros(provider_name, model, usage.input, usage.output)

    meta = dict(ctx.meta)
    if usage.extra:
        meta.update(usage.extra)
    if err_message is not None:
        meta["error_message"] = err_message

    event_id: Optional[UUID] = None
    try:
        event_id = await _persist_event(
            tenant_id=ctx.tenant_id,
            operation=ctx.operation,
            bug_id=ctx.bug_id,
            provider=provider_name,
            model=model,
            prompt_version=ctx.prompt_version,
            tokens_in=usage.input,
            tokens_out=usage.output,
            cost_micros_usd=cost,
            latency_ms=latency_ms,
            confidence=ctx.confidence,
            rationale=ctx.rationale,
            status=status,
            error_kind=err_kind,
            cached=ctx.cached,
            meta=meta,
        )
    except Exception:
        # Observability must never break the user-facing path.
        logger.exception("Failed to persist intelligence_event")

    if pending_exc is not None:
        raise pending_exc
    return text, event_id


def _resolve_provider_name(provider: LLMProvider) -> str:
    settings = getattr(provider, "settings", None)
    raw = getattr(settings, "llm_provider", None) if settings else None
    if isinstance(raw, str) and raw.strip():
        return raw.lower()
    return type(provider).__name__.replace("Provider", "").lower()


def _resolve_model(provider: LLMProvider) -> str:
    settings = getattr(provider, "settings", None)
    if settings is None:
        return "unknown"
    provider_name = _resolve_provider_name(provider)
    raw = getattr(settings, f"{provider_name}_model", None)
    return raw if isinstance(raw, str) and raw.strip() else "unknown"


async def _persist_event(
    *,
    tenant_id: UUID,
    operation: str,
    bug_id: Optional[str],
    provider: str,
    model: str,
    prompt_version: str,
    tokens_in: Optional[int],
    tokens_out: Optional[int],
    cost_micros_usd: Optional[int],
    latency_ms: int,
    confidence: Optional[float],
    rationale: Optional[str],
    status: str,
    error_kind: Optional[str],
    cached: bool,
    meta: dict[str, Any],
) -> UUID:
    if rationale is not None and len(rationale) > _MAX_RATIONALE_CHARS:
        rationale = rationale[:_MAX_RATIONALE_CHARS]

    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO intelligence_event (
                    tenant_id, operation, bug_id, provider, model, prompt_version,
                    tokens_in, tokens_out, cost_micros_usd, latency_ms,
                    confidence, rationale, status, error_kind, cached, meta
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                ) RETURNING id
                """,
                (
                    tenant_id, operation, bug_id, provider, model, prompt_version,
                    tokens_in, tokens_out, cost_micros_usd, latency_ms,
                    confidence, rationale, status, error_kind, cached, Jsonb(meta),
                ),
            )
            row = await cur.fetchone()
            await conn.commit()
            return row[0]
