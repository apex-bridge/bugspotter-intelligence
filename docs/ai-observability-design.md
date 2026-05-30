# AI Observability Design

Production observability for LLM-backed features in `bugspotter-intelligence`. Covers Sprint 1 (confidence + feedback) and Sprint 2 (reasoning + cost log) from the roadmap.

## Goals

- Answer "is the AI helping me?" per-tenant — accuracy, latency, cost.
- Every LLM call is auditable (model, prompt version, latency, tokens, cost, rationale).
- User can mark any AI suggestion correct/incorrect with one click.
- UI can show confidence + "needs review" badge without extra round trip.
- Self-hosted admins see monthly cost & cache hit-rate without grepping logs.

## Non-goals

- Online retraining / fine-tuning (active learning queue is a later sprint).
- Per-call OpenTelemetry tracing — stays Phase 7 in ROADMAP.
- Replacing the offline severity / embedding benchmarks.

## Current state (verified)

- LLM is called from three places, none capture latency/tokens:
  - `src/bugspotter_intelligence/services/reranker.py:57` (search smart-mode)
  - `src/bugspotter_intelligence/services/rule_parser_service.py:356` (NL → dedup rule)
  - `src/bugspotter_intelligence/api/routes/ask.py:28` (Q&A)
- `LLMProvider.generate()` (`llm/base.py:13`) returns `str` only.
- `EnrichmentConfidence` (`api/schemas/responses.py:68`) already models per-field confidence — reuse the pattern, don't reinvent.
- Multi-tenancy is fully wired: `TenantContext` in every request, `tenant_id` on tables, cache keys include tenant.
- Migrations are async functions in `db/migrations.py`, idempotent via `IF NOT EXISTS`, orchestrated by `create_tables()`.
- No Alembic; no observability tables exist.

## Schema

Two new tables. Both partitioned-friendly later if volume requires (not now).

### `intelligence_event` — immutable audit record per LLM call

```sql
CREATE TABLE IF NOT EXISTS intelligence_event (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL,
    operation       TEXT NOT NULL,       -- 'search_rerank' | 'rule_parse' | 'ask' | 'enrich' | 'severity' | 'dedup'
    bug_id          TEXT NULL,           -- nullable: ask/rule_parse have no bug
    provider        TEXT NOT NULL,       -- 'anthropic' | 'openai' | 'ollama' | ...
    model           TEXT NOT NULL,       -- 'claude-sonnet-4-6', 'gemma3:12b', ...
    prompt_version  TEXT NOT NULL,       -- short hash or label, set per call site
    tokens_in       INTEGER NULL,
    tokens_out      INTEGER NULL,
    cost_micros_usd BIGINT NULL,         -- micro-dollars; NULL if unknown (local model)
    latency_ms      INTEGER NOT NULL,
    confidence      REAL NULL,           -- 0..1, NULL if model didn't emit
    rationale       TEXT NULL,           -- capped at 4 KiB; truncate caller-side
    status          TEXT NOT NULL,       -- 'ok' | 'fallback' | 'error'
    error_kind      TEXT NULL,           -- non-null when status != 'ok'
    cached          BOOLEAN NOT NULL DEFAULT FALSE,
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb,  -- see "meta shape" below
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_intel_event_tenant_created
    ON intelligence_event (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_intel_event_tenant_op_created
    ON intelligence_event (tenant_id, operation, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_intel_event_bug
    ON intelligence_event (tenant_id, bug_id) WHERE bug_id IS NOT NULL;
```

Notes:
- `cost_micros_usd` is computed in the wrapper from a small `MODEL_PRICING` constant map; for Ollama and local models stays `NULL` (cost = 0 isn't the same as unknown).
- `rationale` is the model's own short justification when present. Long chain-of-thought stays out of the DB — log it to stdout if needed.
- `prompt_version` is a short label set at each call site (e.g. `"rerank.v2"`, `"rule_parse.2026-05"`). When a prompt changes, the label changes — this is how we'll detect regressions later.
- Hot, aggregatable fields are flat (sum/avg/percentile queries stay fast without expression indexes); long-tail provider- and op-specific details live in `meta`.

**`meta` shape (by-convention, not enforced):**
- Provider extras: `cache_read_input_tokens` (Anthropic), `reasoning_tokens` (OpenAI o-series), `eval_count` / `prompt_eval_count` / `eval_duration_ns` (Ollama).
- Op-specific context:
  - `search_rerank`: `{candidate_count, top_scores: [..], fallback_reason?}`
  - `rule_parse`: `{extracted_rule_kind?, available_integrations_count}`
  - `ask`: `{context_size, top_k}`
- `prompt_variant` for future A/B (sits next to flat `prompt_version`).
- `error_message` when `status='error'` (raw text; `error_kind` is the typed bucket).
- `settings_snapshot` of feature flags / thresholds active at call time.

We deliberately do NOT add a GIN index on `meta` now. Add when a specific `meta` path becomes a frequent filter — and prefer a targeted expression index over a broad `gin (meta jsonb_path_ops)`.

### `intelligence_feedback` — user verdicts on events

```sql
CREATE TABLE IF NOT EXISTS intelligence_feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id    UUID NOT NULL REFERENCES intelligence_event(id) ON DELETE CASCADE,
    tenant_id   UUID NOT NULL,           -- denormalized for fast per-tenant aggregation
    verdict     TEXT NOT NULL,           -- 'correct' | 'incorrect' | 'partial'
    user_ref    TEXT NULL,               -- opaque identifier from caller (email/id), NOT PII-validated
    note        TEXT NULL,               -- capped at 2 KiB
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (event_id, user_ref)          -- one verdict per (event, user)
);

CREATE INDEX IF NOT EXISTS ix_intel_feedback_tenant_created
    ON intelligence_feedback (tenant_id, created_at DESC);
```

Notes:
- Separate table (not column on event) so multiple reviewers can disagree, and so events stay immutable.
- `user_ref` is opaque — caller (BugSpotter backend) decides what to send. We never store names/emails directly; this honors [[no PII in PRs]] and the broader "no PII without consent" rule.

## Capturing the data — one wrapper, three call sites

Add a thin instrumented wrapper around `LLMProvider.generate()`:

```python
# src/bugspotter_intelligence/observability/recorder.py
@dataclass
class CallRecord:
    tenant_id: UUID
    operation: str
    bug_id: str | None
    prompt_version: str
    confidence: float | None = None
    rationale: str | None = None

async def record_generate(
    provider: LLMProvider,
    prompt: str,
    *,
    ctx: CallRecord,
    **kwargs,
) -> str:
    started = time.perf_counter()
    status = "ok"; err = None; tokens_in = tokens_out = None
    try:
        result, usage = await provider.generate_with_usage(prompt, **kwargs)
        tokens_in, tokens_out = usage.input, usage.output
        return result
    except Exception as e:
        status = "error"; err = type(e).__name__
        raise
    finally:
        latency_ms = int((time.perf_counter() - started) * 1000)
        await _persist_event(
            tenant_id=ctx.tenant_id, operation=ctx.operation, bug_id=ctx.bug_id,
            provider=provider.name, model=provider.model,
            prompt_version=ctx.prompt_version,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_micros_usd=_price(provider, tokens_in, tokens_out),
            latency_ms=latency_ms,
            confidence=ctx.confidence, rationale=ctx.rationale,
            status=status, error_kind=err,
        )
```

Required change to `LLMProvider`:
- Add `generate_with_usage(...) -> tuple[str, Usage]` to `llm/base.py:13`. Default impl calls `generate()` and returns `Usage(input=None, output=None)` (preserves existing providers); Anthropic / OpenAI providers override to pull real token counts from the response.

Three call sites become one-liners:
- `reranker.py:57` → wrap with `operation="search_rerank"`, `bug_id=None`, `prompt_version="rerank.v1"`.
- `rule_parser_service.py:356` → `operation="rule_parse"`, `prompt_version="rule_parse.v1"`.
- `ask.py:28` → `operation="ask"`.

For severity & dedup that don't currently go through `generate()` (they go through enrich endpoint elsewhere), add the same wrap when those land.

## Response schema additions

`SearchResponse` (`api/schemas/responses.py:155`) gets:
```python
event_id: UUID | None = None     # so client can submit feedback later
confidence: float | None = None  # overall, not per-field
```

`EnrichBugResponse` already has `confidence` — also add `event_id`.

`ParseNLRuleResponse` already has `model` + `raw_llm_output` — add `event_id`.

The client (`bugspotter-private` / `bugspotter-public` admin UI) uses `event_id` when the user clicks 👍/👎.

## New API endpoints

### Feedback (caller-facing)

```
POST /v1/intelligence/feedback
  body: { event_id, verdict: "correct"|"incorrect"|"partial", user_ref?, note? }
  auth: tenant API key; rejects if event_id.tenant_id != caller tenant
  -> 201 { feedback_id }
```

### Observability (admin)

All under `/admin/observability/*`, require admin API key, filtered by `?tenant_id=` (admin can scope to one tenant or all).

```
GET /admin/observability/summary?tenant_id=&from=&to=
  -> { calls, cost_usd, p50_ms, p95_ms, cache_hit_rate, by_operation: [...] }

GET /admin/observability/events?tenant_id=&operation=&status=&limit=&offset=
  -> paginated event rows

GET /admin/observability/accuracy?tenant_id=&operation=&from=&to=
  -> { feedback_count, correct, incorrect, partial, precision }
```

Cache stats endpoint stays where it is; the summary endpoint references the same Redis source.

## Migrations

Append to `src/bugspotter_intelligence/db/migrations.py`:

```python
async def create_intelligence_event_table(pool: asyncpg.Pool) -> None: ...
async def create_intelligence_feedback_table(pool: asyncpg.Pool) -> None: ...
```

Wire into `create_tables()` in order: events before feedback (FK dependency). Each uses `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` for idempotency, matching existing convention.

## UI integration (lives in `bugspotter-public` / `bugspotter-private`)

Out of scope for this repo, but the contract:
- Search result card renders 👍/👎 buttons; on click → `POST /v1/intelligence/feedback` with the `event_id` from the response.
- Confidence < 0.6 → yellow "needs review" badge; > 0.85 → no badge; in between → muted "AI-suggested".
- Admin dashboard page consumes the three observability endpoints; one chart per operation (latency p95, accuracy, cost).

Thresholds live in admin settings so they're tunable per tenant later.

## Sprint 1 minimum cut (~1 week)

Ship in this order — each step is independently shippable:

1. Schema migration for both tables (no callers yet) — half-day.
2. `generate_with_usage` on base provider + Anthropic + Ollama providers — 1 day.
3. `record_generate` wrapper + retrofit the three call sites — 1 day.
4. `event_id` + `confidence` added to `SearchResponse` — half-day.
5. `POST /v1/intelligence/feedback` + auth check — 1 day.
6. `GET /admin/observability/summary` + `accuracy` — 1 day.
7. UI in `bugspotter-public`: thumbs on search cards + confidence badge — 1–2 days (separate repo).

Sprint 2 (`reasoning` capture + dashboard for cost log) builds on the same tables — no further schema work.

## Open questions

- **Sampling.** At what volume do we sample events rather than record 100%? Proposed: keep 100% for v1; revisit when any tenant exceeds 100k events/day.
- **Rationale source.** For Anthropic, ask the model for a one-sentence justification in the prompt (cheap). For Ollama / local, same — but Gemma's rationales are noisy. Decide per call site; OK to leave `rationale=NULL` initially.
- **Retention.** Default 90 days? Self-hosted admins probably want longer. Add a configurable retention job in Sprint 2.
- **Cost map updates.** `MODEL_PRICING` will drift. Park it in `config/pricing.yaml` so it's editable without code change; document staleness.

## What this enables later

- Phase 4 feedback loop (ROADMAP) — `intelligence_feedback` IS the feedback loop's storage.
- Active learning queue — `SELECT event_id, bug_id FROM intelligence_event WHERE confidence < 0.5 AND tenant_id = ? ORDER BY created_at DESC`.
- Golden-set CI — compare current model's events on a fixed bug set against a baseline snapshot.
- The 30-day-prod-data article from the article pipeline.
