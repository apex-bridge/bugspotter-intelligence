# BugSpotter Intelligence — Service Roadmap

Development roadmap for the `bugspotter-intelligence` RAG service. Each phase produces a releasable, testable milestone.

> **Note:** This roadmap reflects the current Python/FastAPI implementation (not the originally planned Node.js/Fastify stack).

---

## Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Python/FastAPI foundation | ✅ Complete | Async, Pydantic models |
| PostgreSQL + pgvector | ✅ Complete | Docker Compose ready |
| LLM Provider abstraction | ✅ Complete | Factory + Registry pattern |
| Ollama integration | ✅ Complete | Local inference |
| Embedding service | ✅ Complete | Local + OpenAI providers |
| Bug similarity search | ✅ Complete | Cosine similarity via pgvector |
| CQRS services | ✅ Complete | Command/Query separation |
| Mitigation suggestions | ✅ Complete | RAG-based suggestions |
| Unit + Integration tests | ✅ Complete | pytest + testcontainers |
| CI/CD pipeline | ✅ Complete | GitHub Actions |
| API Key Authentication | ✅ Complete | Bearer token auth |
| Multi-Tenant Support | ✅ Complete | tenant_id filtering |
| Admin API | ✅ Complete | Key management endpoints |
| Redis Rate Limiting | ✅ Complete | Sliding window algorithm |

---

## Phase 1: Foundation ✅ COMPLETE

**Goal:** Core infrastructure with embedding generation and similarity detection.

### 1.1 Project Setup ✅
- [x] Python 3.12+ project with FastAPI
- [x] Docker Compose: PostgreSQL 16 + pgvector extension
- [x] Docker Compose: Ollama with llama3.1:8b
- [x] Pydantic Settings for configuration
- [x] Health check endpoint (`/health`)

### 1.2 Database Layer ✅
- [x] `bug_embeddings` table with VECTOR(384)
- [x] IVFFlat index on embedding column
- [x] Connection pooling with psycopg3
- [x] Migration scripts

### 1.3 LLM Provider System ✅
- [x] Abstract `LLMProvider` base class
- [x] Factory pattern with decorator-based registry
- [x] Ollama provider implementation
- [x] Claude provider (Anthropic SDK)
- [x] OpenAI provider

### 1.4 Embedding Service ✅
- [x] Abstract `EmbeddingProvider` interface
- [x] Local provider (sentence-transformers, all-MiniLM-L6-v2)
- [x] OpenAI provider (text-embedding-3-small)
- [x] Factory for provider selection

### 1.5 Similarity Detection ✅
- [x] `POST /api/v1/bugs/analyze` — store bug with embedding
- [x] `GET /api/v1/bugs/{id}/similar` — find similar bugs
- [x] Configurable thresholds (duplicate ≥0.90, related 0.75–0.90)
- [x] Response includes similarity scores

### 1.6 Mitigation Suggestions ✅
- [x] `GET /api/v1/bugs/{id}/mitigation` — AI-powered suggestions
- [x] RAG context from similar resolved bugs
- [x] "Based on similar bugs" attribution

### 1.7 Testing ✅
- [x] Unit tests with mocked dependencies
- [x] Integration tests with testcontainers
- [x] GitHub Actions CI pipeline

**✅ Released: Foundation Complete**

---

## Phase 2: Authentication & Security ✅ COMPLETE

**Goal:** Production-grade security before adding more features.

### 2.1 API Authentication ✅
- [x] API key authentication middleware (Bearer token)
- [x] `api_keys` table: key_hash, tenant_id, name, created_at, last_used, is_active
- [x] Key generation endpoint (admin only)
- [x] Rate limiting per API key (Redis-based, sliding window)

### 2.2 Multi-Tenant Foundation ✅
- [x] Tenant ID extraction from API key
- [x] All queries filtered by tenant_id
- [x] Backwards compatible (NULL tenant_id for legacy data)
- [x] Tenant context via dependency injection

### 2.3 Admin API ✅
- [x] `POST /admin/api-keys` — Create API key
- [x] `GET /admin/api-keys` — List keys (masked)
- [x] `GET /admin/api-keys/{id}` — Get key details
- [x] `DELETE /admin/api-keys/{id}` — Revoke key

### 2.4 Rate Limiting ✅
- [x] Redis integration (docker-compose)
- [x] Sliding window rate limiter
- [x] X-RateLimit-* headers
- [x] 429 response with Retry-After
- [x] Graceful degradation when Redis unavailable

### 2.5 Testing ✅
- [x] Authentication unit tests
- [x] Tenant isolation tests
- [x] Rate limiter tests
- [x] Admin API tests

**✅ Released: v0.2.0 — Secure Foundation**

**Testable Milestone:** API requests require valid API key; tenants cannot see each other's data.

---

## Phase 3: Smart Search & Caching

**Goal:** Enhanced search with LLM reranking and performance optimization.

### 3.1 Enhanced Search Endpoint
- [ ] `POST /api/v1/search` — natural language bug search
- [ ] Query embedding → pgvector ANN search
- [ ] Pagination support (offset, limit, cursor)
- [ ] Filter by status, date range, severity

### 3.2 Smart Search Mode (LLM Rerank)
- [ ] `POST /api/v1/search?mode=smart`
- [ ] Retrieve top-20 → LLM scores relevance → return top-5
- [ ] Fallback to fast mode if LLM unavailable
- [ ] Latency budget (timeout after 10s)

### 3.3 Redis Integration
- [ ] Add Redis to docker-compose.yml
- [ ] Query result caching (TTL: 5min fast, 15min smart)
- [ ] Cache invalidation on new bug submission
- [ ] Embedding cache for repeated texts

### 3.4 Testing & Release
- [ ] Search accuracy benchmarks
- [ ] Latency benchmarks at 10K, 50K, 100K vectors
- [ ] Cache hit rate monitoring
- [ ] **Release v0.3.0** — Smart Search

**Testable Milestone:** Natural language query → ranked results with sub-second response (cached).

---

## Phase 4: Feedback Loop & Learning

**Goal:** Record actual fixes to improve future suggestions. This phase has high ROI.

### 4.1 Resolution Recording
- [ ] `PATCH /api/v1/bugs/{id}/resolution` — already exists, enhance it
- [ ] `verified_resolutions` table: bug_id, tenant_id, actual_root_cause, actual_fix, fix_embedding, resolved_by, resolved_at
- [ ] Generate embedding for resolution text
- [ ] Link to original AI suggestion if any

### 4.2 AI Accuracy Tracking
- [ ] Compare AI suggestion vs actual fix (semantic similarity)
- [ ] `ai_accuracy_score` field (0.0–1.0)
- [ ] Store comparison metadata for analysis

### 4.3 Enhanced RAG Context
- [ ] Include verified resolutions in similarity search
- [ ] Weight recent resolutions higher
- [ ] "Based on your fix for BUG-XXX" attribution
- [ ] Confidence boost when similar verified fix exists

### 4.4 Accuracy Reporting
- [ ] `GET /api/v1/metrics/accuracy` — AI accuracy over time
- [ ] Per-tenant accuracy breakdown
- [ ] Accuracy by bug category/severity

### 4.5 Testing & Release
- [ ] Feedback loop integration tests
- [ ] Accuracy calculation tests
- [ ] Before/after suggestion quality comparison
- [ ] **Release v0.4.0** — Learning System

**Testable Milestone:** Record fix → future similar bugs get suggestions citing that fix.

---

## Phase 5: Root Cause Analysis

**Goal:** Deeper analysis with structured output for complex bugs.

### 5.1 Analysis Endpoint
- [ ] `POST /api/v1/bugs/{id}/analyze` — comprehensive analysis
- [ ] Input: description, stack_trace, console_logs, environment
- [ ] Structured JSON output with schema validation

### 5.2 Analysis Output Schema
```json
{
  "root_cause": "Description of the root cause",
  "confidence": 0.85,
  "affected_components": ["auth-service", "user-model"],
  "related_bugs": ["BUG-123", "BUG-456"],
  "evidence": ["Stack trace shows...", "Similar to..."],
  "suggested_investigation": ["Check logs for...", "Verify that..."]
}
```

### 5.3 Analysis Storage
- [ ] `bug_analyses` table: bug_id, tenant_id, analysis_json, model_used, created_at
- [ ] Cache analysis results (invalidate on bug update)

### 5.4 Confidence Scoring
- [ ] LLM self-assessment + heuristics
- [ ] Factors: context quality, similar bug count, stack trace presence
- [ ] Low confidence flag for human review

### 5.5 Testing & Release
- [ ] Analysis quality evaluation on labeled dataset
- [ ] Structured output parsing tests
- [ ] **Release v0.5.0** — Root Cause Analysis

**Testable Milestone:** Submit bug with stack trace → structured root cause analysis with confidence.

---

## Phase 6: Bug Summarization

**Goal:** Auto-generate concise summaries for triage and reporting.

### 6.1 Summarization Endpoint
- [ ] `POST /api/v1/bugs/{id}/summarize`
- [ ] 2-3 sentence summary of bug report
- [ ] Extract: what happened, where, impact

### 6.2 Summary Storage & Caching
- [ ] `bug_summaries` table: bug_id, summary, model_name, input_hash
- [ ] Skip LLM if input unchanged (hash check)
- [ ] Bulk summarization for backfill

### 6.3 Async Processing
- [ ] ARQ worker for background summarization (foundation exists)
- [ ] `POST /api/v1/bugs/{id}/summarize?async=true`
- [ ] Progress tracking: `GET /api/v1/jobs/{job_id}`
- [ ] Webhook callback on completion

### 6.4 Testing & Release
- [ ] Summary quality evaluation
- [ ] Async flow tests
- [ ] **Release v0.6.0** — Summarization

**Testable Milestone:** Submit bug → receive async summary notification.

---

## Phase 7: Observability & Production Hardening

**Goal:** Production-grade monitoring, logging, and performance.

### 7.1 Structured Logging
- [ ] JSON log format with correlation IDs
- [ ] Tenant context in all logs
- [ ] Request/response logging (sanitized)
- [ ] Log aggregation guidance (ELK, CloudWatch)

### 7.2 Metrics & Monitoring
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] Key metrics: request latency, LLM latency, cache hit rate, queue depth
- [ ] Grafana dashboard templates
- [ ] Alert rules for SLO violations

### 7.3 Distributed Tracing
- [ ] OpenTelemetry integration
- [ ] Trace context propagation
- [ ] LLM call spans with token counts

### 7.4 Performance Optimization
- [ ] Connection pool tuning
- [ ] pgvector index optimization (IVFFlat → HNSW evaluation)
- [ ] LLM request batching where possible
- [ ] Benchmark at 100K bugs/tenant

### 7.5 Error Handling Improvements
- [ ] Retry logic with exponential backoff
- [ ] Circuit breaker for external services
- [ ] Graceful degradation (LLM unavailable → return cached/partial)

### 7.6 Testing & Release
- [ ] Load testing (k6 or locust)
- [ ] Chaos testing (kill services, network partitions)
- [ ] **Release v0.7.0** — Production Hardening

**Testable Milestone:** Full observability stack; survives component failures gracefully.

---

## Phase 8: Trend Analysis (Future)

**Goal:** Identify emerging bug patterns across the tenant's history.

### 8.1 Clustering Job
- [ ] Scheduled job: weekly clustering per tenant
- [ ] K-means or DBSCAN on bug embeddings
- [ ] `bug_clusters` table: tenant_id, cluster_id, bug_ids[], centroid, keywords[]

### 8.2 Cluster Labeling
- [ ] LLM-generated cluster names
- [ ] Keyword extraction from cluster members
- [ ] Example: "Mobile upload timeouts", "Auth token expiry"

### 8.3 Trend Detection
- [ ] Week-over-week cluster size comparison
- [ ] Flag clusters with >20% growth
- [ ] New cluster detection

### 8.4 Alerting
- [ ] `GET /api/v1/trends` — current trends
- [ ] Webhook notification for significant trends
- [ ] Slack/Teams integration via webhooks

### 8.5 Testing & Release
- [ ] Clustering quality evaluation
- [ ] Trend detection accuracy
- [ ] **Release v0.8.0** — Trend Analysis

**Testable Milestone:** Weekly job → trend alerts for growing bug clusters.

---

## Phase 9: Developer & User Responses (Future)

**Goal:** Tailored outputs for different audiences.

### 9.1 Developer Fix Suggestions
- [ ] `POST /api/v1/bugs/{id}/suggest/fix`
- [ ] Code-aware suggestions (diff format)
- [ ] Affected files, complexity estimate
- [ ] RAG from tenant's past fixes

### 9.2 End User Workarounds
- [ ] `POST /api/v1/bugs/{id}/suggest/workaround`
- [ ] Plain English, step-by-step instructions
- [ ] Safe for customer-facing use

### 9.3 Testing & Release
- [ ] Developer review of fix suggestions
- [ ] User comprehension testing
- [ ] **Release v0.9.0** — Audience-Specific Responses

---

## Version Summary

| Version | Phase | Key Feature | Status |
|---------|-------|-------------|--------|
| v0.1.0 | Phase 1 | Foundation & Similarity | ✅ Complete |
| v0.2.0 | Phase 2 | Authentication & Security | 🔜 Next |
| v0.3.0 | Phase 3 | Smart Search & Caching | Planned |
| v0.4.0 | Phase 4 | Feedback Loop & Learning | Planned |
| v0.5.0 | Phase 5 | Root Cause Analysis | Planned |
| v0.6.0 | Phase 6 | Bug Summarization | Planned |
| v0.7.0 | Phase 7 | Production Hardening | Planned |
| v0.8.0 | Phase 8 | Trend Analysis | Future |
| v0.9.0 | Phase 9 | Audience-Specific Responses | Future |
| v1.0.0 | — | Production Ready | Goal |

---

## Infrastructure Requirements

### Development
- Docker Desktop with 8GB+ RAM allocation
- PostgreSQL 16 with pgvector
- Ollama with 8B model (CPU inference OK)
- Redis (from Phase 3)

### Production (Recommended)
- **Compute:** 2+ API instances behind load balancer
- **Database:** Managed PostgreSQL with pgvector (e.g., Supabase, Neon, AWS RDS)
- **LLM Strategy:**
  - Primary: Cloud APIs (Claude, OpenAI) — reliable, no GPU needed
  - Optional: Self-hosted Ollama with 8B models for cost optimization
- **Cache:** Managed Redis (ElastiCache, Upstash)
- **Queue:** Redis-based (ARQ) or managed queue service

### Self-Hosted (Minimal)
- Single server: 16GB RAM minimum
- Docker Compose deployment
- Local PostgreSQL + Ollama (8B models only)
- Note: 70B models require dedicated GPU (A10/A100)

---

## LLM Strategy

### Recommended Approach

| Use Case | Recommended Model | Fallback |
|----------|-------------------|----------|
| Mitigation suggestions | Claude Sonnet / GPT-4o | Ollama llama3.1:8b |
| Summarization | Claude Haiku / GPT-4o-mini | Ollama llama3.1:8b |
| Root cause analysis | Claude Sonnet / GPT-4o | Ollama llama3.1:8b |
| Embeddings | Local (all-MiniLM-L6-v2) | OpenAI text-embedding-3-small |

### Why Cloud-First?
1. **Reliability:** No GPU maintenance, automatic scaling
2. **Quality:** Larger models = better analysis
3. **Cost:** Pay-per-use vs. dedicated GPU ($1-3K/month)
4. **Speed:** No model loading time

### When to Use Local (Ollama)
- Cost-sensitive high-volume workloads
- Air-gapped environments
- Development/testing
- Privacy requirements (data never leaves your infra)

---

## Key Architectural Decisions

### Already Implemented ✅
1. **CQRS Pattern:** Separate command/query services for clarity
2. **Factory + Registry:** Extensible provider system without code changes
3. **Async Everything:** Non-blocking I/O for all external calls
4. **Dependency Injection:** FastAPI's `Depends()` for testability

### Planned
1. **Multi-Tenancy:** API key → tenant_id, RLS policies
2. **Caching Layer:** Redis for queries, embeddings, LLM responses
3. **Background Jobs:** ARQ workers for async processing
4. **Circuit Breakers:** Graceful degradation when services fail

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Similar bug accuracy | >80% relevant in top-5 | Manual evaluation on test set |
| Mitigation usefulness | >60% "helpful" rating | User feedback tracking |
| API latency (p95) | <500ms (cached), <5s (LLM) | Prometheus metrics |
| AI accuracy improvement | +10% after feedback loop | Before/after comparison |
| System availability | 99.9% uptime | Monitoring alerts |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

Priority areas for contribution:
1. Additional LLM provider implementations
2. Test coverage improvements
3. Documentation and examples
4. Performance benchmarks
