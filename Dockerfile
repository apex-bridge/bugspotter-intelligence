# ============================================================================
# BugSpotter Intelligence - Multi-Stage Docker Build
# ============================================================================
# Stage 1: Builder - Install dependencies, package, and download model
# Stage 2: Production - Minimal runtime image
# ============================================================================

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies for native modules (psycopg, bcrypt, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the fully-pinned dependency closure in its own layer, keyed ONLY on
# requirements.lock — copying src/ first would bust this cache on every code
# change and force a full reinstall of heavy deps (torch/CUDA). The lock pins
# every transitive version so the image builds reproducibly; without it,
# pyproject's `>=` ranges float to the newest release on each build (how
# FastAPI 0.137 shipped silently and broke the API, see #43). Regenerate the
# lock per its header.
COPY requirements.lock ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Pin pip too (the last unpinned build tool) for fully deterministic builds —
# 26.1.2 matches the verified prod runtime. setuptools/wheel are pinned in the
# lock; bump all three together.
RUN pip install --no-cache-dir "pip==26.1.2" && \
    pip install --no-cache-dir -r requirements.lock

# Pre-download the active embedding model in the builder stage. Baking
# the model into the image trades a larger image (~3 GB total vs ~1 GB)
# for predictable cold starts: no HF Hub download on first request, no
# OOM-risk window where lazy load races against the worker timeout.
# Placed before the source copy so a code change doesn't re-download it;
# it depends only on sentence-transformers from the lock layer above.
#
# IMPORTANT: this MUST match db/migrations.py target_dim and the active
# EMBEDDING_MODEL passed via env. Currently BAAI/bge-m3 (1024-dim,
# ~2.3 GB on disk). When changing models, update target_dim in
# migrations.py *and* this line in the same commit.
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Then the local package (fast layer) — WITHOUT re-resolving its deps and
# WITHOUT build isolation, so it builds from the lock-pinned setuptools/wheel
# with no PyPI fetch. Only this layer rebuilds when source changes.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps --no-build-isolation .

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM python:3.12-slim AS production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (nologin — no interactive shell needed)
RUN groupadd -g 1001 bugspotter && \
    useradd -u 1001 -g bugspotter -s /sbin/nologin bugspotter

WORKDIR /app

# Copy virtual environment (includes installed package + all dependencies)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded model cache from builder, owned by the runtime
# user up front (avoids a second 2.3 GB layer that a separate `chown
# -R` would otherwise produce). TRANSFORMERS_OFFLINE=1 forbids
# huggingface_hub from making any network calls — the model is on
# disk, we never want to silently re-download or hit Hub rate limits
# at runtime.
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1
COPY --from=builder --chown=bugspotter:bugspotter /app/.cache /app/.cache

# Copy LICENSE (MIT compliance requires inclusion in distributed software)
COPY LICENSE ./

# Switch to non-root user
USER bugspotter

EXPOSE 8000

# start-period covers the lazy-load of BGE-M3 from the baked cache
# (~5-15s cold) plus any first-request overhead.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "bugspotter_intelligence.main:app", "--host", "0.0.0.0", "--port", "8000"]
