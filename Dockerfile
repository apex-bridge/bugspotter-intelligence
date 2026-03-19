# ============================================================================
# BugSpotter Intelligence - Multi-Stage Docker Build
# ============================================================================
# Stage 1: Builder - Install dependencies and prepare the application
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

# Copy dependency files and source (pip install . needs the source)
COPY pyproject.toml ./
COPY src/ ./src/

# Install production dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

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

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package itself (source only, deps already in venv)
RUN pip install --no-cache-dir --no-deps -e .

# Set all model/cache directories to /app/.cache (HuggingFace, sentence-transformers, torch)
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch && \
    chown -R bugspotter:bugspotter /app/.cache

# Switch to non-root user
USER bugspotter

# Pre-download the embedding model at build time so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "bugspotter_intelligence.main:app", "--host", "0.0.0.0", "--port", "8000"]
