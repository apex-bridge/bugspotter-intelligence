"""Database migrations and schema setup"""

import logging
from psycopg import AsyncConnection

logger = logging.getLogger(__name__)


async def create_api_keys_table(conn: AsyncConnection) -> None:
    """Create api_keys table for authentication"""
    async with conn.cursor() as cursor:
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id UUID NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                revoked_at TIMESTAMP,
                rate_limit_per_minute INT DEFAULT 60,
                is_admin BOOLEAN DEFAULT FALSE
            )
        """)

        # Index for listing keys by tenant
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS api_keys_tenant_idx
            ON api_keys(tenant_id)
        """)

        # Partial index for active key lookup by prefix (authentication flow)
        # This covers: WHERE key_prefix = ? AND revoked_at IS NULL
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS api_keys_prefix_active_idx
            ON api_keys(key_prefix) WHERE revoked_at IS NULL
        """)

        # Composite index for listing active keys by tenant
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS api_keys_tenant_active_idx
            ON api_keys(tenant_id, revoked_at)
        """)

        await conn.commit()
        print("✅ api_keys table created successfully")


async def add_tenant_id_to_bug_embeddings(conn: AsyncConnection) -> None:
    """Add tenant_id column to bug_embeddings table (nullable for migration)"""
    async with conn.cursor() as cursor:
        # Check if column already exists
        await cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'bug_embeddings' AND column_name = 'tenant_id'
        """)
        if await cursor.fetchone():
            print("ℹ️  tenant_id column already exists in bug_embeddings")
            return

        # Add tenant_id column (nullable for backwards compatibility)
        await cursor.execute("""
            ALTER TABLE bug_embeddings
            ADD COLUMN tenant_id UUID
        """)

        # Index for tenant filtering
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS bug_embeddings_tenant_idx
            ON bug_embeddings(tenant_id)
        """)

        # Composite index for tenant + status queries
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS bug_embeddings_tenant_status_idx
            ON bug_embeddings(tenant_id, status)
        """)

        await conn.commit()
        print("✅ tenant_id column added to bug_embeddings")


async def add_search_indexes(conn: AsyncConnection) -> None:
    """Add composite index for filtered search queries"""
    async with conn.cursor() as cursor:
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS bug_embeddings_tenant_created_idx
            ON bug_embeddings(tenant_id, created_at)
        """)
        await conn.commit()
        print("✅ Search indexes created")


async def create_intelligence_event_table(conn: AsyncConnection) -> None:
    """Create intelligence_event table — immutable audit record per LLM call"""
    async with conn.cursor() as cursor:
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS intelligence_event (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id       UUID NOT NULL,
                operation       TEXT NOT NULL,
                bug_id          TEXT,
                provider        TEXT NOT NULL,
                model           TEXT NOT NULL,
                prompt_version  TEXT NOT NULL,
                tokens_in       INTEGER,
                tokens_out      INTEGER,
                cost_micros_usd BIGINT,
                latency_ms      INTEGER NOT NULL,
                confidence      REAL,
                rationale       TEXT,
                status          TEXT NOT NULL,
                error_kind      TEXT,
                cached          BOOLEAN NOT NULL DEFAULT FALSE,
                meta            JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS intelligence_event_tenant_created_idx
            ON intelligence_event(tenant_id, created_at DESC)
        """)

        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS intelligence_event_tenant_op_created_idx
            ON intelligence_event(tenant_id, operation, created_at DESC)
        """)

        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS intelligence_event_tenant_bug_idx
            ON intelligence_event(tenant_id, bug_id) WHERE bug_id IS NOT NULL
        """)

        await conn.commit()
        print("✅ intelligence_event table created successfully")


async def create_intelligence_feedback_table(conn: AsyncConnection) -> None:
    """Create intelligence_feedback table — user verdicts on intelligence_event rows"""
    async with conn.cursor() as cursor:
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS intelligence_feedback (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                event_id    UUID NOT NULL REFERENCES intelligence_event(id) ON DELETE CASCADE,
                tenant_id   UUID NOT NULL,
                verdict     TEXT NOT NULL,
                user_ref    TEXT,
                note        TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (event_id, user_ref)
            )
        """)

        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS intelligence_feedback_tenant_created_idx
            ON intelligence_feedback(tenant_id, created_at DESC)
        """)

        # FK index: cascade DELETE on intelligence_event and the accuracy JOIN
        # would otherwise scan the full feedback table.
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS intelligence_feedback_event_idx
            ON intelligence_feedback(event_id)
        """)

        # Plain UNIQUE treats NULLs as distinct, so anonymous feedback
        # (user_ref IS NULL) could insert unlimited duplicate rows. Endpoint
        # normalizes user_ref to a sentinel for belt-and-suspenders dedup;
        # this partial index is the DB-side guard.
        await cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS intelligence_feedback_event_anon_idx
            ON intelligence_feedback (event_id) WHERE user_ref IS NULL
        """)

        await conn.commit()
        print("✅ intelligence_feedback table created successfully")


async def create_tables(conn: AsyncConnection) -> None:
    """
    Create all required tables

    Called during application startup to ensure schema exists
    """
    async with conn.cursor() as cursor:
        # Enable pgvector extension
        await cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Migrate embedding dimension if table exists with wrong size.
        # Use format_type() to reliably get "vector(N)" string, since
        # atttypmod encodes metadata (not raw dimension).
        target_dim = 1024
        await cursor.execute("""
            SELECT format_type(atttypid, atttypmod) AS col_type
            FROM pg_attribute
            WHERE attrelid = to_regclass('bug_embeddings')
            AND attname = 'embedding'
        """)
        row = await cursor.fetchone()
        if row is not None and row[0] is not None:
            col_type = row[0]  # e.g. "vector(384)" or "vector(1024)"
            import re
            dim_match = re.search(r'vector\((\d+)\)', col_type)
            current_dim = int(dim_match.group(1)) if dim_match else None
            if current_dim is not None and current_dim != target_dim:
                logger.info(f"Migrating bug_embeddings.embedding from {current_dim}d to {target_dim}d")
                await cursor.execute("DROP INDEX IF EXISTS bug_embeddings_embedding_idx;")
                await cursor.execute("ALTER TABLE bug_embeddings DROP COLUMN embedding;")
                await cursor.execute(f"ALTER TABLE bug_embeddings ADD COLUMN embedding VECTOR({target_dim});")
                logger.info("Migration complete — existing embeddings dropped. Re-embed all bugs.")

        # Create bug_embeddings table (for fresh installs)
        await cursor.execute("""
                             CREATE TABLE IF NOT EXISTS bug_embeddings
                             (
                                 bug_id
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 title
                                 TEXT
                                 NOT
                                 NULL,
                                 description
                                 TEXT,
                                 status
                                 TEXT
                                 DEFAULT
                                 'open',
                                 resolution
                                 TEXT,
                                 resolution_summary
                                 TEXT,
                                 embedding
                                 VECTOR
                             (
                                 1024
                             ),
                                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                 last_accessed TIMESTAMP
                                 );
                             """)

        # Create indexes
        await cursor.execute("""
                             CREATE INDEX IF NOT EXISTS bug_embeddings_embedding_idx
                                 ON bug_embeddings
                                 USING ivfflat (embedding vector_cosine_ops)
                                 WITH (lists = 100);
                             """)

        await cursor.execute("""
                             CREATE INDEX IF NOT EXISTS bug_embeddings_status_idx
                                 ON bug_embeddings(status);
                             """)

        await cursor.execute("""
                             CREATE INDEX IF NOT EXISTS bug_embeddings_accessed_idx
                                 ON bug_embeddings(last_accessed);
                             """)

        await conn.commit()
        print("✅ bug_embeddings table created successfully")

    # Create api_keys table
    await create_api_keys_table(conn)

    # Add tenant_id to bug_embeddings (migration)
    await add_tenant_id_to_bug_embeddings(conn)

    # Add search indexes
    await add_search_indexes(conn)

    # AI observability tables (Sprint 1)
    await create_intelligence_event_table(conn)
    await create_intelligence_feedback_table(conn)
