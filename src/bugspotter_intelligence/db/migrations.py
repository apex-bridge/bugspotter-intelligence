"""Database migrations and schema setup"""

from psycopg import AsyncConnection


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

        # Index for fast key lookup
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS api_keys_key_hash_idx
            ON api_keys(key_hash)
        """)

        # Index for listing keys by tenant
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS api_keys_tenant_idx
            ON api_keys(tenant_id)
        """)

        # Index for prefix-based key lookup (used in bcrypt validation flow)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS api_keys_key_prefix_idx
            ON api_keys(key_prefix)
        """)

        # Partial index for active keys only
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS api_keys_active_idx
            ON api_keys(key_hash) WHERE revoked_at IS NULL
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


async def create_tables(conn: AsyncConnection) -> None:
    """
    Create all required tables

    Called during application startup to ensure schema exists
    """
    async with conn.cursor() as cursor:
        # Enable pgvector extension
        await cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create bug_embeddings table
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
                                 384
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
