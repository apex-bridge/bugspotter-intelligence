from datetime import datetime
from typing import Optional
from uuid import UUID

from psycopg import AsyncConnection


class BugRepository:
    """Data access layer for bug_embeddings table"""

    @staticmethod
    async def insert_bug(
        conn: AsyncConnection,
        bug_id: str,
        title: str,
        description: Optional[str],
        embedding: list[float],
        tenant_id: Optional[UUID] = None,
    ) -> None:
        """
        Insert or update bug embedding.

        Args:
            conn: Database connection
            bug_id: Unique bug identifier
            title: Bug title
            description: Bug description
            embedding: Vector embedding
            tenant_id: Optional tenant UUID (for multi-tenancy)
        """
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO bug_embeddings
                    (bug_id, title, description, embedding, last_accessed, tenant_id)
                VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (bug_id)
                DO
                UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP,
                    last_accessed = EXCLUDED.last_accessed,
                    tenant_id = COALESCE(EXCLUDED.tenant_id, bug_embeddings.tenant_id)
                """,
                (bug_id, title, description, embedding, datetime.now(), tenant_id),
            )
            await conn.commit()

    @staticmethod
    async def find_similar(
        conn: AsyncConnection,
        embedding: list[float],
        limit: int = 5,
        threshold: float = 0.7,
        tenant_id: Optional[UUID] = None,
        exclude_bug_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Find similar bugs using vector similarity.

        Args:
            conn: Database connection
            embedding: Vector embedding to search with
            limit: Maximum number of results
            threshold: Minimum similarity score (0.0-1.0)
            tenant_id: Filter by tenant (None = no filter for legacy data)
            exclude_bug_id: Bug ID to exclude from results

        Returns:
            List of similar bugs with similarity scores
        """
        async with conn.cursor() as cursor:
            # Build query with optional tenant filter
            query = """
                SELECT bug_id,
                       title,
                       description,
                       status,
                       resolution,
                       1 - (embedding <=> %s::vector) as similarity
                FROM bug_embeddings
                WHERE 1 - (embedding <=> %s::vector) >= %s
                  AND status != 'duplicate'
            """
            params: list = [embedding, embedding, threshold]

            # Add tenant filter if provided
            if tenant_id is not None:
                query += " AND (tenant_id = %s OR tenant_id IS NULL)"
                params.append(tenant_id)

            # Exclude specific bug (useful when finding duplicates)
            if exclude_bug_id is not None:
                query += " AND bug_id != %s"
                params.append(exclude_bug_id)

            query += """
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params.extend([embedding, limit])

            await cursor.execute(query, params)
            rows = await cursor.fetchall()

            return [
                {
                    "bug_id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "status": row[3],
                    "resolution": row[4],
                    "similarity": float(row[5]),
                }
                for row in rows
            ]

    @staticmethod
    async def get_bug(
        conn: AsyncConnection,
        bug_id: str,
        tenant_id: Optional[UUID] = None,
    ) -> Optional[dict]:
        """
        Get a single bug by ID.

        Args:
            conn: Database connection
            bug_id: Bug identifier
            tenant_id: Filter by tenant (None = no filter for legacy data)

        Returns:
            Bug dict if found, None otherwise
        """
        async with conn.cursor() as cursor:
            query = """
                SELECT bug_id,
                       title,
                       description,
                       status,
                       resolution,
                       resolution_summary,
                       created_at,
                       updated_at,
                       tenant_id
                FROM bug_embeddings
                WHERE bug_id = %s
            """
            params: list = [bug_id]

            # Add tenant filter if provided
            if tenant_id is not None:
                query += " AND (tenant_id = %s OR tenant_id IS NULL)"
                params.append(tenant_id)

            await cursor.execute(query, params)
            row = await cursor.fetchone()

            if not row:
                return None

            return {
                "bug_id": row[0],
                "title": row[1],
                "description": row[2],
                "status": row[3],
                "resolution": row[4],
                "resolution_summary": row[5],
                "created_at": row[6],
                "updated_at": row[7],
                "tenant_id": row[8],
            }

    @staticmethod
    async def update_resolution(
        conn: AsyncConnection,
        bug_id: str,
        resolution: str,
        resolution_summary: Optional[str] = None,
        status: str = "resolved",
        tenant_id: Optional[UUID] = None,
    ) -> bool:
        """
        Update bug resolution information.

        Args:
            conn: Database connection
            bug_id: Bug identifier
            resolution: Resolution text
            resolution_summary: Optional AI-generated summary
            status: New status (default: "resolved")
            tenant_id: Filter by tenant (for ownership verification)

        Returns:
            True if updated, False if not found or not owned by tenant
        """
        async with conn.cursor() as cursor:
            query = """
                UPDATE bug_embeddings
                SET resolution         = %s,
                    resolution_summary = %s,
                    status             = %s,
                    updated_at         = CURRENT_TIMESTAMP
                WHERE bug_id = %s
            """
            params: list = [resolution, resolution_summary, status, bug_id]

            # Add tenant filter if provided
            if tenant_id is not None:
                query += " AND (tenant_id = %s OR tenant_id IS NULL)"
                params.append(tenant_id)

            await cursor.execute(query, params)
            await conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    async def search(
        conn: AsyncConnection,
        embedding: list[float],
        *,
        tenant_id: UUID,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> tuple[list[dict], int]:
        """
        Search bugs by vector similarity with filters and pagination.

        Uses cosine similarity ordering with optional status and date filters.
        Returns all results for the tenant ordered by relevance.

        Uses two separate queries for better performance:
        1. COUNT query to get total matching documents
        2. SELECT query with pagination to fetch current page

        Args:
            conn: Database connection
            embedding: Query embedding vector
            tenant_id: Tenant UUID for isolation
            limit: Page size
            offset: Page offset
            status: Optional status filter
            date_from: Optional created_at lower bound
            date_to: Optional created_at upper bound

        Returns:
            Tuple of (result dicts, total matching count)
        """
        async with conn.cursor() as cursor:
            # Build WHERE clause for both queries
            where_conditions = ["(tenant_id = %s OR tenant_id IS NULL)"]
            count_params: list = [tenant_id]

            if status is not None:
                where_conditions.append("status = %s")
                count_params.append(status)

            if date_from is not None:
                where_conditions.append("created_at >= %s")
                count_params.append(date_from)

            if date_to is not None:
                where_conditions.append("created_at <= %s")
                count_params.append(date_to)

            where_clause = " AND ".join(where_conditions)

            # Query 1: Get total count
            count_query = f"""
                SELECT COUNT(*)
                FROM bug_embeddings
                WHERE {where_clause}
            """
            await cursor.execute(count_query, count_params)
            total_row = await cursor.fetchone()
            total = total_row[0] if total_row else 0

            if total == 0:
                return [], 0

            # Query 2: Get paginated results
            select_query = f"""
                SELECT bug_id,
                       title,
                       description,
                       status,
                       resolution,
                       1 - (embedding <=> %s::vector) as similarity,
                       created_at
                FROM bug_embeddings
                WHERE {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s OFFSET %s
            """
            select_params = [embedding] + count_params + [embedding, limit, offset]

            await cursor.execute(select_query, select_params)
            rows = await cursor.fetchall()

            results = [
                {
                    "bug_id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "status": row[3],
                    "resolution": row[4],
                    "similarity": float(row[5]),
                    "created_at": row[6],
                }
                for row in rows
            ]

            return results, total
