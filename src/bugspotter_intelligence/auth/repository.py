"""API Key repository for database operations"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from .models import APIKey


class APIKeyRepository:
    """Data access layer for api_keys table"""

    @staticmethod
    async def insert_key(
        conn: AsyncConnection,
        tenant_id: UUID,
        key_hash: str,
        key_prefix: str,
        name: str,
        rate_limit_per_minute: int = 60,
        is_admin: bool = False,
    ) -> APIKey:
        """
        Insert a new API key.

        Args:
            conn: Database connection
            tenant_id: UUID of the tenant
            key_hash: bcrypt hash of the API key
            key_prefix: First 12 chars of the key for display
            name: Human-readable name
            rate_limit_per_minute: Rate limit for this key
            is_admin: Whether this is an admin key

        Returns:
            The created APIKey object
        """
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                INSERT INTO api_keys
                    (tenant_id, key_hash, key_prefix, name, rate_limit_per_minute, is_admin)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, tenant_id, key_prefix, name, created_at,
                          last_used_at, revoked_at, rate_limit_per_minute, is_admin
                """,
                (tenant_id, key_hash, key_prefix, name, rate_limit_per_minute, is_admin),
            )
            row = await cursor.fetchone()
            await conn.commit()

            return APIKey.model_validate(row)

    @staticmethod
    async def get_by_hash(conn: AsyncConnection, key_hash: str) -> Optional[APIKey]:
        """
        Get API key by its hash.

        Args:
            conn: Database connection
            key_hash: Hash of the API key (legacy method, prefer list_by_prefix)

        Returns:
            APIKey if found, None otherwise

        Note:
            This method is deprecated for bcrypt keys since bcrypt hashes
            are non-deterministic. Use list_by_prefix() and verify_api_key() instead.
        """
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT id, tenant_id, key_prefix, name, created_at,
                       last_used_at, revoked_at, rate_limit_per_minute, is_admin
                FROM api_keys
                WHERE key_hash = %s
                """,
                (key_hash,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return APIKey.model_validate(row)

    @staticmethod
    async def list_by_prefix(
        conn: AsyncConnection, key_prefix: str
    ) -> list[tuple[APIKey, str]]:
        """
        List all active API keys with a matching prefix, including their hashes.

        Args:
            conn: Database connection
            key_prefix: Key prefix to match (e.g., first 12 characters)

        Returns:
            List of (APIKey, key_hash) tuples for verification

        Note:
            Used for bcrypt verification where we can't do direct hash lookup.
            Returns only active (non-revoked) keys.
            The key_hash is returned for verification but not included in APIKey model.
        """
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT id, tenant_id, key_prefix, name, created_at,
                       last_used_at, revoked_at, rate_limit_per_minute, is_admin, key_hash
                FROM api_keys
                WHERE key_prefix = %s AND revoked_at IS NULL
                ORDER BY created_at DESC
                """,
                (key_prefix,),
            )
            rows = await cursor.fetchall()

            return [
                (
                    APIKey.model_validate(row),
                    row['key_hash'],
                )
                for row in rows
            ]

    @staticmethod
    async def list_by_tenant(conn: AsyncConnection, tenant_id: UUID) -> list[APIKey]:
        """
        List all API keys for a tenant.

        Args:
            conn: Database connection
            tenant_id: UUID of the tenant

        Returns:
            List of APIKey objects (may be empty)
        """
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT id, tenant_id, key_prefix, name, created_at,
                       last_used_at, revoked_at, rate_limit_per_minute, is_admin
                FROM api_keys
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            )
            rows = await cursor.fetchall()

            return [APIKey.model_validate(row) for row in rows]

    @staticmethod
    async def update_last_used(conn: AsyncConnection, key_id: UUID) -> None:
        """
        Update the last_used_at timestamp for an API key.

        Args:
            conn: Database connection
            key_id: UUID of the API key
        """
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE api_keys
                SET last_used_at = %s
                WHERE id = %s
                """,
                (datetime.now(), key_id),
            )
            await conn.commit()

    @staticmethod
    async def revoke_key(
        conn: AsyncConnection, key_id: UUID, tenant_id: UUID
    ) -> bool:
        """
        Revoke an API key (soft delete).

        Args:
            conn: Database connection
            key_id: UUID of the API key to revoke
            tenant_id: UUID of the tenant (for ownership verification)

        Returns:
            True if key was revoked, False if not found or not owned by tenant
        """
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE api_keys
                SET revoked_at = %s
                WHERE id = %s AND tenant_id = %s AND revoked_at IS NULL
                """,
                (datetime.now(), key_id, tenant_id),
            )
            await conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    async def get_by_id(
        conn: AsyncConnection, key_id: UUID, tenant_id: UUID
    ) -> Optional[APIKey]:
        """
        Get API key by ID (with tenant verification).

        Args:
            conn: Database connection
            key_id: UUID of the API key
            tenant_id: UUID of the tenant (for ownership verification)

        Returns:
            APIKey if found and owned by tenant, None otherwise
        """
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT id, tenant_id, key_prefix, name, created_at,
                       last_used_at, revoked_at, rate_limit_per_minute, is_admin
                FROM api_keys
                WHERE id = %s AND tenant_id = %s
                """,
                (key_id, tenant_id),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return APIKey.model_validate(row)
