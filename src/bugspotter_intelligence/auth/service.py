"""API Key service for business logic"""

from uuid import UUID

from psycopg import AsyncConnection

from .models import APIKey, TenantContext
from .repository import APIKeyRepository
from .utils import generate_api_key, get_key_prefix, hash_api_key, verify_api_key


class APIKeyService:
    """
    Business logic for API key management.

    Handles key creation, validation, listing, and revocation.
    """

    def __init__(self, key_prefix: str = "bsi_"):
        """
        Initialize the API key service.

        Args:
            key_prefix: Prefix for generated keys (default: "bsi_")
        """
        self.key_prefix = key_prefix
        self.repo = APIKeyRepository()

    async def create_key(
        self,
        conn: AsyncConnection,
        tenant_id: UUID,
        name: str,
        rate_limit_per_minute: int = 60,
        is_admin: bool = False,
    ) -> tuple[APIKey, str]:
        """
        Create a new API key.

        Args:
            conn: Database connection
            tenant_id: UUID of the tenant
            name: Human-readable name for the key
            rate_limit_per_minute: Rate limit for this key
            is_admin: Whether this is an admin key

        Returns:
            Tuple of (APIKey, plain_key). The plain key is only returned once!

        Example:
            >>> api_key, plain_key = await service.create_key(
            ...     conn, tenant_id, "Production API Key"
            ... )
            >>> print(f"Store this key: {plain_key}")
        """
        plain_key = generate_api_key(self.key_prefix)
        key_hash = hash_api_key(plain_key)
        key_prefix = get_key_prefix(plain_key)

        api_key = await self.repo.insert_key(
            conn=conn,
            tenant_id=tenant_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            rate_limit_per_minute=rate_limit_per_minute,
            is_admin=is_admin,
        )

        return api_key, plain_key

    async def validate_key(
        self,
        conn: AsyncConnection,
        plain_key: str,
    ) -> TenantContext | None:
        """
        Validate an API key and return tenant context.

        Args:
            conn: Database connection
            plain_key: The plain text API key from Authorization header

        Returns:
            TenantContext if valid, None if invalid or revoked

        Note:
            Also updates last_used_at timestamp for the key.
            Uses bcrypt.checkpw for secure hash comparison.
        """
        key_prefix = get_key_prefix(plain_key)
        candidates = await self.repo.list_by_prefix(conn, key_prefix)

        # Try to verify against each key with matching prefix
        api_key = None
        for candidate_key, candidate_hash in candidates:
            if verify_api_key(plain_key, candidate_hash):
                api_key = candidate_key
                break

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        # Update last_used timestamp
        await self.repo.update_last_used(conn, api_key.id)

        return TenantContext(
            tenant_id=api_key.tenant_id,
            api_key_id=api_key.id,
            is_admin=api_key.is_admin,
            rate_limit_per_minute=api_key.rate_limit_per_minute,
        )

    async def list_keys(
        self,
        conn: AsyncConnection,
        tenant_id: UUID,
    ) -> list[APIKey]:
        """
        List all API keys for a tenant.

        Args:
            conn: Database connection
            tenant_id: UUID of the tenant

        Returns:
            List of APIKey objects (keys are masked - only prefix shown)
        """
        return await self.repo.list_by_tenant(conn, tenant_id)

    async def revoke_key(
        self,
        conn: AsyncConnection,
        key_id: UUID,
        tenant_id: UUID,
    ) -> bool:
        """
        Revoke an API key.

        Args:
            conn: Database connection
            key_id: UUID of the key to revoke
            tenant_id: UUID of the tenant (for ownership verification)

        Returns:
            True if revoked, False if not found or not owned by tenant
        """
        return await self.repo.revoke_key(conn, key_id, tenant_id)

    async def get_key(
        self,
        conn: AsyncConnection,
        key_id: UUID,
        tenant_id: UUID,
    ) -> APIKey | None:
        """
        Get a specific API key by ID.

        Args:
            conn: Database connection
            key_id: UUID of the key
            tenant_id: UUID of the tenant (for ownership verification)

        Returns:
            APIKey if found and owned by tenant, None otherwise
        """
        return await self.repo.get_by_id(conn, key_id, tenant_id)
