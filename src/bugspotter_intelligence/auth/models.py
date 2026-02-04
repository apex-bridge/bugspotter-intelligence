"""Authentication domain models"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class APIKey(BaseModel):
    """API Key domain model"""

    id: UUID
    tenant_id: UUID
    key_prefix: str = Field(..., description="First 12 chars for display (e.g., 'bsi_abc12345')")
    name: str = Field(..., description="Human-readable name for the key")
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)
    is_admin: bool = False

    @property
    def is_active(self) -> bool:
        """Check if the API key is active (not revoked)"""
        return self.revoked_at is None

    model_config = {"from_attributes": True}


class TenantContext(BaseModel):
    """
    Tenant context passed through request lifecycle.

    This is injected into route handlers via dependency injection
    after API key validation.
    """

    tenant_id: UUID
    api_key_id: UUID
    is_admin: bool = False
    rate_limit_per_minute: int = 60
