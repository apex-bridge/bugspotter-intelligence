"""Authentication module for API key management and tenant isolation"""

from .dependencies import (
    get_api_key_service,
    get_current_tenant,
    get_optional_tenant,
    require_admin,
)
from .models import APIKey, TenantContext
from .repository import APIKeyRepository
from .service import APIKeyService
from .utils import generate_api_key, get_key_prefix, hash_api_key, verify_api_key

__all__ = [
    # Models
    "APIKey",
    "TenantContext",
    # Repository & Service
    "APIKeyRepository",
    "APIKeyService",
    # Dependencies
    "get_api_key_service",
    "get_current_tenant",
    "get_optional_tenant",
    "require_admin",
    # Utils
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "get_key_prefix",
]
