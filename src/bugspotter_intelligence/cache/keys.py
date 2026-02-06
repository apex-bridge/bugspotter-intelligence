"""Cache key builder with namespaced, tenant-scoped keys"""

import hashlib
import json
from uuid import UUID


class CacheKeyBuilder:
    """Builds deterministic, namespaced cache keys with tenant isolation."""

    SEARCH_FAST_NS = "search:fast"
    SEARCH_SMART_NS = "search:smart"
    EMBEDDING_NS = "embed"
    TENANT_VERSION_NS = "tenant:ts"

    @staticmethod
    def search_key(tenant_id: UUID, query_hash: str, mode: str, token: int = 0) -> str:
        """
        Build cache key for search results.

        Format: search:{mode}:{tenant_id}:t{token}:{query_hash}

        The token component ensures old cached entries are ignored
        after tenant data changes (new bug submitted, etc.).
        """
        ns = (
            CacheKeyBuilder.SEARCH_SMART_NS
            if mode == "smart"
            else CacheKeyBuilder.SEARCH_FAST_NS
        )
        return f"{ns}:{tenant_id}:t{token}:{query_hash}"

    @staticmethod
    def embedding_key(text_hash: str) -> str:
        """
        Build cache key for embedding vectors.

        Format: embed:{text_hash}
        """
        return f"{CacheKeyBuilder.EMBEDDING_NS}:{text_hash}"

    @staticmethod
    def tenant_version_key(tenant_id: UUID) -> str:
        """
        Build key for tenant invalidation timestamp.

        Format: tenant:ts:{tenant_id}
        """
        return f"{CacheKeyBuilder.TENANT_VERSION_NS}:{tenant_id}"

    @staticmethod
    def hash_query(query: str, filters: dict) -> str:
        """
        Create deterministic hash of query + filters for use in cache keys.

        Args:
            query: The search query text
            filters: Dict of filter parameters (status, date_from, date_to, limit, offset)

        Returns:
            SHA256 hex digest (first 16 chars for compactness)
        """
        # Sort filters for deterministic ordering
        canonical = json.dumps({"q": query, "f": filters}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
