"""Tests for cache key builder"""

from uuid import uuid4

from bugspotter_intelligence.cache.keys import CacheKeyBuilder


class TestSearchKey:
    """Tests for CacheKeyBuilder.search_key"""

    def test_includes_mode_namespace(self):
        """Should use correct namespace for fast and smart modes"""
        tid = uuid4()
        fast_key = CacheKeyBuilder.search_key(tid, "abc123", mode="fast")
        smart_key = CacheKeyBuilder.search_key(tid, "abc123", mode="smart")

        assert fast_key.startswith("search:fast:")
        assert smart_key.startswith("search:smart:")

    def test_includes_tenant_id(self):
        """Should include tenant_id in key"""
        tid = uuid4()
        key = CacheKeyBuilder.search_key(tid, "abc123", mode="fast")

        assert str(tid) in key

    def test_includes_token(self):
        """Should include invalidation token in key"""
        tid = uuid4()
        key_t0 = CacheKeyBuilder.search_key(tid, "abc123", mode="fast", token=0)
        key_t5 = CacheKeyBuilder.search_key(tid, "abc123", mode="fast", token=5)

        assert ":t0:" in key_t0
        assert ":t5:" in key_t5
        assert key_t0 != key_t5

    def test_different_tenants_produce_different_keys(self):
        """Should produce different keys for different tenants"""
        key1 = CacheKeyBuilder.search_key(uuid4(), "abc123", mode="fast")
        key2 = CacheKeyBuilder.search_key(uuid4(), "abc123", mode="fast")

        assert key1 != key2

    def test_same_inputs_produce_same_key(self):
        """Should produce deterministic keys"""
        tid = uuid4()
        key1 = CacheKeyBuilder.search_key(tid, "abc123", mode="fast", token=1)
        key2 = CacheKeyBuilder.search_key(tid, "abc123", mode="fast", token=1)

        assert key1 == key2


class TestEmbeddingKey:
    """Tests for CacheKeyBuilder.embedding_key"""

    def test_uses_embedding_namespace(self):
        """Should use embed namespace"""
        key = CacheKeyBuilder.embedding_key("abc123")

        assert key.startswith("embed:")

    def test_includes_text_hash(self):
        """Should include the text hash"""
        key = CacheKeyBuilder.embedding_key("deadbeef")

        assert key == "embed:deadbeef"


class TestTenantVersionKey:
    """Tests for CacheKeyBuilder.tenant_version_key"""

    def test_uses_tenant_version_namespace(self):
        """Should use tenant:ts namespace"""
        tid = uuid4()
        key = CacheKeyBuilder.tenant_version_key(tid)

        assert key.startswith("tenant:ts:")
        assert str(tid) in key


class TestHashQuery:
    """Tests for CacheKeyBuilder.hash_query"""

    def test_deterministic_for_same_inputs(self):
        """Should produce same hash for same query and filters"""
        hash1 = CacheKeyBuilder.hash_query("login crash", {"status": "open"})
        hash2 = CacheKeyBuilder.hash_query("login crash", {"status": "open"})

        assert hash1 == hash2

    def test_different_queries_produce_different_hashes(self):
        """Should produce different hashes for different queries"""
        hash1 = CacheKeyBuilder.hash_query("login crash", {})
        hash2 = CacheKeyBuilder.hash_query("signup error", {})

        assert hash1 != hash2

    def test_different_filters_produce_different_hashes(self):
        """Should produce different hashes when filters differ"""
        hash1 = CacheKeyBuilder.hash_query("login crash", {"status": "open"})
        hash2 = CacheKeyBuilder.hash_query("login crash", {"status": "closed"})

        assert hash1 != hash2

    def test_filter_order_does_not_matter(self):
        """Should produce same hash regardless of filter key order"""
        hash1 = CacheKeyBuilder.hash_query("q", {"a": "1", "b": "2"})
        hash2 = CacheKeyBuilder.hash_query("q", {"b": "2", "a": "1"})

        assert hash1 == hash2

    def test_returns_16_char_hex_string(self):
        """Should return a 16-character hex string"""
        result = CacheKeyBuilder.hash_query("test", {})

        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)
