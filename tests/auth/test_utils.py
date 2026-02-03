"""Tests for authentication utility functions"""

import pytest

from bugspotter_intelligence.auth.utils import (
    generate_api_key,
    get_key_prefix,
    hash_api_key,
)


class TestGenerateApiKey:
    """Test suite for generate_api_key function"""

    def test_generates_key_with_default_prefix(self):
        """Should generate key with default 'bsi_' prefix"""
        key = generate_api_key()
        assert key.startswith("bsi_")

    def test_generates_key_with_custom_prefix(self):
        """Should generate key with custom prefix"""
        key = generate_api_key(prefix="test_")
        assert key.startswith("test_")

    def test_generates_unique_keys(self):
        """Should generate unique keys each time"""
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_key_length_is_sufficient(self):
        """Should generate sufficiently long keys for security"""
        key = generate_api_key()
        # prefix (4) + 43 chars from token_urlsafe(32) = ~47 chars
        assert len(key) >= 40

    def test_key_is_url_safe(self):
        """Should generate URL-safe characters"""
        key = generate_api_key()
        # URL-safe base64 only contains alphanumeric, dash, and underscore
        valid_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        )
        key_without_prefix = key[4:]  # Remove 'bsi_'
        assert all(c in valid_chars for c in key_without_prefix)


class TestHashApiKey:
    """Test suite for hash_api_key function"""

    def test_returns_sha256_hex_digest(self):
        """Should return 64-character hex string (SHA256)"""
        key_hash = hash_api_key("test_key")
        assert len(key_hash) == 64
        assert all(c in "0123456789abcdef" for c in key_hash)

    def test_same_key_produces_same_hash(self):
        """Should produce consistent hashes"""
        key = "bsi_test123"
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)
        assert hash1 == hash2

    def test_different_keys_produce_different_hashes(self):
        """Should produce different hashes for different keys"""
        hash1 = hash_api_key("key1")
        hash2 = hash_api_key("key2")
        assert hash1 != hash2

    def test_known_hash_value(self):
        """Should produce known SHA256 hash"""
        # SHA256 of "test" is well-known
        key_hash = hash_api_key("test")
        expected = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        assert key_hash == expected


class TestGetKeyPrefix:
    """Test suite for get_key_prefix function"""

    def test_returns_first_12_chars_by_default(self):
        """Should return first 12 characters"""
        key = "bsi_abc123456789xyz"
        prefix = get_key_prefix(key)
        assert prefix == "bsi_abc12345"
        assert len(prefix) == 12

    def test_returns_custom_length(self):
        """Should return custom number of characters"""
        key = "bsi_abc123456789xyz"
        prefix = get_key_prefix(key, length=8)
        assert prefix == "bsi_abc1"

    def test_returns_full_key_if_shorter_than_length(self):
        """Should return full key if shorter than requested length"""
        key = "short"
        prefix = get_key_prefix(key, length=12)
        assert prefix == "short"

    def test_returns_exact_length_when_equal(self):
        """Should return exact key when length matches"""
        key = "exactly12chr"
        prefix = get_key_prefix(key, length=12)
        assert prefix == key
