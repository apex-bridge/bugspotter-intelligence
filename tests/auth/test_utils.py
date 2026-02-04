"""Tests for authentication utility functions"""

from bugspotter_intelligence.auth.utils import (
    generate_api_key,
    get_key_prefix,
    hash_api_key,
    verify_api_key,
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

    def test_returns_bcrypt_hash(self):
        """Should return bcrypt hash string"""
        key_hash = hash_api_key("test_key")
        # bcrypt hashes are 60 characters and start with $2b$
        assert len(key_hash) == 60
        assert key_hash.startswith("$2b$")

    def test_same_key_produces_different_hashes(self):
        """Should produce different hashes due to random salt"""
        key = "bsi_test123"
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)
        # bcrypt uses random salts, so hashes will differ
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_api_key(key, hash1)
        assert verify_api_key(key, hash2)

    def test_different_keys_produce_different_hashes(self):
        """Should produce different hashes for different keys"""
        hash1 = hash_api_key("key1")
        hash2 = hash_api_key("key2")
        assert hash1 != hash2


class TestVerifyApiKey:
    """Test suite for verify_api_key function"""

    def test_verifies_correct_key(self):
        """Should verify a correct key against its hash"""
        plain_key = "bsi_test_secret_123"
        key_hash = hash_api_key(plain_key)
        assert verify_api_key(plain_key, key_hash) is True

    def test_rejects_incorrect_key(self):
        """Should reject an incorrect key"""
        key_hash = hash_api_key("bsi_correct_key")
        assert verify_api_key("bsi_wrong_key", key_hash) is False

    def test_verifies_multiple_times(self):
        """Should consistently verify the same key"""
        plain_key = "bsi_persistent"
        key_hash = hash_api_key(plain_key)
        # Verify multiple times to ensure consistency
        assert verify_api_key(plain_key, key_hash) is True
        assert verify_api_key(plain_key, key_hash) is True
        assert verify_api_key(plain_key, key_hash) is True


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
