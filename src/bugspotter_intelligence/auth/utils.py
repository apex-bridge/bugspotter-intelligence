"""Authentication utility functions"""

import hashlib
import secrets


def generate_api_key(prefix: str = "bsi_") -> str:
    """
    Generate a secure API key with prefix.

    Args:
        prefix: Prefix for the key (default: "bsi_")

    Returns:
        A secure random API key like "bsi_abc123..."

    Example:
        >>> key = generate_api_key()
        >>> key.startswith("bsi_")
        True
        >>> len(key) > 40
        True
    """
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}{random_part}"


def hash_api_key(key: str) -> str:
    """
    Hash an API key using SHA256.

    Args:
        key: The plain text API key

    Returns:
        SHA256 hex digest of the key

    Note:
        We store only the hash in the database for security.
        The plain key is returned to the user only once on creation.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def get_key_prefix(key: str, length: int = 12) -> str:
    """
    Extract displayable prefix from key.

    Args:
        key: The full API key
        length: Number of characters to include (default: 12)

    Returns:
        First `length` characters of the key for display purposes

    Example:
        >>> get_key_prefix("bsi_abc123456789xyz")
        'bsi_abc12345'
    """
    return key[:length] if len(key) >= length else key
