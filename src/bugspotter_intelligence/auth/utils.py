"""Authentication utility functions"""

import secrets

import bcrypt


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
    Hash an API key using bcrypt with automatic salt generation.

    Args:
        key: The plain text API key

    Returns:
        bcrypt hash of the key (includes salt)

    Note:
        We use bcrypt instead of SHA-256 for security best practices:
        - Automatic per-key salt generation prevents rainbow table attacks
        - Configurable work factor (currently 12) makes brute-force attacks costly
        - Industry standard for credential storage

        The plain key is returned to the user only once on creation.

    Security:
        Even though our API keys are high-entropy random strings generated
        with secrets.token_urlsafe(32), using bcrypt ensures defense-in-depth
        and follows security best practices for any credential storage.
    """
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(key.encode(), salt).decode()


def verify_api_key(key: str, key_hash: str) -> bool:
    """
    Verify an API key against its stored hash.

    Args:
        key: The plain text API key to verify
        key_hash: The stored bcrypt hash

    Returns:
        True if the key matches the hash, False otherwise

    Example:
        >>> hashed = hash_api_key("bsi_secret123")
        >>> verify_api_key("bsi_secret123", hashed)
        True
        >>> verify_api_key("bsi_wrong", hashed)
        False
    """
    return bcrypt.checkpw(key.encode(), key_hash.encode())


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
