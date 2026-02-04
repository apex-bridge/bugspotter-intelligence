"""Shared fixtures for service tests"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from bugspotter_intelligence.cache.service import CacheService
from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.llm import LLMProvider
from bugspotter_intelligence.services.embeddings import EmbeddingProvider


@pytest.fixture
def mock_settings():
    """Mock settings with default values"""
    settings = Settings()
    settings.similarity_threshold = 0.75
    settings.duplicate_threshold = 0.90
    settings.max_similar_bugs = 5
    return settings


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider"""
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(return_value="AI generated suggestion")
    return provider


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider"""
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed = MagicMock(return_value=[0.1] * 384)  # Mock embedding vector
    provider.embed_batch = MagicMock(return_value=[[0.1] * 384, [0.2] * 384])
    provider.dimension = MagicMock(return_value=384)
    provider.provider_name = "mock"
    return provider


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn.cursor = MagicMock(return_value=cursor)
    conn.commit = AsyncMock()
    return conn


@pytest.fixture
def mock_cache_service():
    """Mock CacheService for cache integration tests"""
    cache = MagicMock(spec=CacheService)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.invalidate_tenant = AsyncMock()
    cache.get_tenant_version = AsyncMock(return_value=0)
    return cache