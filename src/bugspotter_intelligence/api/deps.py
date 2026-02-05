"""FastAPI dependency injection"""

import logging

from fastapi import Depends

from bugspotter_intelligence.config import Settings
from bugspotter_intelligence.db.database import get_db_connection
from bugspotter_intelligence.llm import LLMProvider, create_llm_provider
from bugspotter_intelligence.cache import CacheService, get_cache_service
from bugspotter_intelligence.services import BugCommandService, BugQueryService, SearchService
from bugspotter_intelligence.services.embeddings import EmbeddingProvider, LocalEmbeddingProvider
from bugspotter_intelligence.services.reranker import LLMReranker

logger = logging.getLogger(__name__)

# Global singletons
_settings: Settings | None = None
_llm_provider: LLMProvider | None = None
_embedding_provider: EmbeddingProvider | None = None


def get_settings() -> Settings:
    """Get settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_llm_provider() -> LLMProvider:
    """Get LLM provider singleton"""
    global _llm_provider
    if _llm_provider is None:
        settings = get_settings()
        _llm_provider = create_llm_provider(settings)
    return _llm_provider


def get_embedding_provider() -> EmbeddingProvider:
    """Get embedding provider singleton"""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = LocalEmbeddingProvider()
    return _embedding_provider


def get_cache() -> CacheService:
    """Get CacheService singleton"""
    return get_cache_service()


def get_bug_command_service(
    llm_provider: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    cache: CacheService = Depends(get_cache),
) -> BugCommandService:
    """Get BugCommandService instance"""
    return BugCommandService(llm_provider, embedding_provider, cache=cache)


def get_bug_query_service(
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider)
) -> BugQueryService:
    """Get BugQueryService instance"""
    return BugQueryService(settings, llm_provider, embedding_provider)


def get_reranker(
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> LLMReranker:
    """Get LLMReranker instance"""
    return LLMReranker(
        llm_provider,
        timeout_seconds=settings.smart_search_timeout,
    )


def get_search_service(
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    reranker: LLMReranker = Depends(get_reranker),
    cache: CacheService = Depends(get_cache),
) -> SearchService:
    """Get SearchService instance"""
    return SearchService(
        embedding_provider,
        reranker=reranker,
        smart_candidate_limit=settings.smart_search_candidate_limit,
        cache=cache if settings.cache_enabled else None,
        cache_ttl_fast=settings.cache_ttl_fast_search,
        cache_ttl_smart=settings.cache_ttl_smart_search,
    )


__all__ = [
    "get_settings",
    "get_llm_provider",
    "get_embedding_provider",
    "get_cache",
    "get_bug_command_service",
    "get_bug_query_service",
    "get_search_service",
    "get_db_connection",
]
