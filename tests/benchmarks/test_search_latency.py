"""
Search latency benchmarks.

These tests require a running PostgreSQL+pgvector database with test data.
Run with: pytest tests/benchmarks/ -m slow -v
"""

import time

import pytest

slow = pytest.mark.slow


@slow
@pytest.mark.skip(reason="Requires database with seeded test data")
class TestSearchLatency:
    """Latency benchmarks for search at various dataset sizes"""

    async def test_fast_search_10k_vectors(self):
        """Fast search should complete in <100ms at 10K vectors"""
        # Requires: 10K bug_embeddings rows seeded
        # Measure: embed query + ANN search time
        pass

    async def test_fast_search_50k_vectors(self):
        """Fast search should complete in <200ms at 50K vectors"""
        pass

    async def test_filtered_search_10k_vectors(self):
        """Filtered search (status=open) should complete in <150ms at 10K vectors"""
        pass

    async def test_cached_search_latency(self):
        """Cached search should complete in <10ms"""
        pass
