"""Test suite for embedding service"""

import pytest
import numpy as np
from bugspotter_intelligence.services.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider
)


class TestLocalEmbeddingProvider:
    """Test suite for LocalEmbeddingProvider"""

    @pytest.fixture
    def provider(self):
        """Create provider instance for tests"""
        return LocalEmbeddingProvider()

    def test_provider_initialization(self, provider):
        """Should initialize with default model"""
        assert provider.model_name == "BAAI/bge-m3"
        assert provider.provider_name == "local"

    def test_custom_model_initialization(self):
        """Should accept custom model name"""
        provider = LocalEmbeddingProvider(model_name="custom-model")
        assert provider.model_name == "custom-model"

    def test_lazy_loading(self, provider):
        """Should lazy-load the model"""
        # Model should not be loaded yet
        assert provider._model is None

        # Access model property
        model = provider.model

        # Now it should be loaded
        assert provider._model is not None
        assert model == provider._model

    def test_dimension(self, provider):
        """Should return correct embedding dimension"""
        dimension = provider.dimension()
        assert dimension == 1024  # BAAI/bge-m3 has 384 dimensions
        assert isinstance(dimension, int)

    def test_embed_single_text(self, provider):
        """Should generate embedding for single text"""
        text = "App crashes on login"
        embedding = provider.embed(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_empty_text_raises_error(self, provider):
        """Should raise error for empty text"""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            provider.embed("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            provider.embed("   ")

    def test_embed_batch(self, provider):
        """Should generate embeddings for multiple texts"""
        texts = [
            "Login fails",
            "Search broken",
            "Payment error"
        ]

        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 1024 for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)

    def test_embed_batch_empty_list_raises_error(self, provider):
        """Should raise error for empty list"""
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            provider.embed_batch([])

    def test_similar_texts_have_higher_similarity(self, provider):
        """Similar texts should have higher cosine similarity"""
        # Similar texts
        text1 = "Login fails with null pointer"
        text2 = "Login crashes with null reference"

        # Dissimilar text
        text3 = "Payment processing successful"

        emb1 = provider.embed(text1)
        emb2 = provider.embed(text2)
        emb3 = provider.embed(text3)

        # Calculate cosine similarities
        sim_12 = self._cosine_similarity(emb1, emb2)
        sim_13 = self._cosine_similarity(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13
        assert sim_12 > 0.7  # Should be quite similar
        assert sim_13 < 0.5  # Should be less similar

    def test_same_text_has_perfect_similarity(self, provider):
        """Same text should have similarity of 1.0"""
        text = "Login fails with error"

        emb1 = provider.embed(text)
        emb2 = provider.embed(text)

        similarity = self._cosine_similarity(emb1, emb2)

        # Should be very close to 1.0 (allow small floating point errors)
        assert abs(similarity - 1.0) < 0.0001

    def test_embedding_is_deterministic(self, provider):
        """Same text should always produce same embedding"""
        text = "Test consistency"

        emb1 = provider.embed(text)
        emb2 = provider.embed(text)

        # Embeddings should be identical
        assert emb1 == emb2

    def test_batch_vs_single_consistency(self, provider):
        """Batch embedding should match individual embeddings"""
        texts = ["Text A", "Text B", "Text C"]

        # Generate individually
        individual = [provider.embed(text) for text in texts]

        # Generate as batch
        batch = provider.embed_batch(texts)

        # Should be identical
        for ind, bat in zip(individual, batch):
            assert ind == bat

    def test_integration_with_log_extractor(self, provider):
        """Should work with log extractor output"""
        from bugspotter_intelligence.utils.log_extractor import build_embedding_text

        console_logs = [
            {
                "level": "error",
                "message": "TypeError: Cannot read property 'name' of null",
                "stack": "at login.js:42"
            }
        ]

        network_logs = [
            {"url": "/api/login", "method": "POST", "status": 500, "duration": 123}
        ]

        metadata = {"browser": "Chrome", "os": "Windows"}

        embedding_text = build_embedding_text(
            title="Login crashes",
            description="Null pointer error",
            console_logs=console_logs,
            network_logs=network_logs,
            metadata=metadata
        )

        # Should successfully generate embedding
        embedding = provider.embed(embedding_text)

        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))


class TestEmbeddingProviderInterface:
    """Test that LocalEmbeddingProvider implements the interface correctly"""

    def test_implements_embedding_provider(self):
        """Should implement EmbeddingProvider interface"""
        provider = LocalEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_has_required_methods(self):
        """Should have all required methods"""
        provider = LocalEmbeddingProvider()

        assert hasattr(provider, 'embed')
        assert callable(provider.embed)

        assert hasattr(provider, 'embed_batch')
        assert callable(provider.embed_batch)

        assert hasattr(provider, 'dimension')
        assert callable(provider.dimension)

        assert hasattr(provider, 'provider_name')