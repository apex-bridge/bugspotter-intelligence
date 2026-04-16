from functools import lru_cache
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating text embeddings

    Uses sentence-transformers for local embedding generation.
    Can be extended later with:
    - Caching
    - Batch processing
    - Multiple model support
    - OpenAI/Anthropic embeddings
    """

    # Default model - high-quality multilingual embeddings (1024d, ~268ms per embed)
    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(self, model_name: str | None = None):
        """
        Initialize embedding service

        Args:
            model_name: Sentence-transformers model name
                       Defaults to 'BAAI/bge-m3' (1024 dimensions)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # Lazy loading

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load the model (only loads when first used)

        This avoids loading the model at startup if not needed
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name} ({self.dimension} dimensions)")
        return self._model

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.embed("App crashes on login")
            >>> len(embedding)
            1024
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Convert to list of floats
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (more efficient)

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Batch encoding is more efficient
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        """Get the embedding dimension for this model"""
        return self.model.get_sentence_embedding_dimension()


@lru_cache()
def get_embedding_service(model_name: str | None = None) -> EmbeddingService:
    """
    Get cached embedding service instance (singleton)

    This ensures we only load the model once
    """
    return EmbeddingService(model_name)