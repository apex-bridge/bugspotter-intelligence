import logging
from sentence_transformers import SentenceTransformer
from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers

    Self-hosted, no API calls, free
    Can be extended later with OpenAI, Cohere, etc.
    """

    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # Lazy loading

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded: {self.model_name} ({self.dimension()} dimensions)")
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate embedding for single text"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logger.debug(f"Generating embedding (length: {len(text)} chars)")
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        logger.debug(f"Generating {len(texts)} embeddings")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

    @property
    def provider_name(self) -> str:
        return "local"