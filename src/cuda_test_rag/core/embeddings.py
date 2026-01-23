"""Embedding management for document vectorization."""

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from cuda_test_rag.config import Settings


class EmbeddingManager:
    """Manages embedding model for document vectorization."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._embeddings = None

    @property
    def embeddings(self) -> Embeddings:
        """Get or create the embedding model."""
        if self._embeddings is None:
            if self.settings.use_local_embeddings:
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.local_embedding_model,
                )
            else:
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.settings.embedding_model,
                    google_api_key=self.settings.google_api_key,
                )
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self.embeddings.embed_documents(texts)
