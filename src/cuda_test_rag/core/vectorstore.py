"""Vector store management for document storage and retrieval."""

from pathlib import Path

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_core.documents import Document

from cuda_test_rag.config import Settings
from cuda_test_rag.core.embeddings import EmbeddingManager


class VectorStoreManager:
    """Manages the vector store for document storage."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_manager: EmbeddingManager | None = None,
    ):
        self.settings = settings or Settings()
        self.embedding_manager = embedding_manager or EmbeddingManager(self.settings)
        self._vectorstore = None

    @property
    def persist_directory(self) -> Path:
        """Get the persistence directory for the vector store."""
        return Path(self.settings.vectorstore_path)

    @property
    def vectorstore(self) -> Chroma:
        """Get or create the vector store."""
        if self._vectorstore is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._vectorstore = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=self.embedding_manager.embeddings,
                persist_directory=str(self.persist_directory),
            )
        return self._vectorstore

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        return self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search for similar documents."""
        return self.vectorstore.similarity_search(query, k=k)

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.vectorstore.delete_collection()
        self._vectorstore = None
