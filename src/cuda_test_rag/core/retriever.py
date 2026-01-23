"""Document retriever for RAG queries."""

from langchain_core.documents import Document

from cuda_test_rag.config import Settings
from cuda_test_rag.core.vectorstore import VectorStoreManager


class DocumentRetriever:
    """Retrieves relevant documents for RAG queries."""

    def __init__(
        self,
        settings: Settings | None = None,
        vectorstore_manager: VectorStoreManager | None = None,
    ):
        self.settings = settings or Settings()
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager(self.settings)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve relevant documents for a query."""
        k = k or self.settings.retrieval_k
        return self.vectorstore_manager.similarity_search(query, k=k)

    def retrieve_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with similarity scores."""
        k = k or self.settings.retrieval_k
        return self.vectorstore_manager.vectorstore.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int | None = None):
        """Get a LangChain retriever object."""
        k = k or self.settings.retrieval_k
        return self.vectorstore_manager.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
