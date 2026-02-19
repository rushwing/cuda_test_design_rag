"""Document retriever for RAG queries."""

from typing import Optional

from langchain_core.documents import Document

from cuda_test_rag.config import Settings
from cuda_test_rag.core.reranker import BGEReranker, RerankerFactory
from cuda_test_rag.core.vectorstore import VectorStoreManager


class DocumentRetriever:
    """Retrieves relevant documents for RAG queries with optional reranking."""

    def __init__(
        self,
        settings: Settings | None = None,
        vectorstore_manager: VectorStoreManager | None = None,
        reranker: Optional[BGEReranker] = None,
    ):
        self.settings = settings or Settings()
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager(self.settings)

        # Initialize reranker if enabled
        if reranker is not None:
            self.reranker = reranker
        elif self.settings.reranker_type.lower() != "none":
            self.reranker = RerankerFactory.create(
                self.settings.reranker_type,
                self.settings,
                model_name=self.settings.reranker_model,
            )
        else:
            self.reranker = None

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve (before reranking)

        Returns:
            List of relevant documents
        """
        k = k or self.settings.retrieval_k

        # Initial retrieval
        docs = self.vectorstore_manager.similarity_search(query, k=k)

        # Apply reranking if enabled
        if self.reranker is not None:
            top_k = self.settings.reranker_top_k or k
            scored_docs = self.reranker.rerank(query, docs, top_k=top_k)
            docs = [doc for doc, _ in scored_docs]

        return docs

    def retrieve_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with similarity scores (before reranking)."""
        k = k or self.settings.retrieval_k
        return self.vectorstore_manager.vectorstore.similarity_search_with_score(query, k=k)

    def retrieve_with_rerank_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with reranking scores.

        Returns:
            List of (document, rerank_score) tuples
        """
        k = k or self.settings.retrieval_k

        if self.reranker is None:
            # Return similarity scores if no reranker
            return self.vectorstore_manager.vectorstore.similarity_search_with_score(query, k=k)

        docs = self.vectorstore_manager.similarity_search(query, k=k)
        top_k = self.settings.reranker_top_k or k
        return self.reranker.rerank(query, docs, top_k=top_k)

    def get_retriever(self, k: int | None = None):
        """Get a LangChain retriever object."""
        k = k or self.settings.retrieval_k
        return self.vectorstore_manager.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
