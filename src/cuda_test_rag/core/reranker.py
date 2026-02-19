"""Reranker for improving RAG retrieval quality."""

from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from cuda_test_rag.config import Settings


class BGEReranker:
    """BGE Reranker for improving retrieval quality.

    Uses BAAI/bge-reranker-v2-m3 or similar model to rerank
    retrieval results from the vector store.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.settings = settings or Settings()
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
            except ImportError:
                raise ImportError(
                    "FlagEmbedding is required for reranking. "
                    "Install with: uv pip install FlagEmbedding"
                )
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """Rerank documents based on query relevance.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return (default: all)

        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            return []

        # Extract text content for reranking
        doc_texts = [doc.page_content for doc in documents]

        # Compute reranking scores
        scores = self.model.compute_score([query] * len(doc_texts), doc_texts)

        # Pair documents with scores and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        return scored_docs

    def rerank_with_scores(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """Alias for rerank() to maintain compatibility."""
        return self.rerank(query, documents, top_k)


class RerankerFactory:
    """Factory for creating rerankers."""

    @staticmethod
    def create(
        reranker_type: str = "bge",
        settings: Optional[Settings] = None,
        **kwargs,
    ) -> Optional[BGEReranker]:
        """Create a reranker instance.

        Args:
            reranker_type: Type of reranker ("bge" or "none")
            settings: Application settings
            **kwargs: Additional arguments for reranker

        Returns:
            Reranker instance or None
        """
        if reranker_type.lower() == "none":
            return None

        if reranker_type.lower() == "bge":
            model_name = kwargs.get("model_name", "BAAI/bge-reranker-v2-m3")
            return BGEReranker(settings=settings, model_name=model_name)

        raise ValueError(f"Unknown reranker type: {reranker_type}")
