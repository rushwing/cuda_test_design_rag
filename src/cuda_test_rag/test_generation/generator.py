"""CUDA test suite generator using RAG."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from cuda_test_rag.config import Settings
from cuda_test_rag.core.retriever import DocumentRetriever
from cuda_test_rag.test_generation.prompts import TestPromptTemplates


class CUDATestGenerator:
    """Generates CUDA test suites using RAG."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: DocumentRetriever | None = None,
    ):
        self.settings = settings or Settings()
        self.retriever = retriever or DocumentRetriever(self.settings)
        self._llm = None

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Get or create the LLM."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.settings.llm_model,
                google_api_key=self.settings.google_api_key,
                temperature=self.settings.temperature,
            )
        return self._llm

    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a context string."""
        return "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        )

    def generate(self, query: str) -> str:
        """Generate CUDA test suite based on a query."""
        prompt = TestPromptTemplates.get_test_generation_prompt()

        chain = (
            {
                "context": self.retriever.get_retriever() | self._format_docs,
                "query": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(query)

    def generate_with_context(self, query: str, context: str) -> str:
        """Generate CUDA test suite with explicit context."""
        prompt = TestPromptTemplates.get_test_generation_prompt()

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({"context": context, "query": query})

    def retrieve_and_generate(self, query: str, k: int | None = None) -> dict:
        """Retrieve documents and generate tests, returning both."""
        docs = self.retriever.retrieve(query, k=k)
        context = self._format_docs(docs)
        result = self.generate_with_context(query, context)

        return {
            "query": query,
            "retrieved_documents": docs,
            "context": context,
            "generated_tests": result,
        }
