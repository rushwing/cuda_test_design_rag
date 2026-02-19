"""Multi-stage test generation pipeline.

Stage 1: Requirements -> Test Intents (with RAG)
Stage 2: Test Intents -> Test Skeletons (optional RAG for examples)
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from cuda_test_rag.config import Settings
from cuda_test_rag.core.retriever import DocumentRetriever
from cuda_test_rag.test_generation.models import PipelineResult, TestIntentCollection, TestCaseCollection
from cuda_test_rag.test_generation.prompts import TestPromptTemplates


class TestGenerationPipeline:
    """Two-stage pipeline for CUDA test generation.

    Stage 1: Extract test intents from requirements using RAG
    Stage 2: Generate test skeletons from intents (optionally with RAG)
    """

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

    def _format_docs(self, docs: list[Document]) -> str:
        """Format retrieved documents into a context string."""
        return "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        )

    def _get_doc_sources(self, docs: list[Document]) -> list[str]:
        """Extract source paths from documents."""
        return [doc.metadata.get("source", "Unknown") for doc in docs]

    # ==================== Stage 1: Test Intent Generation ====================

    def generate_intents(
        self,
        query: str,
        k: int | None = None,
    ) -> tuple[TestIntentCollection, str, list[str]]:
        """Stage 1: Generate test intents from requirements.

        Args:
            query: User query describing what to test
            k: Number of documents to retrieve

        Returns:
            Tuple of (TestIntentCollection, context_string, doc_sources)
        """
        # Retrieve relevant documents
        docs = self.retriever.retrieve(query, k=k)
        context = self._format_docs(docs)
        doc_sources = self._get_doc_sources(docs)

        # Generate intents using LLM
        prompt = TestPromptTemplates.get_intent_generation_prompt()
        chain = prompt | self.llm | StrOutputParser()

        raw_response = chain.invoke({"context": context, "query": query})

        # Parse the response into structured intents
        intents = TestIntentCollection.from_yaml_string(raw_response, source_query=query)

        return intents, context, doc_sources

    def generate_intents_with_context(
        self,
        query: str,
        context: str,
    ) -> TestIntentCollection:
        """Stage 1 with explicit context (no retrieval).

        Args:
            query: User query describing what to test
            context: Pre-provided context string

        Returns:
            TestIntentCollection with parsed intents
        """
        prompt = TestPromptTemplates.get_intent_generation_prompt()
        chain = prompt | self.llm | StrOutputParser()

        raw_response = chain.invoke({"context": context, "query": query})

        return TestIntentCollection.from_yaml_string(raw_response, source_query=query)

    # ==================== Stage 2: Test Skeleton Generation ====================

    def generate_skeletons(
        self,
        intents: TestIntentCollection,
        additional_context: str = "",
    ) -> str:
        """Stage 2: Generate test skeletons from intents.

        Args:
            intents: Test intents from Stage 1
            additional_context: Optional additional context (e.g., code examples)

        Returns:
            Generated test skeleton code as string
        """
        prompt = TestPromptTemplates.get_skeleton_generation_prompt()
        chain = prompt | self.llm | StrOutputParser()

        intents_str = intents.to_prompt_string()

        return chain.invoke({
            "test_intents": intents_str,
            "context": additional_context or "No additional context provided.",
        })

    def generate_skeletons_with_rag(
        self,
        intents: TestIntentCollection,
        example_query: str = "CUDA test examples and patterns",
        k: int | None = None,
    ) -> tuple[str, str]:
        """Stage 2 with RAG for retrieving code examples.

        Args:
            intents: Test intents from Stage 1
            example_query: Query for retrieving relevant examples
            k: Number of documents to retrieve

        Returns:
            Tuple of (skeleton_code, retrieved_context)
        """
        # Retrieve examples/patterns
        docs = self.retriever.retrieve(example_query, k=k)
        context = self._format_docs(docs)

        skeletons = self.generate_skeletons(intents, additional_context=context)

        return skeletons, context

    # ==================== Full Pipeline ====================

    def run_full_pipeline(
        self,
        query: str,
        k: int | None = None,
        use_rag_for_skeletons: bool = False,
    ) -> PipelineResult:
        """Run the complete two-stage pipeline.

        Args:
            query: User query describing what to test
            k: Number of documents to retrieve
            use_rag_for_skeletons: Whether to use RAG in Stage 2

        Returns:
            PipelineResult with all outputs
        """
        result = PipelineResult(query=query)

        # Stage 1: Generate intents
        intents, context, doc_sources = self.generate_intents(query, k=k)
        result.test_intents = intents
        result.context = context
        result.retrieved_documents = doc_sources
        result.stage_completed = 1

        # Stage 2: Generate skeletons
        if use_rag_for_skeletons:
            skeletons, _ = self.generate_skeletons_with_rag(intents)
        else:
            skeletons = self.generate_skeletons(intents)

        result.test_skeletons = skeletons
        result.stage_completed = 2

        return result

    def run_stage1_only(
        self,
        query: str,
        k: int | None = None,
    ) -> PipelineResult:
        """Run only Stage 1 (intent generation).

        Args:
            query: User query describing what to test
            k: Number of documents to retrieve

        Returns:
            PipelineResult with intents only
        """
        intents, context, doc_sources = self.generate_intents(query, k=k)

        return PipelineResult(
            query=query,
            test_intents=intents,
            context=context,
            retrieved_documents=doc_sources,
            stage_completed=1,
        )

    def run_stage2_only(
        self,
        intents: TestIntentCollection,
        additional_context: str = "",
    ) -> str:
        """Run only Stage 2 (skeleton generation).

        Args:
            intents: Test intents (from Stage 1 or loaded from file)
            additional_context: Optional additional context

        Returns:
            Generated skeleton code
        """
        return self.generate_skeletons(intents, additional_context)

    # ==================== Stage 3: Full C++ Code Generation ====================

    def generate_code(
        self,
        test_cases: str,
        context: str = "",
    ) -> str:
        """Stage 3: Generate complete, compilable C++ test code.

        Args:
            test_cases: Test case specifications (from Stage 2 or loaded from file)
            context: Additional CUDA context/reference

        Returns:
            Generated C++ code
        """
        prompt = TestPromptTemplates.get_code_generation_prompt()
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "test_cases": test_cases,
            "context": context or "No additional context provided.",
        })

    def generate_code_with_rag(
        self,
        test_cases: str,
        context_query: str = "CUDA kernel implementation examples and patterns",
        k: int | None = None,
    ) -> tuple[str, str]:
        """Stage 3 with RAG for retrieving relevant code examples.

        Args:
            test_cases: Test case specifications
            context_query: Query for retrieving relevant examples
            k: Number of documents to retrieve

        Returns:
            Tuple of (generated_code, retrieved_context)
        """
        docs = self.retriever.retrieve(context_query, k=k)
        context = self._format_docs(docs)

        code = self.generate_code(test_cases, context)
        return code, context

    def run_stage3_only(
        self,
        test_cases: str,
        additional_context: str = "",
    ) -> str:
        """Run only Stage 3 (code generation).

        Args:
            test_cases: Test cases (from Stage 2 or loaded from file)
            additional_context: Optional additional context

        Returns:
            Generated C++ code
        """
        return self.generate_code(test_cases, additional_context)

    # ==================== Full 3-Stage Pipeline ====================

    def run_full_pipeline_3stage(
        self,
        query: str,
        k: int | None = None,
        use_rag_for_code: bool = False,
    ) -> PipelineResult:
        """Run the complete three-stage pipeline.

        Args:
            query: User query describing what to test
            k: Number of documents to retrieve
            use_rag_for_code: Whether to use RAG in Stage 3

        Returns:
            PipelineResult with all outputs
        """
        result = PipelineResult(query=query)

        # Stage 1: Generate intents
        intents, context, doc_sources = self.generate_intents(query, k=k)
        result.test_intents = intents
        result.context = context
        result.retrieved_documents = doc_sources
        result.stage_completed = 1

        # Stage 2: Generate test cases (markdown)
        # Convert intents to prompt string for Stage 2
        intents_str = intents.to_prompt_string()
        testcase_prompt = TestPromptTemplates.get_testcase_generation_prompt()
        testcase_chain = testcase_prompt | self.llm | StrOutputParser()
        test_cases_md = testcase_chain.invoke({
            "test_intents": intents_str,
            "context": context,
        })
        result.test_cases = TestCaseCollection(
            test_cases=[],
            source_intents=intents_str,
            raw_response=test_cases_md,
        )
        result.stage_completed = 2

        # Stage 3: Generate C++ code
        if use_rag_for_code:
            code, _ = self.generate_code_with_rag(test_cases_md)
        else:
            code = self.generate_code(test_cases_md, context)

        result.test_code = code
        result.stage_completed = 3

        return result

    # ==================== Persistence ====================

    def save_intents(self, intents: TestIntentCollection, path: Path | str) -> None:
        """Save test intents to a YAML file.

        Args:
            intents: Test intents to save
            path: Output file path
        """
        import yaml

        path = Path(path)
        data = {
            "source_query": intents.source_query,
            "test_intents": [
                {
                    "id": i.id,
                    "category": i.category,
                    "title": i.title,
                    "description": i.description,
                    "requirement_ref": i.requirement_ref,
                    "priority": i.priority,
                    "preconditions": i.preconditions,
                }
                for i in intents.test_intents
            ],
        }
        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    def load_intents(self, path: Path | str) -> TestIntentCollection:
        """Load test intents from a YAML file.

        Args:
            path: Input file path

        Returns:
            TestIntentCollection loaded from file
        """
        path = Path(path)
        content = path.read_text()
        return TestIntentCollection.from_yaml_string(f"```yaml\n{content}\n```")

    def save_skeletons(self, skeletons: str, path: Path | str) -> None:
        """Save test skeletons to a file.

        Args:
            skeletons: Generated skeleton code
            path: Output file path
        """
        path = Path(path)
        path.write_text(skeletons)

    def save_code(self, code: str, path: Path | str) -> None:
        """Save generated C++ code to a file.

        Args:
            code: Generated C++ code
            path: Output file path
        """
        path = Path(path)
        path.write_text(code)
