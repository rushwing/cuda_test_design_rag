"""Document splitting for chunking large documents."""

import re
from typing import Literal

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from cuda_test_rag.config import Settings


class DocumentSplitter:
    """Splits documents into chunks for embedding."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._splitter = None
        self._markdown_splitter = None

    @property
    def splitter(self) -> RecursiveCharacterTextSplitter:
        """Get or create the text splitter."""
        if self._splitter is None:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
        return self._splitter

    @property
    def markdown_splitter(self) -> MarkdownHeaderTextSplitter:
        """Get or create markdown header splitter for semantic chunking."""
        if self._markdown_splitter is None:
            self._markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "h1"),
                    ("##", "h2"),
                ],
                strip_headers=False,
            )
        return self._markdown_splitter

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        return self.splitter.split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        """Split a single text into chunks."""
        return self.splitter.split_text(text)

    def _extract_yaml_frontmatter(self, text: str) -> tuple[dict, str]:
        """Extract YAML front matter from markdown text.

        Returns:
            Tuple of (metadata_dict, content_without_frontmatter)
        """
        pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(pattern, text, re.DOTALL)

        if match:
            try:
                metadata = yaml.safe_load(match.group(1))
                content = text[match.end() :]
                return metadata or {}, content
            except yaml.YAMLError:
                return {}, text
        return {}, text

    def split_markdown_test_cases(
        self,
        documents: list[Document],
        chunk_strategy: Literal["by_test_case", "by_section", "default"] = "by_test_case",
    ) -> list[Document]:
        """Split markdown test case documents with metadata extraction.

        Args:
            documents: List of documents to split
            chunk_strategy:
                - "by_test_case": Keep each test case (# heading) as one chunk (best for few-shot)
                - "by_section": Split by ## headings (medium granularity)
                - "default": Use standard character-based splitting

        Returns:
            List of Document chunks with extracted metadata
        """
        all_chunks = []

        for doc in documents:
            # Extract YAML front matter as metadata
            frontmatter, content = self._extract_yaml_frontmatter(doc.page_content)

            # Merge frontmatter with existing metadata
            base_metadata = {**doc.metadata, **frontmatter}

            if chunk_strategy == "default":
                # Standard splitting
                chunks = self.splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={**base_metadata, "chunk_index": i},
                        )
                    )

            elif chunk_strategy == "by_test_case":
                # Split by # Test Case headers - keeps entire test cases together
                test_case_pattern = r"(?=^# (?:Test Case|.*?Test Case))"
                parts = re.split(test_case_pattern, content, flags=re.MULTILINE)
                parts = [p.strip() for p in parts if p.strip()]

                # If no test case headers found, treat whole doc as one chunk
                if len(parts) <= 1:
                    parts = [content]

                for i, part in enumerate(parts):
                    # Extract test case ID if present
                    tc_id_match = re.search(r"\(?(CUDA-[A-Z]+-\d+)\)?", part)
                    tc_metadata = {}
                    if tc_id_match:
                        tc_metadata["test_case_id"] = tc_id_match.group(1)

                    all_chunks.append(
                        Document(
                            page_content=part,
                            metadata={**base_metadata, **tc_metadata, "chunk_index": i},
                        )
                    )

            elif chunk_strategy == "by_section":
                # Use markdown header splitter for ## sections
                md_chunks = self.markdown_splitter.split_text(content)
                for i, md_chunk in enumerate(md_chunks):
                    chunk_metadata = {**base_metadata, "chunk_index": i}
                    # Add header info from markdown splitter
                    if hasattr(md_chunk, "metadata"):
                        chunk_metadata.update(md_chunk.metadata)
                        chunk_content = md_chunk.page_content
                    else:
                        chunk_content = md_chunk

                    all_chunks.append(
                        Document(page_content=chunk_content, metadata=chunk_metadata)
                    )

        return all_chunks
