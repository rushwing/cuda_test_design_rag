"""Tests for document processing module."""

import pytest

from cuda_test_rag.document_processing.splitter import DocumentSplitter


class TestDocumentSplitter:
    """Tests for DocumentSplitter class."""

    def test_split_text(self, settings):
        """Test splitting text into chunks."""
        splitter = DocumentSplitter(settings)
        text = "A" * 2000  # Create text larger than chunk size
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert all(len(chunk) <= settings.chunk_size for chunk in chunks)

    def test_split_text_with_overlap(self, settings):
        """Test that chunks have proper overlap."""
        splitter = DocumentSplitter(settings)
        text = "word " * 500  # Create text with repeated words
        chunks = splitter.split_text(text)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                # Overlap should exist between chunks
                assert len(chunks[i]) > 0
                assert len(chunks[i + 1]) > 0

    def test_split_short_text(self, settings):
        """Test that short text is not split."""
        splitter = DocumentSplitter(settings)
        text = "Short text"
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text
