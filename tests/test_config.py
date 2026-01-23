"""Tests for configuration module."""

import pytest

from cuda_test_rag.config import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings(google_api_key="test-key")
        assert settings.llm_model == "gemini-2.5-flash"
        assert settings.embedding_model == "models/embedding-001"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.retrieval_k == 4

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = Settings(
            google_api_key="test-key",
            chunk_size=500,
            retrieval_k=8,
        )
        assert settings.chunk_size == 500
        assert settings.retrieval_k == 8

    def test_paths(self):
        """Test path helper methods."""
        settings = Settings(
            google_api_key="test-key",
            docs_path="/custom/docs",
            vectorstore_path="/custom/vectorstore",
        )
        assert str(settings.get_docs_path()) == "/custom/docs"
        assert str(settings.get_vectorstore_path()) == "/custom/vectorstore"
