"""Configuration management using pydantic-settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google Gemini settings
    google_api_key: str = Field(default="", description="Google AI API key")
    llm_model: str = Field(default="gemini-2.5-flash", description="LLM model name")
    embedding_model: str = Field(default="models/embedding-001", description="Embedding model")
    temperature: float = Field(default=0.1, description="LLM temperature")

    # Local embedding settings
    use_local_embeddings: bool = Field(default=False, description="Use local embeddings instead of Google")
    local_embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Local embedding model name")

    # RAG settings
    chunk_size: int = Field(default=1000, description="Document chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap size")
    retrieval_k: int = Field(default=4, description="Number of documents to retrieve")

    # Storage settings
    vectorstore_path: str = Field(
        default="./data/vectorstore",
        description="Path to vector store",
    )
    collection_name: str = Field(
        default="cuda_test_docs",
        description="Vector store collection name",
    )
    docs_path: str = Field(
        default="./data/knowledge_base/docs",
        description="Path to knowledge base documents directory",
    )
    requests_db_path: str = Field(
        default="./data/requests.db",
        description="Path to SQLite database for request tracking",
    )
    data_base_path: str = Field(
        default="./data",
        description="Base path for all data directories",
    )

    def get_docs_path(self) -> Path:
        """Get the documents directory path."""
        return Path(self.docs_path)

    def get_vectorstore_path(self) -> Path:
        """Get the vector store directory path."""
        return Path(self.vectorstore_path)
