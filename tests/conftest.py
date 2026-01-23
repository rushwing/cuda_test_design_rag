"""Pytest configuration and fixtures."""

import pytest

from cuda_test_rag.config import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        google_api_key="test-key",
        vectorstore_path="./test_data/vectorstore",
        docs_path="./test_data/docs",
        collection_name="test_collection",
    )


@pytest.fixture
def sample_document_content():
    """Sample CUDA document content for testing."""
    return """
    # CUDA Memory Management Requirements

    ## Memory Allocation
    - All device memory must be allocated using cudaMalloc
    - Memory allocation failures must be checked and handled
    - Maximum allocation size: 4GB per request

    ## Memory Transfer
    - Host to device transfers use cudaMemcpyHostToDevice
    - Device to host transfers use cudaMemcpyDeviceToHost
    - Async transfers should use cudaMemcpyAsync with streams

    ## Kernel Requirements
    - Block size should not exceed 1024 threads
    - Shared memory usage must be declared
    - All kernels must handle boundary conditions
    """
