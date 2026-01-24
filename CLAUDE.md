# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG client for generating CUDA test suites (GoogleTest C++) from requirements/design documents using Google Gemini.

## Build & Development Commands

```bash
# Install dependencies (use uv, not pip)
uv pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run a single test file
pytest tests/test_config.py

# Run a specific test
pytest tests/test_config.py::test_function_name -v

# Lint and format
black src tests
ruff check src tests

# Type check
mypy src
```

## Architecture

### Multi-Stage Test Generation Pipeline

```
Documents → RAG Retrieval → Rerank [TODO] → Stage 1 (Intents) → Stage 2 (Test Cases) → Stage 3 (Skeletons) [TODO]
```

- **Stage 1**: Extracts WHAT to test and WHY from design docs → outputs YAML with `TestIntent` objects
- **Stage 2**: Generates detailed test case descriptions → outputs test case specifications
- **Stage 3** [TODO]: Generates HOW to test → outputs GoogleTest C++ code skeletons

The intermediate YAML can be reviewed/edited between stages.

### RAG Retrieval

| Component | Current Implementation | TODO |
|-----------|----------------------|------|
| Embeddings | Google GenAI (`models/embedding-001`) or HuggingFace (`all-MiniLM-L6-v2`) | - |
| Vector Store | ChromaDB (persistent) | - |
| Retrieval | Top-K similarity search | - |
| Reranking | **Not implemented** | Cohere Rerank / Cross-Encoder / BGE-Reranker |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLI Entry Point (cli.py)                           │
│  ┌────────────┬──────────────┬─────────────┬──────────────┬──────────────────┐  │
│  │  ingest    │  gen-intents │gen-skeletons│ gen-pipeline │    generate      │  │
│  └─────┬──────┴──────┬───────┴──────┬──────┴───────┬──────┴────────┬─────────┘  │
└────────│─────────────│──────────────│──────────────│───────────────│────────────┘
         │             └──────────────┴──────┬───────┴───────────────┘
         ▼                                   ▼
┌─────────────────────────┐    ┌───────────────────────────────────────────────────┐
│  document_processing/   │    │              test_generation/                     │
│  ┌───────────────────┐  │    │  ┌─────────────────────────────────────────────┐  │
│  │  DocumentLoader   │  │    │  │      TestGenerationPipeline (pipeline.py)   │  │
│  │  (PDF,DOCX,TXT,MD)│  │    │  │  ┌─────────────┐ ┌─────────────┐ ┌────────┐ │  │
│  └─────────┬─────────┘  │    │  │  │  STAGE 1    │ │  STAGE 2    │ │STAGE 3 │ │  │
│            ▼            │    │  │  │  Intents    │→│  Test Cases │→│Skeleton│ │  │
│  ┌───────────────────┐  │    │  │  │  (YAML)     │ │  (Details)  │ │ [TODO] │ │  │
│  │ DocumentSplitter  │  │    │  │  └─────────────┘ └─────────────┘ └────────┘ │  │
│  └─────────┬─────────┘  │    │  └─────────────────────────────────────────────┘  │
└────────────│────────────┘    └───────────────────────────────────────────────────┘
             │                                       │
             ▼                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                            core/ (RAG Layer)                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐    │
│  │  Query → Embed (Google/HF) → ChromaDB Top-K → [Rerank TODO] → Results     │    │
│  └───────────────────────────────────────────────────────────────────────────┘    │
│  EmbeddingManager → VectorStoreManager (ChromaDB) ← DocumentRetriever              │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

- `core/`: RAG fundamentals (embeddings via `langchain-google-genai`, ChromaDB vectorstore, retriever)
- `document_processing/`: Loaders (PDF, DOCX, TXT, MD) and text chunking
- `test_generation/`: Multi-stage pipeline (`pipeline.py`) and legacy single-stage (`generator.py`)
- `cli.py`: Typer CLI entry point

### Key Classes

- `Settings` (config.py): Pydantic settings loaded from `.env`
- `TestGenerationPipeline` (test_generation/pipeline.py): Main orchestrator for multi-stage generation
- `TestIntent`, `TestIntentCollection` (test_generation/models.py): Pydantic models for Stage 1 output
- `CUDATestGenerator` (test_generation/generator.py): Legacy single-stage generator

## Configuration

All settings in `config.py` via pydantic-settings. Key env vars:
- `GOOGLE_API_KEY` (required) - free at https://aistudio.google.com/apikey
- `LLM_MODEL` - default: `gemini-2.5-flash`
- `EMBEDDING_MODEL` - default: `models/embedding-001`

## CLI Usage

```bash
# Ingest documents
cuda-test-rag ingest ./data/knowledge_base/docs

# Generate test cases (using shell script wrapper)
./scripts/gen_test_cases.sh --auto                     # Run both stages
./scripts/gen_test_cases.sh --intent                   # Stage 1 only
./scripts/gen_test_cases.sh --approve REQ-xxx          # Approve intents
./scripts/gen_test_cases.sh --case -r REQ-xxx          # Stage 2 only
./scripts/gen_test_cases.sh --list                     # List requests

# Utility commands
./scripts/clean.sh                                     # Clean generated files
cuda-test-rag search "query"                           # Search documents
cuda-test-rag clear                                    # Clear vector store
```

## Testing Patterns

Tests use `conftest.py` fixtures for `Settings` with mock API key. When writing tests that don't need real API calls, use the `settings` fixture which provides isolated paths.
