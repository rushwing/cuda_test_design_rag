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

### Two-Stage Test Generation Pipeline

```
Documents → RAG Retrieval → Stage 1 (Intents) → Stage 2 (Skeletons)
```

- **Stage 1**: Extracts WHAT to test and WHY from design docs → outputs YAML with `TestIntent` objects
- **Stage 2**: Generates HOW to test → outputs GoogleTest C++ code skeletons

The intermediate YAML can be reviewed/edited between stages.

### Module Structure

- `core/`: RAG fundamentals (embeddings via `langchain-google-genai`, ChromaDB vectorstore, retriever)
- `document_processing/`: Loaders (PDF, DOCX, TXT, MD) and text chunking
- `test_generation/`: Two-stage pipeline (`pipeline.py`) and legacy single-stage (`generator.py`)
- `cli.py`: Typer CLI entry point

### Key Classes

- `Settings` (config.py): Pydantic settings loaded from `.env`
- `TestGenerationPipeline` (test_generation/pipeline.py): Main orchestrator for two-stage generation
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
cuda-test-rag ingest ./data/docs

# Two-stage (recommended)
cuda-test-rag gen-pipeline "test description" -i intents.yaml -s tests.cu

# Separate stages
cuda-test-rag gen-intents "test description" -o intents.yaml
cuda-test-rag gen-skeletons intents.yaml -o tests.cu

# Legacy single-stage
cuda-test-rag generate "test description" -o tests.cu
```

## Testing Patterns

Tests use `conftest.py` fixtures for `Settings` with mock API key. When writing tests that don't need real API calls, use the `settings` fixture which provides isolated paths.
