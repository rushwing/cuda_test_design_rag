# CUDA Test RAG

A RAG (Retrieval-Augmented Generation) client for generating CUDA test suites from requirements and design documents.

## Overview

This tool uses LangChain and Google Gemini (free) to:
1. Ingest and index your CUDA requirements/design documents
2. Retrieve relevant context based on your queries
3. Generate comprehensive CUDA test suites using a **multi-stage pipeline**

## Multi-Stage Generation Pipeline

For more stable and controllable test generation, this tool uses a multi-stage approach:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           Shared RAG Pipeline                                 │
│                                                                               │
│  ┌──────────┐    ┌────────────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  Ingest  │───▶│ Embed (Google/HF)  │───▶│   ChromaDB   │───▶│ Top-K    │  │
│  └──────────┘    └────────────────────┘    └──────────────┘    │ Retrieve │  │
│                                                                └────┬─────┘  │
│                                                                     │        │
│                                            ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▼ ─ ─ ┐  │
│                                            │   Rerank [TODO]             │  │
│                                            │   (Cohere/Cross-Encoder)    │  │
│                                            └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ┘  │
│                                                                     │        │
└─────────────────────────────────────────────────────────────────────┼────────┘
                                                                      │
                    ┌─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│          Stage 1: Requirements → Test Intents (RAG)          │
│  Extract WHAT to test and WHY from design docs               │
└─────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│          Stage 2: Test Intents → Test Cases                  │
│  Generate detailed test case descriptions                    │
└─────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│          Stage 3: Test Cases → Test Skeletons  [TODO]        │
│  Generate GoogleTest C++ code skeletons                      │
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

**Benefits:**
- **Debuggability**: Inspect/validate test intents and cases before code generation
- **Human-in-the-loop**: Engineers can review/edit intents and test cases
- **Cacheability**: Reuse intents for different skeleton styles
- **Retry granularity**: Re-run only failed stages

## Requirements

- Python 3.10+
- Windows 11 (also works on Linux/macOS)
- Google AI API key (free at https://aistudio.google.com/apikey)
- [uv](https://github.com/astral-sh/uv) (recommended for package management)

## Quick Start

### 1. Install uv (if not already installed)

**Windows (PowerShell):**
```powershell
winget install astral-sh.uv
# or
irm https://astral.sh/uv/install.ps1 | iex
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Initialize Virtual Environment

**PowerShell (recommended for Windows):**
```powershell
.\scripts\init_venv.ps1
```

**Git Bash / WSL:**
```bash
./scripts/init_venv.sh
```

### 3. Configure Environment

Get your free API key from https://aistudio.google.com/apikey

Edit `.env` and add your Google AI API key:
```env
GOOGLE_API_KEY=your-google-api-key-here
```

### 4. Activate Virtual Environment

**PowerShell:**
```powershell
.\cuda_venv\Scripts\Activate.ps1
```

**Git Bash:**
```bash
source cuda_venv/Scripts/activate
```

## Usage

### Ingest Documents

Add your requirements and design documents to `data/docs/` or specify a path:

```bash
# Ingest all documents from default directory
cuda-test-rag ingest ./data/docs

# Ingest a specific file
cuda-test-rag ingest ./path/to/requirements.pdf

# Clear existing documents and re-ingest
cuda-test-rag ingest ./data/docs --clear
```

Supported file formats: PDF, DOCX, TXT, Markdown

### Multi-Stage Pipeline (Recommended)

#### Option A: Run Full Pipeline

```bash
# Run both stages at once
cuda-test-rag gen-pipeline "Generate tests for matrix multiplication kernel"

# Save outputs to files
cuda-test-rag gen-pipeline "Test memory management" \
    --intents intents.yaml \
    --skeletons tests.cu

# Use RAG for Stage 2 as well (retrieve code examples)
cuda-test-rag gen-pipeline "Test kernel launch" --rag-stage2
```

#### Option B: Run Stages Separately

```bash
# Stage 1: Generate test intents
cuda-test-rag gen-intents "Test CUDA memory operations" -o intents.yaml

# (Optional) Review and edit intents.yaml manually

# Stage 2: Generate skeletons from intents
cuda-test-rag gen-skeletons intents.yaml -o tests.cu
```

### Legacy Single-Stage Generation

```bash
# Generate tests directly (single LLM call)
cuda-test-rag generate "Generate unit tests for matrix multiplication"
```

### Search Documents

```bash
# Search for relevant documents
cuda-test-rag search "memory management requirements"
```

### Clear Vector Store

```bash
cuda-test-rag clear
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ingest` | Ingest documents into vector store |
| `gen-pipeline` | Run full multi-stage pipeline |
| `gen-intents` | Stage 1: Generate test intents |
| `gen-skeletons` | Stage 2: Generate test skeletons |
| `generate` | Legacy single-stage generation |
| `search` | Search documents |
| `clear` | Clear vector store |

## Project Structure

```
cuda_test_design_rag/
├── pyproject.toml              # Project configuration
├── .env.example                # Environment template
├── scripts/
│   ├── init_venv.sh           # Bash setup script
│   └── init_venv.ps1          # PowerShell setup script
├── data/
│   ├── docs/                  # Place your documents here
│   └── vectorstore/           # ChromaDB storage
├── src/cuda_test_rag/
│   ├── cli.py                 # CLI interface
│   ├── config.py              # Settings management
│   ├── core/                  # RAG core functionality
│   │   ├── embeddings.py      # Embedding management
│   │   ├── retriever.py       # Document retrieval
│   │   └── vectorstore.py     # Vector store operations
│   ├── document_processing/   # Document handling
│   │   ├── loader.py          # File loaders
│   │   └── splitter.py        # Text chunking
│   ├── test_generation/       # Test generation
│   │   ├── generator.py       # Legacy single-stage generator
│   │   ├── pipeline.py        # Multi-stage pipeline
│   │   ├── models.py          # Data models (TestIntent, etc.)
│   │   └── prompts.py         # Prompt templates
│   └── utils/
│       └── logging.py         # Logging setup
└── tests/                     # Unit tests
```

## Configuration

Configuration can be set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Google AI API key (free) |
| `LLM_MODEL` | `gemini-2.5-flash` | LLM model for generation |
| `EMBEDDING_MODEL` | `models/embedding-001` | Embedding model |
| `TEMPERATURE` | `0.1` | Generation temperature |
| `CHUNK_SIZE` | `1000` | Document chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RETRIEVAL_K` | `4` | Number of documents to retrieve |
| `VECTORSTORE_PATH` | `./data/vectorstore` | Vector store location |
| `DOCS_PATH` | `./data/docs` | Documents directory |

## Programmatic Usage

### Multi-Stage Pipeline

```python
from cuda_test_rag.config import Settings
from cuda_test_rag.test_generation import TestGenerationPipeline

settings = Settings()
pipeline = TestGenerationPipeline(settings)

# Run full pipeline
result = pipeline.run_full_pipeline("Test memory operations")
print(f"Generated {len(result.test_intents.test_intents)} intents")
print(result.test_skeletons)

# Or run stages separately
intents, context, sources = pipeline.generate_intents("Test memory ops")
pipeline.save_intents(intents, "intents.yaml")

# Later, load and generate skeletons
intents = pipeline.load_intents("intents.yaml")
skeletons = pipeline.generate_skeletons(intents)
```

### Legacy Single-Stage

```python
from cuda_test_rag.config import Settings
from cuda_test_rag.test_generation import CUDATestGenerator

settings = Settings()
generator = CUDATestGenerator(settings)
result = generator.retrieve_and_generate("Generate tests for cudaMemcpy")
print(result["generated_tests"])
```

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLI Entry Point (cli.py)                           │
│  ┌────────────┬──────────────┬─────────────┬──────────────┬──────────────────┐  │
│  │  ingest    │  gen-intents │gen-skeletons│ gen-pipeline │    generate      │  │
│  │  (docs)    │  (Stage 1)   │ (Stage 2)   │ (Full pipeline)│   (legacy)     │  │
│  └─────┬──────┴──────┬───────┴──────┬──────┴───────┬──────┴────────┬─────────┘  │
└────────│─────────────│──────────────│──────────────│───────────────│────────────┘
         │             │              │              │               │
         ▼             └──────────────┴──────┬───────┴───────────────┘
┌─────────────────┐                          │
│   INGEST FLOW   │                          ▼
└────────┬────────┘            ┌──────────────────────────────────────────────────┐
         │                     │           GENERATION FLOW                        │
         ▼                     └──────────────────────────────────────────────────┘
┌─────────────────────────┐                  │
│  document_processing/   │                  ▼
│  ┌───────────────────┐  │    ┌───────────────────────────────────────────────────┐
│  │  DocumentLoader   │  │    │              test_generation/                     │
│  │  ├─ PDF           │  │    │  ┌─────────────────────────────────────────────┐  │
│  │  ├─ DOCX          │  │    │  │      TestGenerationPipeline (pipeline.py)   │  │
│  │  ├─ TXT           │  │    │  │  ┌─────────────┐ ┌─────────────┐ ┌────────┐ │  │
│  │  └─ MD            │  │    │  │  │  STAGE 1    │ │  STAGE 2    │ │STAGE 3 │ │  │
│  └─────────┬─────────┘  │    │  │  │  Intents    │ │  Test Cases │ │Skeleton│ │  │
│            │            │    │  │  │  (YAML)     │→│  (Details)  │→│ [TODO] │ │  │
│            ▼            │    │  │  └─────────────┘ └─────────────┘ └────────┘ │  │
│  ┌───────────────────┐  │    │  └─────────────────────────────────────────────┘  │
│  │ DocumentSplitter  │  │    │                                                   │
│  │ ├─ by_test_case   │  │    │  ┌─────────────────────────────────────────────┐  │
│  │ ├─ by_section     │  │    │  │  TestPromptTemplates (prompts.py)           │  │
│  │ └─ default        │  │    │  │  ├─ Intent generation prompts               │  │
│  └─────────┬─────────┘  │    │  │  ├─ Test case generation prompts            │  │
│            │            │    │  │  └─ Few-shot examples                       │  │
└────────────│────────────┘    │  └─────────────────────────────────────────────┘  │
             │                 │                                                   │
             │                 │  ┌─────────────────────────────────────────────┐  │
             │                 │  │  Models (models.py)                         │  │
             │                 │  │  ├─ TestIntent, TestIntentCollection        │  │
             │                 │  │  ├─ TestSkeleton, PipelineResult            │  │
             │                 │  │  └─ TestCategory, TestPriority (enums)      │  │
             │                 │  └─────────────────────────────────────────────┘  │
             │                 └───────────────────────────────────────────────────┘
             │                                       │
             ▼                                       │
┌────────────────────────────────────────────────────│──────────────────────────────┐
│                            core/ (RAG Layer)       │                              │
│  ┌───────────────────┐                             ▼                              │
│  │  EmbeddingManager │◄────────────────────────────────────────────┐              │
│  │  (embeddings.py)  │                                             │              │
│  │  ├─ GoogleGenAI   │                                             │              │
│  │  │   Embeddings   │                         ┌───────────────────┴───────────┐  │
│  │  └─ HuggingFace   │                         │   DocumentRetriever           │  │
│  │     (local opt.)  │                         │   (retriever.py)              │  │
│  └─────────┬─────────┘                         │   ├─ retrieve(query, k)       │  │
│            │                                   │   ├─ retrieve_with_scores()   │  │
│            ▼                                   │   └─ get_retriever()          │  │
│  ┌───────────────────┐                         └───────────────┬───────────────┘  │
│  │ VectorStoreManager│◄────────────────────────────────────────┘                  │
│  │  (vectorstore.py) │                                                            │
│  │  ├─ add_documents │          ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│  │  ├─ similarity_   │          │  Reranker [TODO]                             │  │
│  │  │   search       │          │  (Cohere Rerank / Cross-Encoder / BGE)       │  │
│  │  └─ clear         │          └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
│  └─────────┬─────────┘                                                            │
└────────────│──────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐    ┌──────────────────────────────────────┐
│            ChromaDB                    │    │          storage/                    │
│  ┌──────────────────────────────────┐  │    │  ┌──────────────────────────────┐   │
│  │  data/vectorstore/               │  │    │  │  RequestDatabase             │   │
│  │  (Persistent Vector Collection)  │  │    │  │  (database.py)               │   │
│  └──────────────────────────────────┘  │    │  │  └─ SQLite: data/requests.db │   │
└────────────────────────────────────────┘    │  ├──────────────────────────────┤   │
                                              │  │  TestFileManager             │   │
                                              │  │  (file_manager.py)           │   │
                                              │  └──────────────────────────────┘   │
                                              └──────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────┐
│                           External Services (LangChain Stack)                      │
│  ┌────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │ langchain_google_  │  │ langchain_chroma    │  │ langchain_community         │  │
│  │ genai              │  │ └─ Chroma wrapper   │  │ └─ Document loaders         │  │
│  │ ├─ GoogleGenAI     │  └─────────────────────┘  └─────────────────────────────┘  │
│  │ │  Embeddings      │                                                            │
│  │ └─ ChatGoogleGenAI │            ┌─────────────────────────────────────────┐     │
│  │    (LLM)           │◄───────────│  Google Gemini API (gemini-2.5-flash)   │     │
│  └────────────────────┘            └─────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                              Data Flow Summary                                     │
│                                                                                    │
│   Documents ──► Loader ──► Splitter ──► Embeddings ──► ChromaDB                   │
│                                                             │                      │
│   User Query ──────────────────────────────────────────────►│                      │
│                                                             ▼                      │
│                                          ┌──────────────────────────────┐          │
│                                          │   RAG Retrieval (Top-K)      │          │
│                                          │   (Google/HuggingFace Embed) │          │
│                                          └──────────────┬───────────────┘          │
│                                                         ▼                          │
│                                          ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐          │
│                                          │   Rerank [TODO]              │          │
│                                          │   (Cohere/Cross-Encoder)     │          │
│                                          └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘          │
│                                                         ▼                          │
│                                          ┌──────────────────────────────┐          │
│                                          │   Stage 1: Intent Generation │          │
│                                          │   Query + Context → YAML     │          │
│                                          └──────────────┬───────────────┘          │
│                                                         ▼                          │
│                                          ┌──────────────────────────────┐          │
│                                          │   [Optional: Human Review]   │          │
│                                          └──────────────┬───────────────┘          │
│                                                         ▼                          │
│                                          ┌──────────────────────────────┐          │
│                                          │   Stage 2: Test Case Gen     │          │
│                                          │   Intents → Test Cases       │          │
│                                          └──────────────┬───────────────┘          │
│                                                         ▼                          │
│                                          ┌──────────────────────────────┐          │
│                                          │   [Optional: Human Review]   │          │
│                                          └──────────────┬───────────────┘          │
│                                                         ▼                          │
│                                          ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐          │
│                                          │   Stage 3: Skeleton Gen      │          │
│                                          │   Test Cases → C++ [TODO]    │          │
│                                          └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘          │
│                                                         ▼                          │
│                                          ┌──────────────────────────────┐          │
│                                          │   Output: tests.cu           │          │
│                                          └──────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
                        ┌────────────┐
                        │  config.py │  (Settings - loaded by all modules)
                        │  Settings  │
                        └─────┬──────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌─────────────────┐
│     core/     │    │ document_      │    │ test_generation/│
│               │    │ processing/    │    │                 │
│ EmbeddingMgr  │    │                │    │ Pipeline        │
│      │        │    │ Loader         │    │ Generator       │
│      ▼        │    │    │           │    │ Models          │
│ VectorStore   │◄───│    ▼           │    │ Prompts         │
│      │        │    │ Splitter       │    │                 │
│      ▼        │    └────────────────┘    └────────┬────────┘
│ Retriever     │◄──────────────────────────────────┘
└───────────────┘
```

## Development

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black src tests
ruff check src tests
```

### Type Checking

```bash
mypy src
```

## License

MIT
