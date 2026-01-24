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

Create a `.env` file from the example template:

```bash
cp .env.example .env
```

Edit `.env` and configure the following settings:

```env
# Required: Get your free API key from https://aistudio.google.com/apikey
GOOGLE_API_KEY=your-google-api-key-here

# Optional: Use local embeddings instead of Google API (default: false)
# Set to true to use HuggingFace embeddings locally (no API calls for embeddings)
USE_LOCAL_EMBEDDINGS=false

# Optional: Local embedding model (default: all-MiniLM-L6-v2)
# Only used when USE_LOCAL_EMBEDDINGS=true
# Other options: sentence-transformers/all-mpnet-base-v2, BAAI/bge-small-en-v1.5
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**Embedding Options:**

| Option | Pros | Cons |
|--------|------|------|
| Google API (default) | Higher quality, no local resources | Requires API calls |
| Local HuggingFace | No API costs, works offline | Uses local CPU/GPU, slightly lower quality |

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

### 1. Ingest Documents

Add your requirements and design documents to `data/knowledge_base/docs/`:

```bash
cuda-test-rag ingest ./data/knowledge_base/docs
```

Supported file formats: PDF, DOCX, TXT, Markdown

### 2. Configure Test Request

Edit the configuration file `config/test_request_sample.json`:

```json
{
  "test_filters": {
    "gpu_architecture": "Hopper",
    "product_series": "Data Center",
    "module_name": "Memory Management",
    "cuda_solution_version": "12.2",
    "priority": "High",
    "test_type": "Functional"
  },
  "generation_config": {
    "num_few_shot_examples": 3,
    "focus_areas": ["corner cases", "error handling"]
  }
}
```

### 3. Generate Test Cases

```bash
# Auto mode: Run both stages (intents → test cases)
./scripts/gen_test_cases.sh --auto

# Stage 1 only: Generate intents for review
./scripts/gen_test_cases.sh --intent

# Approve intents after review
./scripts/gen_test_cases.sh --approve REQ-xxxxxxxx-xxxxxx

# Stage 2 only: Generate test cases from approved intents
./scripts/gen_test_cases.sh --case -r REQ-xxxxxxxx-xxxxxx

# List all requests
./scripts/gen_test_cases.sh --list
```

### 4. Review Generated Files

Generated files are saved to `data/requests/<request_id>/`:
- `intents.yaml` - Test intents from Stage 1
- `test_cases.md` - Generated test cases from Stage 2

### Utility Commands

```bash
# Clean generated files
./scripts/clean.sh

# Clean everything including vectorstore
./scripts/clean.sh --all

# Preview what would be deleted
./scripts/clean.sh --dry-run
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `./scripts/gen_test_cases.sh --auto` | Run full pipeline (both stages) |
| `./scripts/gen_test_cases.sh --intent` | Stage 1: Generate test intents |
| `./scripts/gen_test_cases.sh --case -r <ID>` | Stage 2: Generate test cases |
| `./scripts/gen_test_cases.sh --approve <ID>` | Approve intents for Stage 2 |
| `./scripts/gen_test_cases.sh --list` | List all requests |
| `./scripts/clean.sh` | Clean generated files |
| `cuda-test-rag ingest <path>` | Ingest documents into vector store |
| `cuda-test-rag search <query>` | Search documents |
| `cuda-test-rag clear` | Clear vector store |

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
| `EMBEDDING_MODEL` | `models/embedding-001` | Google embedding model |
| `USE_LOCAL_EMBEDDINGS` | `false` | Use local HuggingFace embeddings |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model name |
| `TEMPERATURE` | `0.1` | Generation temperature |
| `CHUNK_SIZE` | `1000` | Document chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RETRIEVAL_K` | `4` | Number of documents to retrieve |
| `VECTORSTORE_PATH` | `./data/vectorstore` | Vector store location |
| `DOCS_PATH` | `./data/knowledge_base/docs` | Documents directory |

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
