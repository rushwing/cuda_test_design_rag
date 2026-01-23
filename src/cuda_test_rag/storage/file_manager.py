"""File manager for organizing test case documents."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from cuda_test_rag.config import Settings


class TestFileManager:
    """Manages file organization for test case documents.

    Directory Structure:
    data/
    ├── knowledge_base/          # Original/curated test cases (ingested into RAG)
    │   └── docs/
    │       ├── cuda_test_memory_management.md
    │       └── ...
    ├── requests/                 # Generation requests and outputs
    │   └── 2024/
    │       └── 01/
    │           └── REQ-2024-0123-001/
    │               ├── request.json
    │               ├── few_shot_sources.txt
    │               ├── generated_intents.yaml
    │               └── generated_tests.md
    ├── reviewed/                 # Expert-reviewed test cases
    │   └── 2024/
    │       └── 01/
    │           └── REQ-2024-0123-001/
    │               ├── reviewed_tests.md
    │               └── review_notes.md
    ├── training_ready/           # Approved for training (symlinks or copies)
    │   └── ...
    └── vectorstore/              # ChromaDB
    """

    def __init__(self, settings: Settings | None = None, base_path: str | None = None):
        self.settings = settings or Settings()
        self.base_path = Path(base_path or "./data")
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for subdir in ["knowledge_base/docs", "requests", "reviewed", "training_ready"]:
            (self.base_path / subdir).mkdir(parents=True, exist_ok=True)

    # ==================== Path Generation ====================

    def get_request_dir(self, request_id: str, created_at: datetime | None = None) -> Path:
        """Get the directory path for a request.

        Structure: requests/YYYY/MM/request_id/
        """
        dt = created_at or datetime.utcnow()
        request_dir = self.base_path / "requests" / f"{dt.year}" / f"{dt.month:02d}" / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        return request_dir

    def get_reviewed_dir(self, request_id: str, created_at: datetime | None = None) -> Path:
        """Get the directory path for reviewed files.

        Structure: reviewed/YYYY/MM/request_id/
        """
        dt = created_at or datetime.utcnow()
        reviewed_dir = self.base_path / "reviewed" / f"{dt.year}" / f"{dt.month:02d}" / request_id
        reviewed_dir.mkdir(parents=True, exist_ok=True)
        return reviewed_dir

    def get_knowledge_base_dir(self) -> Path:
        """Get the knowledge base documents directory."""
        return self.base_path / "knowledge_base" / "docs"

    def get_training_ready_dir(self) -> Path:
        """Get the training-ready documents directory."""
        return self.base_path / "training_ready"

    # ==================== Request Files ====================

    def get_request_config_path(self, request_id: str, created_at: datetime | None = None) -> Path:
        """Get path for request configuration JSON."""
        return self.get_request_dir(request_id, created_at) / "request.json"

    def get_generated_intents_path(
        self, request_id: str, created_at: datetime | None = None
    ) -> Path:
        """Get path for generated intents YAML."""
        return self.get_request_dir(request_id, created_at) / "generated_intents.yaml"

    def get_generated_tests_path(
        self, request_id: str, created_at: datetime | None = None
    ) -> Path:
        """Get path for generated test cases markdown."""
        return self.get_request_dir(request_id, created_at) / "generated_tests.md"

    def get_few_shot_sources_path(
        self, request_id: str, created_at: datetime | None = None
    ) -> Path:
        """Get path for few-shot source file list."""
        return self.get_request_dir(request_id, created_at) / "few_shot_sources.txt"

    # ==================== Review Files ====================

    def get_reviewed_tests_path(
        self, request_id: str, created_at: datetime | None = None
    ) -> Path:
        """Get path for expert-reviewed test cases."""
        return self.get_reviewed_dir(request_id, created_at) / "reviewed_tests.md"

    def get_review_notes_path(
        self, request_id: str, created_at: datetime | None = None
    ) -> Path:
        """Get path for review notes."""
        return self.get_reviewed_dir(request_id, created_at) / "review_notes.md"

    # ==================== File Operations ====================

    def save_request_config(self, request_id: str, config: dict, created_at: datetime | None = None) -> Path:
        """Save request configuration to JSON file."""
        import json

        path = self.get_request_config_path(request_id, created_at)
        path.write_text(json.dumps(config, indent=2, default=str))
        return path

    def save_generated_intents(
        self, request_id: str, intents_yaml: str, created_at: datetime | None = None
    ) -> Path:
        """Save generated intents to YAML file."""
        path = self.get_generated_intents_path(request_id, created_at)
        path.write_text(intents_yaml)
        return path

    def save_generated_tests(
        self, request_id: str, tests_md: str, created_at: datetime | None = None
    ) -> Path:
        """Save generated test cases to markdown file."""
        path = self.get_generated_tests_path(request_id, created_at)
        path.write_text(tests_md)
        return path

    def save_few_shot_sources(
        self, request_id: str, source_paths: list[str], created_at: datetime | None = None
    ) -> Path:
        """Save list of few-shot source files used."""
        path = self.get_few_shot_sources_path(request_id, created_at)
        path.write_text("\n".join(source_paths))
        return path

    def save_reviewed_tests(
        self,
        request_id: str,
        reviewed_md: str,
        review_notes: str = "",
        created_at: datetime | None = None,
    ) -> tuple[Path, Optional[Path]]:
        """Save expert-reviewed test cases and notes."""
        tests_path = self.get_reviewed_tests_path(request_id, created_at)
        tests_path.write_text(reviewed_md)

        notes_path = None
        if review_notes:
            notes_path = self.get_review_notes_path(request_id, created_at)
            notes_path.write_text(review_notes)

        return tests_path, notes_path

    # ==================== Training Data ====================

    def promote_to_training(
        self, request_id: str, source_path: Path, created_at: datetime | None = None
    ) -> Path:
        """Copy approved test cases to training-ready directory."""
        dt = created_at or datetime.utcnow()

        # Create a descriptive filename
        dest_filename = f"{dt.strftime('%Y%m%d')}_{request_id}_tests.md"
        dest_path = self.get_training_ready_dir() / dest_filename

        shutil.copy2(source_path, dest_path)
        return dest_path

    def list_training_ready_files(self) -> list[Path]:
        """List all files in training-ready directory."""
        training_dir = self.get_training_ready_dir()
        return sorted(training_dir.glob("*.md"))

    # ==================== Knowledge Base ====================

    def add_to_knowledge_base(self, source_path: Path, new_name: str | None = None) -> Path:
        """Add a reviewed test case to the knowledge base."""
        dest_name = new_name or source_path.name
        dest_path = self.get_knowledge_base_dir() / dest_name

        shutil.copy2(source_path, dest_path)
        return dest_path

    def list_knowledge_base_files(self) -> list[Path]:
        """List all files in knowledge base."""
        kb_dir = self.get_knowledge_base_dir()
        return sorted(kb_dir.glob("*.md"))

    # ==================== Utilities ====================

    def get_relative_path(self, absolute_path: Path) -> str:
        """Get path relative to base_path for database storage."""
        try:
            return str(absolute_path.relative_to(self.base_path))
        except ValueError:
            return str(absolute_path)

    def get_absolute_path(self, relative_path: str) -> Path:
        """Get absolute path from relative path stored in database."""
        return self.base_path / relative_path
