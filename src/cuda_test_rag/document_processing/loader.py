"""Document loaders for various file formats."""

from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


class DocumentLoader:
    """Loads documents from various file formats."""

    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": TextLoader,  # Use TextLoader for markdown (no unstructured dependency)
    }

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a single file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        loader = loader_class(str(path))
        documents = loader.load()

        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = str(path)
            doc.metadata["file_name"] = path.name

        return documents

    def load_directory(self, directory_path: str | Path) -> list[Document]:
        """Load all supported files from a directory."""
        path = Path(directory_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        documents = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    docs = self.load_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return documents
