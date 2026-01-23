"""Logging configuration."""

import logging
import sys

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging with rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    # Set library log levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    return logging.getLogger("cuda_test_rag")
