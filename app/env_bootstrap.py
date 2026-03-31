"""Load repo-root ``.env`` into the process environment (optional ``python-dotenv``)."""

from __future__ import annotations

from pathlib import Path


def load_repo_dotenv() -> None:
    """Parse ``<repo>/.env`` if present. Existing OS environment wins (``override=False``)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent
    path = root / ".env"
    if path.is_file():
        load_dotenv(path, override=False)
