"""Environment-driven configuration.

Everything here was previously a literal repeated across 21 notebooks, which is
why the embedding model, the LLM, and the database port each had two or three
different values depending on which notebook you opened.

The PostgreSQL port defaults to **5433**, not 5432. Port 5432 is commonly taken by
another project's database, and these notebooks create and drop tables -- pointing
them at a neighbour's server is a genuine hazard, not a nuisance.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from .models import DEFAULT_EMBEDDING_ALIAS, DEFAULT_LLM_TAG, canonical_alias

load_dotenv()  # a .env at the repo root wins over nothing, loses to real env vars


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


POSTGRES_CONFIG: dict[str, Any] = {
    "host": _env("RAG_PG_HOST", "localhost"),
    "port": int(_env("RAG_PG_PORT", "5433")),
    "database": _env("RAG_PG_DATABASE", "rag_db"),
    "user": _env("RAG_PG_USER", "postgres"),
    "password": _env("RAG_PG_PASSWORD", "postgres"),
}

# Which models the notebooks use. Override per-notebook by passing an explicit
# alias; override globally with the environment.
EMBEDDING_ALIAS: str = canonical_alias(_env("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_ALIAS))
LLM_TAG: str = _env("RAG_LLM_MODEL", DEFAULT_LLM_TAG)

# Chunking defaults, previously repeated as bare numbers in every notebook.
CHUNK_SIZE: int = int(_env("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(_env("RAG_CHUNK_OVERLAP", "100"))
TOP_K: int = int(_env("RAG_TOP_K", "5"))

# When set, ragkit.embed serves a deterministic fake instead of calling Ollama, so
# the whole notebook suite runs in CI with no server and no model downloads.
USE_FAKE_MODELS: bool = _env("RAG_FAKE_MODELS", "").lower() in {"1", "true", "yes"}


def describe() -> str:
    """One-line summary for notebooks to print, so a run's config is self-documenting."""
    where = f"{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"
    fake = "  [FAKE MODELS]" if USE_FAKE_MODELS else ""
    return f"embedding={EMBEDDING_ALIAS}  llm={LLM_TAG}  postgres={where}{fake}"
