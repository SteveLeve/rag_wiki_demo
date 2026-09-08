"""ragkit - the shared toolkit behind the RAG Wiki Demo notebooks.

This package exists to serve the curriculum, not the other way around. Read
AGENTS.md before adding to it: some duplication between these modules and the
notebooks is deliberate and tested (see tests/test_teaching_parity.py).
"""

from .models import (
    DEFAULT_EMBEDDING_ALIAS,
    DEFAULT_LLM_TAG,
    EMBEDDING_MODELS,
    LANGUAGE_MODELS,
    EmbeddingModel,
    canonical_alias,
    get_embedding_model,
)

__version__ = "3.0.0"

__all__ = [
    "canonical_alias",
    "get_embedding_model",
    "EmbeddingModel",
    "EMBEDDING_MODELS",
    "LANGUAGE_MODELS",
    "DEFAULT_EMBEDDING_ALIAS",
    "DEFAULT_LLM_TAG",
    "__version__",
]
