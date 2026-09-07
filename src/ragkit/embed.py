"""Embedding generation, with a seam for the fake client.

Every notebook previously called ``ollama.embed`` directly in a Python loop, one
text per HTTP round trip. Ollama accepts a batch, so embedding a 1000-chunk
corpus went from 1000 requests to a handful.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import config
from .models import get_embedding_model

__all__ = ["get_client", "embed_texts", "embed_one"]

_client: Any = None


def get_client() -> Any:
    """Return the Ollama client, or the deterministic fake when RAG_FAKE_MODELS is set.

    Tests and CI patch this single seam rather than monkeypatching the ollama
    module, which is what let the old mocks silently miss.
    """
    global _client
    if _client is not None:
        return _client

    if config.USE_FAKE_MODELS:
        from .testing import FakeOllamaClient

        _client = FakeOllamaClient()
        return _client

    import ollama

    if not hasattr(ollama, "embed"):
        raise RuntimeError(
            "The installed ollama package predates embed(). ragkit needs >=0.5 "
            "for batch input and the dimensions= parameter: pip install -U 'ollama>=0.5'"
        )
    _client = ollama
    return _client


def set_client(client: Any) -> None:
    """Override the client. Intended for tests."""
    global _client
    _client = client


def embed_texts(
    texts: Sequence[str],
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed many texts, batching requests rather than looping one at a time.

    Args:
        texts: The strings to embed.
        model: Catalog alias or Ollama tag. Defaults to the configured model.
        dimensions: Truncate output to this width (Matryoshka). None keeps full width.
        batch_size: Texts per request.
    """
    if not texts:
        return []

    spec = get_embedding_model(model or config.EMBEDDING_ALIAS)
    client = get_client()

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        kwargs: dict[str, Any] = {"model": spec.ollama_tag, "input": batch}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        response = client.embed(**kwargs)
        vectors.extend(response["embeddings"])

    expected = dimensions or spec.dimension
    if vectors and len(vectors[0]) != expected:
        raise RuntimeError(
            f"{spec.ollama_tag} returned {len(vectors[0])}-dim vectors, expected {expected}. "
            "Run scripts/preflight.py to re-check the catalog against the server."
        )
    return vectors


def embed_one(text: str, model: str | None = None, dimensions: int | None = None) -> list[float]:
    """Embed a single string. Convenience wrapper over embed_texts."""
    return embed_texts([text], model=model, dimensions=dimensions)[0]
