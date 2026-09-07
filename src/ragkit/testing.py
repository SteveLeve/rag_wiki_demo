"""A deterministic stand-in for Ollama, so notebooks and tests run without it.

The mocks this replaces never worked. Both patched the legacy module-level
``ollama.embeddings`` with a ``prompt=`` signature returning ``{'embedding': ...}``,
while every notebook calls ``ollama.embed(model=, input=)`` and reads
``['embeddings']``. Nothing was ever intercepted, so the "mocked" tests either hit
a live daemon or asserted against errors.

Worse, tests/conftest.py seeded a fixed ``RandomState(42)``, so *every* text
received the *identical* vector. Any similarity assertion built on that was
comparing a vector to itself.

This fake fixes both: vectors derive from content, so similar calls are
distinguishable and similarity is meaningful, and dimensions come from the
catalog, so a fake run still exercises 384/768/1024 rather than masking a
hardcoded 768.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

from .models import get_embedding_model

__all__ = ["FakeOllamaClient", "fake_embedding"]


def fake_embedding(text: str, dimension: int) -> list[float]:
    """A deterministic unit vector derived from the text's content.

    Equal texts give equal vectors; texts sharing a prefix give nearby vectors,
    so cosine similarity is at least weakly meaningful rather than noise.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
    # Expand the digest to the requested width, then unit-normalize.
    raw: list[float] = []
    counter = 0
    while len(raw) < dimension:
        block = hashlib.blake2b(
            digest + counter.to_bytes(4, "big"), digest_size=64
        ).digest()
        raw.extend((b - 127.5) / 127.5 for b in block)
        counter += 1
    raw = raw[:dimension]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


class FakeOllamaClient:
    """Implements the parts of the ollama API that ragkit uses.

    Deliberately mirrors the *current* API shape -- ``embed(model=, input=)``
    returning ``{'embeddings': [[...]]}`` -- so a passing fake run is evidence the
    real call signature is right.
    """

    def __init__(self) -> None:
        self.embed_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []

    def embed(
        self,
        model: str,
        input: str | Sequence[str],  # noqa: A002 - matches the ollama signature
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.embed_calls.append({"model": model, "input": input, **kwargs})
        texts = [input] if isinstance(input, str) else list(input)
        width = dimensions or get_embedding_model(model).dimension
        return {
            "model": model,
            "embeddings": [fake_embedding(t, width) for t in texts],
        }

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.chat_calls.append({"model": model, "messages": messages, **kwargs})
        last = messages[-1]["content"] if messages else ""
        return {
            "model": model,
            "message": {
                "role": "assistant",
                "content": f"[fake completion for: {last[:80]}]",
            },
            "done": True,
        }
