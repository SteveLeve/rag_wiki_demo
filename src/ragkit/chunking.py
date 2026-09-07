"""Splitting documents into retrievable chunks.

`foundation/01-basic-rag-in-memory.ipynb` keeps a teaching copy of chunk_text --
it is the first real algorithm the curriculum shows. Everything downstream
imports from here.
"""

from __future__ import annotations

__all__ = ["chunk_text", "estimate_size_mb"]


def chunk_text(text: str, max_size: int = 1000) -> list[str]:
    """Split text into chunks of at most ~max_size characters.

    Prefers paragraph boundaries, falling back to sentence boundaries when a
    single paragraph is oversized. Chunking on structure rather than a fixed
    character window keeps semantically related sentences together, which is what
    makes the retrieved context readable rather than truncated mid-thought.
    """
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) > max_size:
            if current:
                chunks.append(current.strip())
                current = ""

            if len(paragraph) > max_size:
                # A single paragraph overflows; drop to sentence granularity.
                for sentence in paragraph.split(". "):
                    if len(current) + len(sentence) > max_size:
                        if current:
                            chunks.append(current.strip())
                        current = sentence + ". "
                    else:
                        current += sentence + ". "
            else:
                current = paragraph
        else:
            current += "\n\n" + paragraph if current else paragraph

    if current.strip():
        chunks.append(current.strip())
    return chunks


def estimate_size_mb(items: list[str]) -> float:
    """Approximate the in-memory size of a list of strings, in megabytes."""
    return sum(len(s.encode("utf-8")) for s in items) / (1024 * 1024)
