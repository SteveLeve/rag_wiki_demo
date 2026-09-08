"""Second-pass reranking of retrieved candidates.

Vector search optimises for recall over a large corpus; a reranker re-scores the
handful of survivors with a slower, more accurate model that reads the query and
document *together* rather than embedding them independently.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from . import config, embed

__all__ = ["DEFAULT_CROSS_ENCODER", "load_cross_encoder", "cross_encoder_rerank", "llm_rerank"]

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=4)
def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER):
    """Load a CrossEncoder once and keep it.

    Constructing one costs seconds. Doing it *inside* the per-query rerank
    function -- which is what the notebook used to do -- reloads the model from
    disk for every question, and turns a two-minute evaluation into an eight
    minute one that looks like reranking is inherently slow. It is not; loading
    is.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "cross-encoder reranking needs sentence-transformers. "
            "Install it with: pip install -e '.[advanced]'"
        ) from exc
    return CrossEncoder(model_name)


def cross_encoder_rerank(
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_CROSS_ENCODER,
    text_key: str = "chunk_text",
) -> list[dict[str, Any]]:
    """Rerank candidates with a sentence-transformers CrossEncoder.

    Note ``model_name`` is honoured. The notebook version declared a
    RERANKER_MODEL constant and then shadowed it with a hardcoded default
    argument, so changing the constant had no effect.
    """
    if not candidates:
        return []

    encoder = load_cross_encoder(model_name)
    pairs = [(query, c[text_key]) for c in candidates]
    scores = encoder.predict(pairs)

    ranked = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def llm_rerank(
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_k: int = 5,
    model: str | None = None,
    text_key: str = "chunk_text",
) -> list[dict[str, Any]]:
    """Rerank by asking the local LLM to score each candidate's relevance 0-10.

    Needs no extra model download -- it reuses the generation model the notebooks
    already pull. Slower than a cross-encoder per candidate, which is the point
    the modern-reranking notebook measures rather than asserts.
    """
    if not candidates:
        return []

    client = embed.get_client()
    tag = model or config.LLM_TAG
    scored = []
    for candidate in candidates:
        prompt = (
            "Rate how well the passage answers the question, from 0 to 10.\n"
            "Reply with only the number.\n\n"
            f"Question: {query}\n\nPassage: {candidate[text_key][:1500]}\n\nScore:"
        )
        reply = client.chat(model=tag, messages=[{"role": "user", "content": prompt}])
        scored.append({**candidate, "rerank_score": _parse_score(reply["message"]["content"])})

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def _parse_score(text: str) -> float:
    """Pull the first number out of an LLM reply, clamped to 0-10.

    Small local models add commentary despite instructions, so this tolerates it
    and falls back to 0.0 rather than raising mid-pipeline.
    """
    import re

    match = re.search(r"\d+(?:\.\d+)?", text or "")
    if not match:
        return 0.0
    return max(0.0, min(10.0, float(match.group())))
