"""Similarity search and result fusion.

`foundation/01` keeps a teaching copy of cosine_similarity and retrieve;
`advanced-techniques/07-hybrid-search.ipynb` keeps one of reciprocal_rank_fusion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["cosine_similarity", "retrieve", "bm25_search_postgresql", "reciprocal_rank_fusion"]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors, in [-1, 1].

    Raises on a dimension mismatch rather than returning a meaningless number.
    Silently comparing a 384-dim vector to a 768-dim one is exactly the failure
    the multi-model registry exists to prevent, so it must be loud.
    """
    if len(a) != len(b):
        raise ValueError(
            f"cannot compare vectors of different dimensions: {len(a)} vs {len(b)}. "
            "This usually means two embedding models were mixed in one collection."
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(
    query_embedding: Sequence[float],
    corpus: Sequence[tuple[str, Sequence[float]]],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Return the top_n (chunk, similarity) pairs, most similar first.

    The brute-force in-memory version. `ragkit.store.VectorStore` does the same
    thing in PostgreSQL with an index once the corpus outgrows memory.
    """
    scored = [(chunk, cosine_similarity(query_embedding, emb)) for chunk, emb in corpus]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse several ranked ID lists into one, by reciprocal rank.

    Each list contributes 1/(k + rank) per item. The constant k damps the
    influence of the very top positions so that one confident-but-wrong ranker
    cannot dominate the fusion; 60 is the value from the original RRF paper.

    Works across rankers whose scores are not comparable -- which is the whole
    point when fusing dense-vector scores with BM25 scores.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def bm25_search_postgresql(
    query: str, conn, table_name: str, top_k: int = 10
) -> list[tuple[str, float, int]]:
    """Keyword retrieval over an embeddings table, using PostgreSQL full-text search.

    `ts_rank` is not literally BM25, but it is the same family: term frequency
    damped by document length, computed by the database that already holds the
    text. The point of pairing it with vector search is that it matches exact
    tokens -- product codes, names, error strings -- which a dense embedding
    happily smooths away.

    Returns (chunk_text, relevance, chunk_id) tuples, most relevant first.
    """
    with conn.cursor() as cur:
        # plainto_tsquery, not to_tsquery: it takes arbitrary user text without
        # tripping over operators, so a question mark cannot become a syntax error.
        cur.execute(
            f"""
            SELECT chunk_text,
                   ts_rank(to_tsvector('english', chunk_text),
                           plainto_tsquery('english', %s)) AS relevance,
                   id
            FROM {table_name}
            WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
            ORDER BY relevance DESC
            LIMIT %s
            """,
            (query, query, top_k),
        )
        return [(chunk, float(score), chunk_id) for chunk, score, chunk_id in cur.fetchall()]
