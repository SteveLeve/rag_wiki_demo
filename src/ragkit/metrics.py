"""Retrieval quality metrics.

Canonical implementations. Before September 2026 each of these existed in six
notebook copies; `evaluation-lab/02-evaluation-metrics-framework.ipynb` keeps a
teaching copy (see AGENTS.md), and every other notebook imports from here.

An AST comparison of the six copies found precision_at_k, recall_at_k and
mean_reciprocal_rank to be semantically identical, and the three ndcg_at_k
variants to differ only in whether they imported log2 from math or numpy. All
six ndcg copies, however, shared a genuine bug -- see ndcg_at_k below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "dcg_score",
    "ndcg_at_k",
]


def precision_at_k(retrieved_ids: Sequence[int], relevant_ids: Sequence[int], k: int = 5) -> float:
    """Of the top-K results, what fraction are relevant?

    Answers "how much of what I showed the user was worth showing?"

    Note the denominator is k, not len(retrieved_ids[:k]). Returning fewer than k
    results is itself a failure to fill the slots, and precision should reflect
    that rather than quietly grading on a curve.
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[int], relevant_ids: Sequence[int], k: int = 5) -> float:
    """Of all relevant chunks that exist, what fraction appear in the top-K?

    Answers "how much of what the user needed did I actually find?"
    """
    relevant_set = set(relevant_ids)
    if not relevant_set or k <= 0:
        return 0.0
    hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved_ids: Sequence[int], relevant_ids: Sequence[int]) -> float:
    """1 / (rank of the first relevant result), or 0.0 if none is relevant.

    Answers "how far must the user read before hitting something useful?"
    """
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def dcg_score(relevance_scores: Sequence[float]) -> float:
    """Discounted Cumulative Gain over an ordered list of relevance scores.

    Each hit is worth (2**rel - 1), discounted by log2(rank + 1) so that a
    relevant result found at rank 1 counts for more than the same result at
    rank 10.
    """
    return sum(
        (2**rel - 1) / math.log2(rank + 2)
        for rank, rel in enumerate(relevance_scores)
    )


def ideal_relevance(n_relevant: int, k: int) -> list[float]:
    """The relevance vector of a perfect top-K ranking -- the NDCG denominator.

    The ideal ranking puts as many relevant chunks as actually exist (capped at k)
    into the top slots. If fewer than k relevant chunks exist, the remaining slots
    are genuinely empty: no ranking could have done better, so they score zero in
    both the numerator and the denominator and cancel out.

    Every ndcg_at_k in this repo previously computed its ideal as
    ``sorted(retrieved_relevance, reverse=True)`` -- the best ordering *of what was
    retrieved*. That makes NDCG blind to recall: retrieving 1 relevant chunk out of
    10 and ranking it first scored a perfect 1.000, because the only achievable
    ordering of one hit is that hit at rank 1. Normalizing against the full relevant
    set instead scores that case 0.339, which is the honest number.

    Args:
        n_relevant: How many relevant chunks exist in the ground truth, total.
        k:          The cutoff.

    Returns:
        Relevance scores of an ideal ranking, highest first, length k.
    """
    if k <= 0:
        return []
    hits = min(max(n_relevant, 0), k)
    return [1.0] * hits + [0.0] * (k - hits)


def ndcg_at_k(retrieved_ids: Sequence[int], relevant_ids: Sequence[int], k: int = 5) -> float:
    """Normalized DCG@K: how good is this ranking against the best possible one?

    Answers "is the good stuff near the top, and did I find enough of it?"
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    relevance = [1.0 if cid in relevant_set else 0.0 for cid in retrieved_ids[:k]]

    idcg = dcg_score(ideal_relevance(len(relevant_set), k))
    if idcg == 0:
        return 0.0
    return dcg_score(relevance) / idcg
