"""Tests for the canonical retrieval metrics.

The NDCG cases are the important ones: they pin the behaviour that six notebook
copies previously got wrong, so a regression cannot slip back in.
"""

import math

import pytest

from ragkit.metrics import (
    dcg_score,
    ideal_relevance,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k([7, 8, 9], [1, 2, 3], k=3) == 0.0

    def test_partial(self):
        assert precision_at_k([1, 8, 9], [1, 2, 3], k=3) == pytest.approx(1 / 3)

    def test_denominator_is_k_not_result_count(self):
        # Returning 1 result when 5 were asked for is a failure to fill the slots.
        assert precision_at_k([1], [1, 2, 3], k=5) == pytest.approx(0.2)

    def test_k_zero(self):
        assert precision_at_k([1, 2], [1, 2], k=0) == 0.0


class TestRecallAtK:
    def test_found_all(self):
        assert recall_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0

    def test_found_half(self):
        assert recall_at_k([1, 2, 9, 9], [1, 2, 3, 4], k=4) == 0.5

    def test_no_relevant_defined(self):
        assert recall_at_k([1, 2], [], k=2) == 0.0

    def test_duplicates_in_ground_truth_do_not_inflate(self):
        assert recall_at_k([1], [1, 1, 1], k=5) == 1.0


class TestMeanReciprocalRank:
    def test_first_position(self):
        assert mean_reciprocal_rank([1, 2, 3], [1]) == 1.0

    def test_third_position(self):
        assert mean_reciprocal_rank([8, 9, 1], [1]) == pytest.approx(1 / 3)

    def test_absent(self):
        assert mean_reciprocal_rank([8, 9], [1]) == 0.0


class TestIdealRelevance:
    def test_fewer_relevant_than_k(self):
        assert ideal_relevance(2, 5) == [1.0, 1.0, 0.0, 0.0, 0.0]

    def test_more_relevant_than_k(self):
        assert ideal_relevance(10, 3) == [1.0, 1.0, 1.0]

    def test_none_relevant(self):
        assert ideal_relevance(0, 3) == [0.0, 0.0, 0.0]

    def test_k_zero(self):
        assert ideal_relevance(5, 0) == []


class TestNDCG:
    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k([1, 2, 3], [1, 2, 3], k=3) == pytest.approx(1.0)

    def test_nothing_relevant_scores_zero(self):
        assert ndcg_at_k([7, 8, 9], [1, 2, 3], k=3) == 0.0

    def test_penalises_missing_relevant_chunks(self):
        """The regression this whole module exists for.

        One relevant chunk out of ten, ranked first. The old implementation
        normalised against the retrieved set and returned a perfect 1.0. NDCG must
        see that nine relevant chunks were missed.
        """
        score = ndcg_at_k([1, 90, 91, 92, 93], list(range(1, 11)), k=5)
        assert score == pytest.approx(0.339, abs=1e-3)
        assert score < 1.0

    def test_ranking_still_matters(self):
        """Same recall, better ordering must score higher."""
        early = ndcg_at_k([1, 90, 91, 92, 93], list(range(1, 11)), k=5)
        late = ndcg_at_k([90, 91, 92, 93, 1], list(range(1, 11)), k=5)
        assert early > late

    def test_no_relevant_defined(self):
        assert ndcg_at_k([1, 2], [], k=2) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k([1, 2], [1], k=0) == 0.0

    def test_dcg_discounts_by_position(self):
        assert dcg_score([1, 0]) == pytest.approx(1 / math.log2(2))
        assert dcg_score([0, 1]) == pytest.approx(1 / math.log2(3))
