"""Unit tests for the precision@1 / recall@k primitives.

These don't load an embedder — they exercise the pure ranking-metric
functions on hand-rolled retrieved-id lists.
"""

from matcher.eval import precision_at_1, recall_at_k


def test_precision_at_1_hit():
    assert precision_at_1(["a1", "a2", "a3"], "a1") == 1.0


def test_precision_at_1_miss_when_gold_is_second():
    assert precision_at_1(["a2", "a1", "a3"], "a1") == 0.0


def test_precision_at_1_empty_retrieved():
    assert precision_at_1([], "a1") == 0.0


def test_recall_at_k_hit_at_top():
    assert recall_at_k(["a1", "a2", "a3"], "a1", k=5) == 1.0


def test_recall_at_k_hit_within_window():
    assert recall_at_k(["a2", "a3", "a1", "a4"], "a1", k=5) == 1.0


def test_recall_at_k_miss_outside_window():
    """Gold at position 6 must miss when k=5."""
    retrieved = ["a2", "a3", "a4", "a5", "a6", "a1"]
    assert recall_at_k(retrieved, "a1", k=5) == 0.0


def test_recall_at_k_truncates_to_window():
    """Recall at k=1 only counts the very first hit."""
    assert recall_at_k(["a2", "a1"], "a1", k=1) == 0.0
    assert recall_at_k(["a1", "a2"], "a1", k=1) == 1.0
