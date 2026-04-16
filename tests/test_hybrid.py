"""Lightweight unit tests that don't require downloading a model.

For the embedding-dependent tests, run the demo script.
"""

from matcher.bm25 import Bm25Index, tokenize


def test_tokenize_basic():
    assert tokenize("Heinz Tomato Ketchup 460ML") == ["heinz", "tomato", "ketchup", "460ml"]


def test_bm25_top_candidate_is_exact_match():
    ids = ["a", "b", "c"]
    names = ["Heinz Tomato Ketchup 460ML", "Hellmann's Mayo 450ml", "Coca-Cola 1.5L"]
    idx = Bm25Index(ids, names)
    top = idx.search("Heinz Tomato Ketchup 460ml", k=1)
    assert top[0][0] == "a"
