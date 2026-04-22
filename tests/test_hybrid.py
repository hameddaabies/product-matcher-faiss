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


# ---------------------------------------------------------------------------
# Unit-aware tokenizer tests
# ---------------------------------------------------------------------------


def test_tokenize_decimal_unit_is_single_token():
    """'1.5L' must not be split into ['1', '5l'] — the naive alphanumeric split."""
    tokens = tokenize("Coca-Cola 1.5L")
    assert "1.5l" in tokens, f"expected '1.5l' in {tokens}"
    assert "1" not in tokens
    assert "5l" not in tokens


def test_tokenize_decimal_unit_full_sequence():
    assert tokenize("Coca-Cola 1.5L") == ["coca", "cola", "1.5l"]


def test_tokenize_integer_unit_unchanged():
    """Integer-unit tokens (no decimal) must still be preserved as before."""
    assert tokenize("Philips 300W LED Bulb") == ["philips", "300w", "led", "bulb"]


def test_tokenize_decimal_weight_unit():
    assert tokenize("Oat milk 2.5kg bag") == ["oat", "milk", "2.5kg", "bag"]


def test_bm25_ranks_correct_volume_first():
    """Unit-aware tokenizer lets BM25 distinguish products by volume.

    BM25Okapi IDF is negative when a term appears in >N/2 docs.  We need
    '1.5l' to appear in only 1 of 4 documents so its IDF is positive and it
    lifts the 1.5 L product to the top.
    """
    ids = ["a", "b", "c", "d"]
    names = ["Coca-Cola 1.5L", "Coca-Cola 2L", "Pepsi 2L", "Sprite 2L"]
    idx = Bm25Index(ids, names)
    results = idx.search("Coca-Cola 1.5L", k=1)
    assert results[0][0] == "a", (
        f"Expected Coca-Cola 1.5L ('a') to rank first, got '{results[0][0]}'"
    )
