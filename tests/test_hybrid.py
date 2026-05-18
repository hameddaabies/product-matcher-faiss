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


# ---------------------------------------------------------------------------
# Whitespace-tolerant unit tokenization (retailer-B style: "460 ml", "1.5 L")
# ---------------------------------------------------------------------------


def test_tokenize_integer_unit_with_space():
    """'460 ml' must collapse into the same token as '460ml'."""
    tokens = tokenize("HEINZ KETCHUP - TOMATO (460 ml)")
    assert "460ml" in tokens, f"expected '460ml' in {tokens}"
    assert "460" not in tokens
    assert "ml" not in tokens


def test_tokenize_decimal_unit_with_space():
    tokens = tokenize("Coca Cola 1.5 L bottle")
    assert "1.5l" in tokens, f"expected '1.5l' in {tokens}"


def test_tokenize_full_word_unit_with_space():
    """Unit allowlist covers spelled-out forms like 'grams' and 'liter'."""
    assert "400grams" in tokenize("Nutella 400 grams")
    assert "1.5liter" in tokenize("Coca Cola 1.5 liter")


def test_tokenize_does_not_merge_non_unit_word():
    """A trailing word outside the unit allowlist must stay split.

    Guards against over-eager merging that would lose word-level signal
    (e.g. '100 Pack' must not become a single opaque '100pack' token).
    """
    tokens = tokenize("Lipton Tea Bags 100 Pack")
    assert "100" in tokens
    assert "pack" in tokens
    assert "100pack" not in tokens


# ---------------------------------------------------------------------------
# Unicode letters — European catalogs (Dutch, French, Spanish, German)
# ---------------------------------------------------------------------------


def test_tokenize_preserves_latin_diacritics():
    """'Crème brûlée' must stay two tokens, not shatter into ASCII fragments."""
    assert tokenize("Crème brûlée 250g") == ["crème", "brûlée", "250g"]


def test_tokenize_preserves_tilde_n():
    """'Jalapeño' must survive as one token (not 'jalape' + 'o')."""
    tokens = tokenize("Jalapeño chips")
    assert "jalapeño" in tokens
    assert "jalape" not in tokens


def test_bm25_matches_across_diacritic_variant():
    """Query without diacritics should still find candidate with diacritics
    when other tokens overlap — and exact-diacritic queries must not collapse
    to a different product just because the ASCII fallback ranked higher."""
    ids = ["a", "b"]
    names = ["Crème brûlée dessert 250g", "Vanilla pudding 250g"]
    idx = Bm25Index(ids, names)
    top = idx.search("Crème brûlée 250g", k=1)
    assert top[0][0] == "a"


def test_bm25_matches_across_unit_whitespace_variants():
    """Cross-retailer query: '460 ml' (retailer B) must hit '460ML' (retailer A)."""
    ids = ["a", "b", "c"]
    names = [
        "Heinz Tomato Ketchup 460ML",
        "Heinz Ketchup 1000ml",
        "Hellmann's Mayo 450ml",
    ]
    idx = Bm25Index(ids, names)
    top = idx.search("HEINZ KETCHUP - TOMATO (460 ml)", k=1)
    assert top[0][0] == "a", (
        f"Expected '460 ml' query to rank '460ML' candidate first, got '{top[0][0]}'"
    )
