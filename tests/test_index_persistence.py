"""Tests for HnswIndex.save/load and HybridMatcher.save/load.

Synthetic random vectors and a stub embedder are used so these tests do not
depend on sentence-transformers being downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from matcher.hybrid import HybridMatcher
from matcher.index import HnswIndex


def _make_unit_vecs(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_save_load_roundtrip_preserves_search_results(tmp_path):
    dim = 16
    ids = [f"p{i}" for i in range(20)]
    vecs = _make_unit_vecs(len(ids), dim)

    original = HnswIndex(dim)
    original.add(ids, vecs)
    queries = _make_unit_vecs(3, dim, seed=1)
    expected = original.search(queries, k=5)

    path = tmp_path / "index.faiss"
    original.save(path)
    assert path.exists()
    assert (tmp_path / "index.faiss.ids.json").exists()

    restored = HnswIndex.load(path)
    assert restored.dim == dim
    assert restored.search(queries, k=5) == expected


def test_load_recovers_ids_in_original_order(tmp_path):
    dim = 8
    ids = ["alpha", "beta", "gamma", "delta"]
    vecs = _make_unit_vecs(len(ids), dim)

    idx = HnswIndex(dim)
    idx.add(ids, vecs)
    path = tmp_path / "ids.faiss"
    idx.save(path)

    restored = HnswIndex.load(path)
    assert restored._ids == ids


def test_add_rejects_ids_vecs_count_mismatch():
    """Fewer ids than vectors must fail fast, not silently corrupt the row->id map."""
    dim = 8
    vecs = _make_unit_vecs(4, dim)
    idx = HnswIndex(dim)
    with pytest.raises(AssertionError, match="count mismatch"):
        idx.add(["only", "three", "ids"], vecs)


def test_search_k_larger_than_index_drops_padding_sentinels():
    """k > ntotal must return only real hits, never a phantom from the -1 pad.

    FAISS fills the unused slots of an over-large k request with row ``-1``.
    Unguarded, ``self._ids[-1]`` would resolve that sentinel to the *last*
    id, silently duplicating it as a fake neighbour — so the guard's job is
    to yield exactly the two indexed items, with no repeats.
    """
    dim = 8
    ids = ["a", "b"]
    idx = HnswIndex(dim)
    idx.add(ids, _make_unit_vecs(len(ids), dim))

    hits = idx.search(_make_unit_vecs(1, dim, seed=1), k=5)[0]

    assert len(hits) == len(ids)
    assert {pid for pid, _ in hits} == set(ids)


# ---------------------------------------------------------------------------
# HybridMatcher.save / HybridMatcher.load
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Deterministic fake embedder — a pure function of text, no model download."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            vecs[i] = v / np.linalg.norm(v)
        return vecs


def test_hybrid_matcher_save_load_roundtrip_preserves_match(tmp_path):
    ids = ["a1", "a2", "a3"]
    names = ["Heinz Tomato Ketchup 460ML", "Hellmann's Mayo 450ml", "Coca-Cola 1.5L"]
    embedder = _StubEmbedder(dim=8)

    original = HybridMatcher(ids=ids, names=names, alpha=0.5, top_k_each=10, embedder=embedder)
    path = tmp_path / "hybrid"
    original.save(path)
    assert (tmp_path / "hybrid.matcher.json").exists()

    restored = HybridMatcher.load(path, embedder=embedder)
    query = "HEINZ KETCHUP - TOMATO (460 ml)"
    assert restored.match(query) == original.match(query)


def test_hybrid_matcher_load_restores_alpha_and_top_k(tmp_path):
    embedder = _StubEmbedder(dim=8)
    original = HybridMatcher(
        ids=["a1", "a2"], names=["Foo Bar", "Baz Qux"], alpha=0.3, top_k_each=7, embedder=embedder
    )
    path = tmp_path / "hybrid2"
    original.save(path)

    restored = HybridMatcher.load(path, embedder=embedder)
    assert restored.alpha == 0.3
    assert restored.top_k_each == 7


def test_hybrid_matcher_load_without_embedder_arg_constructs_a_fresh_one(tmp_path, monkeypatch):
    """Omitting ``embedder=`` at load time must fall back to ``matcher.hybrid.Embedder``."""
    import matcher.hybrid as hybrid_module

    embedder = _StubEmbedder(dim=8)
    original = HybridMatcher(ids=["a1"], names=["Solo Product"], embedder=embedder)
    path = tmp_path / "hybrid3"
    original.save(path)

    monkeypatch.setattr(hybrid_module, "Embedder", lambda: embedder)
    restored = HybridMatcher.load(path)
    assert restored.embedder is embedder
