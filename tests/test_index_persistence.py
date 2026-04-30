"""Tests for HnswIndex.save / HnswIndex.load.

Synthetic random vectors are used so the test does not depend on
sentence-transformers being downloaded.
"""

from __future__ import annotations

import numpy as np

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
