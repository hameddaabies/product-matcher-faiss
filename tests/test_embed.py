"""Tests for the streaming encode API.

A stub subclass supplies the vectors so these tests do not depend on
sentence-transformers having downloaded a model.
"""

from __future__ import annotations

import numpy as np
import pytest

from matcher.embed import Embedder


class _StubEmbedder(Embedder):
    """Embedder with the model replaced by a constant-vector encoder."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.batches.append(list(texts))
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        vecs[:, 0] = 1.0
        return vecs


def test_encode_batches_splits_input_with_short_final_block():
    emb = _StubEmbedder()
    blocks = list(emb.encode_batches([f"p{i}" for i in range(7)], batch_size=3))
    assert [b.shape for b in blocks] == [(3, 4), (3, 4), (1, 4)]
    assert emb.batches == [["p0", "p1", "p2"], ["p3", "p4", "p5"], ["p6"]]


def test_encode_batches_pulls_only_one_batch_at_a_time():
    """The point of the API: a lazy source must not be drained up front."""
    consumed: list[int] = []

    def source():
        for i in range(6):
            consumed.append(i)
            yield f"p{i}"

    blocks = _StubEmbedder().encode_batches(source(), batch_size=2)
    next(blocks)
    assert consumed == [0, 1]
    next(blocks)
    assert consumed == [0, 1, 2, 3]


def test_encode_batches_empty_input_yields_nothing():
    emb = _StubEmbedder()
    assert list(emb.encode_batches([], batch_size=4)) == []
    assert emb.batches == []


def test_encode_batches_rejects_nonpositive_batch_size():
    """Validation is eager — a generator would defer it to the first next()."""
    with pytest.raises(AssertionError):
        _StubEmbedder().encode_batches(["a"], batch_size=0)
