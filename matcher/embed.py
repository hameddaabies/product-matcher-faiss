"""Embedding wrapper. Normalizes vectors for cosine-similarity via inner product."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors shape (n, dim)."""
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def encode_batches(
        self, texts: Iterable[str], batch_size: int = 1024
    ) -> Iterator[np.ndarray]:
        """Stream ``texts`` through :meth:`encode`, yielding one block per batch.

        :meth:`encode` holds the catalog in memory twice: once as the input
        list and again as the ``(n, dim)`` result. At 10M SKUs and 768 dims
        that result alone is ~30 GB. This consumes ``texts`` lazily — a
        generator over a DB cursor or a JSONL reader works — and yields
        ``(<=batch_size, dim)`` blocks, so peak memory is one batch rather
        than the whole catalog. Feed each block straight into
        :meth:`HnswIndex.add`, which appends.

        The final block is short when the input length is not a multiple of
        ``batch_size``; empty input yields nothing.
        """
        assert batch_size >= 1, f"batch_size must be >= 1, got {batch_size}"
        return self._encode_batches(iter(texts), batch_size)

    def _encode_batches(self, it: Iterator[str], batch_size: int) -> Iterator[np.ndarray]:
        while batch := list(islice(it, batch_size)):
            yield self.encode(batch)
