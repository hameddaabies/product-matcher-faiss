"""FAISS HNSW index with product-ID mapping.

We use IndexHNSWFlat + an external id->row dict so we can delete / update
individual items later if we want to (pure FAISS HNSW doesn't support removal).
"""

from __future__ import annotations

import faiss
import numpy as np


class HnswIndex:
    def __init__(
        self,
        dim: int,
        *,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self._ids: list[str] = []

    def add(self, ids: list[str], vecs: np.ndarray) -> None:
        assert vecs.shape[1] == self.dim, f"vec dim {vecs.shape[1]} != {self.dim}"
        self.index.add(vecs)
        self._ids.extend(ids)

    def search(self, query_vecs: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        """Return top-k (id, cosine_score) per query row."""
        scores, rows = self.index.search(query_vecs, k)
        out: list[list[tuple[str, float]]] = []
        for score_row, row_row in zip(scores, rows):
            hits: list[tuple[str, float]] = []
            for s, r in zip(score_row, row_row):
                if r == -1:
                    continue
                hits.append((self._ids[r], float(s)))
            out.append(hits)
        return out
