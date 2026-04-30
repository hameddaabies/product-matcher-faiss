"""FAISS HNSW index with product-ID mapping.

We use IndexHNSWFlat + an external id->row dict so we can delete / update
individual items later if we want to (pure FAISS HNSW doesn't support removal).
"""

from __future__ import annotations

import json
from pathlib import Path

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

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index + external ID list to disk.

        Writes two files: ``<path>`` (FAISS binary, via ``faiss.write_index``)
        and ``<path>.ids.json`` (row→id mapping). Both are required by
        :meth:`load`; losing either makes the index unusable.
        """
        path = Path(path)
        faiss.write_index(self.index, str(path))
        _ids_path(path).write_text(json.dumps(self._ids))

    @classmethod
    def load(cls, path: str | Path) -> "HnswIndex":
        """Reconstruct an :class:`HnswIndex` previously written by :meth:`save`.

        HNSW build parameters (``M``, ``efConstruction``) are restored from the
        FAISS binary; ``efSearch`` is preserved as it was at save time.
        """
        path = Path(path)
        index = faiss.read_index(str(path))
        ids = json.loads(_ids_path(path).read_text())
        obj = cls.__new__(cls)
        obj.index = index
        obj.dim = index.d
        obj._ids = list(ids)
        return obj


def _ids_path(path: Path) -> Path:
    return path.with_name(path.name + ".ids.json")
