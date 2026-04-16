"""Hybrid matcher — union candidate sets, score, rank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import Bm25Index
from .embed import Embedder
from .index import HnswIndex


@dataclass
class MatchHit:
    id: str
    name: str
    score: float
    bm25_score: float
    cosine_score: float


class HybridMatcher:
    """Wraps embedding + HNSW + BM25 into a single match() call."""

    def __init__(
        self,
        *,
        ids: list[str],
        names: list[str],
        alpha: float = 0.5,
        top_k_each: int = 20,
        embedder: Embedder | None = None,
    ) -> None:
        assert len(ids) == len(names)
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.top_k_each = top_k_each
        self._id_to_name = dict(zip(ids, names))

        self.embedder = embedder or Embedder()
        self.hnsw = HnswIndex(self.embedder.dim)
        self.hnsw.add(ids, self.embedder.encode(names))
        self.bm25 = Bm25Index(ids, names)

    def match(self, query: str, top_n: int = 5) -> list[MatchHit]:
        q_vec = self.embedder.encode([query])
        semantic = self.hnsw.search(q_vec, self.top_k_each)[0]
        lexical = self.bm25.search(query, self.top_k_each)

        sem_scores = {pid: s for pid, s in semantic}
        lex_scores = {pid: s for pid, s in lexical}
        ids_union = set(sem_scores) | set(lex_scores)

        hits: list[MatchHit] = []
        for pid in ids_union:
            sem = sem_scores.get(pid, 0.0)
            lex = lex_scores.get(pid, 0.0)
            combined = self.alpha * sem + (1.0 - self.alpha) * lex
            hits.append(
                MatchHit(
                    id=pid,
                    name=self._id_to_name[pid],
                    score=combined,
                    bm25_score=lex,
                    cosine_score=sem,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_n]
