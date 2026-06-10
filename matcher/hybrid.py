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


def accept_top1(hits: list[MatchHit], threshold: float) -> MatchHit | None:
    """Apply the accept/decline decision to a *ranked* candidate list.

    Returns the top-ranked hit when its score clears ``threshold`` (inclusive),
    else ``None`` — modelling "no confident match exists in the catalog" rather
    than forcing the query onto its nearest neighbour. ``hits`` must already be
    sorted best-first (as :meth:`HybridMatcher.match` returns them); only the
    rank-1 candidate is considered, mirroring ``matcher.eval.pr_curve``.
    """
    if not hits:
        return None
    top = hits[0]
    return top if top.score >= threshold else None


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

    def best_match(self, query: str, threshold: float = 0.0) -> MatchHit | None:
        """Return the single best candidate, or ``None`` if it scores below ``threshold``.

        The first-class accept/decline decision the README's step 5 describes:
        tune ``threshold`` from a labelled PR sweep (``python -m matcher.eval``),
        then call this for a committed match-or-nothing answer instead of an
        always-populated top-N list. Default ``threshold=0.0`` accepts any
        non-empty result, preserving the old "always return the nearest" shape.
        """
        return accept_top1(self.match(query, top_n=1), threshold)
