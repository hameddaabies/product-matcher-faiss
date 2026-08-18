"""Hybrid matcher — union candidate sets, score, rank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def rank_hits(hits: list[MatchHit]) -> list[MatchHit]:
    """Order candidates best-first, breaking score ties on ``id``.

    Score alone is not a total order. Cross-retailer catalogs routinely carry
    two rows under the same product name, and those score bitwise-identically
    on both the lexical and the semantic side. Because
    :meth:`HybridMatcher.match` pools candidates through a ``set`` union, a
    score-only sort leaves tied candidates in ``set`` iteration order — which
    depends on the per-process ``PYTHONHASHSEED`` and so changes between runs.
    ``best_match`` would then commit to a different product id each run, and a
    threshold swept on one run would not reproduce on the next. Ids are unique
    within the pool, so ``(-score, id)`` is a stable total order.
    """
    return sorted(hits, key=lambda h: (-h.score, h.id))


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
        return rank_hits(hits)[:top_n]

    def best_match(self, query: str, threshold: float = 0.0) -> MatchHit | None:
        """Return the single best candidate, or ``None`` if it scores below ``threshold``.

        The first-class accept/decline decision the README's step 5 describes:
        tune ``threshold`` from a labelled PR sweep (``python -m matcher.eval``),
        then call this for a committed match-or-nothing answer instead of an
        always-populated top-N list. Default ``threshold=0.0`` accepts any
        non-empty result, preserving the old "always return the nearest" shape.
        """
        return accept_top1(self.match(query, top_n=1), threshold)

    def save(self, path: str | Path) -> None:
        """Persist everything needed to restore matching without re-embedding.

        Writes the FAISS binary + id list (via :meth:`HnswIndex.save`)
        alongside a ``<path>.matcher.json`` sidecar holding ``alpha``,
        ``top_k_each``, and the id→name mapping. BM25 rebuilds from those
        names on :meth:`load` — cheap, no model involved — so only the
        embedding *model* needs loading to serve new queries, not the whole
        catalog re-embedded. This is the restart path the README's "Restarts"
        note promises; ``HnswIndex.save``/``load`` alone leaves BM25 and the
        id→name lookup for the caller to reconstruct by hand.
        """
        path = Path(path)
        self.hnsw.save(path)
        _matcher_path(path).write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "top_k_each": self.top_k_each,
                    "id_to_name": self._id_to_name,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path, *, embedder: Embedder | None = None) -> "HybridMatcher":
        """Reconstruct a :class:`HybridMatcher` previously written by :meth:`save`.

        Restores the FAISS HNSW index from disk and rebuilds BM25 from the
        saved names, skipping catalog re-embedding entirely. Pass ``embedder``
        to reuse an already-loaded model; otherwise a fresh one is loaded.
        """
        path = Path(path)
        state = json.loads(_matcher_path(path).read_text())
        id_to_name: dict[str, str] = state["id_to_name"]

        obj = cls.__new__(cls)
        obj.alpha = state["alpha"]
        obj.top_k_each = state["top_k_each"]
        obj._id_to_name = id_to_name
        obj.embedder = embedder or Embedder()
        obj.hnsw = HnswIndex.load(path)
        obj.bm25 = Bm25Index(list(id_to_name.keys()), list(id_to_name.values()))
        return obj


def _matcher_path(path: Path) -> Path:
    return path.with_name(path.name + ".matcher.json")
