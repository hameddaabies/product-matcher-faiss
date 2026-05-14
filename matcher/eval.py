"""Evaluation harness — precision@1 and recall@5 against a labeled set.

    python -m matcher.eval

Loads ``retailer_a`` as the catalog, runs each ``retailer_b`` name through
the matcher, and compares the ranked candidate IDs against the gold pairs
in ``fixtures/gold_pairs.json``.  The metric primitives are exposed as
plain functions so they can be unit-tested without loading an embedder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

from .hybrid import HybridMatcher


def precision_at_1(retrieved_ids: list[str], gold_id: str) -> float:
    """1.0 iff the top-ranked candidate is the gold match."""
    return 1.0 if retrieved_ids and retrieved_ids[0] == gold_id else 0.0


def recall_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> float:
    """1.0 iff the gold match appears anywhere in the top-k candidates."""
    assert k > 0
    return 1.0 if gold_id in retrieved_ids[:k] else 0.0


class _AlphaTunable(Protocol):
    alpha: float

    def match(self, query: str, top_n: int = ...): ...


def sweep_alpha(
    matcher: _AlphaTunable,
    queries: dict[str, str],
    gold: dict[str, str],
    alphas: Iterable[float],
    top_n: int = 5,
) -> list[tuple[float, float, float]]:
    """Sweep the hybrid-mix weight and report (alpha, p@1, r@5) per value.

    Mutates ``matcher.alpha`` between runs.  Embeddings, BM25 indices, and the
    HNSW graph are built once at matcher construction, so the sweep only pays
    for the linear-combine + sort step per alpha — cheap enough to grid-search
    densely on a small labeled set.
    """
    rows: list[tuple[float, float, float]] = []
    for a in alphas:
        matcher.alpha = a
        p1 = 0.0
        r5 = 0.0
        for qid, gold_id in gold.items():
            retrieved = [h.id for h in matcher.match(queries[qid], top_n=top_n)]
            p1 += precision_at_1(retrieved, gold_id)
            r5 += recall_at_k(retrieved, gold_id, 5)
        n = len(gold)
        rows.append((a, p1 / n, r5 / n))
    return rows


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "fixtures"
    retailer_a = _load(root / "retailer_a.json")
    retailer_b = _load(root / "retailer_b.json")
    gold = {row["query_id"]: row["match_id"] for row in _load(root / "gold_pairs.json")}

    matcher = HybridMatcher(
        ids=[r["id"] for r in retailer_a],
        names=[r["name"] for r in retailer_a],
        alpha=0.5,
        top_k_each=10,
    )
    name_by_b = {r["id"]: r["name"] for r in retailer_b}

    p1_total = 0.0
    r5_total = 0.0
    n = 0
    for query_id, gold_id in gold.items():
        hits = matcher.match(name_by_b[query_id], top_n=5)
        retrieved = [h.id for h in hits]
        p1_total += precision_at_1(retrieved, gold_id)
        r5_total += recall_at_k(retrieved, gold_id, 5)
        n += 1

    print(f"queries:     {n}")
    print(f"precision@1: {p1_total / n:.3f}")
    print(f"recall@5:    {r5_total / n:.3f}")


if __name__ == "__main__":
    main()
