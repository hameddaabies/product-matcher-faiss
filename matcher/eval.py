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


def pr_curve(
    scored_predictions: list[tuple[float, bool]],
    thresholds: Iterable[float],
) -> list[tuple[float, float, float]]:
    """Precision/recall of the accept-the-top-1 decision over a threshold sweep.

    Each entry of ``scored_predictions`` is one ``(top1_score, is_correct)``
    pair per query that *has* a gold match: ``top1_score`` is the matcher's
    score for its rank-1 candidate, ``is_correct`` whether that candidate is
    the gold match.  For a threshold ``t`` the matcher *accepts* its top-1 iff
    ``top1_score >= t``, giving:

      * TP — accepted and correct
      * FP — accepted but wrong
      * FN — rejected (a gold match existed but we declined to commit)

    Returns ``(threshold, precision, recall)`` per threshold, where
    ``precision = TP / (TP + FP)`` and ``recall = TP / (queries with gold)``.
    Precision is reported as 1.0 when nothing is accepted (vacuously no wrong
    accepts) so the high-threshold tail of the curve is well-defined.  This is
    the tool the README's "pick the threshold from a labelled set" note needs.
    """
    total_with_gold = len(scored_predictions)
    rows: list[tuple[float, float, float]] = []
    for t in thresholds:
        tp = sum(1 for s, correct in scored_predictions if s >= t and correct)
        fp = sum(1 for s, correct in scored_predictions if s >= t and not correct)
        accepted = tp + fp
        precision = tp / accepted if accepted else 1.0
        recall = tp / total_with_gold if total_with_gold else 0.0
        rows.append((t, precision, recall))
    return rows


def best_threshold(
    pr_rows: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """Pick the F1-maximizing accept threshold from a :func:`pr_curve` sweep.

    Collapses the precision/recall sweep into a single recommended operating
    point: the threshold whose ``(precision, recall)`` maximizes the harmonic
    mean ``F1 = 2·P·R / (P + R)``. Ties break toward the *higher* threshold,
    preferring the more conservative (higher-precision) accept rule when two
    thresholds score equally.

    Rows where ``precision + recall == 0`` score ``F1 = 0.0`` (the metric is
    undefined there — typically the high-threshold tail that accepts nothing).
    Returns ``(threshold, f1)``. Raises ``ValueError`` on an empty sweep, since
    there is then no operating point to choose.
    """
    if not pr_rows:
        raise ValueError("pr_rows is empty; run pr_curve over a non-empty threshold set")
    best_t = pr_rows[0][0]
    best_f1 = -1.0
    for t, precision, recall in pr_rows:
        denom = precision + recall
        f1 = 2.0 * precision * recall / denom if denom else 0.0
        if f1 > best_f1 or (f1 == best_f1 and t >= best_t):
            best_f1 = f1
            best_t = t
    return best_t, best_f1


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
    top1: list[tuple[float, bool]] = []
    for query_id, gold_id in gold.items():
        hits = matcher.match(name_by_b[query_id], top_n=5)
        retrieved = [h.id for h in hits]
        p1_total += precision_at_1(retrieved, gold_id)
        r5_total += recall_at_k(retrieved, gold_id, 5)
        top1.append((hits[0].score, hits[0].id == gold_id) if hits else (0.0, False))
        n += 1

    print(f"queries:     {n}")
    print(f"precision@1: {p1_total / n:.3f}")
    print(f"recall@5:    {r5_total / n:.3f}")

    curve = pr_curve(top1, [i / 10 for i in range(11)])
    print("\nthreshold  precision  recall   (accept top-1 iff score >= threshold)")
    for t, prec, rec in curve:
        print(f"  {t:.1f}       {prec:.3f}     {rec:.3f}")

    rec_t, rec_f1 = best_threshold(curve)
    print(f"\nrecommended threshold: {rec_t:.1f}  (F1={rec_f1:.3f})")


if __name__ == "__main__":
    main()
