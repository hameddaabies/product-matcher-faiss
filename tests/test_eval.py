"""Unit tests for the precision@1 / recall@k primitives.

These don't load an embedder — they exercise the pure ranking-metric
functions on hand-rolled retrieved-id lists.
"""

from dataclasses import dataclass

import pytest

from matcher.eval import (
    best_threshold,
    pr_curve,
    precision_at_1,
    recall_at_k,
    sweep_alpha,
)


def test_precision_at_1_hit():
    assert precision_at_1(["a1", "a2", "a3"], "a1") == 1.0


def test_precision_at_1_miss_when_gold_is_second():
    assert precision_at_1(["a2", "a1", "a3"], "a1") == 0.0


def test_precision_at_1_empty_retrieved():
    assert precision_at_1([], "a1") == 0.0


def test_recall_at_k_hit_at_top():
    assert recall_at_k(["a1", "a2", "a3"], "a1", k=5) == 1.0


def test_recall_at_k_hit_within_window():
    assert recall_at_k(["a2", "a3", "a1", "a4"], "a1", k=5) == 1.0


def test_recall_at_k_miss_outside_window():
    """Gold at position 6 must miss when k=5."""
    retrieved = ["a2", "a3", "a4", "a5", "a6", "a1"]
    assert recall_at_k(retrieved, "a1", k=5) == 0.0


def test_recall_at_k_truncates_to_window():
    """Recall at k=1 only counts the very first hit."""
    assert recall_at_k(["a2", "a1"], "a1", k=1) == 0.0
    assert recall_at_k(["a1", "a2"], "a1", k=1) == 1.0


# ---------------------------------------------------------------------------
# sweep_alpha — exercise without loading an embedder via a stub matcher.
# ---------------------------------------------------------------------------


@dataclass
class _Hit:
    id: str


class _StubMatcher:
    """Returns a different ranking at each alpha so the sweep is observable."""

    def __init__(self, rankings: dict[float, list[str]]) -> None:
        self.alpha = 0.0
        self._rankings = rankings

    def match(self, query: str, top_n: int = 5):
        return [_Hit(pid) for pid in self._rankings[self.alpha][:top_n]]


def test_sweep_alpha_reports_metrics_per_alpha():
    rankings = {
        0.0: ["a1", "a2", "a3"],   # gold at rank 1 → p@1=1.0
        0.5: ["a2", "a1", "a3"],   # gold at rank 2 → p@1=0.0, r@5=1.0
        1.0: ["a3", "a4", "a5"],   # gold absent    → p@1=0.0, r@5=0.0
    }
    matcher = _StubMatcher(rankings)
    rows = sweep_alpha(
        matcher,
        queries={"q1": "anything"},
        gold={"q1": "a1"},
        alphas=[0.0, 0.5, 1.0],
    )
    assert rows == [(0.0, 1.0, 1.0), (0.5, 0.0, 1.0), (1.0, 0.0, 0.0)]


def test_sweep_alpha_mutates_matcher_alpha():
    """The helper must actually flip matcher.alpha — that's how it tunes."""
    matcher = _StubMatcher({0.25: ["a1"], 0.75: ["a1"]})
    sweep_alpha(matcher, queries={"q": "x"}, gold={"q": "a1"}, alphas=[0.25, 0.75])
    assert matcher.alpha == 0.75


# ---------------------------------------------------------------------------
# pr_curve — accept/reject threshold sweep for the match decision.
# ---------------------------------------------------------------------------


def test_pr_curve_recall_drops_as_threshold_rises():
    """Raising the accept threshold drops a correct-but-low-score match."""
    preds = [(0.9, True), (0.4, True)]  # two correct top-1s, different scores
    rows = pr_curve(preds, [0.3, 0.5])
    assert rows == [(0.3, 1.0, 1.0), (0.5, 1.0, 0.5)]


def test_pr_curve_precision_penalizes_accepted_wrong_match():
    """A wrong top-1 above threshold is a false positive → precision 0.5."""
    rows = pr_curve([(0.8, True), (0.8, False)], [0.5])
    assert rows == [(0.5, 0.5, 0.5)]


def test_pr_curve_precision_is_one_when_nothing_accepted():
    """High-threshold tail: no accepts → vacuous precision 1.0, recall 0.0."""
    rows = pr_curve([(0.4, True), (0.3, False)], [0.9])
    assert rows == [(0.9, 1.0, 0.0)]


def test_pr_curve_empty_predictions():
    rows = pr_curve([], [0.5])
    assert rows == [(0.5, 1.0, 0.0)]


# ---------------------------------------------------------------------------
# best_threshold — collapse the PR sweep into one recommended operating point.
# ---------------------------------------------------------------------------


def test_best_threshold_picks_f1_maximizing_row():
    """The interior threshold with balanced P/R beats the lopsided endpoints."""
    rows = [(0.3, 0.5, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 0.5)]
    t, f1 = best_threshold(rows)
    assert t == 0.5
    assert f1 == 1.0


def test_best_threshold_breaks_ties_toward_higher_threshold():
    """Equal F1 → prefer the more conservative (higher-precision) cutoff."""
    rows = [(0.2, 0.5, 1.0), (0.7, 0.5, 1.0)]  # identical F1 at both rows
    t, _ = best_threshold(rows)
    assert t == 0.7


def test_best_threshold_zero_f1_when_nothing_accepted():
    """High-threshold tail (recall 0) is F1-undefined → scored as 0.0, not a div-by-zero."""
    t, f1 = best_threshold([(0.9, 1.0, 0.0)])
    assert t == 0.9
    assert f1 == 0.0


def test_best_threshold_empty_sweep_raises():
    with pytest.raises(ValueError):
        best_threshold([])
