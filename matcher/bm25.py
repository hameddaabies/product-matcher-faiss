"""BM25 candidate generator over product names.

The tokenizer preserves unit expressions — both integer-unit ("460ml", "300W")
and decimal-unit ("1.5L", "2.5kg") — as single tokens.  A naive alphanumeric
split would break "1.5L" into ["1", "5l"], losing the quantity+unit signal.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

# Match decimal-or-integer number immediately followed by a unit string first,
# then fall back to any alphanumeric run.  Order matters: the unit branch must
# come before the plain-alphanumeric branch so "1.5L" is captured whole.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?[A-Za-z]+|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class Bm25Index:
    def __init__(self, ids: list[str], texts: list[str]) -> None:
        assert len(ids) == len(texts)
        self.ids = ids
        self.tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:k]
        max_score = max(float(scores[order[0]]), 1e-9) if len(order) else 1.0
        return [(self.ids[i], float(scores[i]) / max_score) for i in order]
