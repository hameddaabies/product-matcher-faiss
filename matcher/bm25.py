"""BM25 candidate generator over product names.

Simple whitespace tokenizer — for production you'd plug a proper tokenizer
(lowercasing, stripping punctuation, handling units like '460ml').
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


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
