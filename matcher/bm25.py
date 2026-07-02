"""BM25 candidate generator over product names.

The tokenizer preserves unit expressions — both integer-unit ("460ml", "300W")
and decimal-unit ("1.5L", "2.5kg") — as single tokens.  A naive alphanumeric
split would break "1.5L" into ["1", "5l"], losing the quantity+unit signal.

It also collapses a number separated from a known unit by whitespace
("460 ml", "1.5 L", "400 grams") into the same single token, so retailers
that write "460 ml" match retailers that write "460ML".  Only a curated
allowlist of unit suffixes is collapsed; arbitrary "<number> <word>"
sequences like "100 Pack" are left split.
"""

from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

# Curated unit allowlist.  Kept conservative: ambiguous bare letters like
# "m" (meter / million) and "in" (preposition) are intentionally excluded.
_UNITS = (
    "ml", "l", "cl", "liter", "liters", "litre", "litres",
    "kg", "g", "mg", "gram", "grams", "kilo", "kilos",
    "oz", "lb", "lbs", "pound", "pounds",
    "w", "kw", "mw", "kwh", "wh", "mah", "ah",
    "hz", "khz", "mhz", "ghz",
    "cm", "mm", "ft", "inch", "inches",
)
# Sort by length descending so the regex prefers "liters" over "l".
_UNIT_ALT = "|".join(sorted(_UNITS, key=len, reverse=True))
_UNIT_SPACE_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s+({_UNIT_ALT})\b",
    re.IGNORECASE,
)

# Match decimal-or-integer number immediately followed by a unit string first,
# then fall back to any letter-or-digit run.  Order matters: the unit branch
# must come before the plain branch so "1.5L" is captured whole.  The plain
# branch uses ``[^\W_]+`` rather than ``[A-Za-z0-9]+`` so unicode letters
# survive — "Crème", "Jalapeño", "Café" are preserved instead of being
# shattered into ASCII fragments.  Units themselves stay ASCII-only.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?[A-Za-z]+|[^\W_]+")


def tokenize(text: str) -> list[str]:
    # Fold to NFC first: an accented letter can arrive either precomposed
    # ("é" = U+00E9) or decomposed ("e" + U+0301 combining accent).  The two
    # are visually identical but byte-different, and the combining mark is not
    # a word character — so decomposed "Crème" would tokenize to
    # ["cre", "me"] (mark splits the word and the accent is dropped) while the
    # precomposed form yields ["crème"].  Two retailers on different
    # normalization forms would then never match; NFC makes the split stable.
    text = unicodedata.normalize("NFC", text)
    normalized = _UNIT_SPACE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)
    return [t.lower() for t in _TOKEN_RE.findall(normalized)]


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
